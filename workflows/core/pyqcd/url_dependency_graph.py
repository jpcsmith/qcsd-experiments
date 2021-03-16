"""Usage: module [options] [INFILE [PREFIX]]

Filter browser URL request logs and extract dependency graphs.

INFILE is a gzipped json stream containing the browser fetch results,
one per line.  PREFIX is the prefix to append to the filename of each
of the output dependency graphs and defaults to 'x'.  Each graph is a
CSV adjancency list of the pairs (dependency URL, URL), and are written
to the files PREFIX000.csv, PREFIX001.csv ...

Options:
    --groupby-connection
        Group the URLs for each request by the logged connection ID
        instead of by the URL's origin as is done in neqo-client.

    -v, --verbose
        Output more log messages.
"""
import json
import gzip
import logging
import urllib.parse
from collections import Counter, defaultdict
from typing import Set, Tuple, Optional, Dict, Iterator, Any
from itertools import chain

import doceasy
import pandas as pd

import pyqcd

#: The minimum number of URLs in each dependency graph
N_MININMUM_URLS = 6
#: Allow error codes for too many requests (429) and server timeout (522) since
#: these are transient.
ALLOWED_HTTP_ERRORS = [429, 522, ]

_LOGGER = logging.getLogger("url-dep-graph")


def is_target(node, edges) -> bool:
    """Return true iff the node is a target in the graph."""
    for (_, target) in edges:
        if node == target:
            return True
    return False


def _extract_edges(log_message, edges):
    """Add an edge for the referer, initiator, and any scripts executed."""
    if log_message["method"] != "Network.requestWillBeSent":
        return

    request = log_message["params"]["request"]
    if request["method"] != "GET":
        return

    # Add an edge for the initiator if specified
    initiator = log_message["params"]["initiator"]
    if "url" in initiator:
        assert initiator["url"] is not None
        edges.add((initiator["url"], request["url"]))

    # Add for each JS script that was traversed as they're indirect requirements
    if "stack" in initiator:
        for stack_entry in initiator["stack"]["callFrames"]:
            edges.add((stack_entry["url"], request["url"]))

    # Add an edge for the referer defaulting to None if no edge was added for
    # this request URL yet.
    if "Referer" in request["headers"]:
        edges.add((request["headers"]["Referer"], request["url"]))
    elif not is_target(request["url"], edges):
        edges.add((None, request["url"]))


def is_valid_edge(edge, valid_nodes) -> bool:
    """Returns true iff all non-None edges are in valid_nodes."""
    dependency, target = edge
    return ((dependency is None or dependency in valid_nodes)
            and (target in valid_nodes))


class _UrlGrouper:
    """Group URLs by either their connection id or origin."""
    def __init__(self, groupby_connection: bool):
        self._connections: Dict[Any, Set[str]] = defaultdict(set)
        self._use_origin = not groupby_connection

    def maybe_add(self, log_message):
        """Track the URL in the log message if it represents a received
        response.
        """
        if log_message["method"] != "Network.responseReceived":
            return

        response = log_message["params"]["response"]
        if (response["status"] >= 400 and response["status"] not in
                ALLOWED_HTTP_ERRORS):
            return

        key = response["connectionId"]
        if self._use_origin:
            parse = urllib.parse.urlsplit(response["url"])
            key = (parse.scheme, parse.netloc)

        self._connections[key].add(response["url"])

    def get_group_by_origin(self, url: str) -> Set[str]:
        """Returns the group of URLs that are associated with
        the origin of provided URL.
        """
        parse = urllib.parse.urlsplit(url)

        if self._use_origin:
            return self._connections[(parse.scheme, parse.netloc)]

        # Sometimes the above URL and the URLs in the connections differ by
        # a suffix as /, /#, or /#!/ (angular), so only consider the netloc.
        base_url = "{parse.scheme}://{parse.netloc}"
        return next(urls for urls in self._connections.values()
                    if any(url.startswith(base_url) for url in urls))


def _to_adjacency_list(
    fetch_output, groupby_connection: bool
) -> Set[Tuple[Optional[str], str]]:
    """Return an adjacency list of (dependency, url) for the provided
    fetch results.
    """
    assert fetch_output["status"] == "success"

    # Set of (dependency, url) forming the edges of the graph
    edges: Set[Tuple[Optional[str], str]] = set()

    connections = _UrlGrouper(groupby_connection)

    # Dictionary of (scheme, host, port) -> URL to determine which URLs share
    # the same origin. Used instead of the connectionId since this is the method
    # that neqo-client uses.

    for log_message in fetch_output["http_trace"]:
        _extract_edges(log_message["message"]["message"], edges)
        connections.maybe_add(log_message["message"]["message"])
    assert edges

    # Select the connection group that contains the final url
    valid_nodes = connections.get_group_by_origin(fetch_output["final_url"])

    # Drop edges that contain an invalid node
    final_edges = set(e for e in edges if is_valid_edge(e, valid_nodes))

    return _check_graph(final_edges, fetch_output["final_url"])


def _check_graph(edges, url: str):
    """Check our assumptions about the resulting graph from the adjacency
    list.
    """
    # Skip checking if the edge list is empty
    if not edges:
        return edges

    root_nodes = set(s for (s, t) in edges) - set(t for (s, t) in edges)
    if len(root_nodes) > 1:
        _LOGGER.warning("Dropping %r as there are two root nodes: %r", url,
                        root_nodes)
        return set()

    # If there is a single non-None root node, add the None root node
    if len(root_nodes) == 1 and None not in root_nodes:
        root = root_nodes.pop()
        _LOGGER.debug("Adding root edge None->%s for %r", root, url)
        edges.add((None, root))

    # Check that there is only URL which is initiated by the "user"
    n_initiated = sum(1 for e in edges if e[0] is None)
    if n_initiated != 1:
        _LOGGER.warning("Dropping %r as there are %d user-initated nodes.",
                        url, n_initiated)
        return set()

    return edges


def to_adjacency_list(fetch_output_generator, **kwargs) -> Iterator[set]:
    """Filter and generate non-empty adjacency lists from the input
    generated of fetch results.
    """
    seen_urls = set()
    duplicate_counter: Counter = Counter()
    n_insufficient = 0

    for fetch_output in fetch_output_generator:
        url = fetch_output["final_url"]

        if fetch_output["status"] != "success":
            _LOGGER.debug(
                "Dropping %r with a status of %r.", url, fetch_output["status"])
            continue

        if not url.startswith("https"):
            _LOGGER.warning("Dropping %r as it is not HTTPS.", url)
            continue

        if url in seen_urls:
            _LOGGER.debug("Dropping %r as it was already encountered.", url)
            duplicate_counter[url] += 1
            continue

        if not (edges := _to_adjacency_list(fetch_output, **kwargs)):
            continue

        n_urls = len(set(chain.from_iterable(edges)))
        # The +1 accounts for the presence of the empty root URL
        if n_urls < (N_MININMUM_URLS + 1):
            _LOGGER.debug("Dropping %r as it has only %d/%d required URLs.",
                          url, n_urls, N_MININMUM_URLS)
            n_insufficient += 1
            continue

        yield edges
        seen_urls.add(url)

    _LOGGER.info("Dropped duplicates: %s.", dict(duplicate_counter))
    _LOGGER.info("Dropped %d URLs with less than the %d minimum URLs.",
                 n_insufficient, N_MININMUM_URLS)


def main(infile: str, prefix: str, verbose: bool, groupby_connection: bool):
    """Filter browser URL request logs and extract dependency graphs."""
    pyqcd.init_logging(int(verbose) + 1)
    _LOGGER.info("Running with arguments: %s.", locals())

    file_id = -1
    with gzip.open(infile, mode="r") as json_lines:
        for file_id, edges in enumerate(
            to_adjacency_list((json.loads(line) for line in json_lines),
                              groupby_connection=groupby_connection)
        ):
            frame = pd.DataFrame(edges, columns=["dependency", "url"])
            frame[["url", "dependency"]].to_csv(
                f"{prefix}{file_id:03d}.csv", header=False, index=False)

    _LOGGER.info("Script complete. Extracted %d dependency graphs.", file_id+1)


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, {
        "INFILE": str,
        "PREFIX": doceasy.Or(str, doceasy.Use(lambda _: "x")),
        "--verbose": bool,
        "--groupby-connection": bool
    }))
