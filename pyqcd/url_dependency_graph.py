import json
import gzip
import logging
from typing import Set, Tuple, Optional, Dict

import pandas as pd

import pyqcd


_LOGGER = logging.getLogger("url-dep-graph")


def is_target(node, edges) -> bool:
    """Return true iff the node is a target in the graph."""
    for (src, target) in edges:
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


def _same_domain(edge: Tuple[Optional[str], str], base_url: str) -> bool:
    """Return true iff the target and dependency URLs are on the same
    domain as the base_url or the target is and there is no dependency.
    """
    dependency, target = edge

    if not target.startswith(base_url):
        _LOGGER.debug("Dropping edge %r for url %r due to target at a "
                      "different domain.", edge, base_url)
        return False

    if not (dependency is None or dependency.startswith(base_url)):
        _LOGGER.debug("Dropping edge %r for url %r due to a dependency from a "
                      "different domain.", edge, base_url)
        return False
    return True


def _extract_connection(log_message, connections):
    if log_message["method"] != "Network.responseReceived":
        return

    response = log_message["params"]["response"]
    urls_on_conn = connections.setdefault(response["connectionId"], set())

    assert response["url"] not in urls_on_conn
    urls_on_conn.add(response["url"])


def to_adjacency_list(fetch_output) -> Set[Tuple[Optional[str], str]]:
    """Return an adjacency list of (dependency, url) for the provided
    fetch results.
    """
    assert fetch_output["status"] == "success"

    edges: Set[Tuple[Optional[str], str]] = set()
    connections: Dict[int, Set[str]] = dict()

    for log_message in fetch_output["http_trace"]:
        _extract_edges(log_message["message"]["message"], edges)
        _extract_connection(log_message["message"]["message"], connections)
    assert edges

    breakpoint()

    # If the original URL is a prefix of the final URL, use the original URL
    base_url = fetch_output["url"]
    if not fetch_output["final_url"].startswith(base_url):
        base_url = fetch_output["final_url"]
        # Add a root edge for our new base URL, so that we can start the
        # collection from this URL instead
        edges.add((None, base_url))

    final_edges = set(edge for edge in edges if _same_domain(edge, base_url))
    if not final_edges:
        _LOGGER.warning("No edges left after domain filtering for url: %r, "
                        "final_url: %r",
                        fetch_output["url"], fetch_output["final_url"])

    return _check_graph(final_edges)


def _check_graph(edges):
    """Check our assumptions about the resulting graph from the adjacency
    list, namely:
        - the resulting graph is rooted at the None node,
    """
    # Skip checking if the edge list is empty
    if not edges:
        return edges

    # Check that the graph is rooted at the None node
    dep_only = set(s for (s, t) in edges) - set(t for (s, t) in edges)
    assert len(dep_only) == 1
    assert None in dep_only

    # Check that there is only URL which is initiated by the "user"
    n_initiated = sum(1 for e in edges if e[0] is None)
    assert n_initiated == 1

    return edges


def main(infile: str, prefix: str, log_level: int = 1):
    pyqcd.init_logging(log_level)

    with gzip.open(infile, mode="r") as json_lines:
        entries = filter(
            lambda x: x["status"] == "success",
            (json.loads(line) for line in json_lines)
        )

        for fetch_output in entries:
            if edges := to_adjacency_list(fetch_output):
                frame = pd.DataFrame(edges, columns=["dependency", "url"])


main("results/determine-url-deps/browser-logs.json.gz", "/tmp/x")
