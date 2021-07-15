"""Usage: module [options] PREFIX INFILE...

Filter browser URL request logs and extract dependency graphs.

INFILE is a gzipped json stream containing the browser fetch results,
one per line.  PREFIX is the prefix to append to the filename of each
of the output dependency graphs and defaults to 'x'.  Each graph is a
CSV adjancency list of the pairs (dependency URL, URL), and are written
to the files PREFIX000.csv, PREFIX001.csv ...

Options:
    --no-origin-filter
        Disable filtering by the web-page origin, leaving graph nodes
        from all origins.

    -v, --verbose
        Output more log messages.
"""
import json
import logging
import fileinput
import urllib.parse
from collections import Counter
from typing import Set, Optional, Iterator, Dict, List, Any
from pathlib import Path
import networkx as nx

import common
from common import doceasy

#: The minimum number of URLs in each dependency graph
N_MININMUM_URLS = 1
#: Allow error codes for too many requests (429) and server timeout (522) since
#: these are transient.
ALLOWED_HTTP_ERRORS = [429, 522, ]

_LOGGER = logging.getLogger("url-dep-graph")


def origin(url: str) -> str:
    """Return the origin of the URL."""
    parts = urllib.parse.urlsplit(url)
    return f"{parts[0]}://{parts[1]}"


class InvalidGraphError(RuntimeError):
    """Raised when construction would result in an empty graph."""


def _rget(mapping, keys: List[Any], default=None):
    """Recrusvie get."""
    assert isinstance(keys, list)
    for key in keys[:-1]:
        mapping = mapping.get(key, {})
    return mapping.get(keys[-1], default)


class _DependencyGraph:
    def __init__(self, browser_log, origin_: Optional[str] = None):
        self.logs = browser_log
        self.origin = origin_
        self.graph = nx.DiGraph()
        #: A mapping of URLs to node_ids to account for redirections
        self._url_node_ids: Dict[str, List[str]] = dict()
        self._ignored_requests: Set[str] = set()

        self._construct()

    def _construct(self):
        msgs = self.logs["http_trace"]
        msgs.sort(key=lambda msg: msg["timestamp"])

        for msg in msgs:
            msg = msg["message"]["message"]

            if msg["method"] == "Network.requestWillBeSent":
                self._handle_request(msg)
            elif msg["method"] == "Network.responseReceived":
                self._handle_response(msg)
            elif msg["method"] == "Network.dataReceived":
                self._handle_data(msg)
            else:
                continue

        if self_loops := list(nx.nodes_with_selfloops(self.graph)):
            raise InvalidGraphError(f"Graph contains self loops: {self_loops}")
        if loops := list(nx.simple_cycles(self.graph)):
            raise InvalidGraphError(f"Graph contains loops: {loops}")

        if self.origin is not None:
            to_drop = [node for node, node_origin in self.graph.nodes("origin")
                       if node_origin != self.origin]
            self.graph.remove_nodes_from(to_drop)

            if len(self.graph) == 0:
                raise InvalidGraphError(
                    f"Origin filtering would result in an empty graph:"
                    f" {self.origin}")

        self.graph = nx.relabel_nodes(self.graph, mapping={
            node: i for (i, node) in enumerate(self.graph)
        })

    def to_json(self) -> str:
        """Convert the graph to json."""
        return json.dumps(nx.node_link_data(self.graph), indent=2)

    def roots(self) -> List[str]:
        """Return the roots of the graph."""
        return [node for node, degree in self.graph.in_degree() if degree == 0]

    def _add_node(self, node_id, url, type_):
        if url not in self._url_node_ids:
            self._url_node_ids[url] = [node_id, ]
        elif node_id not in self._url_node_ids[url]:
            self._url_node_ids[url].append(node_id)
        # If this is a redirection it will change the details of the node
        self.graph.add_node(
            node_id, url=url, done=False, type=type_, origin=origin(url),
            content_length=None, data_length=0,
        )

    def _handle_data(self, msg):
        assert msg["method"] == "Network.dataReceived"
        node_id = msg["params"]["requestId"]
        if (
            node_id in self._ignored_requests
            or node_id not in self.graph.nodes
        ):
            return

        self.graph.nodes[node_id]["data_length"] += _rget(
            msg, ["params", "dataLength"], 0)

    def _handle_response(self, msg):
        assert msg["method"] == "Network.responseReceived"
        node_id = msg["params"]["requestId"]
        if node_id in self._ignored_requests:
            return
        self.graph.nodes[node_id]["done"] = True

        size = _rget(
            msg, ["params", "response", "headers", "content-length"], None
        )
        if size is not None:
            self.graph.nodes[node_id]["content_length"] = int(size)

    def _find_node_by_url(self, url: str) -> Optional[str]:
        """Find the most recent node associated with a get request."""
        # TODO: Check these nodes for which is completed?
        if not (node_ids := self._url_node_ids.get(url, [])):
            return None
        if nid := next(
            (nid for nid in node_ids if self.graph.nodes[nid]["done"]), None
        ):
            return nid
        return node_ids[-1]

    def _add_origin_dependency(self, dep: str, node_id):
        """Add a dependency for node_id to the root with the same
        origin as dep.
        """
        root_node = next((root for root in self.roots()
                          if self.graph.nodes[root]["origin"] == origin(dep)),
                         None)
        assert root_node is not None
        self.graph.add_edge(root_node, node_id)

    def _add_dependency(self, dep, node_id) -> bool:
        if not dep.startswith("http"):
            return True
        if dep_node := self._find_node_by_url(dep):
            self.graph.add_edge(dep_node, node_id)
            return True
        return False

    def _handle_request(self, msg):
        assert msg["method"] == "Network.requestWillBeSent"

        request = msg["params"]["request"]
        node_id = msg["params"]["requestId"]

        if (request["method"] != "GET"
                or not request["url"].startswith("https://")):
            self._ignored_requests.add(node_id)
            return

        self._add_node(node_id, request["url"], msg["params"]["type"])

        if msg["params"]["documentURL"] != request["url"]:
            if not self._add_dependency(msg["params"]["documentURL"], node_id):
                _LOGGER.debug(
                    "Unable to find documentURL dependency of %r: %r",
                    request["url"], msg["params"]["documentURL"])

        if initiator_url := msg["params"]["initiator"].get("url", None):
            if not self._add_dependency(initiator_url, node_id):
                _LOGGER.debug(
                    "Unable to find initiator dependency of %r: %r",
                    request["url"], initiator_url)

        if stack := msg["params"]["initiator"].get("stack", None):
            for stack_frame in stack["callFrames"]:
                if not self._add_dependency(stack_frame["url"], node_id):
                    _LOGGER.debug(
                        "Unable to find documentURL dependency of %r: %r",
                        request["url"], stack_frame["url"])


def extract_graphs(
    fetch_output_generator, use_origin: bool = True
) -> Iterator[nx.DiGraph]:
    """Filter and generate non-empty graphs from the input
    generated of fetch results.
    """
    seen_urls = set()
    dropped_urls: Dict[str, Counter] = {
        "duplicate": Counter(),
        "disconnected": Counter(),
        "insufficient": Counter(),
        "empty": Counter()
    }

    for result in fetch_output_generator:
        url = result["final_url"] or result["url"]

        if result["status"] != "success":
            _LOGGER.debug(
                "Dropping %r with a status of %r.", url, result["status"])
            continue
        if not url.startswith("https"):
            _LOGGER.debug("Dropping %r as it is not HTTPS.", url)
            continue
        if url in seen_urls:
            _LOGGER.debug("Dropping %r as it was already encountered.", url)
            dropped_urls["duplicate"][url] += 1
            continue

        try:
            graph = _DependencyGraph(
                result, origin(url) if use_origin else None)
        except InvalidGraphError as err:
            _LOGGER.debug("Dropping %r: %s", url, err)
            dropped_urls["empty"][url] += 1
            continue

        if len(graph.roots()) > 1:
            _LOGGER.debug("Dropping %r as it is disconnected.", url)
            dropped_urls["disconnected"][url] += 1
        elif len(graph.graph.nodes) < N_MININMUM_URLS:
            _LOGGER.debug("Dropping %r as it has only %d/%d required URLs.",
                          url, len(graph.graph.nodes), N_MININMUM_URLS)
            dropped_urls["insufficient"][url] += 1
        else:
            yield graph
            seen_urls.add(url)

    for type_, counters in dropped_urls.items():
        _LOGGER.debug("Dropped %s urls: %s", type_, dict(counters))
        _LOGGER.info("Dropped %s urls: %d", type_, sum(counters.values()))


def main(infile: List[str], prefix: str, verbose: bool, no_origin_filter: bool):
    """Filter browser URL request logs and extract dependency graphs."""
    common.init_logging(verbosity=int(verbose) + 1)
    _LOGGER.info("Running with arguments: %s.", locals())

    file_id = -1
    with fileinput.input(files=infile, openhook=fileinput.hook_compressed) \
            as json_lines:
        results = (json.loads(line) for line in json_lines)

        for file_id, graph in enumerate(
            extract_graphs(results, use_origin=not no_origin_filter)
        ):
            path = Path(f"{prefix}{file_id:04d}.json")
            if not path.is_file():
                path.write_text(graph.to_json())
            else:
                _LOGGER.info("Refusing to overwrite: %s", path)
    _LOGGER.info("Script complete. Extracted %d dependency graphs.", file_id+1)


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, {
        "PREFIX": doceasy.Or(str, doceasy.Use(lambda _: "x")),
        "INFILE": [str],
        "--no-origin-filter": bool,
        "--verbose": bool,
    }))
