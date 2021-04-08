"""Usage: module [options] [INFILE [PREFIX]]

Filter browser URL request logs and extract dependency graphs.

INFILE is a gzipped json stream containing the browser fetch results,
one per line.  PREFIX is the prefix to append to the filename of each
of the output dependency graphs and defaults to 'x'.  Each graph is a
CSV adjancency list of the pairs (dependency URL, URL), and are written
to the files PREFIX000.csv, PREFIX001.csv ...

Options:
    -v, --verbose
        Output more log messages.
"""
import json
import gzip
import logging
import urllib.parse
from collections import Counter
from typing import Set, Optional, Iterator, Dict, List
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


class _DependencyGraph:
    def __init__(self, browser_log, origin_: Optional[str] = None):
        self.logs = browser_log
        self.origin = origin_
        self.graph = nx.DiGraph()
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
            else:
                continue
        assert len(list(nx.simple_cycles(self.graph))) == 0
        assert len(list(nx.nodes_with_selfloops(self.graph))) == 0

        if origin is not None:
            to_drop = [node for node, node_origin in self.graph.nodes("origin")
                       if node_origin != self.origin]
            self.graph.remove_nodes_from(to_drop)
            assert len(self.graph) > 0

        self.graph = nx.relabel_nodes(self.graph, mapping={
            node: i for (i, node) in enumerate(self.graph)
        })

    def to_json(self) -> str:
        """Convert the graph to json."""
        return json.dumps(nx.node_link_data(self.graph), indent=2)

    def roots(self) -> List[str]:
        """Return the roots of the graph."""
        return [node for node, degree in self.graph.in_degree() if degree == 0]

    def _handle_response(self, msg):
        assert msg["method"] == "Network.responseReceived"
        node_id = msg["params"]["requestId"]
        if node_id in self._ignored_requests:
            return
        self.graph.nodes[node_id]["done"] = True

    def _find_node_by_url(self, url: str) -> Optional[str]:
        """Find the most recent node associated with a get request."""
        nodes = [node for (node, data) in self.graph.nodes(data=True)
                 if data['url'] == url and data["done"]]
        if not nodes:
            return None
        # Return the last node that was added
        return nodes[-1]

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

        if request["method"] != "GET" or not request["url"].startswith("http"):
            self._ignored_requests.add(node_id)
            return

        self.graph.add_node(
            node_id, url=request["url"], done=False, type=msg["params"]["type"],
            origin=origin(request["url"]))

        if msg["params"]["documentURL"] != request["url"]:
            if not self._add_dependency(msg["params"]["documentURL"], node_id):
                _LOGGER.warning(
                    "Unable to find documentURL dependency of %r: %r",
                    request["url"], msg["params"]["documentURL"]
                )

        if initiator_url := msg["params"]["initiator"].get("url", None):
            assert self._add_dependency(initiator_url, node_id)

        if stack := msg["params"]["initiator"].get("stack", None):
            for stack_frame in stack["callFrames"]:
                assert self._add_dependency(stack_frame["url"], node_id)


def extract_graphs(fetch_output_generator) -> Iterator[nx.DiGraph]:
    """Filter and generate non-empty graphs from the input
    generated of fetch results.
    """
    seen_urls = set()
    dropped_urls: Dict[str, Counter] = {
        "duplicate": Counter(),
        "disconnected": Counter(),
        "insufficient": Counter(),
    }

    for result in fetch_output_generator:
        url = result["final_url"]

        if result["status"] != "success":
            _LOGGER.debug(
                "Dropping %r with a status of %r.", url, result["status"])
            continue
        if not url.startswith("https"):
            _LOGGER.warning("Dropping %r as it is not HTTPS.", url)
            continue
        if url in seen_urls:
            _LOGGER.debug("Dropping %r as it was already encountered.", url)
            dropped_urls["duplicate"][url] += 1
            continue

        graph = _DependencyGraph(result, origin(url))

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

    for type_ in ("duplicate", "disconnected", "insufficient"):
        counters = dropped_urls[type_]
        _LOGGER.debug("Dropped %s urls: %s", type_, dict(counters))
        _LOGGER.info("Dropped %s urls: %d", type_, sum(counters.values()))


def main(infile: str, prefix: str, verbose: bool):
    """Filter browser URL request logs and extract dependency graphs."""
    common.init_logging(int(verbose) + 1)
    _LOGGER.info("Running with arguments: %s.", locals())

    file_id = -1
    with gzip.open(infile, mode="r") as json_lines:
        results = (json.loads(line) for line in json_lines)

        for file_id, graph in enumerate(extract_graphs(results)):
            Path(f"{prefix}{file_id:03d}.json").write_text(graph.to_json())
    _LOGGER.info("Script complete. Extracted %d dependency graphs.", file_id+1)


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, {
        "INFILE": str,
        "PREFIX": doceasy.Or(str, doceasy.Use(lambda _: "x")),
        "--verbose": bool,
    }))
