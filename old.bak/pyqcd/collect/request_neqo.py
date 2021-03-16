""" Calls the neqo-client in the neqo-qcd repository to perform QUIC requests
    both shaped and unshaped to provided urls
"""

# pylint: disable=too-many-arguments
import os
import csv
import sys
import time
import signal
from contextlib import contextmanager
# determine OS
from sys import platform
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from typing import Optional
from invoke import task

NEQO_LOG = "neqo_transport=info,debug"
LOCAL_NETLOC = "{}:7443".format(
    "localhost" if platform == "linux" else "host.docker.internal")
DEFAULT_NETLOC = {
    "vanilla": "vanilla.neqo-test.com:7443",
    "weather": "weather.neqo-test.com:7443",
}

@contextmanager
def capture(
    conn,
    out_pcap: str,
    interface: Optional[str] = "lo" if platform=="linux" else "lo0",
    filter_: str = "port 7443"
):
    """Create a context manager for capturing a trace with tshark.
    """
    iface_flag = ("" if platform=="linux" else f"-i en2") if interface is None else f"-i {interface}"
    promise = conn.run(
        f"tshark {iface_flag} -f '{filter_}' -w '{out_pcap}'",
        asynchronous=True, echo=True)
    time.sleep(3)

    try:
        yield promise
    finally:
        os.kill(promise.runner.process.pid, signal.SIGTERM)
        promise.join()


def _load_urls(name: str, netloc: str, local: bool):
    netloc = netloc or (LOCAL_NETLOC if local else DEFAULT_NETLOC[name])
    urls = Path(f"urls/{name}.txt").read_text().split()
    urls = [urlunsplit(urlsplit(url)._replace(scheme="https", netloc=netloc))
            for url in urls]
    return urls


@task
def neqo_request(conn, name, netloc=None, local=False, shaping=True, log=False):
    """Request the URLs associated with a domain.

    Use the default local netloc by passing specifying local as true.

    The argument netloc is an optional network locality such as
    localhost:443 that replaces the default.
    """
    urls = ' '.join(_load_urls(name, netloc, local))

    client_binary = {
        "linux": "../target/debug/neqo-client",
        "darwin": ("docker exec -w $PWD -e LD_LIBRARY_PATH=$PWD/../target"
                   "/debug/build/neqo-crypto-044e50838ff4228a/out/dist/Debug"
                   "/lib/ -e SSLKEYLOGFILE=out.log")
                    + (" -e RUST_LOG=")+(NEQO_LOG if log else "")
                    + (" -e CSDEF_NO_SHAPING=")+("" if shaping else "yes")
                    + (" neqo-qcd ../target/debug/neqo-client")
    }[platform]

    conn.run(f"{client_binary} {urls}", echo=True, env={
        "SSLKEYLOGFILE": "out.log",
        "RUST_LOG": NEQO_LOG if log else "",
        "CSDEF_NO_SHAPING": "" if shaping else "yes"
    })