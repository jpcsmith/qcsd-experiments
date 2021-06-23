"""Wrapper to run neqo-client binary."""
import os
import re
import logging
import functools
import contextlib
import subprocess
from subprocess import PIPE
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Union, Tuple, NamedTuple, Optional, Dict, List, BinaryIO
from ipaddress import IPv4Address, IPv6Address, ip_address

from lab.sniffer import tcpdump

import common.pcap

_LOGGER = logging.getLogger(__name__)

#: A filelike object for redirection. Accepts values of
#: subprocess.PIPE/STDERR/DEVNULL/STDOUT which are integers
FileLike = Union[None, str, Path, BinaryIO, int]

TempFile = functools.partial(NamedTemporaryFile, dir=Path.cwd())

Endpoint = NamedTuple('Endpoint', [
    ('ip', Union[IPv4Address, IPv6Address]), ('port', int)
])


class NeqoCompletedProcess(subprocess.CompletedProcess):
    # pylint: disable=too-few-public-methods
    """The result of running the Neqo client.

    Attributes:
        local_endpoint :
            The local UDP ip and port initiating the connection.
        remote_endpoint :
            The remote UDP ip and port initiating the connection.
        pcap :
            The pcapng of the captured traffic.
    """
    def __init__(self, result: subprocess.CompletedProcess):
        super().__init__(
            result.args, result.returncode, result.stdout, result.stderr
        )
        self.local_endpoint: Optional[Endpoint] = None
        self.remote_endpoint: Optional[Endpoint] = None
        self.pcap: Optional[bytes] = None


def run(
    neqo_args: List[str],
    *,
    check: bool = True,
    stdout: FileLike = None,
    stderr: FileLike = None,
    pcap: FileLike = None,
    env: Optional[Dict] = None,
    neqo_exe: Optional[List[str]] = None,
    tcpdump_kw: Optional[Dict] = None,
) -> NeqoCompletedProcess:
    """Run neqo-client and capture the communication traffic."""
    env = env or dict()
    tcpdump_kw = tcpdump_kw or dict()
    tcpdump_kw.setdefault("capture_filter", "udp")

    if (
        not isinstance(pcap, (str, Path, BinaryIO))
        and pcap != subprocess.PIPE
        and pcap is not None
    ):
        raise ValueError(f"Unsupported pcap value: {pcap!r}")

    with contextlib.ExitStack() as stack:
        if "SSLKEYLOGFILE" not in env:
            keylog = stack.enter_context(TempFile(mode="r")).name
        keylog = env.setdefault("SSLKEYLOGFILE", keylog)

        if isinstance(stderr, (str, Path)):
            stderr = stack.enter_context(open(stderr, mode="wb"))
        if isinstance(stdout, (str, Path)):
            stdout = stack.enter_context(open(stdout, mode="wb"))

        with tcpdump(**tcpdump_kw) as sniffer:
            result = _run_neqo(neqo_args, neqo_exe=neqo_exe, check=check,
                               stdout=stdout, stderr=stderr, env=env)

        # Stdout is always set by _run_neqo. Extract it and clear it if not
        # explicity requested
        stdout_data = result.stdout.decode("utf-8")
        if stdout != PIPE and stderr != subprocess.STDOUT:
            result.stdout = None
        result = NeqoCompletedProcess(result)

        # Endpoints and pcap is not attached on failure
        if result.returncode == 0:
            (result.local_endpoint, result.remote_endpoint) = \
                extract_endpoints(stdout_data)

        # None is the only case in which we do not use the PCAP
        if result.returncode == 0 and pcap is not None:
            assert pcap == PIPE or isinstance(pcap, (str, Path))
            pcap_bytes = common.pcap.embed_tls_keys(sniffer.pcap(), keylog)
            if isinstance(pcap, (str, Path)):
                Path(pcap).write_bytes(pcap_bytes)
            else:
                result.pcap = pcap_bytes
    return result


def _run_neqo(
    neqo_args: List[str],
    *,
    check: bool,
    stdout: Union[None, BinaryIO, int],
    stderr: Union[None, BinaryIO, int],
    env: Dict,
    neqo_exe: Optional[List[str]] = None,
):
    """Run NEQO and record its output and related files."""
    cmd = (neqo_exe or ["neqo-client"]) + neqo_args

    with TempFile(mode="rb") as output_file:
        _LOGGER.debug("Running NEQO with additional env vars: %s", env)
        # Update the process environment with the received env
        env = {**os.environ, **env}

        # Escape any arguments that have spaces
        cmd_str = " ".join([f"'{x}'" if " " in x else x for x in cmd])
        cmd_str = f"set -o pipefail; {cmd_str} | tee {output_file.name}"
        _LOGGER.debug("Running NEQO with command: %r", cmd_str)

        result = subprocess.run(
            cmd_str,
            # We need the shell so that we can do redirection
            shell=True, executable='bash', env=env,
            # Raise an exception on program error codes
            check=check,
            # Redirect stdout and stderr to the files if set
            stdout=stdout, stderr=stderr
        )

        if result.stdout is None:
            result.stdout = output_file.read()
        return result


def extract_endpoints(neqo_output: str) -> Tuple[Endpoint, Endpoint]:
    """Extract the endpoints from the log output."""
    conn_lines = [line for line in neqo_output.split("\n")
                  if line.startswith("H3 Client connecting:")]

    if len(conn_lines) == 0:
        raise ValueError("Output does not contain an endpoint message.")
    if len(conn_lines) > 1:
        raise ValueError(f"Multiple connections in Neqo output: {conn_lines}.")

    pattern = (r"(?:(?P<{end}ver>V4|V6)\()?"
               r"\[?(?P<{end}ip>[.\d]+|[:\dA-Fa-f]+)\]?:(?P<{end}port>\d+)"
               r"(?({end}ver)\))")
    pattern = "{} -> {}".format(
        pattern.format(end="l"), pattern.format(end="r"))
    match = re.search(pattern, conn_lines[0])

    if not match:
        raise ValueError(f"Unable to parse connection line: {conn_lines[0]}")

    return (Endpoint(ip_address(match["lip"]), int(match["lport"])),
            Endpoint(ip_address(match["rip"]), int(match["rport"])))
