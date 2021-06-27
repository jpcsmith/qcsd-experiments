"""Wrapper to run neqo-client binary."""
import os
import re
import logging
import functools
import contextlib
import subprocess
from subprocess import PIPE, Popen
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
    timeout: Optional[float] = None,
) -> Tuple[subprocess.CompletedProcess, Optional[bytes]]:
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
            result = _run_neqo(
                neqo_args, check=check, stdout=stdout, stderr=stderr, env=env,
                neqo_exe=neqo_exe, timeout=timeout
            )

        pcap_bytes: Optional[bytes] = None
        if result.returncode == 0 and pcap is not None:
            # None is the only case in which we do not use the PCAP
            assert pcap == PIPE or isinstance(pcap, (str, Path))
            pcap_bytes = common.pcap.embed_tls_keys(sniffer.pcap(), keylog)
            if isinstance(pcap, (str, Path)):
                Path(pcap).write_bytes(pcap_bytes)
                pcap_bytes = None

    return (result, pcap_bytes)


def _run_neqo(
    neqo_args: List[str],
    *,
    check: bool,
    stdout: Union[None, BinaryIO, int],
    stderr: Union[None, BinaryIO, int],
    env: Dict,
    neqo_exe: Optional[List[str]],
    timeout: Optional[float],
):
    """Run NEQO in a subprocess.

    Enables terminating subprocesses in a docker-safe manner while
    immitating subprocess.run.
    """
    _LOGGER.debug("Running NEQO with additional env vars: %s", env)
    # Update the process environment with the received env
    env = {**os.environ, **env}

    cmd = (neqo_exe or ["neqo-client"]) + neqo_args
    _LOGGER.debug("Running NEQO with command: %s", cmd)

    with Popen(cmd, env=env, stdout=stdout, stderr=stderr) as process:
        try:
            out, err = process.communicate(timeout=timeout)  # type:ignore
        except subprocess.TimeoutExpired:
            # Try SIGTERM before resorting to SIGKILL. If neqo is being run in a
            # docker container, we cannot just kill the docker run process as
            # the signal will then not be propagated.
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        except:  # noqa=E922 Including KeyboardInterrupt, communicate handled it
            process.kill()
            raise

        retcode = process.poll()
        assert retcode is not None, "didnt we kill the process?"

        if check and retcode:
            raise subprocess.CalledProcessError(
                retcode, process.args, output=out, stderr=err)
    return subprocess.CompletedProcess(process.args, retcode, out, err)


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
