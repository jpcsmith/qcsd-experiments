"""Wrapper to run neqo-client binary."""
import os
import logging
import functools
import contextlib
import subprocess
from subprocess import PIPE, Popen
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    Union, Tuple, NamedTuple, Optional, Dict, List, BinaryIO, Callable
)
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
    filter_server_address: bool = False,
) -> Tuple[subprocess.CompletedProcess, Optional[bytes]]:
    """Run neqo-client and capture the communication traffic."""
    if filter_server_address and stdout != PIPE:
        raise ValueError("'filter_neqo_ips' requires piped stdout")

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
        if pcap is not None:
            # None is the only case in which we do not use the PCAP
            assert pcap == PIPE or isinstance(pcap, (str, Path))
            pcap_bytes = sniffer.pcap()
            assert pcap_bytes is not None

            if Path(keylog).read_text().strip():
                # Filtering must occur before embedding of the secrets, as
                # tshark cannot write a PCAP with secrets
                if filter_server_address:
                    pcap_bytes = _filter_neqo_ips(pcap_bytes, result.stdout)
                pcap_bytes = common.pcap.embed_tls_keys(pcap_bytes, keylog)

            if isinstance(pcap, (str, Path)):
                Path(pcap).write_bytes(pcap_bytes)
                pcap_bytes = None

    return (result, pcap_bytes)


def run_alongside(
    neqo_args: List[str],
    other_subprocess: Callable[[], subprocess.Popen],
    *,
    stdout: FileLike = None,
    stderr: FileLike = None,
    env: Optional[Dict] = None,
    neqo_exe: Optional[List[str]] = None,
    tcpdump_kw: Optional[Dict] = None,
    timeout: Optional[float] = None,
    skip_neqo: bool = False,
) -> bytes:
    """Run both neqo-client and other_subprocess and capture the traffic.

    The other_subprocess is called within the packet capture context
    and should immediately return a Popen object. Neqo is then called
    after which the Popen is allowed to complete and then its return
    code is checked and a CalledProcessError is raised for non-zero
    error codes.

    Timeout is not applied to the other_subprocess. If skip_neqo is True,
    then this only run the other_subprocess.

    Returns only the pcap bytes.
    """
    env = env or dict()
    tcpdump_kw = tcpdump_kw or dict()
    tcpdump_kw.setdefault("capture_filter", "udp")

    with contextlib.ExitStack() as stack:
        if "SSLKEYLOGFILE" not in env:
            keylog = stack.enter_context(TempFile(mode="r")).name
        keylog = env.setdefault("SSLKEYLOGFILE", keylog)

        if isinstance(stderr, (str, Path)):
            stderr = stack.enter_context(open(stderr, mode="wb"))
        if isinstance(stdout, (str, Path)):
            stdout = stack.enter_context(open(stdout, mode="wb"))

        with tcpdump(**tcpdump_kw) as sniffer:
            popen = other_subprocess()
            try:
                if not skip_neqo:
                    _run_neqo(
                        neqo_args, check=True, stdout=stdout, stderr=stderr,
                        env=env, neqo_exe=neqo_exe, timeout=timeout
                    )
                else:
                    _LOGGER.debug("Skipping NEQO")
            except subprocess.SubprocessError:
                popen.terminate()
                raise
            finally:
                # Always wait for the process to terminate, whether we're
                # leaving by an exception or not
                (outs, errs) = popen.communicate()
            if popen.returncode != 0:
                raise subprocess.CalledProcessError(
                    popen.returncode, ["???"], output=outs, stderr=errs)

        pcap_bytes = sniffer.pcap()
        assert pcap_bytes is not None
    return pcap_bytes


def _filter_neqo_ips(pcap, stdout) -> bytes:
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8")
    (local, remote) = extract_endpoints(stdout)

    dst_ver = "ipv6" if remote.ip.version == 6 else "ip"
    # Filter to packets from the remote ip and local port
    # Exclude the local IP as this may change depending on the vantage point
    display_filter = (
        f"{dst_ver}.addr=={remote.ip} and udp.port=={remote.port}"
        f" and udp.port=={local.port}"
    )
    return common.pcap.filter_pcap(pcap, display_filter)


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


def _parse_endpoint(addr: str) -> Endpoint:
    if addr[:3] in ["V4(", "V6("]:
        addr = addr[3:].replace(")", "")
    if addr.startswith("["):
        addr = addr[1:].replace("]", "")

    ipaddr, port = addr.rsplit(":", maxsplit=1)
    return Endpoint(ip_address(ipaddr), int(port))


def extract_endpoints(neqo_output: str) -> Tuple[Endpoint, Endpoint]:
    """Extract the endpoints from the log output."""
    conn_lines = [line for line in neqo_output.split("\n")
                  if line.startswith("H3 Client connecting:")]

    if len(conn_lines) == 0:
        raise ValueError("Output does not contain an endpoint message.")
    if len(conn_lines) > 1:
        raise ValueError(f"Multiple connections in Neqo output: {conn_lines}.")

    # Split the line into the separate IPs
    laddr, _, raddr = conn_lines[0].split(" ")[3:]
    return _parse_endpoint(laddr), _parse_endpoint(raddr)
