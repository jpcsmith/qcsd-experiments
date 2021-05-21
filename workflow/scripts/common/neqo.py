"""Wrapper to run neqo-client binary."""
import os
import re
import sys
import logging
import contextlib
import subprocess
from pathlib import Path
from subprocess import PIPE
from tempfile import NamedTemporaryFile
from typing import Union, Tuple, NamedTuple
from ipaddress import IPv4Address, IPv6Address, ip_address

from lab.sniffer import TCPDumpPacketSniffer

_LOGGER = logging.getLogger(__name__)


def run(
    neqo_args,
    pcap_file: Union[str, Path, None],
    ignore_errors: bool,
    stdout=None,
    stderr=None,
    env=None,
) -> bool:
    """Run neqo-client while capturing the traffic.

    Return True iff the capture was a success.
    """
    with NamedTemporaryFile(mode="r") as keylog:
        with tcpdump(capture_filter="udp") as sniffer:
            (output, is_success) = _run_neqo(
                neqo_args, keylog_file=keylog.name, ignore_errors=ignore_errors,
                stdout=stdout, stderr=stderr, env=env,
            )

        if pcap_file is not None:
            pcap_bytes = _filter_packets(
                sniffer.pcap(), output, ignore_errors=(not is_success))
            pcap_bytes = _embed_tls_keys(pcap_bytes, keylog.name)

            with open(pcap_file, mode="wb") as pcap:
                pcap.write(pcap_bytes)

        return is_success


def is_run_successful(stdout_file: Union[str, Path]) -> bool:
    """Return true if the output of neqo run indicates a success."""
    # We check for the tag ">>> SUCCESS <<<" at the end of the file
    with open(stdout_file, mode="rb") as file_:
        file_.seek(-30, os.SEEK_END)
        return b">>> SUCCESS <<<" in file_.read()


def is_run_almost_successful(
    stdout_file: Union[str, Path], remaining: int
) -> bool:
    """Return true iff the run was successful or had at most
    `remaining` packets remaining to collect.
    """
    if is_run_successful(stdout_file):
        return True

    tag = "[FlowShaper] final remaining packets:"
    with open(stdout_file, mode="r") as stdout:
        count = min(
            (int(line[len(tag):]) for line in stdout if line.startswith(tag)),
            default=None
        )
        return count is not None and count <= remaining


@contextlib.contextmanager
def tcpdump(*args, **kwargs):
    """Sniff packets within a context manager using tcpdump.
    """
    sniffer = TCPDumpPacketSniffer(*args, **kwargs)
    sniffer.start()
    try:
        yield sniffer
    finally:
        sniffer.stop()


def _run_neqo(
    neqo_args, keylog_file: str, ignore_errors: bool,
    stdout=None, stderr=None, env=None,
):
    """Run NEQO and record its output and related files."""
    with contextlib.ExitStack() as stack:
        output_file = stack.enter_context(NamedTemporaryFile(mode="rt"))

        if isinstance(stdout, (str, Path)):
            stdout = stack.enter_context(open(stdout, mode="w"))
        if isinstance(stderr, (str, Path)):
            stderr = stack.enter_context(open(stderr, mode="w"))

        env = env or {}
        env["SSLKEYLOGFILE"] = keylog_file
        process_env = os.environ.copy()
        process_env.update(env)
        _LOGGER.info("Running NEQO with additional env vars: %s", env)

        is_success = False
        args = " ".join([(f"'{x}'" if " " in x else x) for x in neqo_args])
        cmd = f"set -o pipefail; neqo-client {args} | tee {output_file.name}"
        _LOGGER.info("Running NEQO with command: %r", cmd)
        try:
            subprocess.run(
                cmd,
                # We need the shell so that we can do redirection
                shell=True, executable='bash', env=process_env,
                # Raise an exception on program error codes
                check=True,
                # Redirect stdout and stderr to the files if set
                stdout=stdout, stderr=stderr
            )
            is_success = True
        except subprocess.CalledProcessError as err:
            if not ignore_errors:
                raise
            _LOGGER.error(err)
            is_success = False

        if ignore_errors:
            print(">>> SUCCESS <<<" if is_success else ">>> FAILURE <<<",
                  file=(sys.stdout if stdout is None else stdout))
        return (output_file.read(), is_success)


def _embed_tls_keys(pcap_bytes: bytes, keylog_file: str) -> bytes:
    """Embed TLS keys and return a pcapng file with the embedded
    secrets.
    """
    with open(keylog_file, "r") as keylog:
        assert len(keylog.read().strip()) > 0

    result = subprocess.run([
        "editcap",
        "--inject-secrets", f"tls,{keylog_file}",
        "-F", "pcapng",
        "-", "-"
    ], check=True, input=pcap_bytes, stdout=PIPE)

    assert result.stdout is not None
    return result.stdout


Endpoint = NamedTuple('Endpoint', [
    ('ip', Union[IPv4Address, IPv6Address]), ('port', int)
])


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


def _filter_packets(
    pcap_bytes: bytes, neqo_output: str, ignore_errors: bool
) -> bytes:
    """Filter the packets to the IPs and ports extracted from
    neqo_output and return a pcapng with the data.
    """
    try:
        (local, remote) = extract_endpoints(neqo_output)
    except ValueError:
        if not ignore_errors:
            raise
        display_filter = "udp.port"
    else:
        dst_ver = "ipv6" if remote.ip.version == 6 else "ip"
        # Filter to packets from the remote ip and local port
        # Exclude the local IP as this may change depending on the vantage point
        display_filter = (
            f"{dst_ver}.addr=={remote.ip} and udp.port=={remote.port}"
            f" and udp.port=={local.port}"
        )

    result = subprocess.run([
        "tshark", "-r", "-", "-w", "-", "-F", "pcapng", "-Y", display_filter
    ], check=True, input=pcap_bytes, stdout=PIPE)

    # Ensure that the result is neither none nor empty
    assert result.stdout
    return result.stdout
