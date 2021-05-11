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
from dataclasses import dataclass
from typing import Union, Tuple, NamedTuple
from ipaddress import IPv4Address, IPv6Address, ip_address

from lab.sniffer import TCPDumpPacketSniffer


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


@dataclass
class NeqoResult:
    """Various output files and data from a NEQO run."""
    output: str
    is_success: bool


def run_neqo(
    neqo_args, keylog_file: str, ignore_errors: bool,
    stdout=None, stderr=None, env=None,
) -> NeqoResult:
    """Run NEQO and record its output and related files."""
    is_success = True

    with contextlib.ExitStack() as stack:
        output_file = stack.enter_context(NamedTemporaryFile(mode="rt"))

        if isinstance(stdout, (str, Path)):
            stdout = stack.enter_context(open(stdout, mode="w"))
        if isinstance(stderr, (str, Path)):
            stderr = stack.enter_context(open(stderr, mode="w"))

        process_env = os.environ.copy()
        if env is not None:
            process_env.update(env)
        process_env["SSLKEYLOGFILE"] = keylog_file

        args = " ".join([(f"'{x}'" if " " in x else x) for x in neqo_args])
        try:
            subprocess.run(
                f"set -o pipefail; neqo-client {args} | tee {output_file.name}",
                # We need the shell so that we can do redirection
                shell=True, executable='bash', env=process_env,
                # Raise an exception on program error codes
                check=True,
                # Redirect stdout and stderr to the files if set
                stdout=stdout, stderr=stderr
            )
            if ignore_errors:
                print(">>> SUCCESS <<<",
                      file=(sys.stdout if stdout is None else stdout))
        except subprocess.CalledProcessError as err:
            if not ignore_errors:
                raise
            logging.getLogger(__name__).error(err)
            print(">>> FAILURE <<<",
                  file=(sys.stdout if stdout is None else stdout))
            is_success = False

        return NeqoResult(output=output_file.read(), is_success=is_success)


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


def main(
    neqo_args,
    pcap_file: Union[str, Path, None],
    ignore_errors: bool,
    stdout=None,
    stderr=None,
    env=None,
):
    """Run neqo-client while capturing the traffic."""
    with NamedTemporaryFile(mode="r") as keylog:
        with tcpdump(capture_filter="udp") as sniffer:
            result = run_neqo(
                neqo_args, keylog_file=keylog.name, ignore_errors=ignore_errors,
                stdout=stdout, stderr=stderr, env=env,
            )

        if pcap_file is not None:
            pcap_bytes = _filter_packets(sniffer.pcap(), result.output,
                                         ignore_errors=(not result.is_success))
            pcap_bytes = _embed_tls_keys(pcap_bytes, keylog.name)

            with open(pcap_file, mode="wb") as pcap:
                pcap.write(pcap_bytes)
