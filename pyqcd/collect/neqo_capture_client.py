#!/usr/bin/env python3
"""Usage: module [options] -- [NEQO_ARGS]...

Run neqo-client with the specified arguments provided to NEQO_ARGS,
and capture and filter the trace to the IP and ports specified in
the output from NEQO.

The call to neqo-client inherits the process's environment.

Options:
    --pcap-file filename
        Write the decrypted and filtered pcap data in pcapng format to
        filename.
"""
import os
import re
import contextlib
import subprocess
from subprocess import PIPE
from tempfile import NamedTemporaryFile
from dataclasses import dataclass
from typing import Optional, Union, Tuple, NamedTuple
from ipaddress import IPv4Address, IPv6Address, ip_address

import doceasy
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


def run_neqo(neqo_args, keylog_file: str) -> NeqoResult:
    """Run NEQO and record its output and related files."""
    with NamedTemporaryFile(mode="rt") as output_file:
        env = os.environ.copy()
        env["SSLKEYLOGFILE"] = keylog_file

        args = " ".join(neqo_args)
        subprocess.run(
            f"set -o pipefail; neqo-client {args} | tee {output_file.name}",
            # We need the shell so that we can do redirection
            shell=True, executable='bash', env=env,
            # Raise an exception on program error codes
            check=True,
        )

        return NeqoResult(
            output=output_file.read()
        )


def _embed_tls_keys(pcap_bytes: bytes, keylog_file: str) -> bytes:
    """Embed TLS keys and return a pcapng file with the embedded
    secrets.
    """
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


def _filter_packets(pcap_bytes: bytes, neqo_output: str) -> bytes:
    """Filter the packets to the IPs and ports extracted from
    neqo_output and return a pcapng with the data.
    """
    (local, remote) = extract_endpoints(neqo_output)

    dst_ver = "ipv6" if remote.ip.version == 6 else "ip"
    result = subprocess.run([
        "tshark",
        "-r", "-",
        "-w", "-", "-F", "pcapng",
        # Filter to packets from the remote ip and local port
        # Exclude the local IP as this may change depending on the vantage point
        "-Y", (f"{dst_ver}.addr=={remote.ip} and udp.port=={remote.port} "
               f"and udp.port=={local.port}")
    ], check=True, input=pcap_bytes, stdout=PIPE)

    # Ensure that the result is neither none nor empty
    assert result.stdout
    return result.stdout


def main(neqo_args, pcap_file: Optional[str]):
    """Run neqo-client while capturing the traffic."""
    with NamedTemporaryFile(mode="r") as keylog:
        with tcpdump(capture_filter="udp") as sniffer:
            result = run_neqo(neqo_args, keylog_file=keylog.name)

        if pcap_file is not None:
            pcap_bytes = _filter_packets(sniffer.pcap(), result.output)
            pcap_bytes = _embed_tls_keys(pcap_bytes, keylog.name)

            with open(pcap_file, mode="wb") as pcap:
                pcap.write(pcap_bytes)


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, doceasy.Schema({
        "NEQO_ARGS": [str],
        "--pcap-file": doceasy.Or(None, str)
    }, ignore_extra_keys=True)))
