#!/usr/bin/env python3
"""Usage: neqo [options] -- [NEQO_ARGS]...

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
import sys
import contextlib
import subprocess
from subprocess import PIPE
from tempfile import NamedTemporaryFile
from dataclasses import dataclass
from typing import Optional, Union, Tuple, NamedTuple
from ipaddress import IPv4Address, IPv6Address, ip_address

from common import doceasy, neqo


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


def run_neqo(neqo_args, keylog_file: str, ignore_errors: bool) -> NeqoResult:
    """Run NEQO and record its output and related files."""
    is_success = True

    with NamedTemporaryFile(mode="rt") as output_file:
        env = os.environ.copy()
        env["SSLKEYLOGFILE"] = keylog_file

        args = " ".join([(f"'{x}'" if " " in x else x) for x in neqo_args])
        try:
            subprocess.run(
                f"set -o pipefail; neqo-client {args} | tee {output_file.name}",
                # We need the shell so that we can do redirection
                shell=True, executable='bash', env=env,
                # Raise an exception on program error codes
                check=True,
            )
            if ignore_errors:
                print(">>> SUCCESS <<<")
        except subprocess.CalledProcessError as err:
            if not ignore_errors:
                raise
            print(err, file=sys.stderr)
            print(">>> FAILURE <<<")
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


def main(neqo_args, pcap_file: Optional[str], ignore_errors: bool):
    """Run neqo-client while capturing the traffic."""
    with NamedTemporaryFile(mode="r") as keylog:
        with tcpdump(capture_filter="udp") as sniffer:
            result = run_neqo(neqo_args, keylog_file=keylog.name,
                              ignore_errors=ignore_errors)

        if pcap_file is not None:
            pcap_bytes = _filter_packets(sniffer.pcap(), result.output,
                                         ignore_errors=(not result.is_success))
            pcap_bytes = _embed_tls_keys(pcap_bytes, keylog.name)

            with open(pcap_file, mode="wb") as pcap:
                pcap.write(pcap_bytes)


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, doceasy.Schema({
        "NEQO_ARGS": [str],
        "--pcap-file": doceasy.Or(None, str),
        "--ignore-errors": bool,
    }, ignore_extra_keys=True)))
