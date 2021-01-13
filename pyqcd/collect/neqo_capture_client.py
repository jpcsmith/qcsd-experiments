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
from typing import Optional

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


def _filter_packets(pcap_bytes: bytes, neqo_output: str) -> bytes:
    """Filter the packets to the IPs and ports extracted from
    neqo_output and return a pcapng with the data.
    """
    match = re.search(
        r"^H3 Client connecting: "
        r"(?P<src_ver>V[46])\(\[+(?P<src_ip>.*)\]+:(?P<src_port>\d+)\) "
        r"-> (?P<dst_ver>V[46])\(\[+(?P<dst_ip>.*)\]+:(?P<dst_port>\d+)\)$",
        neqo_output, flags=re.MULTILINE
    )
    assert match

    dst_ver = "ipv6" if match["dst_ver"] == "V6" else "ip"
    result = subprocess.run([
        "tshark",
        "-r", "-",
        "-w", "-", "-F", "pcapng",
        # Filter to packets from the remote ip and local port
        # Exclude the local IP as this may change depending on the vantage point
        "-Y", " and ".join([
            f"{dst_ver}.addr=={match['dst_ip']}",
            f"udp.port=={match['dst_port']}",
            f"udp.port=={match['src_port']}",
        ])
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
