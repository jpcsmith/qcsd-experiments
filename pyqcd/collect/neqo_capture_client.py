#!/usr/bin/env python3
"""Usage: module [options] -- [NEQO_ARGS]...

Options:
    --pcap-file filename
        Write the captured and filtered p

"""
import contextlib
import subprocess
from tempfile import NamedTemporaryFile

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


def main(neqo_args, pcap_file: str):
    """Run neqo-client while capturing the traffic."""
    with tcpdump(capture_filter="udp") as sniffer, \
            NamedTemporaryFile(mode="rt") as output_file:

        args = " ".join(neqo_args)
        subprocess.run(
            f"neqo-client {args} | tee {output_file.name}",
            # We need the shell so that we can do redirection
            shell=True, executable='bash',
            # Inherit the environment variables of the the python process
            env=None,
            # Raise an exception on program error codes
            check=True,
        )

    with open(pcap_file, mode="wb") as pcap:
        pcap.write(sniffer.pcap())


# TODO: Need to dump SSH keys
main(["--dummy-urls", "https://vanilla.neqo-test.com:7443/img/2nd-big-item.jpg",
      "https://vanilla.neqo-test.com:7443/css/bootstrap.min.css",
      "https://vanilla.neqo-test.com:7443/img/3rd-item.jpg",
      "https://vanilla.neqo-test.com:7443/img/4th-item.jpg",
      "https://vanilla.neqo-test.com:7443/img/5th-item.jpg",
      "https://vanilla.neqo-test.com:7443/css/bootstrap.min.css",
      "https://vanilla.neqo-test.com:7443/img/3rd-item.jpg",
      "https://vanilla.neqo-test.com:7443/img/4th-item.jpg",
      "https://vanilla.neqo-test.com:7443/img/5th-item.jpg",
      ], pcap_file="out.pcap")
