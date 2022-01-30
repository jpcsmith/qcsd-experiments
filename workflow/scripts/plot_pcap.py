#!/usr/bin/env python3
import io
import sys
import ipaddress
from subprocess import run
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main(pcap_file: str, log_file: str):
    plot_data = _to_frame(pcap_file, log_file)

    fig, axes = plt.subplots(1, 1)
    sns.scatterplot(
        data=plot_data, x="time", y="dsize", ax=axes,
        hue="port", style="port",
    )
    fig.savefig("/tmp/plot.png", dpi=300)


def _to_frame(pcap_file: str, log_filename: str) -> pd.DataFrame:
    with open(log_filename, mode="r") as log_file:
        addrs = [
            line.split("->")[1].strip().rsplit(":", maxsplit=1)[0].strip("[]")
            for line in log_file if "H3 Client connecting" in line
        ]

    ipv4_addrs = {addr for addr in addrs if _is_ipv4(addr)}
    ipv6_addrs = {addr for addr in addrs if not _is_ipv4(addr)}

    ip_filters = []
    if ipv4_addrs:
        ip_filters.append("ip.addr in {{ {} }}".format(" ".join(ipv4_addrs)))
    if ipv6_addrs:
        ip_filters.append("ipv6.addr in {{ {} }}".format(" ".join(ipv6_addrs)))
    filter_string = "udp and ({})".format(" or ".join(ip_filters))

    stdout = run([
        "tshark", "-r", pcap_file, "-Tfields",
        "-e", "udp.srcport", "-e", "udp.dstport", "-e", "frame.time_relative",
        "-e", "udp.length",
        filter_string
    ], check=True, capture_output=True).stdout

    plot_data = pd.read_csv(
        io.BytesIO(stdout), sep="\t", header=None,
        names=["srcport", "dstport", "time", "size"]
    )
    mask = (plot_data["srcport"] == 443)

    plot_data.loc[~mask, "port"] = plot_data[~mask]["srcport"]
    plot_data.loc[mask, "port"] = plot_data[mask]["dstport"]
    plot_data["port"] = plot_data["port"].astype(int)

    plot_data["dsize"] = plot_data["size"]
    plot_data.loc[plot_data["srcport"] == 443, "dsize"] *= -1

    return plot_data


def _is_ipv4(addr: str) -> bool:
    address = ipaddress.ip_address(addr)
    return isinstance(address, ipaddress.IPv4Address)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
