#!/usr/bin/env python3
"""Usage: module [options] -- [INPUTS]...

Measures the distance between two dummy traces using Pearson correlation.
Traces are sampled into 5ms intervals.

Inputs:
    Defended Trace
    Dummy Schedule
    Dummy ids list

Options:
    --output-file filename
        Save the dummy packets trace and rolling window pearson graph
        to filename.
    --window-size int
        Size of the rolling window used to calculate Pearson's correlation
        in the traces.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import doceasy


def load_cover_trace(filename):
    data = pd.read_csv(filename)
    data = data.sort_values(by="timestamp").reset_index(drop=True)

    # Ensure that the trace starts with packet number zero
    assert data.loc[0, "packet_number"] == 0
    data["timestamp"] = (data["timestamp"] - data.loc[0, "timestamp"]) * 1e3

    # Drop all packets with packet number zero as they're only there for the starting time
    data = data[data["packet_number"] != 0].reset_index(drop=True)

    data = data.rename(columns={"timestamp": "time",
                                "chaff_traffic": "length"})
    data = data[["time", "length", "other_traffic",
                 "packet_length", "is_outgoing"]]
    data["time"] = pd.to_datetime(data["time"], unit="ms")

    data = data.groupby("is_outgoing").resample("5ms", on="time", origin="epoch").sum().drop(columns="is_outgoing")
    # data["time"] = (data["time"]- dt.datetime(1970,1,1)).dt.total_seconds()
    return data.xs(True)["length"], data.xs(False)["length"]


def load_schedule(filename):
    data = pd.read_csv(filename, header=None, names=["time", "length"])
    data["is_outgoing"] = data["length"] > 0
    data["length"] = data["length"].abs()
    data = data.sort_values(by="time").reset_index(drop=True)

    data["time"] = pd.to_datetime(data["time"], unit="s")

    data = data.groupby("is_outgoing").resample(
                                        "5ms", on="time", origin="epoch"
                                        )["length"].sum()

    return data.xs(True), data.xs(False)


def main(inputs, output_file):
    """Measure Pearson correlation between two traces.
        Outputs graph of rolling window Pearson."""

    (outgoing, incoming) = load_cover_trace(inputs[0])
    (baseline_out, baseline_in) = load_schedule(inputs[1])

    r_window_size = 25

    df_tx = pd.concat([outgoing, baseline_out], keys=["Defended", "Baseline"], axis=1).fillna(0)
    df_rx = pd.concat([-incoming, -baseline_in], keys=["Defended", "Baseline"], axis=1).fillna(0)
    bins_rx = df_rx.index.values.astype(float)
    bins_tx = df_tx.index.values.astype(float)

    rolling_rx = df_rx['Defended'].rolling(window=r_window_size, center=True).corr(df_rx['Baseline'])
    rolling_tx = df_tx['Defended'].rolling(window=r_window_size, center=True).corr(df_tx['Baseline'])

    f, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 6))
    ax[0].scatter(x=df_rx.index, y=df_rx["Defended"], label="Defended", s=1.0, marker='1')
    ax[0].scatter(x=df_rx.index, y=df_rx["Baseline"], label="Baseline", s=1.0, marker='2')
    ax[0].scatter(x=df_tx.index, y=df_tx["Defended"], label="Defended", s=1.0, marker='1')
    ax[0].scatter(x=df_tx.index, y=df_tx["Baseline"], label="Baseline", s=1.0, marker='2')
    ax[0].set(xlabel='ms', ylabel='packet size')
    ax[0].legend()
    rolling_tx.plot(x=bins_tx, ax=ax[1])
    ax[1].set(xlabel='ms', ylabel='Pearson r TX')
    rolling_rx.plot(x=bins_rx, ax=ax[2])
    ax[2].set(xlabel='ms', ylabel='Pearson r RX')
    f.savefig(output_file, dpi=300, bbox_inches="tight")

    r, p = stats.pearsonr(df_tx["Defended"], df_tx["Baseline"])
    print(f"Scipy computed Pearson r TX: {r} and p-value: {p}")
    r, p = stats.pearsonr(df_rx["Defended"], df_rx["Baseline"])
    print(f"Scipy computed Pearson r RX: {r} and p-value: {p}")


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, doceasy.Schema({
        "INPUTS": [str],
        "--output-file": doceasy.Or(None, str)
    }, ignore_extra_keys=True)))
