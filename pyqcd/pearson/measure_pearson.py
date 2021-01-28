#!/usr/bin/env python3
"""Usage: module [options] -- [INPUTS]...

Measures the distance between two dummy traces using Pearson correlation.
Traces are sampled into 1ms intervals.

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


def load_cover_trace(filename, dummy_streams):
    data = pd.read_csv(filename, delimiter=";")
    data = data.sort_values(by="frame.time_epoch").reset_index(drop=True)

    dummy_ids = pd.read_csv(dummy_streams, header=None, names=["id"])["id"]

    # Ensure that the trace starts with packet number zero
    assert data.loc[0, "quic.packet_number"] == 0
    start_time = data.loc[0, "frame.time_epoch"]

    # Drop all packets with packet number zero
    # as they're only there for the starting time
    data = data[data["quic.packet_number"] != 0].reset_index(drop=True)

    size_if_dummy = lambda v: int(v) if (int(v) in dummy_ids) else 0
    data["length"] = data["quic.stream.length"].map(
        lambda x: 0 if x is np.nan else sum(size_if_dummy(v) for v in x.split(","))
    )
    data["length"] += data["quic.padding_length"].fillna(0)
    data["length"] = data["length"].astype(int)
    data["is_outgoing"] = ~data["udp.srcport"].isin((443, 80))

    data["time"] = (data["frame.time_epoch"] - start_time) * 1e3

    data = data[["time", "length", "is_outgoing"]]

    data["time"] = pd.to_datetime(data["time"], unit="ms")
    data = data.groupby("is_outgoing").resample(
                                        "1ms", on="time", origin="epoch"
                                        )["length"].sum()
    return data.xs(True), data.xs(False)


def load_schedule(filename):
    data = pd.read_csv(filename, header=None, names=["time", "length"])
    data["is_outgoing"] = data["length"] > 0
    data["length"] = data["length"].abs()
    data = data.sort_values(by="time").reset_index(drop=True)

    data["time"] = pd.to_datetime(data["time"], unit="s")

    data = data.groupby("is_outgoing").resample(
                                        "1ms", on="time", origin="epoch"
                                        )["length"].sum()

    return data.xs(True), data.xs(False)


def main(inputs, output_file):
    """Measure Pearson correlation between two traces.
        Outputs graph of rolling window Pearson."""

    (outgoing, incoming) = load_cover_trace(inputs[0], inputs[2])
    (baseline_out, baseline_in) = load_schedule(inputs[1])

    r_window_size = 100

    df = pd.concat(
                [outgoing, baseline_out],
                keys=["Defended", "Baseline"],
                axis=1
            ).fillna(0)
    bins = np.arange(len(df))

    rolling_r = df['Defended'].rolling(
                                window=r_window_size,
                                center=True
                                ).corr(df['Baseline'])
    # display(df)
    # display(rolling_r)

    f, ax = plt.subplots(2, 1, sharex=True, figsize=(14, 6))
    ax[0].scatter(x=bins,
                  y=df["Defended"],
                  label="Defended",
                  s=1.3,
                  marker='1'
                  )
    ax[0].scatter(x=bins,
                  y=df["Baseline"],
                  label="Baseline",
                  s=1.3,
                  marker='2'
                  )
    ax[0].set(xlabel='ms', ylabel='packet size')
    ax[0].legend()
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='ms', ylabel='Pearson r')
    f.savefig(output_file, dpi=300, bbox_inches="tight")

    r, p = stats.pearsonr(df["Defended"], df["Baseline"])
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, doceasy.Schema({
        "INPUTS": [str],
        "--output-file": doceasy.Or(None, str)
    }, ignore_extra_keys=True)))
