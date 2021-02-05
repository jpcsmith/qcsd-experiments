#!/usr/bin/env python3
"""Usage: module [options] -- [INPUTS]...

Measures the distance between an undefended trace and
the target trace.A target trace is composed of the baseline
plus the scheduled chaff traffic.
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

def load_schedule(filename):
    print("Reading schedule: {}".format(filename))
    data = pd.read_csv(filename, header=None, names=["time", "length"])
    data["is_outgoing"] = data["length"] > 0
    data["length"] = data["length"].abs()
    data = data.sort_values(by="time").reset_index(drop=True)

    data["time"] = pd.to_datetime(data["time"], unit="s")

    data = data.groupby("is_outgoing").resample(
                                        "0.01ms", on="time", origin="epoch"
                                        )["length"].sum()

    return data.xs(True), data.xs(False)

def load_trace(filename):
    print("Reading trace: {}".format(filename))
    data = pd.read_csv(filename)
    assert data.loc[0, "packet_number"] == 0
    data["timestamp"] = (data["timestamp"] - data.loc[0, "timestamp"]) * 1e3
    data = data.rename(columns={"timestamp": "time",
                                "packet_length": "length"})
    data = data[["time", "length", "is_outgoing"]]
    data["time"] = pd.to_datetime(data["time"], unit="ms")

    data = data.groupby("is_outgoing").resample("0.01ms", on="time", origin="epoch").sum().drop(columns="is_outgoing")
    # data["time"] = (data["time"]- dt.datetime(1970,1,1)).dt.total_seconds()
    return data.xs(True)["length"], data.xs(False)["length"]


def make_target_trace(baseline, schedule):
    """Takes a baseline trace and a chaff traffic schedule and
        adds them into a target trace
    """
    target = baseline.append(schedule, ignore_index=False).sort_index()

    return target


def rolling_pearson(df_tx, df_rx):
    """Measures the Pearson's distance with rolling window"""

    r_window_size = 25

    rolling_rx = df_rx['Defended'].rolling(
                                    window=r_window_size,
                                    center=True).corr(df_rx['Target'])
    rolling_tx = df_tx['Defended'].rolling(
                                    window=r_window_size,
                                    center=True).corr(df_tx['Target'])

    return rolling_tx, rolling_rx


def main(inputs, output_file):
    """Loads two traces, build target and measure pearson
    """

    # Load traces
    (outgoing, incoming) = load_trace(inputs[0])
    print(outgoing, incoming)
    print("***\n")
    (baseline_out, baseline_in) = load_trace(inputs[1])
    print(baseline_out, baseline_in)
    print("***\n")
    (schedule_out, schedule_in) = load_schedule(inputs[2])
    print(schedule_out, schedule_in)

    # Create target trace: baseline + chaff
    print("***\n")
    print("Target trace:")
    target_out = make_target_trace(baseline_out, schedule_out)
    target_in = make_target_trace(baseline_in, schedule_in)
    print(target_out, target_in)
    df_tx = pd.concat([outgoing, target_out], keys=["Defended", "Target"], axis=1).fillna(0)
    df_rx = pd.concat([-incoming, -target_in], keys=["Defended", "Target"], axis=1).fillna(0)
    bins_rx = df_rx.index.values.astype(float)
    bins_tx = df_tx.index.values.astype(float)
    # TODO resample here

    outgoing = outgoing.resample("5ms", origin="epoch").sum()
    incoming = incoming.resample("5ms", origin="epoch").sum()
    target_out = target_out.resample("5ms", origin="epoch").sum()
    target_in = target_in.resample("5ms", origin="epoch").sum()

    # measure pearson
    (rolling_tx, rolling_rx) = rolling_pearson(df_tx, df_rx)

    # plot values
    f, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 6))
    ax[0].scatter(x=df_rx.index, y=df_rx["Defended"], label="Defended", s=1.0, marker='1')
    ax[0].scatter(x=df_rx.index, y=df_rx["Target"], label="Target", s=1.0, marker='2')
    ax[0].scatter(x=df_tx.index, y=df_tx["Defended"], label="Defended", s=1.0, marker='1')
    ax[0].scatter(x=df_tx.index, y=df_tx["Target"], label="Target", s=1.0, marker='2')
    ax[0].set(ylabel='packet size')
    ax[0].legend()
    rolling_tx.plot(x=bins_tx, ax=ax[1])
    ax[1].set(xlabel='ms', ylabel='Pearson r TX')
    rolling_rx.plot(x=bins_rx, ax=ax[2])
    ax[2].set(xlabel='ms', ylabel='Pearson r RX')
    f.savefig("results/pearson/front/007_0/full_test.png", dpi=300, bbox_inches="tight")
    # f.show()

    r, p = stats.pearsonr(df_tx["Defended"], df_tx["Target"])
    print(f"Scipy computed Pearson r TX: {r} and p-value: {p}")
    r, p = stats.pearsonr(df_rx["Defended"], df_rx["Target"])
    print(f"Scipy computed Pearson r RX: {r} and p-value: {p}")



if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, doceasy.Schema({
        "INPUTS": [str],
        "--output-file": doceasy.Or(None, str)
    }, ignore_extra_keys=True)))
