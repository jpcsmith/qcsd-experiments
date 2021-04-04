#!/usr/bin/env python3
"""Usage: module [options] -- [INPUTS]...

Measures the distance between an undefended trace and
the target trace.A target trace is composed of the baseline
plus the scheduled chaff traffic.

Inputs:
    Defended Trace
    Target Schedule

Options:
    --output-plot filename
        Save the dummy packets trace and rolling window pearson graph
        to filename.
    --output-json filename
        Save the pearsons results to json file
    --window-size int
        Size of the rolling window used to calculate Pearson's correlation
        in the traces.
    --sample-rate float
        rate of sampling packets in ms
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import doceasy
import json

LAG_MAX_MS = 250


def load_schedule(filename, rate):
    print("Reading schedule: {}".format(filename))
    data = pd.read_csv(filename, header=None, names=["time", "length"])
    data["is_outgoing"] = data["length"] > 0
    data["length"] = data["length"].abs()
    data = data.sort_values(by="time").reset_index(drop=True)

    data["time"] = pd.to_datetime(data["time"], unit="s")

    # sampling rate
    sampling = str(rate)+"ms"
    data = data.groupby("is_outgoing").resample(
                                        sampling, on="time", origin="epoch"
                                        )["length"].sum()

    return data.xs(True), data.xs(False)


def load_trace(filename, rate):
    print("Reading trace: {}".format(filename))
    data = pd.read_csv(filename)
    assert data.loc[0, "packet_number"] == 0
    data["timestamp"] = (data["timestamp"] - data.loc[0, "timestamp"]) * 1e3
    data = data.rename(columns={"timestamp": "time",
                                "packet_length": "length"})
    data = data[["time", "length", "is_outgoing"]]
    data["time"] = pd.to_datetime(data["time"], unit="ms")

    sampling = str(rate)+"ms"
    data = data.groupby("is_outgoing").resample(
                                        sampling, on="time", origin="epoch"
                                        ).sum().drop(columns="is_outgoing")

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


def lagged_crosscorr(s1, s2, lag=0):
    r, p = stats.pearsonr(s1.shift(lag).fillna(0), s2)
    return r


def main(inputs, output_plot, output_json, window_size=25, sample_rate=5):
    """Loads two traces, build target and measure pearson
    """

    # Load traces
    (outgoing, incoming) = load_trace(inputs[0], float(sample_rate))
    print(outgoing, incoming)
    print("***\n")
    (schedule_out, schedule_in) = load_schedule(inputs[1], float(sample_rate))
    print(schedule_out, schedule_in)

    r_window_size = int(window_size)

    # concat traces
    df_tx = pd.concat([outgoing, schedule_out], keys=["Defended", "Target"], axis=1).fillna(0)
    df_rx = pd.concat([-incoming, -schedule_in], keys=["Defended", "Target"], axis=1).fillna(0)
    bins_rx = df_rx.index.values.astype(float)
    bins_tx = df_tx.index.values.astype(float)

    # measure pearson
    # using the lagged pearson correlation
    max_lag = int(LAG_MAX_MS / float(sample_rate))
    rs = [lagged_crosscorr(df_rx['Defended'],
                           df_rx["Target"],
                           lag) for lag in range(0, max_lag+1)]
    offset = np.argmax(rs)-np.ceil(len(rs)/2)
    print("Offset RX: {}".format(offset))
    df_rx["Defended"] = df_rx["Defended"].shift(int(offset)).fillna(0)

    rolling_rx = df_rx['Defended'].rolling(window=r_window_size, center=True).corr(df_rx['Target'])
    rolling_tx = df_tx['Defended'].rolling(window=r_window_size, center=True).corr(df_tx['Target'])

    f, ax = plt.subplots(3, 1, sharex=True, figsize=(14, 6))
    # ax[0].scatter(x=df_rx.index, y=df_rx["Defended"], label="Defended", s=1.0, marker='1')
    # ax[0].scatter(x=df_rx.index, y=df_rx["Target"], label="Target", s=1.0, marker='2')
    # ax[0].scatter(x=df_tx.index, y=df_tx["Defended"], label="Defended", s=1.0, marker='1')
    # ax[0].scatter(x=df_tx.index, y=df_tx["Target"], label="Target", s=1.0, marker='2')
    # ax[0].set(xlabel='ms', ylabel='packet size')
    ax[0].scatter(x=schedule_out.index, y=schedule_out, label="Schedule", s=1.0, marker='2')
    ax[0].scatter(x=outgoing.index, y=outgoing, label="Defended", s=1.0, marker='1')
    ax[0].scatter(x=schedule_in.index, y=-schedule_in, label="Schedule", s=1.0, marker='2')
    ax[0].scatter(x=incoming.index, y=-incoming, label="Defended", s=1.0, marker='1')
    ax[0].set_ylim(bottom=-750*(np.log(sample_rate)), top=1200)
    ax[0].legend()
    rolling_tx.plot(x=bins_tx, ax=ax[1])
    ax[1].set(xlabel='ms', ylabel='Pearson r TX')
    rolling_rx.plot(x=bins_rx, ax=ax[2])
    ax[2].set(xlabel='ms', ylabel='Pearson r RX')
    f.savefig(output_plot, dpi=300, bbox_inches="tight")

    r_tx, p_tx = stats.pearsonr(df_tx["Defended"], df_tx["Target"])
    print(f"Scipy computed Pearson r TX: {r_tx} and p-value: {p_tx}")
    r_rx, p_rx = stats.pearsonr(df_rx["Defended"], df_rx["Target"])
    print(f"Scipy computed Pearson r RX: {r_rx} and p-value: {p_rx}")

    # save results as json
    data = {
        'TX': {
            'stats': (r_tx, p_tx),
            'rolling': list(rolling_tx),
        },
        'RX': {
            'stats': (r_rx, p_rx),
            'rolling': list(rolling_rx),
        }
    }

    with open(output_json, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, doceasy.Schema({
        "INPUTS": [str],
        "--output-plot": doceasy.Or(None, str),
        "--output-json": str,
        "--window-size": doceasy.Or(None, str),
        "--sample-rate": doceasy.Or(None, str)
    }, ignore_extra_keys=True)))
