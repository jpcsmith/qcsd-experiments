#!/usr/bin/env python3

import numpy as np
import pandas as pd


def load_chaff_trace(filename, rate=1):
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

    # sampling rate
    data = data.groupby("is_outgoing").sum()
    # data["time"] = (data["time"]- dt.datetime(1970,1,1)).dt.total_seconds()
    return data


def load_trace(filename):
    data = pd.read_csv(filename)
    assert data.loc[0, "packet_number"] == 0
    data["timestamp"] = (data["timestamp"] - data.loc[0, "timestamp"]) * 1e3
    data = data.rename(columns={"timestamp": "time",
                                "packet_length": "length"})
    data = data[["time", "length", "is_outgoing"]]
    data["time"] = pd.to_datetime(data["time"], unit="ms")

    data = data.groupby("is_outgoing").sum()
    # data["time"] = (data["time"]- dt.datetime(1970,1,1)).dt.total_seconds()
    return data


def load_schedule(filename, rate=1):
    data = pd.read_csv(filename, header=None, names=["time", "length"])
    data["is_outgoing"] = data["length"] > 0
    data["length"] = data["length"].abs()
    data = data.sort_values(by="time").reset_index(drop=True)

    data["time"] = pd.to_datetime(data["time"], unit="s")
    # sampling rate
    data = data.groupby("is_outgoing").sum()

    return data


def trace_overhead(front_cover_traff_csv,
                   trace_csv,
                   front_traff_csv,
                   schedule_csv):

    defended = load_chaff_trace(front_cover_traff_csv)
    defended_out = defended.xs(True)["length"]
    defended_in = defended.xs(False)["length"]
    defended = defended_out + defended_in

    baseline = load_trace(trace_csv)
    baseline_out = baseline.xs(True)["length"]
    baseline_in = baseline.xs(False)["length"]
    baseline = baseline_out + baseline_in

    def_full = load_trace(front_traff_csv)
    def_full_out = def_full.xs(True)["length"]
    def_full_in = def_full.xs(False)["length"]
    def_full = def_full_out + def_full_in

    schedule_chaff = load_schedule(schedule_csv)
    schedule_out = schedule_chaff.xs(True)['length']
    schedule_in = schedule_chaff.xs(False)['length']
    schedule = schedule_out + schedule_in

    df = pd.DataFrame([[defended_out, def_full_out, baseline_out,
                        schedule_out],
                       [defended_in, def_full_in, baseline_in,
                        schedule_in],
                       [defended, def_full, baseline, schedule]],
                      columns=["defended", "defended-full", "baseline", "schedule"])
    df = df.rename(index={0: 'TX', 1: 'RX', 2: 'Total'})

    df['target'] = df['baseline'] + df['schedule']

    df["diff-schedule"] = df["defended"] - df["schedule"]
    df["diff-target"] = df["defended-full"] - df["target"]
    df["diff-full"] = df["defended-full"] - df["baseline"]
    df["diff-baseline"] = df["target"] - df["baseline"]
    df['r_schedule'] = df["diff-schedule"] / df['schedule']
    df['r_target'] = df["diff-target"] / df['target']
    df['r_full'] = df["diff-full"] / df['baseline']
    df['r_baseline'] = df["diff-baseline"] / df['baseline']

    return df
