"""Functions for constructing and binning timeseries from traces and
csvs.
"""
import io
import subprocess
from typing import Union
from pathlib import Path

import pandas as pd


def from_csv(filename: str) -> pd.DataFrame:
    """Create a timeseries from the schedule in filename.

    The schedule must be a headerless CSV file with the first column
    being timestamps in seconds, and the second being the signed packet
    sizes.
    """
    data = pd.read_csv(
        filename, header=None, sep=",", names=["time", "length"])

    data["is_outgoing"] = data["length"] > 0
    data["length"] = data["length"].abs()
    data["time"] = pd.to_timedelta(data["time"], unit="s")

    return _from_trace(data)


def from_trace(trace: pd.DataFrame, length_col: str = "length") -> pd.DataFrame:
    """
    The trace is a dataframe with the columns time (float) in seconds,
    length (unsigned int), and is_outgoing (bool)
    """
    assert "time" in trace.columns
    assert length_col in trace.columns
    assert "is_outgoing" in trace.columns

    data = trace[["time", length_col, "is_outgoing"]]
    if length_col != "length":
        data = data.rename(columns={length_col: "length"})
    data = data.assign(time=pd.to_timedelta(data["time"], unit="s"))

    # Set the time relative to the earliest entry
    data["time"] -= data["time"].min()

    return _from_trace(data)


def from_pcap(filename: Union[str, Path]) -> pd.DataFrame:
    """Return a timeseries from a pcap file.

    The lengths correspond to the IP length fields.
    """
    command = [
        "tshark", "-r", str(filename),
        "-T", "fields", "-E", "separator=,",
        "-e", "frame.time_epoch", "-e", "udp.length", "-e", "udp.srcport"
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    data = pd.read_csv(
        io.BytesIO(result.stdout), names=["time", "length", "is_outgoing"]
    )

    data["is_outgoing"] = data["is_outgoing"] != 443
    data["time"] = pd.to_timedelta(data["time"], unit="s")
    # Set the time relative to the earliest entry
    data["time"] -= data["time"].min()

    return _from_trace(data)


def _from_trace(data: pd.DataFrame) -> pd.DataFrame:
    # Handle the possibility that there are duplicate times by already summing
    data = data.groupby(["time", "is_outgoing"]).sum()
    # Convert to a single column for incoming and a single for outgoing
    return (data.unstack(fill_value=0)
                .droplevel(0, axis=1)
                .rename(columns={True: "out", False: "in"}))


def resample(timeseries: pd.DataFrame, rate: str) -> pd.DataFrame:
    """Return the timeseries resampled at the specified rate. Rate is
    a frequency string such as '50ms' that can be passed to
    pandas.DataFrame.resample().
    """
    # Ensure that the lowest entry of the schedule is at time zero
    if pd.Timedelta(0) not in timeseries.index:
        zeroes = 0 if len(timeseries.shape) == 1 else [0, 0]
        timeseries.loc[pd.Timedelta(0)] = zeroes

    # Resample the data starting from the first entry
    return timeseries.resample(rate, origin="start", label="left",
                               closed="left", convention="start").sum()
