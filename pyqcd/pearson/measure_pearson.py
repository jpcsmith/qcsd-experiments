import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats
import doceasy


def load_cover_trace(filename):
    data = pd.read_csv(filename, delimiter=";")
    data = data.sort_values(by="frame.time_epoch").reset_index(drop=True)

    # Ensure that the trace starts with packet number zero
    assert data.loc[0, "quic.packet_number"] == 0
    start_time = data.loc[0, "frame.time_epoch"]

    # Drop all packets with packet number zero
    # as they're only there for the starting time
    data = data[data["quic.packet_number"] != 0].reset_index(drop=True)

    # TODO: Should not include stremas which are not a part of the dummy traffic
    data["length"] = data["quic.stream.length"].map(
        lambda x: 0 if x is np.nan else sum(int(v) for v in x.split(","))
    )
    data["length"] += data["quic.padding_length"].fillna(0)
    data["length"] = data["length"].astype(int)
    data["is_outgoing"] = ~data["udp.srcport"].isin((443, 80))

    data["time"] = (data["frame.time_epoch"] - start_time) * 1e3

    data = data[["time", "length", "is_outgoing"]]

    data["time"] = pd.to_datetime(data["time"], unit="ms")
    data = data.groupby("is_outgoing").resample("5ms", on="time", origin="epoch")["length"].sum()
    return data.xs(True), data.xs(False)


def load_schedule(filename):
    data = pd.read_csv(filename, header=None, names=["time", "length"])
    data["is_outgoing"] = data["length"] > 0
    data["length"] = data["length"].abs()
    data = data.sort_values(by="time").reset_index(drop=True)

    data["time"] = pd.to_datetime(data["time"], unit="s")

    data = data.groupby("is_outgoing").resample("5ms", on="time", origin="epoch")["length"].sum()

    return data.xs(True), data.xs(False)


def main(inputs, output_file: Optional[str]):
    """Measure Pearson correlation between two traces.
        Outputs graph of rolling window Pearson."""

    print(inputs)
    print(output_file)

    # (outgoing, incoming) = load_cover_trace(
    #     "../results/collect/front_defended/100/front_cover_traffic.csv"
    #     )
    # (baseline_out, baseline_in) = load_schedule(
    #     "../results/collect/front_defended/100/schedule.csv"
    #     )

    # figure, axes = plt.subplots(1, 1, figsize=(12, 4))
    # outgoing_shift = outgoing + 50
    # display(outgoing_shift)
    # sns.scatterplot(x="time", y="length", data=outgoing_shift.to_frame().reset_index(), marker='.', ax=axes, label="Outgoing Shaped")
    # sns.scatterplot(x="time", y="length", data=baseline_out.to_frame().reset_index(), marker='.', ax=axes, label="Baseline")

    # figure.savefig("/tmp/out.png", dpi=300, bbox_inches="tight")

    # r_window_size = 24

    # df = pd.concat([outgoing, baseline_out], keys=["Defended", "Baseline"], axis=1).fillna(0)
    # bins= np.arange(len(df))

    # rolling_r = df['Defended'].rolling(window=r_window_size, center=True).corr(df['Baseline'])
    # display(df)
    # display(rolling_r)

    # f, ax = plt.subplots(2, 1, sharex=True, figsize=(14, 6))
    # ax[0].scatter(x=bins,
    #               y=df["Defended"],
    #               label="Defended",
    #               s=1.0,
    #               marker='1'
    #               )
    # ax[0].scatter(x=bins,
    #               y=df["Baseline"],
    #               label="Baseline",
    #               s=1.0,
    #               marker='2'
    #               )
    # ax[0].set(xlabel='ms', ylabel='packet size')
    # ax[0].legend()
    # rolling_r.plot(ax=ax[1])
    # ax[1].set(xlabel='ms', ylabel='Pearson r')
    # f.show()

    # r, p = stats.pearsonr(df["Defended"], df["Baseline"])
    # print(f"Scipy computed Pearson r: {r} and p-value: {p}")


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, doceasy.Schema({
        "INPUTS": [str],
        "--output-file": doceasy.Or(None, str)
    }, ignore_extra_keys=True)))
