#!/usr/bin/env python3
"""Usage: module [options] -- [INPUTS]...

Aggregates pearsons values generated from "measure_pearson.py"
into a single plot. Analyses distribution of values

Inputs:
    list of "res.json" files generated by "measure_pearson.py"

Options:
    --output-plot filename
        Save the distribution of pearsons values to filename
    --output-json filename
        Save aggregated results to json file
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import collections.abc
import json
import doceasy

N_BINS_NORMAL = 150


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def plot_normal(data, ax, title):
    ax.set_xlim(left=-0.125, right=1.0)
    count_, bins, _ = ax.hist(data, bins=N_BINS_NORMAL, density=True)
    # fitting curve
    mu, std = norm.fit(data)
    xs = np.arange(-0.125, 1.0, 0.01)
    ax2 = ax.twinx()
    ax2.plot(
             xs,
             norm.cdf(xs, loc=mu, scale=std),
             linewidth=2, color='r', label=f'$\mu={mu:.3f}$,\n$stdev={std:.3f}$'
            )
    ax2.set_ylabel(f"$CDF_{{\mu,\sigma}}$")
    ax2.legend()

    ax.axvline(mu, color='k', linestyle='dashed', linewidth=1)
    ax2.axhline(norm.cdf(mu, loc=mu, scale=std),
               color='k', linestyle='dashed', linewidth=1)
    ax.annotate(f'{mu:.2f}', xy=(mu, 0), xytext=(mu, -1.5))
    ax.set_title(title)
    ax.grid()


def aggregate_and_means(data):
    means = np.array([])
    for (s_id, reps) in data.items():
        avg = np.nanmean(list(reps.values()))
        means = np.append(means, avg)
        # count how many non-nan values
        r_count = (~np.isnan(list(reps.values()))).sum()
        data[s_id] = {'count': int(r_count), 'mean': avg, 'reps': reps}

    return means, data


def main(inputs, output_plot, output_json):
    # load data
    N = len(inputs)
    pearsons_TX = {}
    pearsons_RX = {}
    for path in inputs:
        (sample_id, rep_id) = path.split(sep='/')[10].split(sep='_')
        with open(path, "r") as json_file:
            data = json.load(json_file)
            (r_tx, _) = data['TX']['stats']
            (r_rx, _) = data['RX']['stats']
            pearsons_TX = update(pearsons_TX, {sample_id: {rep_id: r_tx}})
            pearsons_RX = update(pearsons_RX, {sample_id: {rep_id: r_rx}})

    (tx_means, pearsons_TX) = aggregate_and_means(pearsons_TX)
    (rx_means, pearsons_RX) = aggregate_and_means(pearsons_RX)

    # save aggregated data
    with open(output_json, "w") as f:
        json.dump({'TX': pearsons_TX, 'RX': pearsons_RX}, f)

    # cleanup values
    tx_means = tx_means[~np.isnan(tx_means)]
    rx_means = rx_means[~np.isnan(rx_means)]
    assert (not np.isnan(tx_means).any())
    assert (not np.isnan(rx_means).any())

    f, ax = plt.subplots(3, 1, figsize=(10, 12))
    # plot dummy TX
    plot_normal(tx_means, ax[0], 'client -> server chaff')
    # plot dummy RX
    plot_normal(rx_means, ax[1], 'server -> client chaff')
    # plot all
    plot_normal(np.append(rx_means, tx_means),
                ax[2],
                "overall chaff packets")
    ax[2].set_xlabel("Pearson score")
    f.tight_layout()
    f.savefig(output_plot, dpi=300,  bbox_inches="tight")


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, doceasy.Schema({
        "INPUTS": [str],
        "--output-plot": doceasy.Or(None, str),
        "--output-json": str
    }, ignore_extra_keys=True)))

