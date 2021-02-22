#!/usr/bin/env python3
"""Usage: module [options] -- [INPUTS]...

Aggregates pearsons values generated from "measure_pearson.py"
into a single plot. Analyses distribution of values

Inputs:
    list of "overhead.csv" files generated by "front_overhead.py"

Options:
    --output-plot filename
        Save the distribution of pearsons values to filename
    --output-json filename
        Save aggregated results to json file
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
import collections.abc
import json
import doceasy


N_BINS_NORMAL = 20

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# include argument 'direction' = TX | RX for selecting insiede the obj
# then keep a bigger df, adds row each iteration of data with values from [r_schedule, r_target, r_...]
# put index as rep_id
# get mean of every column and return means
def aggreagate(data, direction):
    res = {}
    for (s_id, dfs) in data.items():
        print("Sample: ", s_id)
        reps = {'r_schedule': [], 'r_target': [], 'r_full': [], 'r_baseline': []}
        for (rep_id, df) in dfs.items():
            # print(df.loc[direction])
            s = df.loc[direction]
            # print(list(s.loc[['r_schedule', 'r_target', 'r_full', 'r_baseline']]))
            reps['r_schedule'].append(s['r_schedule'])
            reps['r_target'].append(s['r_target'])
            reps['r_full'].append(s['r_full'])
            reps['r_baseline'].append(s['r_baseline'])
        # print(reps)
        s_means = pd.DataFrame(reps).mean(axis=0)
        res[s_id] = s_means
    print(pd.DataFrame.from_dict(res, orient='index'))
    return pd.DataFrame.from_dict(res, orient='index')


def plot_normal(data, ax, title):
    ax.set_xlim(left=-0.125, right=1.0)
    count_, bins, _ = ax.hist(data, bins=N_BINS_NORMAL, density=True)
    # fitting curve
    mu, std = norm.fit(data)
    xs = np.arange(-0.125, 1.0, 0.01)
    ax.plot(
             xs,
             norm.cdf(xs, loc=mu, scale=std),
             linewidth=2, color='r', label=f'$\mu={mu:.3f}$,\n$stdev={std:.3f}$'
            )
    ax.axvline(mu, color='k', linestyle='dashed', linewidth=1)
    ax.axhline(norm.cdf(mu, loc=mu, scale=std),
               color='k', linestyle='dashed', linewidth=1)
    ax.annotate(f'{mu:.2f}', xy=(mu, 0), xytext=(mu, -0.5))
    ax.set_title(title)
    ax.set_ylabel(f"$CDF_{{\mu,\sigma}}$")
    ax.legend()
    ax.grid()


def main(inputs, output_plot, output_json):

    print(inputs)
    print("*******")
    print(output_plot)
    print(output_json)

    foo = pd.read_csv(inputs[0], index_col=0)
    print(foo.loc['TX','defended'])


    dfs = {}
    for path in inputs:
        print(dfs)
        print("***")
        (sample_id, rep_id) = path.split(sep='/')[3].split(sep='_')
        data = pd.read_csv(path, index_col=0)
        dfs = update(dfs, {sample_id: {rep_id: data}})

    # print("\n")
    # print(dfs['021']['1']['defended']['TX'])

    # with open(output_json, "w") as f:
    #     # need to serialize dataframes to dump as json
    #     json.dump(dfs, f)

    # aggregate all results into two big dataframe.
    # TX :
    #      | 'r_schedule_avg' | 'r_target_avg'  | ...
    # -----|------------------]-----------------]----
    # '020'|      0.20        |       0.3       | ...
    # '021 |      0.34        |       0.1       | ...
    #  ... |       ...        |       ...       | ...
    # RX :
    # (same)
    # also store count and mean into dfs like in pearson

    tx_means = aggreagate(dfs, 'TX')
    rx_means = aggreagate(dfs, 'RX')
    tot_means = aggreagate(dfs, 'Total')
    print(tx_means)
    with open(output_json, "w") as f:
        json.dump(tot_means.to_json(orient='index'), f)

    f, ax = plt.subplots(3, 1, figsize=(10, 12))
    # plot means TX
    plot_normal(tx_means['r_target'], ax[0], 'client -> server overhead')
    # plot means RX
    plot_normal(rx_means['r_target'], ax[1], 'server -> client overhead')
    # plot all
    # plot_normal(np.append(rx_means, tx_means),
    #             ax[2],
    #             "overall overhead packets")
    # ax[2].set_xlabel("Pearson score")

    f.savefig(output_plot, dpi=300,  bbox_inches="tight")


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, doceasy.Schema({
        "INPUTS": [str],
        "--output-plot": doceasy.Or(None, str),
        "--output-json": str
    }, ignore_extra_keys=True)))

