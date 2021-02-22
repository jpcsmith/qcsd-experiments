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
import collections.abc
import json
import doceasy

PALETTE = {'blue': '#4898de',
           'purple': '#a7a3e0',
           'fucsia': '#dab4da',
           'pink': '#f4cddb',
           'rosepink': '#ffb7b6',
           'coral': '#ffae75',
           'orange': '#f7b819'}

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def aggreagate(data, direction):
    res = {}
    for (s_id, dfs) in data.items():
        # print("Sample: ", s_id)
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
    # print(pd.DataFrame.from_dict(res, orient='index'))
    return pd.DataFrame.from_dict(res, orient='index')


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def main(inputs, output_plot, output_json):
    # construct dictonary from csvs
    dfs = {}
    for path in inputs:
        (sample_id, rep_id) = path.split(sep='/')[3].split(sep='_')
        data = pd.read_csv(path, index_col=0)
        dfs = update(dfs, {sample_id: {rep_id: data}})

    # aggregate all results into two big dataframe.
    # TX :
    #      | 'r_schedule_avg' | 'r_target_avg'  | ...
    # -----|------------------]-----------------]----
    # '020'|      0.20        |       0.3       | ...
    # '021 |      0.34        |       0.1       | ...
    #  ... |       ...        |       ...       | ...
    # RX :
    # (same)
    tx_means = aggreagate(dfs, 'TX')
    rx_means = aggreagate(dfs, 'RX')
    tot_means = aggreagate(dfs, 'Total')
    
    # save results to json
    resdict = {
                'TX': tx_means.to_dict(),
                'RX': rx_means.to_dict(),
                'Total': tot_means.to_dict()
                }
    with open(output_json, "w") as f:
        json.dump(resdict, f)

    # BOXPLOT
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    ticks = ['TX', 'RX', 'Tot']
    # overhead over target trace
    ax[0].boxplot([tx_means['r_target']*100,
                   rx_means['r_target']*100,
                   tot_means['r_target']*100],
                  labels=ticks, manage_ticks=True,
                  sym='')
    ax[0].grid()
    ax[0].set_ylabel('Overhead [%]')
    ax[0].set_title(f'$(|defended| - |target|) / |target|$')

    # comparison QCD vs FRONT theoretical 
    data_qcd = [tx_means['r_full']*100,
                rx_means['r_full']*100,
                tot_means['r_full']*100]
    data_front = [tx_means['r_baseline']*100,
                  rx_means['r_baseline']*100,
                  tot_means['r_baseline']*100]
    bpl = ax[1].boxplot(data_qcd,
                        positions=np.array(range(len(data_qcd)))*2.0-0.4,
                        # labels=ticks, manage_ticks=True,
                        widths=0.6)
    bpr = ax[1].boxplot(data_front,
                        positions=np.array(range(len(data_front)))*2.0+0.4,
                        # labels=ticks, manage_ticks=True,
                        widths=0.6)
    set_box_color(bpl, PALETTE['blue'])
    set_box_color(bpr, PALETTE['coral'])
    # draw temporary red and blue lines and use them to create a legend
    ax[1].plot([], c=PALETTE['blue'], label='QCD FRONT')
    ax[1].plot([], c=PALETTE['coral'], label='Front (theoretical)')
    ax[1].set_xticks(range(0, len(ticks) * 2, 2))
    ax[1].set_xticklabels(ticks)
    ax[1].legend()
    ax[1].grid()
    ax[1].set_ylabel('Overhead [%]')
    ax[1].set_title('QCD vs. FRONT overhead over undefended traffic.')

    f.savefig(output_plot, dpi=300,  bbox_inches="tight")


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, doceasy.Schema({
        "INPUTS": [str],
        "--output-plot": doceasy.Or(None, str),
        "--output-json": str
    }, ignore_extra_keys=True)))
