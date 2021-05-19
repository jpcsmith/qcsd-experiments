import random

#: Specify constraints on the wildcards
wildcard_constraints:
    sample_id="\d{4}",
    rep_id="\d{2}",
    excess_msd="\d+",


rule excess_msd__collect:
    """Collect a sample of chaff-only mode for constant rate schedule and excess MSD."""
    input:
        url_dep="results/determine-url-deps/dependencies/{sample_id}.json",
        schedule="results/excess-msd/constant_{rate}Mbps_{duration}s_{interval}ms.csv".format(
            **config["experiment"]["excess_msd"]["padding"]),
    output:
        control="results/excess-msd/{excess_msd}/{sample_id}_{rep_id}/control.stdout.txt",
        control_pcap="results/excess-msd/{excess_msd}/{sample_id}_{rep_id}/control.pcapng",
        defended="results/excess-msd/{excess_msd}/{sample_id}_{rep_id}/defended.stdout.txt",
        defended_pcap="results/excess-msd/{excess_msd}/{sample_id}_{rep_id}/defended.pcapng",
    log:
        "results/excess-msd/{excess_msd}/{sample_id}_{rep_id}/script.log",
        control="results/excess-msd/{excess_msd}/{sample_id}_{rep_id}/control.log",
        defended="results/excess-msd/{excess_msd}/{sample_id}_{rep_id}/defended.log",
    params:
        pad_only_mode="true",
        msd_limit_excess="{excess_msd}"
    threads: 2
    script:
        "../scripts/collect_trace.py"


rule excess_msd__score:
    """Compare the schedule to the defended traces."""
    input:
        control=rules.excess_msd__collect.output["control"],
        control_pcap=rules.excess_msd__collect.output["control_pcap"],
        defended=rules.excess_msd__collect.output["defended"],
        defended_pcap=rules.excess_msd__collect.output["defended_pcap"],
        schedule=rules.excess_msd__collect.input["schedule"],
    output:
        "results/excess-msd/{excess_msd}/{sample_id}_{rep_id}/scores.csv"
    log:
        "results/excess-msd/{excess_msd}/{sample_id}_{rep_id}/scores.log"
    params:
        pad_only=True,
        sample_id="{excess_msd}/{sample_id}_{rep_id}",
        **config["experiment"]["scores"]
    script:
        "../scripts/calculate_score.py"


def excess_msd__all_score__inputs(wildcards):
    dep_directory = checkpoints.url_dependency_graphs.get(**wildcards).output[0]
    sample_ids = glob_wildcards(dep_directory + "/{sample_id}.json").sample_id
    return expand(rules.excess_msd__score.output, sample_id=sample_ids, rep_id="00",
                  excess_msd=config["experiment"]["excess_msd"]["excess_values"])


rule excess_msd__all_score:
    """Compute scores for a FRONT evaluation on all of the dependency graphs"""
    input:
        excess_msd__all_score__inputs
    output:
        "results/excess-msd/scores.csv"
    run:
        pd.concat(
            [pd.read_csv(f) for f in input], ignore_index=True
        ).to_csv(output[0], index=False)


# def excess_msd_collect_all__inputs(wildcards):
#     dep_directory = checkpoints.url_dependency_graphs.get(**wildcards).output[0]
#     sample_ids = glob_wildcards(dep_directory + "/{sample_id}.json").sample_id
#
#     emsd_config = config["experiment"]["excess_msd"]
#     repetitions = [f"{rep:02d}" for rep in range(emsd_config["repetitions"])]
#     inputs = expand(rules.excess_msd_collect.output["pcap"], sample_id=sample_ids,
#                     rep_id=repetitions, excess_msd=emsd_config["excess_values"])
#
#     # Shuffle the order so that like experiments are less likely to run sequentially
#     random.Random(278912).shuffle(inputs)
#     return inputs
#
#
# rule excess_msd_collect_all:
#     """Collect multiple samples for the excess_msd experiment according to the
#     configuration."""
#     input:
#         excess_msd_collect_all__inputs
#
#
# rule excess_msd_plot:
#     """Plot the result of the excess_msd experiment."""
#     input:
#         excess_msd_collect_all__inputs,
#         schedule="results/excess_msd/constant_{rate}Mbps_{duration}s_{interval}ms.csv".format(
#             **config["experiment"]["excess_msd"]["padding"]),
#     output:
#         trend_path=report("results/plots/excess-msd-success-rate.png",
#                           category="Excess MSD Experiment"),
#         heatmap_path=report("results/plots/excess-msd-heatmap.png",
#                             category="Excess MSD Experiment"),
#         pearson_path=report("results/plots/excess-msd-pearson.png",
#                             category="Excess MSD Experiment"),
#     params:
#         base_dir="results/excess_msd",
#         values=config["experiment"]["excess_msd"]["excess_values"]
#     script:
#         "../scripts/plot_excess_msd.py"
