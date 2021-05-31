import random
from pathlib import Path


#: Specify constraints on the wildcards
wildcard_constraints:
    sample_id="\d{4,6}",
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
    excess_msd = config["experiment"]["excess_msd"]

    # Get the sample ids for which we have dependency graphs
    dep_directory = checkpoints.url_dependency_graphs.get(**wildcards).output[0]
    sample_ids = glob_wildcards(dep_directory + "/{sample_id}.json").sample_id

    # Limit the number of samples that will be collected
    sample_ids = sample_ids[:excess_msd["max_samples"]]

    return expand(
        rules.excess_msd__score.output, sample_id=sample_ids, rep_id="00",
        excess_msd=excess_msd["excess_values"]
    )


rule excess_msd__all_score:
    """Compute scores for a FRONT evaluation on all of the dependency graphs"""
    input:
        excess_msd__all_score__inputs
    output:
        "results/excess-msd/scores.csv"
    run:
        pd.concat(
            [pd.read_csv(f) for f in input if Path(f).stat().st_size != 0], ignore_index=True
        ).to_csv(output[0], index=False)


rule excess_msd__plot:
    input:
        rules.excess_msd__all_score.output
    output:
        heatmap="results/plots/excess-msd-heatmap.pdf",
        trend="results/plots/excess-msd-trend.pdf",
        scores="results/plots/excess-msd-scores.pdf",
    notebook:
        "../notebooks/plot-excess-msd-results.ipynb"
