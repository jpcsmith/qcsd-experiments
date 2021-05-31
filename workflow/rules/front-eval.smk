from pathlib import Path


rule front_eval__collect:
    """Collect a control and defended sample with the FRONT defence."""
    input:
        url_dep="results/determine-url-deps/dependencies/{sample_id}.json"
    output:
        control="results/front-eval/{sample_id}_{rep_id}/control.stdout.txt",
        control_pcap="results/front-eval/{sample_id}_{rep_id}/control.pcapng",
        front="results/front-eval/{sample_id}_{rep_id}/front.stdout.txt",
        front_pcap="results/front-eval/{sample_id}_{rep_id}/front.pcapng",
        schedule="results/front-eval/{sample_id}_{rep_id}/schedule.csv",
    log:
        general="results/front-eval/{sample_id}_{rep_id}/script.log",
        control="results/front-eval/{sample_id}_{rep_id}/control.log",
        front="results/front-eval/{sample_id}_{rep_id}/front.log",
    params:
        # Assume at most 100 reps and construct seed as sample-id * 100 + rep
        seed=lambda w: int(w["sample_id"]) * 100 + int(w["rep_id"])
    threads: 2
    script:
        "../scripts/run_front_experiment.py"


rule front_eval__score:
    """Compare the schedule to the defended traces."""
    input:
        control=rules.front_eval__collect.output["control"],
        control_pcap=rules.front_eval__collect.output["control_pcap"],
        defended=rules.front_eval__collect.output["front"],
        defended_pcap=rules.front_eval__collect.output["front_pcap"],
        schedule=rules.front_eval__collect.output["schedule"],
    output:
        "results/front-eval/{sample_id}_{rep_id}/scores.csv"
    log:
        "results/front-eval/{sample_id}_{rep_id}/scores.log"
    params:
        pad_only=True,
        sample_id="{sample_id}_{rep_id}",
        **config["experiment"]["scores"]
    script:
        "../scripts/calculate_score.py"


def front_eval__all_score__inputs(wildcards):
    dep_directory = checkpoints.url_dependency_graphs.get(**wildcards).output[0]
    sample_ids = glob_wildcards(dep_directory + "/{sample_id}.json").sample_id

    # Limit the samples
    sample_ids = sample_ids[:config["experiment"]["front_single_eval"]["max_samples"]]
    return expand(rules.front_eval__score.output, sample_id=sample_ids, rep_id="00")


rule front_eval__all_score:
    """Compute scores for a FRONT evaluation on all of the dependency graphs"""
    input:
        front_eval__all_score__inputs
    output:
        "results/front-eval/scores.csv"
    run:
        pd.concat(
            [pd.read_csv(f) for f in input if Path(f).stat().st_size != 0], ignore_index=True
        ).to_csv(output[0], index=False)


rule front_eval__plot:
    input:
        rules.front_eval__all_score.output
    output:
        "results/plots/front-eval-plot.pdf"
    notebook:
        "../notebooks/plot-scores.ipynb"
