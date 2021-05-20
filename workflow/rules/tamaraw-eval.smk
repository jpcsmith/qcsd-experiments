rule tamaraw_eval__collect:
    """Collect a control and defended sample with the Tamaraw defence."""
    input:
        url_dep="results/determine-url-deps/dependencies/{sample_id}.json"
    output:
        control="results/tamaraw-eval/{sample_id}_{rep_id}/control.stdout.txt",
        control_pcap="results/tamaraw-eval/{sample_id}_{rep_id}/control.pcapng",
        tamaraw="results/tamaraw-eval/{sample_id}_{rep_id}/tamaraw.stdout.txt",
        tamaraw_pcap="results/tamaraw-eval/{sample_id}_{rep_id}/tamaraw.pcapng",
        schedule="results/tamaraw-eval/{sample_id}_{rep_id}/schedule.csv",
    log:
        general="results/tamaraw-eval/{sample_id}_{rep_id}/script.log",
        control="results/tamaraw-eval/{sample_id}_{rep_id}/control.log",
        tamaraw="results/tamaraw-eval/{sample_id}_{rep_id}/tamaraw.log",
    threads: 2
    script:
        "../scripts/collect_tamaraw.py"


rule tamaraw_eval__score:
    """Compare the schedule to the defended traces."""
    input:
        control=rules.tamaraw_eval__collect.output["control"],
        control_pcap=rules.tamaraw_eval__collect.output["control_pcap"],
        defended=rules.tamaraw_eval__collect.output["tamaraw"],
        defended_pcap=rules.tamaraw_eval__collect.output["tamaraw_pcap"],
        schedule=rules.tamaraw_eval__collect.output["schedule"],
    output:
        "results/tamaraw-eval/{sample_id}_{rep_id}/scores.csv"
    log:
        "results/tamaraw-eval/{sample_id}_{rep_id}/scores.log"
    params:
        pad_only=False,
        sample_id="{sample_id}_{rep_id}",
        **config["experiment"]["scores"]
    script:
        "../scripts/calculate_score.py"


def tamaraw_eval__all_score__inputs(wildcards):
    dep_directory = checkpoints.url_dependency_graphs.get(**wildcards).output[0]
    sample_ids = glob_wildcards(dep_directory + "/{sample_id}.json").sample_id

    # Limit the samples
    sample_ids = sample_ids[:config["experiment"]["tamaraw_single_eval"]["max_samples"]]
    return expand(rules.tamaraw_eval__score.output, sample_id=sample_ids, rep_id="00")


rule tamaraw_eval__all_score:
    """Compute scores for a FRONT evaluation on all of the dependency graphs"""
    input:
        tamaraw_eval__all_score__inputs
    output:
        "results/tamaraw-eval/scores.csv"
    run:
        pd.concat(
            [pd.read_csv(f) for f in input], ignore_index=True
        ).to_csv(output[0], index=False)


rule tamaraw_eval__plot:
    input:
        rules.tamaraw_eval__all_score.output
    output:
        "results/plots/tamaraw-eval-plot.pdf"
    notebook:
        "../notebooks/plot-scores.ipynb"
