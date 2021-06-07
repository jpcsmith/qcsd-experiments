wildcard_constraints:
    mode="delay|drop",


rule delay_vs_drop__collect_tamaraw:
    """Collect a control and defended sample with the Tamaraw defence."""
    input:
        url_dep="results/determine-url-deps/dependencies/{sample_id}.json"
    output:
        control="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/control.stdout.txt",
        control_pcap="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/control.pcapng",
        tamaraw="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/defended.stdout.txt",
        tamaraw_pcap="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/defended.pcapng",
        schedule="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/schedule.csv",
    log:
        general="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/script.log",
        control="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/control.log",
        tamaraw="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/defended.log",
    threads: 2
    params:
        drop_unsat_events=lambda w: True if w["mode"] == "drop" else False
    script:
        "../scripts/collect_tamaraw.py"


rule delay_vs_drop__collect_front:
    """Collect a control and defended sample with the FRONT defence."""
    input:
        url_dep="results/determine-url-deps/dependencies/{sample_id}.json"
    output:
        control="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/control.stdout.txt",
        control_pcap="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/control.pcapng",
        front="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/defended.stdout.txt",
        front_pcap="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/defended.pcapng",
        schedule="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/schedule.csv",
    log:
        general="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/script.log",
        control="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/control.log",
        front="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/defended.log",
    params:
        # Assume at most 100 reps and construct seed as sample-id * 100 + rep
        seed=lambda w: int(w["sample_id"]) * 100 + int(w["rep_id"]) + 287038,
        drop_unsat_events=lambda w: w["mode"] == "drop"
    threads: 2
    script:
        "../scripts/run_front_experiment.py"


rule delay_vs_drop__score:
    """Compare the schedule to the defended traces."""
    input:
        control="results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/control.stdout.txt",
        control_pcap="results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/control.pcapng",
        defended="results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/defended.stdout.txt",
        defended_pcap="results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/defended.pcapng",
        schedule="results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/schedule.csv",
    output:
        "results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/scores.csv"
    log:
        "results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/scores.log"
    params:
        pad_only=lambda w: w["defence"] == "front",
        sample_id="{mode}/{defence}/{sample_id}_{rep_id}",
        **config["experiment"]["scores"]
    script:
        "../scripts/calculate_score.py"


def delay_vs_drop__all_score__inputs(wildcards):
    dep_directory = checkpoints.url_dependency_graphs.get(**wildcards).output[0]
    sample_ids = glob_wildcards(dep_directory + "/{sample_id}.json").sample_id

    # Limit the samples
    sample_ids = sample_ids[:config["experiment"]["delay_vs_drop"]["max_samples"]]
    return expand(
        rules.delay_vs_drop__score.output, sample_id=sample_ids, rep_id="00",
        mode=["delay", "drop"], defence=["front", "tamaraw"]
    )


rule delay_vs_drop__all_score:
    """Compute scores for a FRONT evaluation on all of the dependency graphs"""
    input:
        delay_vs_drop__all_score__inputs
    output:
        "results/delay-vs-drop/scores.csv"
    run:
        pd.concat(
            [pd.read_csv(f) for f in input], ignore_index=True
        ).to_csv(output[0], index=False)
