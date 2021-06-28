delay_vs_drop_config = config["experiment"]["delay_vs_drop"]

# TODO: Remove .10chaff
wildcard_constraints:
    mode="delay|drop|delay\..*",
    sim_suffix="(\.sim)?"


rule delay_vs_drop__collect_tamaraw:
    """Collect the defended samples with the Tamaraw defence."""
    input:
        "results/webpage-graphs/graphs/"
    output:
        directory("results/delay-vs-drop/tamaraw/{mode}/dataset/")
    log:
        "results/delay-vs-drop/tamaraw/{mode}/dataset.log"
    threads:
        workflow.cores
    params:
        neqo_args=lambda w: [
            "--header", "user-agent", config["user_agent"],
            "--defence", "tamaraw",
            "--tamaraw-modulo", str(delay_vs_drop_config["tamaraw"]["pad_multiple"]),
            "--tamaraw-rate-in", str(delay_vs_drop_config["tamaraw"]["rate_in"]),
            "--tamaraw-rate-out", str(delay_vs_drop_config["tamaraw"]["rate_out"]),
            "--defence-packet-size", str(delay_vs_drop_config["tamaraw"]["packet_size"]),
            "--drop-unsat-events", ("true" if w["mode"] == "drop" else "false"),
            "--select-padding-by-size",
        ],
        n_monitored=delay_vs_drop_config["n_monitored"],
        n_instances=delay_vs_drop_config["n_instances"],
    script:
        "../scripts/run_collection.py"


rule delay_vs_drop__dataset:
    input:
        "results/delay-vs-drop/{defence}/{mode}/dataset/"
    output:
        temp("results/delay-vs-drop/{defence}/{mode}/dataset{sim_suffix}.h5")
    params:
        n_monitored=delay_vs_drop_config["n_monitored"],
        n_instances=delay_vs_drop_config["n_instances"],
        simulate=lambda w: bool(w["sim_suffix"]),
    script:
        "../scripts/create_dataset.py"


rule delay_vs_drop__features:
    input:
        rules.delay_vs_drop__dataset.output
    output:
        "results/delay-vs-drop/{defence}/{mode}/features{sim_suffix}.h5"
    log:
        "results/delay-vs-drop/{defence}/{mode}/features{sim_suffix}.log"
    shell:
        "workflow/scripts/extract-features {input} {output} 2> {log}"


rule delay_vs_drop__splits:
    """Create train-test-validation splits of the dataset."""
    input:
        rules.delay_vs_drop__features.output
    output:
        "results/delay-vs-drop/{defence}/{mode}/predict{sim_suffix}/split-0.json"
    params:
        seed=delay_vs_drop_config["splits"]["seed"],
        n_folds=delay_vs_drop_config["splits"]["n_folds"],
        validation_size=delay_vs_drop_config["splits"]["validation_size"]
    script:
        "../scripts/split_dataset.py"


rule delay_vs_drop__predictions:
    input:
        dataset=rules.delay_vs_drop__features.output,
        splits=rules.delay_vs_drop__splits.output,
    output:
        "results/delay-vs-drop/{defence}/{mode}/predict{sim_suffix}/{classifier}-0.csv"
    log:
        "results/delay-vs-drop/{defence}/{mode}/predict{sim_suffix}/{classifier}-0.log"
    params:
        classifier="{classifier}",
        classifier_args=lambda w: ("--classifier-args n_jobs=4,feature_set=kfp"
                                   if w["classifier"] == "kfp" else "")
    threads:
        lambda w: min(workflow.cores, 16) if w["classifier"] != "kfp" else 4
    shell:
        "workflow/scripts/evaluate-classifier {params.classifier_args}"
        " {params.classifier} {input.dataset} {input.splits} {output} 2> {log}"


# rule delay_vs_drop__collect_tamaraw:
#     """Collect a control and defended sample with the Tamaraw defence."""
#     input:
#         url_dep="results/determine-url-deps/dependencies/{sample_id}.json"
#     output:
#         control="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/control.stdout.txt",
#         control_pcap="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/control.pcapng",
#         tamaraw="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/defended.stdout.txt",
#         tamaraw_pcap="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/defended.pcapng",
#         schedule="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/schedule.csv",
#     log:
#         general="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/script.log",
#         control="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/control.log",
#         tamaraw="results/delay-vs-drop/{mode}/tamaraw/{sample_id}_{rep_id}/defended.log",
#     threads: 2
#     params:
#         drop_unsat_events=lambda w: True if w["mode"] == "drop" else False
#     script:
#         "../scripts/collect_tamaraw.py"
#
#
# rule delay_vs_drop__collect_front:
#     """Collect a control and defended sample with the FRONT defence."""
#     input:
#         url_dep="results/determine-url-deps/dependencies/{sample_id}.json"
#     output:
#         control="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/control.stdout.txt",
#         control_pcap="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/control.pcapng",
#         front="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/defended.stdout.txt",
#         front_pcap="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/defended.pcapng",
#         schedule="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/schedule.csv",
#     log:
#         general="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/script.log",
#         control="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/control.log",
#         front="results/delay-vs-drop/{mode}/front/{sample_id}_{rep_id}/defended.log",
#     params:
#         # Assume at most 100 reps and construct seed as sample-id * 100 + rep
#         seed=lambda w: int(w["sample_id"]) * 100 + int(w["rep_id"]) + 287038,
#         drop_unsat_events=lambda w: w["mode"] == "drop"
#     threads: 2
#     script:
#         "../scripts/run_front_experiment.py"
#
#
# rule delay_vs_drop__score:
#     """Compare the schedule to the defended traces."""
#     input:
#         control="results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/control.stdout.txt",
#         control_pcap="results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/control.pcapng",
#         defended="results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/defended.stdout.txt",
#         defended_pcap="results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/defended.pcapng",
#         schedule="results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/schedule.csv",
#     output:
#         "results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/scores.csv"
#     log:
#         "results/delay-vs-drop/{mode}/{defence}/{sample_id}_{rep_id}/scores.log"
#     params:
#         pad_only=lambda w: w["defence"] == "front",
#         sample_id="{mode}/{defence}/{sample_id}_{rep_id}",
#         **config["experiment"]["scores"]
#     script:
#         "../scripts/calculate_score.py"
#
#
# def delay_vs_drop__all_score__inputs(wildcards):
#     dep_directory = checkpoints.url_dependency_graphs.get(**wildcards).output[0]
#     sample_ids = glob_wildcards(dep_directory + "/{sample_id}.json").sample_id
#
#     # Limit the samples
#     sample_ids = sample_ids[:config["experiment"]["delay_vs_drop"]["max_samples"]]
#     return expand(
#         rules.delay_vs_drop__score.output, sample_id=sample_ids, rep_id="00",
#         mode=["delay", "drop"], defence=["front", "tamaraw"]
#     )
#
#
# rule delay_vs_drop__all_score:
#     """Compute scores for a FRONT evaluation on all of the dependency graphs"""
#     input:
#         delay_vs_drop__all_score__inputs
#     output:
#         "results/delay-vs-drop/scores.csv"
#     run:
#         pd.concat(
#             [pd.read_csv(f) for f in input], ignore_index=True
#         ).to_csv(output[0], index=False)
