rule ml_eval__collect_tamaraw:
    input:
        "results/determine-url-deps/dependencies/"
    output:
        directory("results/ml-eval/tamaraw/")
    log:
        "results/ml-eval/collect-tamaraw.log"
    threads:
        workflow.cores
    params:
        neqo_args=[
            "--header", "user-agent", config["user_agent"],
            "--tamaraw-modulo", "100", "--tamaraw-rate-in", "5", "--tamaraw-rate-out", "20",
            "--defence-packet-size", "750",
        ],
        n_regions=1,
        n_clients_per_region=2,
        n_instances=2,
        n_monitored=3,
        n_unmonitored=3,
        max_failures=3,
    script:
        "../scripts/run_collection.py"



# rule ml_eval__collect:
#     """Collect control, FRONT-defended, and Tamaraw-defended PCAPs."""
#     input:
#         url_dep="results/determine-url-deps/dependencies/{sample_id}.json"
#     output:
#         # Control output
#         control="results/ml-eval/dataset/{sample_id}_{rep_id}/control.stdout.txt",
#         control_pcap="results/ml-eval/dataset/{sample_id}_{rep_id}/control.pcapng",
#         # FRONT defence outputs
#         front="results/ml-eval/dataset/{sample_id}_{rep_id}/front.stdout.txt",
#         front_pcap="results/ml-eval/dataset/{sample_id}_{rep_id}/front.pcapng",
#         front_schedule="results/ml-eval/dataset/{sample_id}_{rep_id}/front-schedule.csv",
#         # Tamaraw defence outputs
#         tamaraw="results/ml-eval/dataset/{sample_id}_{rep_id}/tamaraw.stdout.txt",
#         tamaraw_pcap="results/ml-eval/dataset/{sample_id}_{rep_id}/tamaraw.pcapng",
#         tamaraw_schedule="results/ml-eval/dataset/{sample_id}_{rep_id}/tamaraw-schedule.csv",
#     log:
#         "results/ml-eval/dataset/{sample_id}_{rep_id}/script.log",
#         control="results/ml-eval/dataset/{sample_id}_{rep_id}/control.log",
#         front="results/ml-eval/dataset/{sample_id}_{rep_id}/front.log",
#         tamaraw="results/ml-eval/dataset/{sample_id}_{rep_id}/tamaraw.log",
#     params:
#         # Assume at most 1000 reps and construct seed as (sample-id * 1000) + rep
#         seed=lambda w: int(w["sample_id"]) * 1000 + int(w["rep_id"])
#     threads: 2
#     script:
#         "../scripts/collect_ml_sample.py"
#
#
# def ml_eval__dataset__inputs(wildcards, use_shortcut=True):
#     """Input function to select samples for the monitored and unmonitored datasets."""
#     ml_eval = config["experiment"]["ml_eval"]
#
#     if use_shortcut:
#         from pathlib import Path
#         sample_ids = Path("results/determine-url-deps/dependency-ids.txt").read_text().split()
#     else:
#         # Get the sample ids for which we have dependency graphs
#         dep_directory = checkpoints.url_dependency_graphs.get(**wildcards).output[0]
#         sample_ids = glob_wildcards(dep_directory + "/{sample_id}.json").sample_id
#
#     # Determine the IDs of the samples to use for the monitored and unmonitored
#     n_monitored = ml_eval["monitored"]
#     n_mon_collect = int(n_monitored["samples"] * (1 + n_monitored["extra"]) + 0.5)
#
#     n_unmonitored = ml_eval["unmonitored"]
#     n_unmon_collect = int(n_unmonitored["samples"] * (1 + n_unmonitored["extra"]) + 0.5)
#
#     n_required = n_mon_collect + n_unmon_collect
#     assert len(sample_ids) >= n_required, (
#         f"not enough urls for dataset, reqiured: {n_required}, available: {len(sample_ids)}"
#     )
#
#     inputs_ = {}
#     rep_ids = [f"{i:02d}" for i in range(ml_eval["monitored"]["instances"])]
#     inputs_["monitored"] = expand(
#         rules.ml_eval__collect.output, sample_id=sample_ids[:n_mon_collect],
#         rep_id=rep_ids,
#     )
#
#     rep_ids = [f"{i:02d}" for i in range(ml_eval["unmonitored"]["instances"])]
#     inputs_["unmonitored"] = expand(
#         rules.ml_eval__collect.output, rep_id=rep_ids,
#         sample_id=sample_ids[n_mon_collect:(n_mon_collect + n_unmon_collect)],
#     )
#
#     return inputs_
#
#
# rule ml_eval__dataset:
#     """Combine the samples into an HDF5 dataset."""
#     output:
#         "results/ml-eval/{sim_prefix}{defence}-dataset.h5",
#     log:
#         "results/ml-eval/{sim_prefix}{defence}-dataset.log"
#     params:
#         defence="{defence}",
#         simulate=lambda w: bool(w["sim_prefix"]),
#         inputs=ml_eval__dataset__inputs
#     wildcard_constraints:
#         sim_prefix="(sim-)?"
#     threads: 16
#     script:
#         "../scripts/create_datasets.py"
#
#
# rule ml_eval__filtered_dataset:
#     """Remove packets below the configured threshold from the monitored and
#     unmonitored traces."""
#     input:
#         "results/ml-eval/{basename}.h5"
#     output:
#         "results/ml-eval/{basename}-filtered.h5"
#     params:
#         size=config["experiment"]["ml_eval"]["min_packet_size"]
#     shell:
#         "workflow/scripts/remove-small-packets {params.size} {input} {output}"
