rule ml_eval__collect:
    input:
        url_dep="results/determine-url-deps/dependencies/{sample_id}.json"
    output:
        # Control output
        control="results/ml-eval/dataset/{sample_id}_{rep_id}/control.stdout.txt",
        control_pcap="results/ml-eval/dataset/{sample_id}_{rep_id}/control.pcapng",
        # FRONT defence outputs
        front="results/ml-eval/dataset/{sample_id}_{rep_id}/front.stdout.txt",
        front_pcap="results/ml-eval/dataset/{sample_id}_{rep_id}/front.pcapng",
        front_schedule="results/ml-eval/dataset/{sample_id}_{rep_id}/front-schedule.csv",
        # Tamaraw defence outputs
        tamaraw="results/ml-eval/dataset/{sample_id}_{rep_id}/tamaraw.stdout.txt",
        tamaraw_pcap="results/ml-eval/dataset/{sample_id}_{rep_id}/tamaraw.pcapng",
        tamaraw_schedule="results/ml-eval/dataset/{sample_id}_{rep_id}/tamaraw-schedule.csv",
    log:
        "results/ml-eval/dataset/{sample_id}_{rep_id}/script.log",
        control="results/ml-eval/dataset/{sample_id}_{rep_id}/control.log",
        front="results/ml-eval/dataset/{sample_id}_{rep_id}/front.log",
        tamaraw="results/ml-eval/dataset/{sample_id}_{rep_id}/tamaraw.log",
    threads: 2
    script:
        "../scripts/collect_ml_sample.py"


def ml_eval__all_collect__inputs(wildcards):
    ml_eval = config["experiment"]["ml_eval"]

    # Get the sample ids for which we have dependency graphs
    dep_directory = checkpoints.url_dependency_graphs.get(**wildcards).output[0]
    sample_ids = glob_wildcards(dep_directory + "/{sample_id}.json").sample_id

    # Collect 1.XX times the samples to account for failures, rounded up
    n_mon_collect = int(ml_eval["n_monitored"]  * (1 + ml_eval["collect_extra"]) + 0.5)
    n_mon_collect_rep = ml_eval["n_per_monitored_inst"]
    n_unmon_collect = int(ml_eval["n_unmonitored"]  * (1 + ml_eval["collect_extra"]) + 0.5)

    return {
        "monitored": expand(
          rules.ml_eval__colect.output, sample_id=sample_ids[:n_mon_collect],
          rep_id=map("{:02d}".format, range(n_mon_collect_rep))
        ),
        "unmonitored": expand(
          rules.ml_eval__colect.output, rep_id="00",
          sample_id=sample_ids[n_mon_collect:(n_mon_collect + n_unmon_collect)]
        )
    }
