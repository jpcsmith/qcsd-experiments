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


def front_eval__score__inputs(wildcards):
    dep_directory = checkpoints.url_dependency_graphs.get(**wildcards).output[0]
    sample_ids = glob_wildcards(dep_directory + "/{sample_id}.json").sample_id

    sample_ids = ["0000", "0001", "0002", "0003",]  # TODO: Remove
    files = rules.front_eval__collect.output
    return {
        "control": expand(files["control"], sample_id=sample_ids, rep_id="00"),
        "control_pcap": expand(files["control_pcap"], sample_id=sample_ids, rep_id="00"),
        "defended": expand(files["front"], sample_id=sample_ids, rep_id="00"),
        "defended_pcap": expand(files["front_pcap"], sample_id=sample_ids, rep_id="00"),
        "schedule": expand(files["schedule"], sample_id=sample_ids, rep_id="00")
    }


rule front_eval__score:
    """Compare the schedule to the defended traces."""
    input:
        unpack(front_eval__score__inputs)
    output:
        "results/front-eval/scores.csv"
    params:
        **config["experiment"]["front_single_eval"]["scores"]
    script:
        "../scripts/calculate_score.py"
