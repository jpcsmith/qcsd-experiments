rule front_eval_collect:
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


rule front_eval_score:
    """Compare the schedule to the defended traces."""
    input:
        schedule=rules.front_eval_collect.output["schedule"],
        # Derived from the pcaps and logs by a rule in common.smk
        defence="results/front-eval/{sample_id}_{rep_id}/front-composition.csv",
    output:
        temp("results/front-eval/{sample_id}_{rep_id}/scores.csv")
    params:
        **config["experiment"]["front_single_eval"]["scores"]
    script:
        "../scripts/calculate_score.py"
