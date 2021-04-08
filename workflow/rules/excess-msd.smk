#: Specify constraints on the wildcards
wildcard_constraints:
    sample_id="\d{3}",
    rep_id="\d{2}",
    excess_msd="\d+",


rule excess_msd_collect:
    """Collect a sample of chaff-only mode for constant rate schedule and excess MSD."""
    input:
        url_dep="results/determine-url-deps/dependencies/{sample_id}.json",
        schedule="results/excess_msd/constant_{rate}Mbps_{duration}s_{interval}ms.csv".format(
            **config["experiment"]["excess_msd"]["padding"]),
    output:
        stdout="results/excess_msd/{excess_msd}/{sample_id}_{rep_id}/stdout.txt",
        pcap="results/excess_msd/{excess_msd}/{sample_id}_{rep_id}/trace.pcapng",
    log:
        "results/excess_msd/{excess_msd}/{sample_id}_{rep_id}/stderr.txt"
    params:
        msd_limit_excess="{excess_msd}"
    threads: 2
    shell:
        "RUST_LOG={config[neqo_log_level]}"
        " python3 workflow/scripts/neqo_capture_client.py --pcap-file {output.pcap}"
        " --ignore-errors"
        " --"
        " --target-trace {input.schedule} --msd-limit-excess {params.msd_limit_excess}"
        " --header user-agent 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'"
        " --pad-only-mode true --url-dependencies-from {input.url_dep}"
        " > {output.stdout} 2> {log}"


def excess_msd_collect_all__inputs(wildcards):
    dep_directory = checkpoints.url_dependency_graphs.get(**wildcards).output[0]
    sample_ids = glob_wildcards(dep_directory + "/{sample_id}.json").sample_id

    emsd_config = config["experiment"]["excess_msd"]
    repetitions = [f"{rep:02d}" for rep in range(emsd_config["repetitions"])]
    return expand(rules.excess_msd_collect.output["pcap"], sample_id=sample_ids,
                  rep_id=repetitions, excess_msd=emsd_config["excess_values"])


rule excess_msd_collect_all:
    input:
        excess_msd_collect_all__inputs
