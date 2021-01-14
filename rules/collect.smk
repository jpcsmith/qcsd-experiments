#: Allow only 1 simultaneous packet capture by default
workflow.global_resources.setdefault("cap_iface",  1)


rule collect_front_defended:
    """Collect defended QUIC traces shaped with the FRONT defence."""
    input:
        "results/determine-url-deps/dependencies/{sample_id}.csv"
    output:
        stdout="results/collect/front_defended/{sample_id}/stdout.txt",
        dummy_ids="results/collect/front_defended/{sample_id}/dummy_streams.txt",
        sampled_schedule="results/collect/front_defended/{sample_id}/schedule.csv",
        pcap="results/collect/front_defended/{sample_id}/trace.pcapng",
    log:
        "results/collect/front_defended/{sample_id}/stderr.txt"
    resources:
        cap_iface=1
    shell: """\
        CSDEF_DUMMY_ID={output.dummy_ids} CSDEF_DUMMY_SCHEDULE={output.sampled_schedule} \
        python3 -m pyqcd.collect.neqo_capture_client --pcap-file {output.pcap} \
                -- --url-dependencies-from {input} > {output.stdout} 2> {log}
        """


def collect_front_defended__all_input(wildcards):
    input_dir = checkpoints.url_dependencies__csv.get(**wildcards).output[0]
    sample_ids = glob_wildcards(input_dir + "/{sample_id}.csv").sample_id

    return expand(rules.collect_front_defended.output["pcap"], sample_id=sample_ids)


rule collect_front_defended__all:
    """Determines the number of URLs samples to be collected and starts the
    collection."""
    input: collect_front_defended__all_input
    message: "rule collect_front_defended__all:\n\tConvenience method for collecting the FRONT defended samples"
