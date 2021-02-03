#: Allow only 1 simultaneous packet capture by default
workflow.global_resources.setdefault("cap_iface",  1)

#: Specify constraints on the wildcards
wildcard_constraints:
    sample_id="\d+",
    rep_id="\d+"

rule collect_front_defended:
    """Collect defended QUIC traces shaped with the FRONT defence."""
    input:
        "results/determine-url-deps/dependencies/{sample_id}.csv"
    output:
        stdout="results/collect/front_defended/{sample_id}_{rep_id}/stdout.txt",
        dummy_ids="results/collect/front_defended/{sample_id}_{rep_id}/dummy_streams.txt",
        sampled_schedule="results/collect/front_defended/{sample_id}_{rep_id}/schedule.csv",
        pcap="results/collect/front_defended/{sample_id}_{rep_id}/trace.pcapng",
    log:
        "results/collect/front_defended/{sample_id}_{rep_id}/stderr.txt"
    resources:
        cap_iface=1
    shell: """\
        CSDEF_DUMMY_ID={output.dummy_ids} CSDEF_DUMMY_SCHEDULE={output.sampled_schedule} \
        RUST_LOG=neqo=debug python3 -m pyqcd.collect.neqo_capture_client \
            --pcap-file {output.pcap} -- --url-dependencies-from {input} \
            > {output.stdout} 2> {log}
        """

rule collect_front_baseline:
    """Collect normal QUIC traces as baseline."""
    input:
        "results/determine-url-deps/dependencies/{sample_id}.csv"
    output:
        stdout="results/collect/front_baseline/{sample_id}_{rep_id}/stdout.txt",
        pcap="results/collect/front_baseline/{sample_id}_{rep_id}/trace.pcapng",
    log:
        "results/collect/front_baseline/{sample_id}_{rep_id}/stderr.txt"
    resources:
        cap_iface=1
    shell: """\
        CSDEF_NO_SHAPING=1 RUST_LOG=neqo=debug \
        python3 -m pyqcd.collect.neqo_capture_client --pcap-file {output.pcap} \
                -- --url-dependencies-from {input} > {output.stdout} 2> {log}
        """

rule front_trace_csv:
    input:
        dummy_ids=rules.collect_front_defended.output["dummy_ids"],
        pcap=rules.collect_front_defended.output["pcap"]
    output:
        "results/collect/front_defended/{sample_id}/front_cover_traffic.csv"
    shell: """\
        tshark -r {input.pcap} \
            -Y "quic.stream.stream_id in {{$(<{input.dummy_ids})}} \
                or quic.padding_length > 0 or quic.packet_number == 0" \
            -Tfields -e frame.time_epoch -e quic.stream.length -e quic.length -e quic.stream.stream_id \
            -e quic.padding_length -e udp.srcport -e quic.packet_number \
            -E header=y -E separator=\; > {output}
        """

def collect_front_defended__all_input(wildcards):
    input_dir = checkpoints.url_dependencies__csv.get(**wildcards).output[0]
    sample_ids = glob_wildcards(input_dir + "/{sample_id}.csv").sample_id

    return expand(rules.collect_front_defended.output["pcap"], sample_id=sample_ids,
                  rep_id=range(config["collect_reps"]))

def collect_front_baseline__all_input(wildcards):
    input_dir = checkpoints.url_dependencies__csv.get(**wildcards).output[0]
    sample_ids = glob_wildcards(input_dir + "/{sample_id}.csv").sample_id

    return expand(rules.collect_front_baseline.output["pcap"], sample_id=sample_ids,
                  rep_id=range(config["collect_reps"]))



rule collect_front_defended__all:
    """Determines the number of URLs samples to be collected and starts the
    collection."""
    input: collect_front_defended__all_input
    message: "rule collect_front_defended__all:\n\tConvenience method for collecting the FRONT defended samples"

rule collect_front_baseline__all:
    """Determines the number of URLs samples to be collected and starts the
    collection."""
    input: collect_front_baseline__all_input
    message: "rule collect_front_baseline__all:\n\tConvenience method for collecting the FRONT baseline samples"
