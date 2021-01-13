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
    shell: """\
        CSDEF_DUMMY_ID={output.dummy_ids} CSDEF_DUMMY_SCHEDULE={output.sampled_schedule} \
        python3 -m pyqcd.collect.neqo_capture_client --pcap-file {output.pcap} \
                -- --url-dependencies-from {input} > {output.stdout} 2> {log}
        """
