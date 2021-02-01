rule icrc_collect_defended:
    """Collect defended QUIC traces shaped with the FRONT defence."""
    input:
        deps="results/determine-url-deps/dependencies/{sample_id}.csv",
        target="notebooks/icrc.csv"
    output:
        stdout="results/icrc-demo/{sample_id}/stdout.txt",
        dummy_ids="results/icrc-demo/{sample_id}/dummy_streams.txt",
        sampled_schedule="results/icrc-demo/{sample_id}/schedule.csv",
        pcap="results/icrc-demo/{sample_id}/trace.pcapng",
    log:
        "results/icrc-demo/{sample_id}/stderr.txt"
    resources:
        cap_iface=1
    shell: """\
        CSDEF_DUMMY_ID={output.dummy_ids} CSDEF_DUMMY_SCHEDULE={output.sampled_schedule} \
        CSDEF_INPUT_TRACE={input.target} RUST_LOG=neqo=trace \
        python3 -m pyqcd.collect.neqo_capture_client --pcap-file {output.pcap} \
            -- --url-dependencies-from {input.deps} > {output.stdout} 2> {log}
        """

rule icrc_front_chaff_csv:
    input:
        pcap=rules.icrc_collect_defended.output["pcap"],
        dummy_ids=rules.icrc_collect_defended.output["dummy_ids"]
    output:
        "results/icrc-demo/{sample_id}/cover_traffic.csv"
    run:
        import pandas as pd
        from pyqcd.parse import parse_quic

        pd.DataFrame(
            parse_quic.parse_chaff_traffic(str(input.pcap), str(input.dummy_ids))
        ).to_csv(str(output), header=True, index=False)
