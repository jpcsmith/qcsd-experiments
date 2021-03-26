#: Allow only 1 simultaneous packet capture by default
workflow.global_resources.setdefault("cap_iface",  1)

# select the CSDEF config
# csdef_config ="../neqo-qcd/neqo-csdef/src/config.toml"
csdef_config ="front-config/config2.toml"

#: Specify constraints on the wildcards
wildcard_constraints:
    sample_id="\d+",
    rep_id="\d+"

ruleorder: successful_collection > collect_front_defended

rule create_chaff_schedule:
    """Creates a schedule for chaff traffic sampled from Rayleigh."""
    params:
        seed=0xBADDCAFE
    output:
        schedule="results/collect/front_defended/{sample_id}_{rep_id}/chaff_schedule.csv",
        rnd_seed="results/collect/front_defended/{sample_id}_{rep_id}/rnd_seed.txt"
    run:
        from pyqcd.rayleigh import sample_rayleigh
        import toml
        with open(csdef_config, "r") as f:
            front_config=toml.loads(f.read())['front_defence']
        with open(output.rnd_seed, "w") as flog:
            flog.write(f'{params.seed}\n')
        sample_rayleigh.create_trace( seed=int(params.seed),
                                      N_TX=front_config['n_client_packets'],
                                      N_RX=front_config['n_server_packets'],
                                      W_min=front_config['peak_minimum'],
                                      W_max=front_config['peak_maximum'],
                                      size=front_config['packet_size'],
                                      outcsv=output.schedule
                                     )

rule create_chaff_schedule_fast:
    """Creates the chaff schedule once for sample 000_0 then copies it to all folders."""
    input:
        schedule="results/collect/front_defended/000_0/chaff_schedule.csv",
        seed="results/collect/front_defended/000_0/rnd_seed.txt"
    shell:"""\
        find results/collect/front_defended/* -type d -exec cp {input.schedule} {{}} \;
        find results/collect/front_defended/* -type d -exec cp {input.seed} {{}} \;
        """


checkpoint collect_front_defended:
    """Collect defended QUIC traces shaped with the FRONT defence."""
    input:
        url_dep="results/determine-url-deps/dependencies/{sample_id}.csv",
        schedule="results/collect/front_defended/{sample_id}_{rep_id}/chaff_schedule.csv"
    output:
        stdout="results/collect/front_defended/{sample_id}_{rep_id}/stdout.txt",
        dummy_ids="results/collect/front_defended/{sample_id}_{rep_id}/dummy_streams.txt",
        sampled_schedule="results/collect/front_defended/{sample_id}_{rep_id}/schedule.csv",
        pcap="results/collect/front_defended/{sample_id}_{rep_id}/trace.pcapng",
        success = "results/collect/front_defended/{sample_id}_{rep_id}/success_collect",
    log:
        "results/collect/front_defended/{sample_id}_{rep_id}/stderr.txt"
    resources:
        cap_iface=1
    shell: """\
        CSDEF_DUMMY_ID={output.dummy_ids} CSDEF_INPUT_TRACE={input.schedule} CSDEF_DUMMY_SCHEDULE={output.sampled_schedule} \
        RUST_LOG=neqo_transport=info,debug python3 -m pyqcd.collect.neqo_capture_client \
            --pcap-file {output.pcap} -- --url-dependencies-from {input.url_dep} \
            > {output.stdout} 2> {log}
        
        if tshark -r {output.pcap} -Y 'quic.frame_type in {{0x1c..0x1d}}' -Tfields -e 'quic.cc.reason_phrase' | grep -q -E 'kthx4shaping|kthxbye' ; then
            touch {output.success};
        fi
        """

checkpoint collect_front_baseline:
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
        CSDEF_NO_SHAPING=1 RUST_LOG=neqo_transport=info,debug \
        python3 -m pyqcd.collect.neqo_capture_client --pcap-file {output.pcap} \
                -- --url-dependencies-from {input} > {output.stdout} 2> {log}
        """

checkpoint collect_tamaraw_defended:
    """Collect defended QUIC traces shaped with the Tamaraw defence."""
    input:
        url_dep="results/determine-url-deps/dependencies/{sample_id}.csv",
        schedule="results/collect/tamaraw_defended/{sample_id}_{rep_id}/shape_schedule.csv"
    output:
        stdout="results/collect/tamaraw_defended/{sample_id}_{rep_id}/stdout.txt",
        dummy_ids="results/collect/tamaraw_defended/{sample_id}_{rep_id}/dummy_streams.txt",
        sampled_schedule="results/collect/tamaraw_defended/{sample_id}_{rep_id}/schedule.csv",
        pcap="results/collect/tamaraw_defended/{sample_id}_{rep_id}/trace.pcapng",
        success = "results/collect/tamaraw_defended/{sample_id}_{rep_id}/success_collect",
    log:
        "results/collect/tamaraw_defended/{sample_id}_{rep_id}/stderr.txt"
    resources:
        cap_iface=1
    shell: """\
        CSDEF_DUMMY_ID={output.dummy_ids} CSDEF_INPUT_TRACE_S={input.schedule} CSDEF_DUMMY_SCHEDULE={output.sampled_schedule} \
        CSDEF_SHAPER_CONFIG={csdef_config} \
        RUST_LOG=neqo_transport=info,debug python3 -m pyqcd.collect.neqo_capture_client \
            --pcap-file {output.pcap} -- --url-dependencies-from {input.url_dep} \
            > {output.stdout} 2> {log}
        
        if tshark -r {output.pcap} -Y 'quic.frame_type in {{0x1c..0x1d}}' -Tfields -e 'quic.cc.reason_phrase' | grep -q -E 'kthx4shaping|kthxbye' ; then
            touch {output.success};
        fi
        """

rule front_baseline_csv:
    """Parse baseline trace into csv file """
    input:
        pcap=rules.collect_front_baseline.output["pcap"]
    output:
        "results/collect/front_baseline/{sample_id}_{rep_id}/trace.csv"
    run:
        import pandas as pd
        from pyqcd.parse import parse_quic

        pd.DataFrame(
            parse_quic.parse_all_traffic(str(input.pcap))
        ).to_csv(str(output), header=True, index=False)

rule front_trace_csv:
    """Parses all packets in defended trace to csv"""
    input:
        pcap=rules.collect_front_defended.output["pcap"]
    output:
        "results/collect/front_defended/{sample_id}_{rep_id}/front_traffic.csv"
    run:
        import pandas as pd
        from pyqcd.parse import parse_quic

        pd.DataFrame(
            parse_quic.parse_all_traffic(str(input.pcap))
        ).to_csv(str(output), header=True, index=False)

rule front_chaff_csv:
    """Extracts the chaff traffic from a pcap of defended trace and saves it in a csv"""
    input:
        dummy_ids=rules.collect_front_defended.output["dummy_ids"],
        pcap=rules.collect_front_defended.output["pcap"]
    output:
        "results/collect/front_defended/{sample_id}_{rep_id}/front_cover_traffic.csv"
    run:
        import pandas as pd
        from pyqcd.parse import parse_quic

        pd.DataFrame(
            parse_quic.parse_chaff_traffic(str(input.pcap), str(input.dummy_ids))
        ).to_csv(str(output), header=True, index=False)

rule tamaraw_target_csv:
    """Creates the tamaraw target schedule from the baseline"""
    input:
        baseline=rules.front_baseline_csv.output[0],
    output:
        "results/collect/tamaraw_defended/{sample_id}_{rep_id}/shape_schedule.csv"
    log:
        "results/collect/tamaraw_defended/{sample_id}_{rep_id}/shape_schedule.txt"
    run:
        import pandas as pd
        from pyqcd.tamaraw import tamaraw as tw

        pd.DataFrame(
            tw.create_target(str(input.baseline))
        ).to_csv(str(output), index=False, header=False)

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

def collect_front_defended_single__all_input(wildcards):
    input_dir = checkpoints.url_dependencies__csv.get(**wildcards).output[0]
    sample_ids = glob_wildcards(input_dir + "/{sample_id}.csv").sample_id

    return expand(rules.collect_front_defended.output["pcap"], sample_id=sample_ids,
                  rep_id=0)

def collect_front_baseline_single__all_input(wildcards):
    input_dir = checkpoints.url_dependencies__csv.get(**wildcards).output[0]
    sample_ids = glob_wildcards(input_dir + "/{sample_id}.csv").sample_id

    return expand(rules.collect_front_baseline.output["pcap"], sample_id=sample_ids,
                  rep_id=0)


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

rule collect_front_defended_single__all:
    """Determines the number of URLs samples to be collected and starts the
    collection."""
    input: collect_front_defended_single__all_input
    message: "rule collect_front_defended__all:\n\tConvenience method for collecting the FRONT defended samples"

rule collect_front_baseline_single__all:
    """Determines the number of URLs samples to be collected and starts the
    collection."""
    input: collect_front_baseline_single__all_input
    message: "rule collect_front_baseline__all:\n\tConvenience method for collecting the FRONT baseline samples"

rule successful_collection:
    """From the trace pcap, determines if the collection completed successfully,
    creating a telltale empty file."""
    input: "results/collect/front_defended/{sample_id}_{rep_id}/trace.pcapng"
    output: "results/collect/front_defended/{sample_id}_{rep_id}/success_collect"
    log: "results/collect/front_defended/{sample_id}_{rep_id}/succ_log.txt"
    shell: """\
        if tshark -r {input} -Y 'quic.frame_type in {{0x1c..0x1d}}' -Tfields -e 'quic.cc.reason_phrase' | grep -q kthx4shaping ; then
            touch {output};
        fi
    """

def successful_collect_input__all(wildcards):
    collect_dir = rules.collect_front_defended.output[3] # points to the pcap
    s_ids = glob_wildcards(collect_dir).sample_id
    s_rep = glob_wildcards(collect_dir).rep_id
    

    return expand(rules.successful_collection.output[0], zip, sample_id=s_ids,
            rep_id=s_rep)

rule successful_collection__all:
    "Determines the samples that already have a trace.pcapng output and creates the success file if correct"
    input: successful_collect_input__all
    message: "To run with '-k' flag or it will stop at the first unsuccess"