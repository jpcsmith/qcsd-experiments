debug_config = {
    "n_monitored": 1,
    "n_instances": 100,
    "n_unmonitored": 100,
}

wildcard_constraints:
    sim_suffix="(\.sim)?",
    suffix="(\.sim|\.filtered)?",
    n_packets="\d+",

rule debug__collect_undefended:
    input:
        "results/webpage-graphs/graphs/"
    output:
        directory("results/debug/undefended/dataset/")
    log:
        "results/debug/undefended/dataset.log"
    threads:
        workflow.cores
    params:
        neqo_args=lambda w: [
            "--header", "user-agent", config["user_agent"],
        ],
        **debug_config,
    script:
        "../scripts/run_collection.py"


rule debug__collect_no_change:
    input:
        "results/webpage-graphs/graphs/"
    output:
        directory("results/debug/defended/dataset/")
    log:
        "results/debug/defended/dataset.log"
    threads:
        workflow.cores
    params:
        neqo_args=lambda w: [
            "--header", "user-agent", config["user_agent"],
            "--defence", "tamaraw",
            "--tamaraw-modulo", "300",
            "--tamaraw-rate-in", "5",
            "--tamaraw-rate-out", "20",
            "--defence-packet-size", "750",
        ],
        **debug_config,
    script:
        "../scripts/run_collection.py"


rule debug__collect_tail_wait:
    input:
        "results/webpage-graphs/graphs/"
    output:
        directory("results/debug/tail-wait/dataset/")
    log:
        "results/debug/tail-wait/dataset.log"
    threads:
        workflow.cores
    params:
        neqo_args=lambda w: [
            "--header", "user-agent", config["user_agent"],
            "--defence", "tamaraw",
            "--tamaraw-modulo", "300",
            "--tamaraw-rate-in", "5",
            "--tamaraw-rate-out", "20",
            "--defence-packet-size", "750",
            "--tail-wait", "100",
        ],
        **debug_config,
    script:
        "../scripts/run_collection.py"


rule debug__collect_full_packet:
    input:
        "results/webpage-graphs/graphs/"
    output:
        directory("results/debug/full-packet/dataset/")
    log:
        "results/debug/full-packet/dataset.log"
    threads:
        workflow.cores
    params:
        neqo_args=lambda w: [
            "--header", "user-agent", config["user_agent"],
            "--defence", "tamaraw",
            "--tamaraw-modulo", "300",
            "--tamaraw-rate-in", "5",
            "--tamaraw-rate-out", "20",
            "--defence-packet-size", "1350",
        ],
        **debug_config,
    script:
        "../scripts/run_collection.py"


rule debug__collect_buflo:
    input:
        "results/webpage-graphs/graphs/",
        schedule="results/debug/buflo/constant_1.2Mbps_5s_5ms.csv"
    output:
        directory("results/debug/buflo/dataset/")
    log:
        "results/debug/buflo/dataset.log"
    threads:
        workflow.cores
    params:
        neqo_args=lambda w: [
            "--header", "user-agent", config["user_agent"],
            "--defence", "schedule",
            "--target-trace", "results/debug/buflo/constant_1.2Mbps_5s_5ms.csv",
            "--target-trace-type", "chaff-and-shape",
        ],
        **debug_config,
    script:
        "../scripts/run_collection.py"


rule debug__dataset:
    input:
        "results/debug/{factor}/dataset/"
    output:
        "results/debug/{factor}/dataset{sim_suffix}.h5"
    params:
        **debug_config,
        simulate=lambda w: bool(w["sim_suffix"]),
    script:
        "../scripts/create_dataset.py"


rule debug__filtered_dataset:
    input:
        "results/debug/{factor}/dataset.h5"
    output:
        "results/debug/{factor}/dataset.filtered.h5"
    params:
        size=150,
    shell:
        "workflow/scripts/remove-small-packets {params.size} {input} {output}"


ruleorder: debug__buflo_limit_packets > debug__filtered_dataset
rule debug__buflo_limit_packets:
    input:
        "results/debug/buflo/dataset{suffix}.h5"
    output:
        "results/debug/buflo-{n_packets}/dataset{suffix}.h5"
    params:
        n_packets=lambda w: int(w["n_packets"])
    run:
        shell("cp {input} {output}")
        import h5py
        with h5py.File(str(output[0]), mode="a") as h5file:
            for i in range(len(h5file["labels"])):
                h5file["sizes"][i] = h5file["sizes"][i][:params["n_packets"]]
                h5file["timestamps"][i] = h5file["timestamps"][i][:params["n_packets"]]


ruleorder: debug__buflo_no_handshake > debug__filtered_dataset
rule debug__buflo_no_handshake:
    input:
        "results/debug/buflo/dataset{suffix}.h5"
    output:
        "results/debug/buflo-hsk-{n_packets}/dataset{suffix}.h5"
    params:
        n_packets=lambda w: int(w["n_packets"])
    run:
        shell("cp {input} {output}")
        import h5py
        with h5py.File(str(output[0]), mode="a") as h5file:
            for i in range(len(h5file["labels"])):
                h5file["sizes"][i] = h5file["sizes"][i][params["n_packets"]:]
                h5file["timestamps"][i] = h5file["timestamps"][i][params["n_packets"]:]
                # Make all times relative to the new first timestamp
                h5file["timestamps"][i] -= h5file["timestamps"][i][0]
                h5file["timestamps"][i][0] = 0


ruleorder: debug__buflo_directioned > debug__filtered_dataset
rule debug__buflo_directioned:
    input:
        "results/debug/{factor}/dataset{suffix}.h5"
    output:
        "results/debug/{factor}-{direction}/dataset{suffix}.h5"
    params:
        direction="{direction}"
    wildcard_constraints:
        direction="(in|out)"
    run:
        shell("cp {input} {output}")
        import h5py
        import numpy
        with h5py.File(str(output[0]), mode="a") as h5file:
            for i in range(len(h5file["labels"])):
                mask = h5file["sizes"][i] > 0 if params["direction"] == "out" else h5file["sizes"][i] < 0
                h5file["sizes"][i] = h5file["sizes"][i][mask]
                h5file["timestamps"][i] = h5file["timestamps"][i][mask]
                # Make all times relative to the new first timestamp
                h5file["timestamps"][i] -= h5file["timestamps"][i][0]
                h5file["timestamps"][i][0] = 0

                dummy_packets = [500, 500] if params["direction"] == "in" else [-500, -500]
                h5file["sizes"][i] = numpy.insert(h5file["sizes"][i], 1, dummy_packets)
                h5file["timestamps"][i] = numpy.insert(h5file["timestamps"][i], 1, [0.0000001, 0.0000002])


rule debug__features:
    input:
        "results/debug/{factor}/dataset{suffix}.h5"
    output:
        "results/debug/{factor}/features{suffix}.h5"
    log:
        "results/debug/{factor}/features{suffix}.log"
    shell:
        "workflow/scripts/extract-features {input} {output} 2> {log}"


rule debug__splits:
    """Create train-test-validation splits of the dataset."""
    input:
        "results/debug/{factor}/features{suffix}.h5"
    output:
        "results/debug/{factor}/predict{suffix}/split-0.json"
    params:
        seed=42,
        n_folds=5,
        validation_size=0.1,
    script:
        "../scripts/split_dataset.py"


rule debug__predictions:
    input:
        dataset=rules.debug__features.output,
        splits=rules.debug__splits.output,
    output:
        "results/debug/{factor}/predict{suffix}/{classifier}-0.csv"
    log:
        "results/debug/{factor}/predict{suffix}/{classifier}-0.log"
    params:
        classifier="{classifier}",
        classifier_args=lambda w: ("--classifier-args n_jobs=4,feature_set=kfp"
                                   if w["classifier"] == "kfp" else "")
    threads:
        lambda w: min(workflow.cores, 16) if w["classifier"] != "kfp" else 4
    shell:
        "workflow/scripts/evaluate-classifier {params.classifier_args}"
        " {params.classifier} {input.dataset} {input.splits} {output} 2> {log}"




rule debug__all:
    input:
        [expand("results/debug/{factor}/predict{suffix}/{classifier}-0.csv",
            classifier=["kfp", "dfnet"],
            suffix=["", ".filtered"],
            # suffix=["", ".sim", ".filtered"],
            factor=["defended", "tail-wait", "full-packet", "buflo", "buflo-10",
                    "buflo-20", "buflo-50", "buflo-100", "buflo-200", "buflo-1000",
                    "buflo-hsk-10", "buflo-hsk-20", "buflo-in", "buflo-out"]),
         expand("results/debug/undefended/predict{suffix}/{classifier}-0.csv",
             classifier=["kfp", "dfnet"],
             suffix=["", ".filtered"])
         ]
