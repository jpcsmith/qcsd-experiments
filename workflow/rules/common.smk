def get_threads_for_classifier(wildcards) -> int:
    """Returns the number of threads to use for the classifier specified
    in the wildcards as `classifier`.
    """
    return 128 if wildcards["classifier"] != "kfp" else 4


def to_memory_per_core(mem_mb: int):
    """Return a function that computes the memory per thread/core."""
    def _memory_per_core(wildcards, input, threads) -> int:
        return int(mem_mb / threads)
    return _memory_per_core


def build_neqo_args(exp_config):
    def _builder(wildcards):
        if wildcards["defence"] == "front":
            args = exp_config["front"]
            result = [
                "--defence", "front",
                "--defence-packet-size", args["packet_size"],
                "--front-max-client-pkts", args["max_client_packets"],
                "--front-max-server-pkts", args["max_server_packets"],
                "--front-peak-max", args["peak_maximum"],
                "--front-peak-min", args["peak_minimum"],
            ]
            if "use_empty_resources" in args:
              result += ["--use-empty-resources", str(args["use_empty_resources"]).lower()]
            return result
        if wildcards["defence"] == "tamaraw":
            args = exp_config["tamaraw"]
            result = [
                "--defence", "tamaraw",
                "--defence-packet-size", args["packet_size"],
                "--tamaraw-rate-in", args["rate_in"],
                "--tamaraw-rate-out", args["rate_out"],
                "--tamaraw-modulo", args["packet_multiple"],
            ]
            if "msd_limit_excess" in args:
              result += ["--msd-limit-excess", args["msd_limit_excess"]]
            if "use_empty_resources" in args:
              result += ["--use-empty-resources", str(args["use_empty_resources"]).lower()]
            return result
        if wildcards["defence"] == "undefended":
            return ["--defence", "none"]
        raise ValueError("Unsupported defence: %r", wildcards["defence"])
    return _builder


localrules: combine_varcnn_predictions
rule combine_varcnn_predictions:
    """Combines VarCNN time and sizes predictions."""
    input:
        sizes="{path}/varcnn-sizes-{i}.csv",
        times="{path}/varcnn-time-{i}.csv"
    output:
        "{path}/varcnn-{i}.csv"
    wildcard_constraints:
        i="\d+"
    run:
        import pandas as pd
        sizes = pd.read_csv(input["sizes"]).set_index("y_true", append=True)
        times = pd.read_csv(input["times"]).set_index("y_true", append=True)
        combined = (sizes + times) / 2
        combined = combined.reset_index(level="y_true", drop=False)
        combined.to_csv(output[0], header=True, index=False)
