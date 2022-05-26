import pandas as pd


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
            if "msd_limit_excess" in args:
                result += ["--msd-limit-excess", args["msd_limit_excess"]]
            if "use_empty_resources" in args:
                result += ["--use-empty-resources", str(args["use_empty_resources"]).lower()]
            if "max_udp_payload_size" in args:
                result += ["--max-udp-payload-size", args["max_udp_payload_size"]]
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
            if "max_udp_payload_size" in args:
                result += ["--max-udp-payload-size", args["max_udp_payload_size"]]

            if "add_noise" in args:
                result += [
                    "--add-noise",
                    "--noise-chance", args["noise_chance"],
                    f"--noise-bound-lower={args['noise_bound_lower']}",
                    f"--noise-bound-upper={args['noise_bound_upper']}",
                ]
            return result
        if wildcards["defence"] == "undefended":
            return ["--defence", "none"]
        raise ValueError("Unsupported defence: %r", wildcards["defence"])
    return _builder


def combine_varcnn_predictions(input, output):
    """Combine two varcnn prediction files and write to the first output."""
    sizes = pd.read_csv(input["sizes"]).set_index("y_true", append=True)
    times = pd.read_csv(input["times"]).set_index("y_true", append=True)
    combined = (sizes + times) / 2
    combined = combined.reset_index(level="y_true", drop=False)
    combined.to_csv(output[0], header=True, index=False)


rule predict__kfp:
    """Perform hyperparameter validation and predictions for the k-FP classifier
    (pattern rule)."""
    output:
        "{path}/classifier~kfp/predictions.csv",
        feature_importance="{path}/classifier~kfp/feature-importances.csv"
    input:
        "{path}/classifier~kfp/features.csv"
    log:
        "{path}/classifier~kfp/predictions.log",
        cv_results="{path}/classifier~kfp/cv-results.csv",
    threads:
        workflow.cores
    shell:
        "workflow/scripts/evaluate_tuned_kfp.py --verbose 0 --n-jobs {threads}"
        " --cv-results-path {log[cv_results]} --feature-importance {output[feature_importance]}"
        " {input} > {output[0]} 2> {log[0]}"


rule predict__varcnn:
    """Perform hyperparameter validation and predictions for either the sizes or time
    component of the Var-CNN classifier (pattern rule)."""
    output:
        "{path}/classifier~varcnn-{feature_type}/hyperparams~{hyperparams}/predictions.csv"
    input:
        "{path}/dataset.h5"
    log:
        "{path}/classifier~varcnn-{feature_type}/hyperparams~{hyperparams}/predictions.log"
    threads:
        workflow.cores
    shell:
        "workflow/scripts/evaluate_tuned_varcnn.py --hyperparams {wildcards.hyperparams}"
        " {wildcards.feature_type} {input} > {output} 2> {log}"


rule predict__dfnet:
    """Perform hyperparameter validation and predictions for the Deep Fingerprinting
    classifier (pattern rule)."""
    output:
        "{path}/classifier~dfnet/hyperparams~{hyperparams}/predictions.csv"
    input:
        "{path}/dataset.h5"
    log:
        "{path}/classifier~dfnet/hyperparams~{hyperparams}/predictions.log"
    threads:
        workflow.cores
    shell:
        "workflow/scripts/evaluate_tuned_df.py --hyperparams {wildcards.hyperparams}"
        " {input} > {output} 2> {log}"


rule extract_features__kfp:
    """Pre-extract the k-FP features as this can be time-consuming (pattern rule)."""
    output:
        "{path}/classifier~kfp/features.csv"
    input:
        "{path}/dataset.h5"
    log:
        "{path}/classifier~kfp/features.log"
    threads: 12
    shell:
        "workflow/scripts/extract_kfp_features.py {input} > {output} 2> {log}"


rule simulate__padded_packets:
    """Create a simulated dataset where incoming packet sizes are hidden."""
    output:
        "{path}/defence~padded-{defence}/strategy~{strategy}/direction~{direction}/dataset.h5"
    input:
        "{path}/defence~{defence}/dataset.h5"
    shell:
        "workflow/scripts/hide_sizes.py --direction {wildcards.direction}"
        " {wildcards.strategy} {input} {output}"
