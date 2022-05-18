# TODO: Need to do dynamic seeds for the noise
tamaraw_defaults = {
    "rate_in": 5,
    "rate_out": 20,
    "packet_multiple": 300,
    "packet_size": 1200,
    "msd_limit_excess": 1500,
}

debug_config = {
    "noised_tamaraw-0.2": {
        "tamaraw": {
            **tamaraw_defaults,
            "add_noise": True,
            "noise_chance": 0.2,
            "noise_bound_lower": -800,
            "noise_bound_upper": 160,
        },
    },
    "noised_tamaraw-0.4": {
        "tamaraw": {
            **tamaraw_defaults,
            "add_noise": True,
            "noise_chance": 0.4,
            "noise_bound_lower": -800,
            "noise_bound_upper": 160,
        },
    },
    "higher_pkt_size": {
        "tamaraw": {
            **tamaraw_defaults,
            "packet_size": 1360
        }
    },
    "max_udp_payload_size": {
        "tamaraw": {
            **tamaraw_defaults,
            "max_udp_payload_size": 1272
        }
    }
}


rule tamaraw_debug__next_run:
    input:
        "results/tamaraw-debug/higher_pkt_size/10x50+0/classifier~kfp/predictions.csv",
        "results/tamaraw-debug/noised_tamaraw-0.2/10x50+0/classifier~kfp/predictions.csv",
        "results/tamaraw-debug/noised_tamaraw-0.4/10x50+0/classifier~kfp/predictions.csv",


rule tamaraw_debug__collect__to_binary:
    """Convert a collected file-based dataset to a binary dataset for faster reads
    (pattern rule)."""
    output:
        "results/tamaraw-debug/{setting}/{n_mon}x{n_samples}+{n_unmon}/dataset.h5"
    input:
        "results/tamaraw-debug/{setting}/{n_mon}x{n_samples}+{n_unmon}/dataset/"
    params:
        n_monitored=lambda w: int(w["n_mon"]),
        n_unmonitored=lambda w: int(w["n_unmon"]),
        n_instances=lambda w: int(w["n_samples"]),
    script:
        "../scripts/create_dataset.py"


def setting_neqo_args(wildcards):
    w_copy = dict(wildcards)
    w_copy["defence"] = "tamaraw"
    setting_conf = debug_config[wildcards["setting"]]
    args_str = build_neqo_args(setting_conf)(w_copy)
    print(args_str)
    return args_str


rule tamaraw_debug__collect:
    """Collect samples for the single-connection machine-learning evaluation
    (pattern rule)."""
    output:
        directory("results/tamaraw-debug/{setting}/{n_mon}x{n_samples}+{n_unmon}/dataset/")
    input:
        "results/webpage-graphs/graphs/"
    log:
        "results/tamaraw-debug/{setting}/{n_mon}x{n_samples}+{n_unmon}/dataset.log"
    threads:
        workflow.cores
    params:
        neqo_args=setting_neqo_args,
        n_monitored=lambda w: int(w["n_mon"]),
        n_unmonitored=lambda w: int(w["n_unmon"]),
        n_instances=lambda w: int(w["n_samples"]),
        max_failures=1,
    script:
        "../scripts/run_collection.py"
