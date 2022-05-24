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
    "noised_tamaraw-0.4-skewed-beta": {
        "tamaraw": {
            **tamaraw_defaults,
            "add_noise": True,
            "noise_chance": 0.4,
            "noise_bound_lower": -900,
            "noise_bound_upper": 0,
            "max_udp_payload_size": 1272
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
            "packet_size": 1250,
            "max_udp_payload_size": 1272
        }
    },
    "lower_msd_excess": {
        "tamaraw": {
            **tamaraw_defaults,
            "msd_limit_excess": 500,
            "packet_size": 1250,
            "max_udp_payload_size": 1272
        }
    },
    "multiple_of_1000": {
        "tamaraw": {
            **tamaraw_defaults,
            "packet_multiple": 1000,
            "packet_size": 1250,
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
        max_failures=1,
        timeout=60,
        configfile=workflow.configfiles[0]
    shell:
        "workflow/scripts/run_collectionv2.py --configfile {params.configfile}"
        " --n-monitored {wildcards.n_mon} --n-instances {wildcards.n_samples}"
        " --n-unmonitored {wildcards.n_unmon} --max-failures {params.max_failures}"
        " --timeout {params.timeout} {input} {output} -- {params.neqo_args} 2> {log}"


def temp__plot__inputs(wildcards, flatten: bool = False):
    defence = wildcards["defence"]
    result = {
        "$k$-FP (Single)": {
            "QCSD": f"results/ml-eval-conn/defence~{defence}/classifier~kfp/predictions.csv",
            "Simulated": f"results/ml-eval-conn/defence~simulated-{defence}/classifier~kfp/predictions.csv",
            "Undef.": f"results/ml-eval-conn/defence~undefended/classifier~kfp/predictions.csv",
        },
        "$k$-FP (Multi)": {
            "QCSD": f"results/ml-eval-mconn/defence~{defence}/classifier~kfp/predictions.csv",
            "Undef.": f"results/ml-eval-mconn/defence~undefended/classifier~kfp/predictions.csv",
        },
        "Tamaraw-100x100+10000": {
            "QCSD": "results/tamaraw-debug/max_udp_payload_size/100x100+10000/classifier~kfp/predictions.csv"
        }
    }
    if defence == "front":
        result["$k$-FP (Brows)"] = {
            "QCSD": f"results/ml-eval-brows/defence~{defence}/classifier~kfp/predictions.csv",
            "Undef.": f"results/ml-eval-brows/defence~undefended/classifier~kfp/predictions.csv",
        }

    if flatten:
        result = [v for values in result.values() for v in values.values()]
    return result


rule temp__plot:
    """Plot the defended, simulated, and undefended settings for the given defence for
    all classifiers (pattern rule)."""
    output:
        "results/plots/debug-plot-{defence}.png",
    input:
        lambda w: temp__plot__inputs(w, flatten=True)
    params:
        layout=temp__plot__inputs,
        line_styles={"QCSD": "solid", "Simulated": "dotted", "Undef.": "dashdot"},
        with_legend=lambda w: w["defence"] == "front"
    notebook:
        "../notebooks/result-analysis-curve.ipynb"
