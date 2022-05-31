simulate_kws = {
    "default": config["experiment"]["default"]["front_config"],
    "larger_packets": {
        **config["experiment"]["default"]["front_config"],
        "packet_size": 1250
    },
    "more_packets": {
        **config["experiment"]["default"]["front_config"],
        "max_client_packets": 1300,
        "max_server_packets": 1300,
    },
    "lower_peak": {
        **config["experiment"]["default"]["front_config"],
        "peak_minimum": 0.2,
        "peak_maximum": 3,
    },
}


rule debug__plot:
    """Plot the defended, simulated, and undefended settings for the given defence for
    all classifiers (pattern rule)."""
    output:
        "debug/debug-brows-{defence}.png",
    input:
        "debug/defence~simulated-front/setting~default/classifier~kfp/predictions.csv",
        "debug/defence~simulated-front/setting~larger_packets/classifier~kfp/predictions.csv",
        "debug/defence~simulated-front/setting~more_packets/classifier~kfp/predictions.csv",
        "debug/defence~simulated-front/setting~lower_peak/classifier~kfp/predictions.csv",
        "debug/defence~simulated-front/setting~default/classifier~dfnet/hyperparams~n_packets=10000/predictions.csv",
        "debug/defence~simulated-front/setting~larger_packets/classifier~dfnet/hyperparams~n_packets=10000/predictions.csv",
        "debug/defence~simulated-front/setting~more_packets/classifier~dfnet/hyperparams~n_packets=10000/predictions.csv",
        "debug/defence~simulated-front/setting~lower_peak/classifier~dfnet/hyperparams~n_packets=10000/predictions.csv",
    params:
        layout={
            "$k$-FP": {
                "default": "debug/defence~simulated-front/setting~default/classifier~kfp/predictions.csv",
                "larger_packets": "debug/defence~simulated-front/setting~larger_packets/classifier~kfp/predictions.csv",
                "more_packets": "debug/defence~simulated-front/setting~more_packets/classifier~kfp/predictions.csv",
                "lower_peak": "debug/defence~simulated-front/setting~lower_peak/classifier~kfp/predictions.csv",
            },
            "DF": {
                "default": "debug/defence~simulated-front/setting~default/classifier~dfnet/hyperparams~n_packets=10000/predictions.csv",
                "larger_packets": "debug/defence~simulated-front/setting~larger_packets/classifier~dfnet/hyperparams~n_packets=10000/predictions.csv",
                "more_packets": "debug/defence~simulated-front/setting~more_packets/classifier~dfnet/hyperparams~n_packets=10000/predictions.csv",
                "lower_peak": "debug/defence~simulated-front/setting~lower_peak/classifier~dfnet/hyperparams~n_packets=10000/predictions.csv",
            }

        },
        line_styles={"default": "dotted", "larger_packets": "solid", "more_packets": "dashdot", "lower_peak": "dashed"},
        with_legend=lambda w: w["defence"] == "front"
    notebook:
        "../notebooks/result-analysis-curve.ipynb"


rule debug__simulated_dataset:
    output:
        "results/debug/defence~simulated-front/setting~{setting}/dataset.h5"
    input:
        "results/ml-eval-brows/defence~undefended/dataset/"
    params:
        n_monitored=100,
        n_instances=100,
        n_unmonitored=10000,
        simulate="front",
        simulate_kws=lambda w: {
            **simulate_kws[w["setting"]],
            "seed": 297,
        }
    wildcard_constraints:
        setting="|".join(simulate_kws.keys())
    script:
        "../scripts/create_dataset.py"
