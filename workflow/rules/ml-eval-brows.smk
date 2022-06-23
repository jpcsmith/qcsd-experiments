ml_eb_config = config["experiment"]["ml_eval_brows"]


rule ml_eval_brows__all:
    """Generate the FRONT plot in the browser setting."""
    input:
        "results/plots/ml-eval-brows-front.png"


def ml_eval_brows__plot__inputs(wildcards, flatten: bool = False):
    defence = wildcards["defence"]
    base = "results/ml-eval-brows"
    result = {
        "$k$-FP": {
            "QCSD": f"{base}/defence~{defence}/classifier~kfp/predictions.csv",
            "Simulated": f"{base}/defence~simulated-{defence}/classifier~kfp/predictions.csv",
            "Undef.": f"{base}/defence~undefended/classifier~kfp/predictions.csv",
        },
        "DF": {
            "QCSD": f"{base}/defence~{defence}/classifier~dfnet/hyperparams~tune/predictions.csv",
            "Simulated": f"{base}/defence~simulated-{defence}/classifier~dfnet/hyperparams~tune/predictions.csv",
            "Undef.": f"{base}/defence~undefended/classifier~dfnet/hyperparams~n_packets=5000/predictions.csv",
        },
        "Var-CNN": {
            "QCSD": f"{base}/defence~{defence}/classifier~varcnn/predictions.csv",
            "Simulated": f"{base}/defence~simulated-{defence}/classifier~varcnn/predictions.csv",
            "Undef.": f"{base}/defence~undefended/classifier~varcnn/predictions.csv"
        },
    }
    if flatten:
        result = [v for values in result.values() for v in values.values()]
    return result


def ml_eval_brows__combine_varcnn__inputs(wildcards):
    base = "results/ml-eval-brows/" + wildcards["path"]
    if "undefended" in wildcards["path"]:
        hparams = ml_ec_config["hyperparams"]["undefended"]
    elif "front" in wildcards["path"]:
        hparams = ml_ec_config["hyperparams"]["front"]
    elif "tamaraw" in wildcards["path"]:
        hparams = ml_ec_config["hyperparams"]["tamaraw"]
    else:
        raise ValueError(f"Unsupported defence: {wildcards['path']}")

    return {
        feature_type: f"{base}/classifier~{tag}/hyperparams~{hparams[tag]}/predictions.csv"
        for feature_type, tag in [("times", "varcnn-time"), ("sizes", "varcnn-sizes")]
    }


rule ml_eval_brows__combine_varcnn:
    """Combine VarCNN time and sizes predictions in the browser setting (pattern rule)."""
    output:
        "results/ml-eval-brows/{path}/classifier~varcnn/predictions.csv"
    input:
        unpack(ml_eval_brows__combine_varcnn__inputs)
    run:
        combine_varcnn_predictions(input, output)


rule ml_eval_brows__simulated_dataset:
    """Create a simulated dataset based on a collected dataset (pattern rule)."""
    output:
        "results/ml-eval-brows/defence~simulated-front/dataset.h5"
    input:
        "results/ml-eval-brows/defence~undefended/dataset/"
    params:
        **ml_eb_config["dataset"],
        simulate="front",
        simulate_kws={**ml_eb_config["front"], "seed": 298}
    script:
        "../scripts/create_dataset.py"


rule ml_eval_brows__plot:
    """Plot the defended and undefended settings for FRONT and all classifiers in the
    browser setting (static rule)."""
    output:
        "results/plots/ml-eval-brows-{defence}.png",
        "results/plots/ml-eval-brows-{defence}.pgf"
    input:
        lambda w: ml_eval_brows__plot__inputs(w, flatten=True)
    params:
        layout=ml_eval_brows__plot__inputs,
        line_styles={"QCSD": "solid", "Undef.": "dashdot", "Simulated": "dotted"},
        with_legend=True,
    notebook:
        "../notebooks/result-analysis-curve.ipynb"


rule ml_eval_brows__collect__to_binary:
    """Convert a collected file-based dataset to a binary dataset for faster reads
    (pattern rule)."""
    output:
        "results/ml-eval-brows/defence~{defence}/dataset.h5"
    input:
        "results/ml-eval-brows/defence~{defence}/dataset/"
    params:
        **ml_eb_config["dataset"],
    script:
        "../scripts/create_dataset.py"


rule ml_eval_brows__collect:
    """Collect samples for the browser machine-learning evaluation (pattern rule)."""
    output:
        directory("results/ml-eval-brows/defence~{defence}/dataset/")
    input:
        "results/webpage-graphs/graphs/"
    log:
        "results/ml-eval-brows/defence~{defence}/dataset.log"
    threads:
        workflow.cores
    params:
        neqo_args=build_neqo_args(ml_eb_config),
        n_monitored=ml_eb_config["dataset"]["n_monitored"],
        n_instances=ml_eb_config["dataset"]["n_instances"],
        n_unmonitored=ml_eb_config["dataset"]["n_unmonitored"],
        skip_neqo=lambda w: w["defence"] == "undefended"
    script:
        "../scripts/run_browser_collection.py"
