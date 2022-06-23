ml_ec_config = config["experiment"]["ml_eval_conn"]


rule ml_eval_conn__all:
    """Create all the plots for the single-connection evaluations (static rule)."""
    input:
        "results/plots/ml-eval-conn-tamaraw.png",
        "results/plots/ml-eval-conn-front.png",


def ml_eval_conn__plot__inputs(wildcards, flatten: bool = False):
    defence = wildcards["defence"]
    hyperparams = ml_ec_config["hyperparams"]
    base = "results/ml-eval-conn"

    result = {
        "$k$-FP": {
            "QCSD": f"{base}/defence~{defence}/classifier~kfp/predictions.csv",
            "Simulated": f"{base}/defence~simulated-{defence}/classifier~kfp/predictions.csv",
            "Undef.": f"{base}/defence~undefended/classifier~kfp/predictions.csv",
        },
        "DF": {
            "QCSD": f"{base}/defence~{defence}/classifier~dfnet/hyperparams~{hyperparams[defence]['dfnet']}/predictions.csv",
            "Simulated": f"{base}/defence~simulated-{defence}/classifier~dfnet/hyperparams~{hyperparams[defence]['dfnet']}/predictions.csv",
            "Undef.": f"{base}/defence~undefended/classifier~dfnet/hyperparams~{hyperparams['undefended']['dfnet']}/predictions.csv",
        },
        "Var-CNN": {
            "QCSD": f"{base}/defence~{defence}/classifier~varcnn/predictions.csv",
            "Simulated": f"{base}/defence~simulated-{defence}/classifier~varcnn/predictions.csv",
            "Undef.": f"{base}/defence~undefended/classifier~varcnn/predictions.csv"
        }
    }
    if flatten:
        result = [v for values in result.values() for v in values.values()]
    return result


rule ml_eval_conn__plot:
    """Plot the defended, simulated, and undefended settings for the given defence for
    all classifiers (pattern rule)."""
    output:
        "results/plots/ml-eval-conn-{defence}.png",
        "results/plots/ml-eval-conn-{defence}.pgf",
    input:
        lambda w: ml_eval_conn__plot__inputs(w, flatten=True)
    params:
        layout=ml_eval_conn__plot__inputs,
        line_styles={"QCSD": "solid", "Simulated": "dotted", "Undef.": "dashdot"},
        with_legend=lambda w: w["defence"] == "front"
    notebook:
        "../notebooks/result-analysis-curve.ipynb"


def ml_eval_conn__combine_varcnn__inputs(wildcards):
    base = "results/ml-eval-conn/" + wildcards["path"]
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


rule ml_eval_conn__combine_varcnn:
    """Combine VarCNN time and sizes predictions (pattern rule)."""
    output:
        "results/ml-eval-conn/{path}/classifier~varcnn/predictions.csv"
    input:
        unpack(ml_eval_conn__combine_varcnn__inputs)
    run:
        combine_varcnn_predictions(input, output)


rule ml_eval_conn__simulated_dataset:
    """Create a simulated dataset based on a collected dataset (pattern rule)."""
    output:
        "results/ml-eval-conn/defence~simulated-{defence}/dataset.h5"
    input:
        lambda w: (
            "results/ml-eval-conn/defence~tamaraw/dataset/" if w["defence"] == "tamaraw"
            else "results/ml-eval-conn/defence~undefended/dataset/"
        )
    params:
        **ml_ec_config["dataset"],
        simulate="{defence}",
        simulate_kws=lambda w: (
            {**ml_ec_config["front"], "seed": 297} if w["defence"] == "front" else {}
        )
    script:
        "../scripts/create_dataset.py"


rule ml_eval_conn__collect__to_binary:
    """Convert a collected file-based dataset to a binary dataset for faster reads
    (pattern rule)."""
    output:
        "results/ml-eval-conn/defence~{defence}/dataset.h5"
    input:
        "results/ml-eval-conn/defence~{defence}/dataset/"
    params:
        **ml_ec_config["dataset"],
    script:
        "../scripts/create_dataset.py"


rule ml_eval_conn__collect:
    """Collect samples for the single-connection machine-learning evaluation
    (pattern rule)."""
    output:
        directory("results/ml-eval-conn/defence~{defence}/dataset/")
    input:
        "results/webpage-graphs/graphs/"
    log:
        "results/ml-eval-conn/defence~{defence}/dataset.log"
    threads:
        workflow.cores
    params:
        neqo_args=build_neqo_args(ml_ec_config),
        n_monitored=ml_ec_config["dataset"]["n_monitored"],
        n_instances=ml_ec_config["dataset"]["n_instances"],
        n_unmonitored=ml_ec_config["dataset"]["n_unmonitored"],
    script:
        "../scripts/run_collection.py"
