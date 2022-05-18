ml_emc_config = config["experiment"]["ml_eval_mconn"]


rule ml_eval_mconn__all:
    """Create all the plots for the multi-connection evaluations (static rule)."""
    input:
        "results/plots/ml-eval-mconn-tamaraw.png",
        "results/plots/ml-eval-mconn-front.png"


def ml_eval_mconn__plot__inputs(wildcards, flatten: bool = False):
    defence = wildcards["defence"]
    base = "results/ml-eval-mconn"
    result = {
        title: {
            "QCSD": f"{base}/defence~{defence}/classifier~{classifier}/predictions.csv",
            "Undef.": f"{base}/defence~undefended/classifier~{classifier}/predictions.csv",
        }
        for (classifier, title) in [
            ("kfp", "$k$-FP"), ("varcnn", "Var-CNN"), ("varcnn-time", "Var-CNN(T)"),
            ("varcnn-sizes", "Var-CNN(S)"),
        ]
    }
    if flatten:
        result = [v for values in result.values() for v in values.values()]
    return result


rule ml_eval_mconn__plot:
    """Plot the defended and undefended settings for the given defence for all
    classifiers in the multiconnection setting (pattern rule)."""
    output:
        "results/plots/ml-eval-mconn-{defence}.png",
        "results/plots/ml-eval-mconn-{defence}.pgf"
    input:
        lambda w: ml_eval_mconn__plot__inputs(w, flatten=True)
    params:
        layout=ml_eval_mconn__plot__inputs,
        line_styles={"QCSD": "solid", "Undef.": "dashdot"},
        with_legend=lambda w: w["defence"] == "front",
    notebook:
        "../notebooks/result-analysis-curve.ipynb"


rule ml_eval_mconn__collect__to_binary:
    """Convert a collected file-based dataset to a binary dataset for faster reads
    (pattern rule)."""
    output:
        "results/ml-eval-mconn/defence~{defence}/dataset.h5"
    input:
        "results/ml-eval-mconn/defence~{defence}/dataset/"
    params:
        **ml_emc_config["dataset"],
    script:
        "../scripts/create_dataset.py"


rule ml_eval_mconn__collect:
    """Collect samples for the multiconnection machine-learning evaluation
    (pattern rule)."""
    output:
        directory("results/ml-eval-mconn/defence~{defence}/dataset/")
    input:
        "results/webpage-graphs-mconn/graphs/"
    log:
        "results/ml-eval-mconn/defence~{defence}/dataset.log"
    threads:
        workflow.cores
    params:
        neqo_args=build_neqo_args(ml_emc_config),
        n_monitored=ml_emc_config["dataset"]["n_monitored"],
        n_instances=ml_emc_config["dataset"]["n_instances"],
        n_unmonitored=ml_emc_config["dataset"]["n_unmonitored"],
        use_multiple_connections=True,
    script:
        "../scripts/run_collection.py"
