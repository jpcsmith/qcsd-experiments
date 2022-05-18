ml_eb_config = config["experiment"]["ml_eval_brows"]


rule ml_eval_brows__all:
    input:
        "results/plots/ml-eval-brows-front.png"


def ml_eval_brows__plot__inputs(wildcards, flatten: bool = False):
    defence = wildcards["defence"]
    base = "results/ml-eval-brows"
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
        line_styles={"QCSD": "solid", "Undef.": "dashdot"},
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
