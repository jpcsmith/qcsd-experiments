ml_eb_config = config["experiment"]["ml_eval_brows"]


rule ml_eval_brows__plot:
    """Plot the defended and undefended settings for FRONT and all classifiers in the
    browser setting (static rule)."""
    output:
        "results/plots/ml-eval-brows-front.png"
    input:
        expand(
            "results/ml-eval-brows/defence~{setting}/classifier~{classifier}/predictions.csv",
            setting=["front", "undefended"], classifier=["kfp", "varcnn"]
        )
    params:
        with_legend=True,
        with_simulated=False
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
