ml_emc_config = config["experiment"]["ml_eval_mconn"]


rule ml_eval_mconn__all:
    """Create all the plots for the multi-connection evaluations (static rule)."""
    input:
        "results/plots/ml-eval-mconn-tamaraw.png",
        "results/plots/ml-eval-mconn-front.png"


rule ml_eval_mconn__all__padded:
    input:
        expand([
            "results/ml-eval-mconn/defence~{defence}/classifier~kfp/predictions.csv",
            ("results/ml-eval-mconn/defence~padded-{defence}/strategy~{strategy}"
                "/direction~{direction}/classifier~kfp/predictions.csv"),
        ], defence=["tamaraw", "front"], strategy=["padE", "pad500", "padMTU"],
        direction=["in", "out", "both"])


rule ml_eval_mconn__plot:
    """Plot the defended and undefended settings for the given defence for all
    classifiers in the multiconnection setting (pattern rule)."""
    output:
        "results/plots/ml-eval-mconn-{defence}.png",
        "results/plots/ml-eval-mconn-{defence}.pgf"
    input:
        expand([
            "results/ml-eval-mconn/defence~{{defence}}/classifier~{classifier}/predictions.csv",
            "results/ml-eval-mconn/defence~undefended/classifier~{classifier}/predictions.csv",
        ], classifier=["kfp", "varcnn"]),
    params:
        with_legend=lambda w: w["defence"] == "front",
        with_simulated=False
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
