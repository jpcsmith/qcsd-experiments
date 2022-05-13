ml_ec_config = config["experiment"]["ml_eval_conn"]


rule ml_eval_conn__all:
    """Create all the plots for the single-connection evaluations (static rule)."""
    input:
        "results/plots/ml-eval-conn-tamaraw.png",
        "results/plots/ml-eval-conn-front.png",


rule ml_eval_conn__plot:
    """Plot the defended, simulated, and undefended settings for the given defence for
    all classifiers (pattern rule)."""
    output:
        "results/plots/ml-eval-conn-{defence}.png",
        "results/plots/ml-eval-conn-{defence}.pgf",
    input:
        expand([
            "results/ml-eval-conn/defence~{{defence}}/classifier~{classifier}/predictions.csv",
            "results/ml-eval-conn/defence~simulated-{{defence}}/classifier~{classifier}/predictions.csv",
            "results/ml-eval-conn/defence~undefended/classifier~{classifier}/predictions.csv",
        ], classifier=["kfp"])
    params:
        with_legend=lambda w: w["defence"] == "front"
    notebook:
        "../notebooks/result-analysis-curve.ipynb"


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
