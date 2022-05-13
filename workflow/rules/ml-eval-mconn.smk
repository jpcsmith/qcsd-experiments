ml_emc_config = config["experiment"]["ml_eval_mconn"]


rule ml_eval_mconn__all:
    """Create all the plots for the multi-connection evaluations (static rule)."""
    input:
        # "results/plots/ml-eval-mconn-tamaraw.png",
        "results/plots/ml-eval-mconn-front.png"


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


rule ml_eval_mconn__predict__kfp:
    """Perform hyperparameter validation and predictions for the k-FP classifier
    (pattern rule)."""
    output:
        "results/ml-eval-mconn/{path}/classifier~kfp/predictions.csv"
    input:
        "results/ml-eval-mconn/{path}/classifier~kfp/features.h5"
    log:
        "results/ml-eval-mconn/{path}/classifier~kfp/predictions.log",
        cv_results="results/ml-eval-mconn/{path}/classifier~kfp/cv-results.csv",
    threads:
        workflow.cores
    shell:
        "workflow/scripts/evaluate_tuned_kfp.py --verbose 0 --n-jobs {threads}"
        " --cv-results-path {log[cv_results]} {input} > {output} 2> {log[0]}"


rule ml_eval_mconn__predict__varcnn:
    """Perform hyperparameter validation and predictions for either the sizes or time
    component of the Var-CNN classifier (pattern rule)."""
    output:
        "results/ml-eval-mconn/{path}/classifier~varcnn-{feature_type}/predictions.csv"
    input:
        "results/ml-eval-mconn/{path}/dataset.h5"
    log:
        "results/ml-eval-mconn/{path}/classifier~varcnn-{feature_type}/predictions.log"
    threads:
        get_threads_for_classifier({"classifier": "varcnn"})
    shell:
        "workflow/scripts/evaluate_tuned_varcnn.py --verbose 0"
        " {wildcards.feature_type} {input} > {output} 2> {log}"


rule ml_eval_mconn__extract_features__kfp:
    """Pre-extract the k-FP features as this can be time-consuming (pattern rule)."""
    output:
        "results/ml-eval-mconn/{path}/classifier~kfp/features.h5"
    input:
        "results/ml-eval-mconn/{path}/dataset.h5"
    log:
        "results/ml-eval-mconn/{path}/classifier~kfp/features.log"
    threads: 12
    shell:
        "workflow/scripts/extract_kfp_features.py {input} > {output} 2> {log}"


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
