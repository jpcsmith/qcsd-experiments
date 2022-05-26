ml_emc_config = config["experiment"]["ml_eval_mconn"]


rule ml_eval_mconn__all:
    """Create all the plots for the multi-connection evaluations (static rule)."""
    input:
        "results/plots/ml-eval-mconn-tamaraw.png",
        "results/plots/ml-eval-mconn-front.png"


def ml_eval_mconn__plot__inputs(wildcards, flatten: bool = False):
    defence = wildcards["defence"]
    base = "results/ml-eval-mconn"
    hyperparams = ml_emc_config["hyperparams"]
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


rule ml_eval_mconn__simulated_dataset:
    """Create a simulated dataset based on a collected dataset (pattern rule)."""
    output:
        "results/ml-eval-mconn/defence~simulated-{defence}/dataset.h5"
    input:
        lambda w: (
            "results/ml-eval-mconn/defence~tamaraw/dataset/" if w["defence"] == "tamaraw"
            else "results/ml-eval-mconn/defence~undefended/dataset/"
        )
    params:
        **ml_emc_config["dataset"],
        simulate="{defence}",
        simulate_kws=lambda w: (
            {**ml_emc_config["front"], "seed": 297} if w["defence"] == "front" else {}
        )
    script:
        "../scripts/create_dataset.py"


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
        line_styles={"QCSD": "solid", "Simulated": "dotted", "Undef.": "dashdot"},
        with_legend=lambda w: w["defence"] == "front",
    notebook:
        "../notebooks/result-analysis-curve.ipynb"


def ml_eval_mconn__combine_varcnn__inputs(wildcards):
    base = "results/ml-eval-mconn/" + wildcards["path"]
    if "undefended" in wildcards["path"]:
        hparams = ml_emc_config["hyperparams"]["undefended"]
    elif "front" in wildcards["path"]:
        hparams = ml_emc_config["hyperparams"]["front"]
    elif "tamaraw" in wildcards["path"]:
        hparams = ml_emc_config["hyperparams"]["tamaraw"]
    else:
        raise ValueError(f"Unsupported defence: {wildcards['path']}")

    return {
        feature_type: f"{base}/classifier~{tag}/hyperparams~{hparams[tag]}/predictions.csv"
        for feature_type, tag in [("times", "varcnn-time"), ("sizes", "varcnn-sizes")]
    }


rule ml_eval_mconn__combine_varcnn:
    output:
        "results/ml-eval-mconn/{path}/classifier~varcnn/predictions.csv"
    input:
        unpack(ml_eval_mconn__combine_varcnn__inputs)
    run:
        combine_varcnn_predictions(input, output)


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
        timeout=120,
        max_failures=3,
        configfile=workflow.configfiles[0],
    shell:
        "workflow/scripts/run_collectionv2.py --configfile {params.configfile}"
        " --n-monitored {params.n_monitored} --n-instances {params.n_instances}"
        " --n-unmonitored {params.n_unmonitored} --max-failures {params.max_failures}"
        " --timeout {params.timeout} --use-multiple-connections"
        " {input} {output} -- {params.neqo_args} 2> {log}"
