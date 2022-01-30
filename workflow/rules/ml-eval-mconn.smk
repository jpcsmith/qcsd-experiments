ml_emc_config = config["experiment"]["ml_eval_mconn"]

rule ml_eval_mconn__collect:
    input:
        "results/webpage-graphs-mconn/graphs/"
    output:
        directory("results/ml-eval-mconn/{defence}/dataset/")
    log:
        "results/ml-eval-mconn/{defence}/dataset.log"
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


rule ml_eval_mconn__dataset:
    input:
        "results/ml-eval-mconn/{defence}/dataset/"
    output:
        "results/ml-eval-mconn/{defence}/dataset.h5"
    params:
        **ml_emc_config["dataset"],
    script:
        "../scripts/create_dataset.py"


# rule ml_eval_mconn__simulated_tamaraw:
#     input:
#         "results/ml-eval-mconn/tamaraw/dataset/"
#     output:
#         "results/ml-eval-mconn/simulated-tamaraw/dataset.h5"
#     params:
#         **ml_emc_config["dataset"],
#         simulate="tamaraw",
#     script:
#         "../scripts/create_dataset.py"
#
#
# rule ml_eval_mconn__simulated_front:
#     input:
#         "results/ml-eval-mconn/undefended/dataset/"
#     output:
#         "results/ml-eval-mconn/simulated-front/dataset.h5"
#     params:
#         **ml_emc_config["dataset"],
#         simulate="front",
#         simulate_kws={**ml_emc_config["front"], "seed": 297},
#     script:
#         "../scripts/create_dataset.py"


rule ml_eval_mconn__features:
    input:
        "results/ml-eval-mconn/{path}/dataset.h5"
    output:
        "results/ml-eval-mconn/{path}/features.h5"
    log:
        "results/ml-eval-mconn/{path}/features.log"
    shell:
        "workflow/scripts/extract-features {input} {output} 2> {log}"


rule ml_eval_mconn__splits:
    """Create train-test-validation splits of the dataset."""
    input:
        "results/ml-eval-mconn/{path}/features.h5"
    output:
        "results/ml-eval-mconn/{path}/split/split-0.json"
    params:
        seed=ml_emc_config["splits"]["seed"],
        n_folds=ml_emc_config["splits"]["n_folds"],
        validation_size=ml_emc_config["splits"]["validation_size"],
    script:
        "../scripts/split_dataset.py"


rule ml_eval_mconn__predictions:
    input:
        dataset="results/ml-eval-mconn/{path}/features.h5",
        splits="results/ml-eval-mconn/{path}/split/split-0.json"
    output:
        "results/ml-eval-mconn/{path}/predict/{classifier}-0.csv"
    log:
        "results/ml-eval-mconn/{path}/predict/{classifier}-0.log"
    params:
        classifier="{classifier}",
        classifier_args=lambda w: ("--classifier-args n_jobs=4,feature_set=kfp"
                                   if w["classifier"] == "kfp" else "")
    threads:
        lambda w: min(workflow.cores, 16) if w["classifier"] != "kfp" else 4
    shell:
        "workflow/scripts/evaluate-classifier {params.classifier_args}"
        " {params.classifier} {input.dataset} {input.splits} {output} 2> {log}"


rule ml_eval_mconn__plot:
    input:
        expand([
            "results/ml-eval-mconn/{{defence}}/predict/{classifier}-0.csv",
            "results/ml-eval-mconn/simulated-{{defence}}/predict/{classifier}-0.csv",
            "results/ml-eval-mconn/undefended/predict/{classifier}-0.csv",
        ], classifier=ml_emc_config["classifiers"])
    output:
        "results/plots/ml-eval-mconn-{defence}.png"
    params:
        with_legend=lambda w: w["defence"] == "front"
    notebook:
        "../notebooks/result-analysis-curve.ipynb"


rule ml_eval_mconn__all:
    input:
        "results/plots/ml-eval-mconn-tamaraw.png",
        "results/plots/ml-eval-mconn-front.png"
