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
    threads: 1
    resources:
        mem_mb=to_memory_per_core(16_000)
    script:
        "../scripts/create_dataset.py"


rule ml_eval_mconn__features:
    input:
        "results/ml-eval-mconn/{path}/dataset.h5"
    output:
        "results/ml-eval-mconn/{path}/features.h5"
    log:
        "results/ml-eval-mconn/{path}/features.log"
    threads: 64
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


ruleorder: combine_varcnn_predictions > ml_eval_mconn__predictions
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
        get_threads_for_classifier
    resources:
        mem_mb=to_memory_per_core(32_000),
        time_min=480
    shell:
        "workflow/scripts/evaluate-classifier {params.classifier_args}"
        " {params.classifier} {input.dataset} {input.splits} {output} 2> {log}"


rule ml_eval_mconn__tuned_varcnn_predict:
    input:
        "results/ml-eval-mconn/{path}/dataset.h5"
    output:
        "results/ml-eval-mconn/{path}/tuned-predict/varcnn-{feature_type}.csv"
    log:
        "results/ml-eval-mconn/{path}/tuned-predict/varcnn-{feature_type}.log"
    threads:
        get_threads_for_classifier({"classifier": "varcnn"})
    shell:
        "workflow/scripts/evaluate_tuned_varcnn.py --verbose 0"
        " {wildcards.feature_type} {input} > {output} 2> {log}"

rule ml_eval_mconn__tuned_kfp_predict:
    input:
        "results/ml-eval-mconn/{path}/features.h5"
    output:
        "results/ml-eval-mconn/{path}/tuned-predict/kfp.csv"
    log:
        "results/ml-eval-mconn/{path}/tuned-predict/kfp.log",
        cv_results="results/ml-eval-mconn/{path}/tuned-predict/cv-results-kfp.csv",
    threads:
        workflow.cores
    shell:
        "workflow/scripts/evaluate_tuned_kfp.py --verbose 0 --n-jobs {threads}"
        " --cv-results-path {log[cv_results]} {input} > {output} 2> {log[0]}"


rule ml_eval_mconn__tuned_all:
    input:
        expand([
            "results/ml-eval-mconn/{defence}/tuned-predict/kfp.csv",
            "results/ml-eval-mconn/undefended/tuned-predict/kfp.csv",
        ], defence=["tamaraw", "front"])


rule ml_eval_mconn__plot:
    input:
        expand([
            "results/ml-eval-mconn/{{defence}}/predict/{classifier}-0.csv",
            "results/ml-eval-mconn/undefended/predict/{classifier}-0.csv",
        ], classifier=ml_emc_config["classifiers"])
    output:
        "results/plots/ml-eval-mconn-{defence}.png"
    params:
        with_legend=lambda w: w["defence"] == "front",
        with_simulated=False
    notebook:
        "../notebooks/result-analysis-curve.ipynb"


rule ml_eval_mconn__all:
    input:
        "results/plots/ml-eval-mconn-tamaraw.png",
        "results/plots/ml-eval-mconn-front.png"
