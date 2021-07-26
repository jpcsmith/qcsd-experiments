ml_eb_config = config["experiment"]["ml_eval_brows"]


rule ml_eval_brows__collect:
    input:
        "results/webpage-graphs/graphs/"
    output:
        directory("results/ml-eval-brows/{defence}/dataset/")
    log:
        "results/ml-eval-brows/{defence}/dataset.log"
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


rule ml_eval_brows__dataset:
    input:
        "results/ml-eval-brows/{defence}/dataset/"
    output:
        "results/ml-eval-brows/{defence}/dataset.h5"
    params:
        **ml_eb_config["dataset"],
    script:
        "../scripts/create_dataset.py"


rule ml_eval_brows__simulated_front:
    input:
        "results/ml-eval-brows/undefended/dataset/"
    output:
        "results/ml-eval-brows/simulated-front/dataset.h5"
    params:
        **ml_eb_config["dataset"],
        simulate="front",
        simulate_kws={**ml_eb_config["front"], "seed": 297},
    script:
        "../scripts/create_dataset.py"


rule ml_eval_brows__filtered_dataset:
    input:
        "results/ml-eval-brows/{path}/dataset.h5"
    output:
        "results/ml-eval-brows/{path}/filtered/dataset.h5"
    params:
        size=ml_eb_config["min_pkt_size"],
    shell:
        "workflow/scripts/remove-small-packets {params.size} {input} {output}"


rule ml_eval_brows__features:
    input:
        "results/ml-eval-brows/{path}/dataset.h5"
    output:
        "results/ml-eval-brows/{path}/features.h5"
    log:
        "results/ml-eval-brows/{path}/features.log"
    shell:
        "workflow/scripts/extract-features {input} {output} 2> {log}"


rule ml_eval_brows__splits:
    """Create train-test-validation splits of the dataset."""
    input:
        "results/ml-eval-brows/{path}/features.h5"
    output:
        "results/ml-eval-brows/{path}/split/split-0.json"
    params:
        seed=ml_eb_config["splits"]["seed"],
        n_folds=ml_eb_config["splits"]["n_folds"],
        validation_size=ml_eb_config["splits"]["validation_size"],
    script:
        "../scripts/split_dataset.py"


rule ml_eval_brows__predictions:
    input:
        dataset="results/ml-eval-brows/{path}/features.h5",
        splits="results/ml-eval-brows/{path}/split/split-0.json"
    output:
        "results/ml-eval-brows/{path}/predict/{classifier}-0.csv"
    log:
        "results/ml-eval-brows/{path}/predict/{classifier}-0.log"
    params:
        classifier="{classifier}",
        classifier_args=lambda w: ("--classifier-args n_jobs=4,feature_set=kfp"
                                   if w["classifier"] == "kfp" else "")
    threads:
        lambda w: min(workflow.cores, 16) if w["classifier"] != "kfp" else 4
    shell:
        "workflow/scripts/evaluate-classifier {params.classifier_args}"
        " {params.classifier} {input.dataset} {input.splits} {output} 2> {log}"


rule ml_eval_brows__plot:
    input:
        expand([
            "results/ml-eval-brows/{{defence}}/{{filtered}}predict/{classifier}-0.csv",
            "results/ml-eval-brows/simulated-{{defence}}/{{filtered}}predict/{classifier}-0.csv",
            "results/ml-eval-brows/undefended/{{filtered}}predict/{classifier}-0.csv",
        ], classifier=ml_eb_config["classifiers"])
    output:
        "results/plots/{filtered}ml-eval-brows-{defence}.png"
    wildcard_constraints:
        filtered="(filtered/)?"
    params:
        with_legend=True,
        with_simulated=False
    notebook:
        "../notebooks/result-analysis-curve.ipynb"


rule ml_eval_brows__all:
    input:
        "results/plots/ml-eval-brows-front.png"
