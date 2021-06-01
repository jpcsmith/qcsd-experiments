rule ml_eval__features:
    """Extract the features for the different classifiers from a dataset."""
    input:
        "results/ml-eval/{basename}.h5"
    output:
        "results/ml-eval/features/{basename}.h5"
    log:
        "results/ml-eval/features/{basename}.log"
    shell:
        "workflow/scripts/extract-features {input} {output} > {log}"


rule ml_eval__splits:
    """Create train-test-validation splits of the dataset."""
    input:
        "results/ml-eval/features/{basename}.h5"
    output:
        expand("results/ml-eval/splits/{{basename}}/{i}.json",
               i=range(config["experiment"]["ml_eval"]["splits"]["n_folds"]))
    params:
        n_folds=config["experiment"]["ml_eval"]["splits"]["n_folds"],
        seed=config["experiment"]["ml_eval"]["splits"]["seed"],
        validation_size=config["experiment"]["ml_eval"]["splits"]["validation_size"]
    script:
        "../scripts/split_dataset.py"


rule ml_eval__predictions:
    """Create predictions for a split of the dataset for a specific classifier."""
    input:
        dataset="results/ml-eval/features/{basename}.h5",
        splits="results/ml-eval/splits/{basename}/{i}.json"
    output:
        "results/ml-eval/predictions/{basename}/{classifier}-{i}.csv"
    log:
        "results/ml-eval/predictions/{basename}/{classifier}-{i}.log"
    params:
        classifier="{classifier}",
        classifier_args=lambda w: ("--classifier-args n_jobs=4,feature_set=kfp"
                                   if w["classifier"] == "kfp" else "")
    threads:
        lambda w: workflow.cores if w["classifier"] != "kfp" else 4
    shell:
        "workflow/scripts/evaluate-classifier {params.classifier_args}"
        " {params.classifier} {input.dataset} {input.splits} {output}"


rule ml_eval__plots:
    input:
        expand(
            ["results/ml-eval/predictions/{defence}-dataset/{classifier}-{i}.csv",
             "results/ml-eval/predictions/sim-{defence}-dataset/{classifier}-{i}.csv"],
            defence=["front", "tamaraw"],
            classifier=["kfp"],
            i=range(config["experiment"]["ml_eval"]["splits"]["n_folds"])
        )
