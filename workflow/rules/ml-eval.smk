wildcard_constraints:
    filename="[a-zA-Z0-9_-]+"


rule ml_eval__features:
    input:
        "results/ml-eval/{filename}.h5"
    output:
        "results/ml-eval/features/{filename}.h5"
    shell:
        "workflow/scripts/extract-features {input} {output}"


rule ml_eval__splits:
    input:
        "results/ml-eval/features/{filename}.h5"
    output:
        "results/ml-eval/splits/{filename}.json"
    params:
        n_folds=config["experiment"]["ml_eval"]["splits"]["n_folds"],
        seed=config["experiment"]["ml_eval"]["splits"]["seed"],
        validation_size=config["experiment"]["ml_eval"]["splits"]["validation_size"]
    script:
        "../scripts/split_dataset.py"


rule ml_eval__single_split:
    input:
        rules.ml_eval__splits.output
    output:
        "results/ml-eval/splits/{filename}.json.d/{i}"
    params:
        lineno=lambda w: int(w["i"]) + 1
    shell:
        "sed -n {params.lineno}p {input} > {output}"


rule ml_eval__train_test_classifier:
    input:
        dataset="results/ml-eval/features/{filename}.h5",
        splits="results/ml-eval/splits/{filename}.json.d/{i}"
    output:
        "results/ml-eval/predictions/{filename}/{classifier}-{i}.csv"
    log:
        "results/ml-eval/predictions/{filename}/{classifier}-{i}.log"
    params:
        classifier="{classifier}",
        classifier_args=lambda w: ("--classifier-args n_jobs=4,feature_set=kfp"
                                   if w["classifier"] == "kfp" else "")
    threads:
        lambda w: workflow.cores if w["classifier"] != "kfp" else 4
    shell:
        "workflow/scripts/evaluate-classifier {params.classifier_args}"
        " {params.classifier} {input.dataset} {input.splits} {output}"
