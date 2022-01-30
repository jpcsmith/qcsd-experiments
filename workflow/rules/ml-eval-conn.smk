ml_ec_config = config["experiment"]["ml_eval_conn"]


rule ml_eval_conn__collect:
    input:
        "results/webpage-graphs/graphs/"
    output:
        directory("results/ml-eval-conn/{defence}/dataset/")
    log:
        "results/ml-eval-conn/{defence}/dataset.log"
    threads:
        workflow.cores
    params:
        neqo_args=build_neqo_args(ml_ec_config),
        n_monitored=ml_ec_config["dataset"]["n_monitored"],
        n_instances=ml_ec_config["dataset"]["n_instances"],
        n_unmonitored=ml_ec_config["dataset"]["n_unmonitored"],
    script:
        "../scripts/run_collection.py"


rule ml_eval_conn__dataset:
    input:
        "results/ml-eval-conn/{defence}/dataset/"
    output:
        "results/ml-eval-conn/{defence}/dataset.h5"
    params:
        **ml_ec_config["dataset"],
    script:
        "../scripts/create_dataset.py"


rule ml_eval_conn__simulated_tamaraw:
    input:
        "results/ml-eval-conn/tamaraw/dataset/"
    output:
        "results/ml-eval-conn/simulated-tamaraw/dataset.h5"
    params:
        **ml_ec_config["dataset"],
        simulate="tamaraw",
    script:
        "../scripts/create_dataset.py"


rule ml_eval_conn__simulated_front:
    input:
        "results/ml-eval-conn/undefended/dataset/"
    output:
        "results/ml-eval-conn/simulated-front/dataset.h5"
    params:
        **ml_ec_config["dataset"],
        simulate="front",
        simulate_kws={**ml_ec_config["front"], "seed": 297},
    script:
        "../scripts/create_dataset.py"


rule ml_eval_conn__filtered_dataset:
    input:
        "results/ml-eval-conn/{path}/dataset.h5"
    output:
        "results/ml-eval-conn/{path}/filtered/dataset.h5"
    params:
        size=ml_ec_config["min_pkt_size"],
    shell:
        "workflow/scripts/remove-small-packets {params.size} {input} {output}"


rule ml_eval_conn__features:
    input:
        "results/ml-eval-conn/{path}/dataset.h5"
    output:
        "results/ml-eval-conn/{path}/features.h5"
    log:
        "results/ml-eval-conn/{path}/features.log"
    threads: 64
    shell:
        "workflow/scripts/extract-features {input} {output} 2> {log}"


rule ml_eval_conn__splits:
    """Create train-test-validation splits of the dataset."""
    input:
        "results/ml-eval-conn/{path}/features.h5"
    output:
        "results/ml-eval-conn/{path}/split/split-0.json"
    params:
        seed=ml_ec_config["splits"]["seed"],
        n_folds=ml_ec_config["splits"]["n_folds"],
        validation_size=ml_ec_config["splits"]["validation_size"],
    script:
        "../scripts/split_dataset.py"


rule ml_eval_conn__predictions:
    input:
        dataset="results/ml-eval-conn/{path}/features.h5",
        splits="results/ml-eval-conn/{path}/split/split-0.json"
    output:
        "results/ml-eval-conn/{path}/predict/{classifier}-0.csv"
    log:
        "results/ml-eval-conn/{path}/predict/{classifier}-0.log"
    params:
        classifier="{classifier}",
        classifier_args=lambda w: ("--classifier-args n_jobs=4,feature_set=kfp"
                                   if w["classifier"] == "kfp" else "")
    threads:
        get_threads_for_classifier
    resources:
        mem_mb=lambda wildcards, input, threads: int(16000 / threads),
        time_min=360
    shell:
        "workflow/scripts/evaluate-classifier {params.classifier_args}"
        " {params.classifier} {input.dataset} {input.splits} {output} 2> {log}"


rule ml_eval_conn__plot:
    input:
        expand([
            "results/ml-eval-conn/{{defence}}/{{filtered}}predict/{classifier}-0.csv",
            "results/ml-eval-conn/simulated-{{defence}}/{{filtered}}predict/{classifier}-0.csv",
            "results/ml-eval-conn/undefended/{{filtered}}predict/{classifier}-0.csv",
        ], classifier=ml_ec_config["classifiers"])
    output:
        "results/plots/{filtered}ml-eval-conn-{defence}.png"
    wildcard_constraints:
        filtered="(filtered/)?"
    params:
        with_legend=lambda w: w["defence"] == "front"
    notebook:
        "../notebooks/result-analysis-curve.ipynb"


rule ml_eval_conn__all:
    input:
        [f"results/plots/{filtered}ml-eval-conn-{defence}.png"
         for defence in ["tamaraw", "front"]
         # for filtered in ["filtered/", ""]
         for filtered in [""]
        ]
    # input:
    #     [f"results/ml-eval-conn/{defence}/{filtered}predict/{classifier}-0.csv"
    #      for defence in ["tamaraw", "front", "undefended", "simulated-tamaraw",
    #                      "simulated-front"]
    #      for classifier in ml_ec_config["classifiers"]
    #      for filtered in ["filtered/", ""]
    #     ]


