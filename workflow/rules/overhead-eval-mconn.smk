ohead_mconn_config = config["experiment"]["overhead_eval_mconn"]

rule overhead_eval_mconn__table:
    input:
        front="results/overhead-eval-mconn/front.csv",
        tamaraw="results/overhead-eval-mconn/tamaraw.csv",
    output:
        "results/tables/overhead-eval-mconn.tex"
    script:
        "../scripts/overhead_to_latex.py"


rule overhead_eval_mconn__calculate:
    input:
        "results/overhead-eval-mconn/{defence}/dataset/"
    output:
        "results/overhead-eval-mconn/{defence}.csv"
    log:
        "results/overhead-eval-mconn/{defence}.log"
    threads:
        workflow.cores
    params:
        defence="{defence}",
        tamaraw_config=ohead_mconn_config["tamaraw"]
    script:
        "../scripts/calculate_overhead.py"


rule overhead_eval_mconn__collect:
    """Collect control and defended samples for a given defence."""
    input:
        "results/webpage-graphs-mconn/graphs/"
    output:
        directory("results/overhead-eval-mconn/{defence}/dataset/")
    log:
        "results/overhead-eval-mconn/{defence}/dataset.log"
    threads:
        workflow.cores
    params:
        neqo_args=build_neqo_args(ohead_mconn_config),
        n_unmonitored=ohead_mconn_config["n_samples"],
        n_monitored=0,
        max_failures=1,
        use_multiple_connections=True,
    script:
        "../scripts/run_paired_collection.py"

