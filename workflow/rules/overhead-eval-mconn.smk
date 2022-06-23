ohead_mconn_config = config["experiment"]["overhead_eval_mconn"]


rule overhead_eval_mconn__table:
    """Generate the table for the multiconnection overhead evaluation (static rule)."""
    input:
        front="results/overhead-eval-mconn/front.csv",
        tamaraw="results/overhead-eval-mconn/tamaraw.csv",
    output:
        "results/tables/overhead-eval-mconn.tex"
    script:
        "../scripts/overhead_to_latex.py"


rule overhead_eval_mconn__calculate:
    """Calculate overheads in the multiconnection setting (pattern rule)."""
    input:
        "results/overhead-eval-mconn/{defence}/dataset/"
    output:
        "results/overhead-eval-mconn/{defence}.csv"
    log:
        "results/overhead-eval-mconn/{defence}.log"
    threads:
        workflow.cores
    params:
        tamaraw_config=json.dumps(ohead_mconn_config["tamaraw"])
    shell:
        "workflow/scripts/calculate_overhead.py {wildcards.defence} {input}"
        " --n-jobs {threads} --tamaraw-config '{params.tamaraw_config}'"
        " > {output} 2> {log}"


rule overhead_eval_mconn__collect:
    """Collect control and defended samples for a given defence (pattern rule)."""
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

