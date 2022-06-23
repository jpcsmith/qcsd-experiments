import json


rule overhead_eval__calculate:
    """Calculate the overhead for a specified defence (pattern rule)."""
    output:
        "results/overhead-eval/{defence}.csv"
    input:
        "results/shaping-eval/{defence}/dataset/"
    log:
        "results/overhead-eval/{defence}.log"
    threads:
        workflow.cores
    params:
        tamaraw_config=json.dumps(config["experiment"]["shaping_eval"]["tamaraw"])
    shell:
      "workflow/scripts/calculate_overhead.py --tamaraw-config '{params.tamaraw_config}'"
      " --n-jobs {threads} {wildcards.defence} {input} > {output} 2> {log}"


rule overhead_eval__table:
    """Place the overhead evaluation results in a LaTeX table (static rule)."""
    output:
        "results/tables/overhead-eval.tex"
    input:
        front="results/overhead-eval/front.csv",
        tamaraw="results/overhead-eval/tamaraw.csv",
    script:
        "../scripts/overhead_to_latex.py"
