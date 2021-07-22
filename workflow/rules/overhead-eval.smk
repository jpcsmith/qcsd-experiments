rule overhead_eval__calculate:
    input:
        "results/shaping-eval/{defence}/dataset/"
    output:
        "results/overhead-eval/{defence}.csv"
    log:
        "results/overhead-eval/{defence}.log"
    threads:
        workflow.cores
    params:
        is_pad_only=lambda w: w["defence"] == "front"
    script:
        "../scripts/calculate_overhead.py"


rule overhead_eval__table:
    input:
        front="results/overhead-eval/front.csv",
        tamaraw="results/overhead-eval/tamaraw.csv",
    output:
        "results/tables/overhead-eval.tex"
    script:
        "../scripts/overhead_to_latex.py"
