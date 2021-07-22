shaping_config = config["experiment"]["shaping_eval"]


rule shaping_eval__collect:
    """Collect a control and defended samples for a given defence."""
    input:
        "results/webpage-graphs/graphs/"
    output:
        directory("results/shaping-eval/{defence}/dataset/")
    log:
        "results/shaping-eval/{defence}/dataset.log"
    threads:
        workflow.cores
    params:
        neqo_args=build_neqo_args(shaping_config),
        n_monitored=0,
        n_unmonitored=shaping_config["n_samples"],
        max_failures=1,
    script:
        "../scripts/run_paired_collection.py"


rule shaping_eval__score:
    """Calculate the scores for the collected samples."""
    input:
        rules.shaping_eval__collect.output
    output:
        "results/shaping-eval/{defence}/scores.csv"
    log:
        "results/shaping-eval/{defence}/scores.log"
    params:
        defence="{defence}",
        ts_offset=shaping_config["scores"]["ts_offset"],
        resample_rates=shaping_config["scores"]["resample_rates"],
        filter_below=shaping_config["scores"]["min_pkt_size"],
        lcss_eps=shaping_config["scores"]["lcss_eps"],
    threads:
        max(workflow.global_resources.get("mem_mb", 1000) // 30_000, 1)
    resources:
        mem_mb=lambda w, input, threads: threads * 30_000
    script:
        "../scripts/calculate_score.py"


rule shaping_eval__plot:
    """Generate box-plots from the calculated scores."""
    input:
        rules.shaping_eval__score.output
    output:
        "results/plots/shaping-eval-{defence}.png"
    params:
        with_legend=lambda w: w["defence"] == "tamaraw"
    notebook:
        "../notebooks/plot-scores.ipynb"


rule shaping_eval__all:
    """Run the entire shaping-eval pipeline to create all the plots."""
    input:
        "results/plots/shaping-eval-front.png",
        "results/plots/shaping-eval-tamaraw.png"
