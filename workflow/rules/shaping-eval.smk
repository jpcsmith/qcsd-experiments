shaping_config = config["experiment"]["shaping_eval"]


rule shaping_eval__collect:
    """Collect a control and defended samples for a given defence (pattern rule)."""
    output:
        directory("results/shaping-eval/{defence}/dataset/")
    input:
        "results/webpage-graphs/graphs/"
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
    """Calculate the scores for the collected samples (pattern rule)."""
    output:
        "results/shaping-eval/{defence}/scores.csv"
    input:
        rules.shaping_eval__collect.output
    log:
        "results/shaping-eval/{defence}/scores.log"
    params:
        defence="{defence}",
        ts_offset=shaping_config["scores"]["ts_offset"],
        resample_rates=shaping_config["scores"]["resample_rates"],
        lcss_eps=lambda w: shaping_config["scores"]["lcss_eps"][w["defence"]]
    threads:
        max(workflow.global_resources.get("mem_mb", 1000) // 30_000, 1)
    resources:
        mem_mb=lambda w, input, threads: threads * 30_000
    script:
        "../scripts/calculate_score.py"


rule shaping_eval__plot:
    """Generate box-plots from the calculated scores (pattern rule)."""
    output:
        "results/plots/shaping-eval-{defence}.png"
    input:
        rules.shaping_eval__score.output
    params:
        with_legend=lambda w: w["defence"] == "front",
        ylabels_at=lambda w: 0.5 if w["defence"] == "tamaraw" else 0.25
    notebook:
        "../notebooks/plot-scores.ipynb"


rule shaping_eval__all:
    """Run the entire shaping-eval pipeline to create all the plots (static rule)."""
    input:
        "results/plots/shaping-eval-front.png",
        "results/plots/shaping-eval-tamaraw.png"
