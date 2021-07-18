shaping_config = config["experiment"]["shaping_eval"]


def build_neqo_args(exp_config):
    def _builder(wildcards):
        if wildcards["defence"] == "front":
            args = exp_config["front"]
            return [
                "--defence", "front",
                "--defence-packet-size", args["packet_size"],
                "--front-max-client-pkts", args["max_client_packets"],
                "--front-max-server-pkts", args["max_server_packets"],
                "--front-peak-max", args["peak_maximum"],
                "--front-peak-min", args["peak_minimum"],
            ]
        if wildcards["defence"] == "tamaraw":
            args = exp_config["tamaraw"]
            return [
                "--defence", "tamaraw",
                "--defence-packet-size", args["packet_size"],
                "--tamaraw-rate-in", args["rate_in"],
                "--tamaraw-rate-out", args["rate_out"],
                "--tamaraw-modulo", args["packet_multiple"],
            ]
        raise ValueError("Unsupported defence: %r", wildcards["defence"])
    return _builder


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
        max(workflow.global_resources.get("mem_mb", 1000) // 10_000, 2)
    resources:
        mem_mb=lambda w, input, threads: threads * 10_000
    script:
        "../scripts/calculate_score.py"


rule shaping_eval__plot:
    """Generate box-plots from the calculated scores."""
    input:
        rules.shaping_eval__score.output
    output:
        "results/plots/shaping-eval-{defence}.png"
    notebook:
        "../notebooks/plot-scores.ipynb"


rule shaping_eval__all:
    """Run the entire shaping-eval pipeline to create all the plots."""
    input:
        "results/plots/shaping-eval-front.png",
        "results/plots/shaping-eval-tamaraw.png"
