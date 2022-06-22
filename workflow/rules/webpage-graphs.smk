rule depfetch__input_batch:
    """Split version scan results into batches for the URL dependency
    fetches (pattern rule)."""
    output:
        "results/webpage-graphs/batches/batch-{i}.csv"
    input:
        "results/version-scan/scan-results.filtered.csv"
    params:
        start=lambda w: 1 + int(w["i"]) * config["webpage_graphs"]["batch_size"],
        batch_size=config["webpage_graphs"]["batch_size"]
    shell:
        "set +o pipefail;"
        " tail -n +{params.start} {input} | head -n {params.batch_size} > {output}"


rule depfetch__url_dependencies:
    """Fetch URL dependencies for a batch of domains with Chromium (pattern rule)."""
    output:
        protected("results/webpage-graphs/browser-logs/batch-{i}.json.gz")
    input:
        rules.depfetch__input_batch.output,
    log:
        "results/webpage-graphs/browser-logs/batch-{i}.log"
    threads: 2
    resources:
        mem_mb=1500
    shell:
        "workflow/scripts/docker-dep-fetch --ranks --max-attempts 1 {input} 2> {log}"
        " | gzip --stdout > {output}"


rule depfetch__webpage_graphs:
    """Extract the dependency graph from the browser results (static rule)."""
    output:
        directory("results/webpage-graphs/graphs")
    input:
        expand(
            rules.depfetch__url_dependencies.output,
           i=range(config["webpage_graphs"]["n_batches"])
        )
    log:
        "results/webpage-graphs/graphs.log"
    shell:
        "mkdir -p {output}"
        " && python3 workflow/scripts/url_dependency_graph.py '{output}/' {input} 2> {log}"
