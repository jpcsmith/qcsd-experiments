rule depfetch_mconn__input_batch:
    """Create a single batch for the URL dependency fetches for multiple
    connections."""
    input:
        "results/version-scan-mconn/scan-results.filtered.csv"
    output:
        "results/webpage-graphs-mconn/batches/batch-{i}.csv"
    params:
        start=lambda w: 1 + int(w["i"]) * config["webpage_graphs"]["batch_size"],
        batch_size=config["webpage_graphs"]["batch_size"]
    shell:
        "set +o pipefail;"
        " tail -n +{params.start} {input} | head -n {params.batch_size} > {output}"


rule depfetch_mconn__url_dependencies:
    """Fetch the URL dependencies with the browser for a split of URLs
    for multiple connections."""
    input:
        rules.depfetch_mconn__input_batch.output,
    output:
        protected("results/webpage-graphs-mconn/browser-logs/batch-{i}.json.gz")
    log:
        "results/webpage-graphs-mconn/browser-logs/batch-{i}.log"
    threads: 2
    resources:
        mem_mb=1500
    shell:
        "workflow/scripts/docker-dep-fetch --ranks --max-attempts 1 --force-quic-on-all"
        " {input} 2> {log} | gzip --stdout > {output}"


rule depfetch_mconn__webpage_graphs:
    """Extract the dependency graph from the browser results for
    multiple connections."""
    input:
        expand(rules.depfetch_mconn__url_dependencies.output,
               i=range(config["webpage_graphs"]["n_batches"]))
    output:
        directory("results/webpage-graphs-mconn/graphs")
    log:
        "results/webpage-graphs-mconn/graphs.log"
    shell:
        "mkdir -p {output}"
        " && python3 workflow/scripts/url_dependency_graph.py --no-origin-filter"
        " '{output}/' {input} 2> {log}"
