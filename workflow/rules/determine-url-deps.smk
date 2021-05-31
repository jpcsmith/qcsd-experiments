rule depfetch_browser_image:
    """Build the browser image for determining URL dependencies."""
    input:
        ancient("resources/docker/dep-fetch/Dockerfile"),
        ancient("resources/docker/dep-fetch/fetch-script"),
        ancient("resources/docker/dep-fetch/requirements.txt")
    output:
        touch("results/determine-url-deps/.dep-fetch-build.done")
    shell: "docker build --tag dep-fetch resources/docker/dep-fetch/"


rule depfetch__input_batch:
    """Create a single batch for the URL dependency fetches."""
    input:
        "results/version-scan/scan-results.filtered.csv"
    output:
        "results/determine-url-deps/input-urls/batch-{i}.csv"
    params:
        start=lambda w: 1 + int(w["i"]) * config["url_deps"]["batch_size"],
        batch_size=config["url_deps"]["batch_size"]
    shell:
        "set +o pipefail;"
        " tail -n +{params.start} {input} | head -n {params.batch_size} > {output}"


rule depfetch__url_dependencies:
    """Fetch the URL dependencies with the browser for a split of URLs."""
    input:
        domains=rules.depfetch__input_batch.output,
        image_done=rules.depfetch_browser_image.output
    output:
        protected("results/determine-url-deps/browser-logs/batch-{i}.json.gz")
    log:
        "results/determine-url-deps/browser-logs/batch-{i}.log"
    threads: 2
    resources:
        mem_mb=1500
    shell:
        "workflow/scripts/docker-dep-fetch --ranks --max-attempts 1 {input.domains}"
        " 2> {log} | gzip --stdout > {output}"


checkpoint url_dependency_graphs:
    """Extract the dependency graph from the browser results."""
    input:
        expand(rules.depfetch__url_dependencies.output, i=range(config["url_deps"]["n_batches"]))
    output:
        prefix=directory("results/determine-url-deps/dependencies")
    log:
        "results/determine-url-deps/dependencies.log"
    shell:
        "mkdir -p {output}"
        " && python3 workflow/scripts/url_dependency_graph.py '{output.prefix}/' {input}"
        " 2> {log}"
