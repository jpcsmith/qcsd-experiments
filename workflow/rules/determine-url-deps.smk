scattergather:
    depfetch=16


rule depfetch_browser_image:
    """Build the browser image for determining URL dependencies."""
    input:
        "resources/docker/dep-fetch/Dockerfile",
        "resources/docker/dep-fetch/fetch-script",
        "resources/docker/dep-fetch/requirements.txt"
    output:
        touch("results/determine-url-deps/.dep-fetch-build.done")
    shell: "docker build --tag dep-fetch resources/docker/dep-fetch/"


rule depfetch_urls:
    "Extract the domains URLs to be fetched and prepend with https://."
    input:
        "results/version-scan/scan-results.filtered.csv"
    output:
        "results/determine-url-deps/urls.txt"
    params:
        max_urls=config["max_fetch_urls"]
    shell:
        "set +o pipefail; "
        "cut -d, -f2 {input} | sed '1d' | head -n {params.max_urls}"
        " | sed 's/^/https:\/\//' > {output}"


rule depfetch_split:
    """Create the splits for the URL dependency fetches."""
    input:
        rules.depfetch_urls.output
    output:
        scatter.depfetch("results/determine-url-deps/urls.txt.d/{scatteritem}.txt")
    run:
        for i, name in enumerate(output):
            shell("split -n'l/%d/%d' {input} > %s" % (i+1, len(output), output[i]))


rule url_dependencies__part:
    """Fetch the URL dependencies with the browser for a split of URLs."""
    input:
        domains="results/determine-url-deps/urls.txt.d/{scatteritem}.txt",
        image_done=rules.depfetch_browser_image.output
    output:
        "results/determine-url-deps/browser-logs.json.d/{scatteritem}.json"
    log:
        "results/determine-url-deps/browser-logs.json.d/{scatteritem}.log"
    threads: 2
    shell:
        "workflow/scripts/docker-dep-fetch --max-attempts 1 {input.domains}"
        " > {output} 2> {log}"


rule url_dependencies:
    """Combine the URL dependency parts into a single file."""
    input:
        gather.depfetch("results/determine-url-deps/browser-logs.json.d/{scatteritem}.json")
    output:
        protected("results/determine-url-deps/browser-logs.json.gz")
    shell: "cat {input} | gzip --to-stdout > {output}"


checkpoint url_dependency_graphs:
    """Extract the dependency graph from the browser results."""
    input:
        rules.url_dependencies.output
    output:
        prefix=directory("results/determine-url-deps/dependencies")
    log:
        "results/determine-url-deps/dependencies.log"
    shell:
        "mkdir -p {output}"
        " && python3 workflow/scripts/url_dependency_graph.py {input} '{output.prefix}/'"
        " 2> {log}"
