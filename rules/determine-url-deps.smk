scattergather:
    depfetch=16


rule depfetch_browser_image:
    """Build the browser image for determining URL dependencies."""
    input:
        "docker/dep-fetch/Dockerfile",
        "docker/dep-fetch/fetch-script",
        "docker/dep-fetch/requirements.txt"
    output:
        touch("results/determine-url-deps/.dep-fetch-build.done")
    shell: "docker build --tag dep-fetch docker/dep-fetch/"


rule depfetch_urls:
    "Extract the domains URLs to be fetched and prepend with https://."
    input:
        "results/profile-domains/filtered-domains.csv"
    output:
        "results/determine-url-deps/urls.txt"
    params:
        max_urls=config["max_fetch_urls"]
    shell: """\
        cut -d, -f2 {input} | sed '1d' | head -n {params.max_urls} \
            | sed 's/^/https:\/\//' > {output}
        """


rule depfetch_split:
    """Create the splits for the URL dependency fetches."""
    input:
        rules.depfetch_urls.output
    output:
        scatter.depfetch("results/determine-url-deps/urls.txt.d/{scatteritem:02d}.txt")
    params:
        num_output=lambda _: len(scatter.depfetch("{scatteritem}"))
    shell: """\
        split --suffix-length=2 -d --number=l/{params.num_output} \
            {input} 'results/determine-url-deps/filter-domains.d/'
        """


rule url_dependencies__part:
    """Fetch the URL dependencies with the browser for a split of URLs."""
    input:
        domains="results/determine-url-deps/urls.txt.d/{scatteritem}.txt",
        image_done=rules.depfetch_browser_image.output
    output:
        "results/determine-url-deps/browser-logs.json.d/{scatteritem}.json"
    log:
        "results/determine-url-deps/browser-logs.json.d/{scatteritem}.log"
    shell: "scripts/docker-dep-fetch {input} > {output} 2> {log}"


rule url_dependencies:
    """Combine the URL dependency parts into a single file."""
    input:
        gather.depfetch("results/determine-url-deps/browser-logs.json.d/{scatteritem:02d}.json")
    output:
        protected("results/determine-url-deps/browser-logs.json.gz")
    shell: "cat {input} | gzip --to-stdout > {output}"
