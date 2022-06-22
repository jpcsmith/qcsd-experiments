rule quic_version_scan:
    """Scan the an Alexa domain list for QUIC support (static rule)."""
    output:
        protected("results/version-scan/scan-results.csv")
    input:
        config["alexa_list"]
    log:
        "results/version-scan/scan-results.log"
    script:
        "../scripts/version_scan.py"


rule quic_version_scan__filtered:
    """Filter QUIC scan to domains supporting the QUIC versions specified
    in the config, while accounting for similar domains (static rule)."""
    output:
        "results/version-scan/scan-results.filtered.csv",
        "results/plots/frequent-prefixes.png",
        "results/plots/frequent-slds.png",
    input:
        rules.quic_version_scan.output
    log:
        "results/version-scan/scan-results.filtered.log"
    params:
        versions=config["quic_versions"],
        sld_domains=config["frequent_slds"],
    script:
        "../scripts/filter_versions.py"
