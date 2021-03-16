rule quic_version_scan:
    """Scan the an Alexa domain list for QUIC support."""
    input:
        config["alexa_list"]
    output:
        protected("results/version-scan/scan-results.csv")
    log:
        "results/version-scan/scan-results.log"
    script:
        "../scripts/version_scan.py"


rule quic_version_scan__filtered:
    """Filter QUIC scan to domains supporting the QUIC versions
    specified in the config, while accounting for similar domains."""
    input:
        rules.quic_version_scan.output
    output:
        "results/version-scan/scan-results.filtered.csv",
        report("results/plots/frequent-prefixes.png",
               caption="../report/frequent-prefixes.rst",
               category="QUIC Scan"),
        report("results/plots/frequent-slds.png",
               caption="../report/frequent-slds.rst",
               category="QUIC Scan"),
    log:
        "results/version-scan/scan-results.filtered.log"

    params:
        versions=config["quic_versions"],
        sld_domains=config["frequent_slds"],
    script:
        "../scripts/filter_versions.py"
