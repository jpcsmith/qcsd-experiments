rule quic_version_scan:
    """Scan the an Alexa domain list for QUIC support."""
    input:
        config["alexa_list"]
    output:
        protected("results/profile-domains/scan-results.csv")
    log:
        "results/profile-domains/scan-results.log"
    shell: "python3 -m pyqcd.profile_domains.scan {input} {output} 2> {log}"


rule filtered_domains:
    """Filter QUIC scan to domains supporting the QUIC versions
    specified in the config, while accounting for similar domains."""
    input:
        rules.quic_version_scan.output
    output:
        "results/profile-domains/filtered-domains.csv"
    log:
        "results/profile-domains/filtered-domains.log"
    params:
        versions=[f"--versions {ver}" for ver in config["supported_versions"]],
        sld_domains=[f"--sld-domains {sld}" for sld in config["clamp_sld_domains"]]
    shell: """\
        python3 -m pyqcd.profile_domains.filter_versions \
            {params.versions} {params.sld_domains} {input} {output} 2> {log}
        """
