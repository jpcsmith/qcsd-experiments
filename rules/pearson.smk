rule pearson_dummy:
    """Measures Pearson correlation of defended trace with the dummy schedule"""
    input:
        defended="results/collect/front_defended/{sample_id}/front_cover_traffic.csv",
        baseline="results/collect/front_defended/{sample_id}/schedule.csv"
    output:
        graph="results/pearson/front/{sample_id}/rolling_pearson.png",
        stdout="results/pearson/front/{sample_id}/stdout.txt"
    log:
        "results/pearson/front/{sample_id}/stderr.txt"
    shell: """\
        python3 -m pyqcd.pearson.measure_pearson --output-file {output.graph} \
        -- {input.defended} {input.baseline} > {output.stdout} 2> {log}
        """