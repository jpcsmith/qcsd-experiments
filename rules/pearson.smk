rule pearson_front_dummy:
    """Measures Pearson correlation of defended trace with the dummy schedule"""
    input:
        defended="results/collect/front_defended/{sample_id}/front_cover_traffic.csv",
        baseline="results/collect/front_defended/{sample_id}/schedule.csv"
    output:
        plot="results/pearson/front/{sample_id}/rolling_pearson.png",
        stdout="results/pearson/front/{sample_id}/stdout.txt"
    log:
        "results/pearson/front/{sample_id}/stderr.txt"
    shell: """\
        python3 -m pyqcd.pearson.measure_pearson --output-file {output.plot} \
        -- {input.defended} {input.baseline} > {output.stdout} 2> {log}
        """

def pearson_front_dummy__all_input(wildcards):
    input_dir = checkpoints.url_dependencies__csv.get(**wildcards).output[0]
    sample_ids = glob_wildcards(input_dir + "/{sample_id}.csv").sample_id

    return expand(rules.pearson_front_dummy.output["plot"], sample_id=sample_ids)

rule pearson_front_dummy__all:
    """Determine the number of samples collected and measures the
    Pearson correaltion for the dummy traffic for each."""
    input: pearson_front_dummy__all_input
    message: "rule pearson_front_dummy__all:\n\tMeasure all traces."