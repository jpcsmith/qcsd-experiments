rule test_overhead:
    """Testing overhead script """
    input:
        schedule="results/collect/front_defended/020_2/schedule.csv",
        defended="results/collect/front_defended/020_2/front_cover_traffic.csv",
        baseline="results/collect/front_baseline/020_2/trace.csv",
        defended_full="results/collect/front_defended/020_2/front_traffic.csv",
    shell: """\
        python3 -m pyqcd.overhead.front_overhead -- \
        {input.defended} {input.baseline} {input.defended_full} {input.schedule}
        """

rule front_overhead:
    """Testing overhead script """
    input:
        schedule="results/collect/front_defended/{sample_id}_{rep_id}/schedule.csv"
    output:
        stdout="results/overhead/front/{sample_id}_{rep_id}/stdout.txt"
    log: "results/overhead/front/{sample_id}_{rep_id}/stderr.txt"
    shell: """\
        python3 -m pyqcd.overhead.front_overhead -- {input.schedule} > {output.stdout} 2> {log}
        """
