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
    """Measure overhead traffic for front defense"""
    input:
        schedule="results/collect/front_defended/{sample_id}_{rep_id}/schedule.csv",
        defended="results/collect/front_defended/{sample_id}_{rep_id}/front_cover_traffic.csv",
        baseline="results/collect/front_baseline/{sample_id}_{rep_id}/trace.csv",
        defended_full="results/collect/front_defended/{sample_id}_{rep_id}/front_traffic.csv",
    output: "results/overhead/front/{sample_id}_{rep_id}/overhead.csv"
    log: "results/overhead/front/{sample_id}_{rep_id}/stderr.txt"
    run:
        import pandas as pd
        from pyqcd.overhead import front_overhead

        pd.DataFrame(
            front_overhead.simple_overhead(input.defended,
                           input.baseline,
                           input.defended_full,
                           input.schedule
                          )
        ).to_csv(str(output), header=True, index=True)