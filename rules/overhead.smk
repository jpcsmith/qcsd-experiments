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

def front_overhead__all_input(wildcards):
    input_bas = rules.collect_front_baseline.output[1] # trace.pacapng
    s_ids_bas = glob_wildcards(input_bas).sample_id
    r_ids_bas = glob_wildcards(input_bas).rep_id
    input_def = rules.collect_front_defended.output[1] # trace.pacapng
    s_ids_def = glob_wildcards(input_def).sample_id
    r_ids_def = glob_wildcards(input_def).rep_id

    intersection = list(set(zip(s_ids_bas, r_ids_bas))
                        & set(zip(s_ids_def, r_ids_def))
                        )
    s_ids = [i for (i,_) in intersection]
    r_ids = [i for (_,i) in intersection]
    return expand(rules.front_overhead.output[0], zip, sample_id=s_ids, rep_id=r_ids)

rule front_overhead__all:
    """Gives front overhead for all collected traces """
    input: front_overhead__all_input
    message: "rule front_overhead__all:\n\tmeasure all overheads"

def overhead_front_aggreagted_input(wildcards):
    overhead_dir = rules.front_overhead.output[0] # points to the csv
    s_ids = glob_wildcards(overhead_dir).sample_id
    s_rep = glob_wildcards(overhead_dir).rep_id
    

    return expand(rules.front_overhead.output[0], zip, sample_id=s_ids,
            rep_id=s_rep)

rule overhead_front_aggregated:
    """Aggregates results from front_overhead and plot distribution of mean of results"""
    input: overhead_front_aggreagted_input
    output:
        plot = "results/overhead/front/aggregate/overhead_distribution.png",
        json = "results/overhead/front/aggregate/res.json",
        stdout = "results/overhead/front/aggregate/stdout.txt"
    log: "results/overhead/front/aggregate/stderr.txt"
    shell: """\
        python3 -m pyqcd.overhead.aggregate_overhead --output-plot {output.plot} --output-json {output.json} \
        -- {input} > {output.stdout} 2> {log}
    """

rule test_expand:
    run:
        input_bas = rules.collect_front_baseline.output[1] # trace.pacapng
        s_ids_bas = glob_wildcards(input_bas).sample_id
        r_ids_bas = glob_wildcards(input_bas).rep_id
        input_def = rules.collect_front_defended.output[1] # trace.pacapng
        s_ids_def = glob_wildcards(input_def).sample_id
        r_ids_def = glob_wildcards(input_def).rep_id

        intersection = list(set(zip(s_ids_bas, r_ids_bas))
                            & set(zip(s_ids_def, r_ids_def))
                            )
        s_ids = [i for (i,_) in intersection]
        r_ids = [i for (_,i) in intersection]

        print(
            expand(rules.front_overhead.output[0], zip, sample_id=s_ids, rep_id=r_ids)
        )