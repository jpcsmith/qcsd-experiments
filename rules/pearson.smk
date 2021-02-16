rule pearson_front_dummy:
    """Measures Pearson correlation of defended trace with the dummy schedule"""
    input:
        defended="results/collect/front_defended/{sample_id}_{rep_id}/front_cover_traffic.csv",
        baseline="results/collect/front_defended/{sample_id}_{rep_id}/schedule.csv",
        dummy_ids="results/collect/front_defended/{sample_id}_{rep_id}/dummy_streams.txt"
    output:
        plot="results/pearson/front/{sample_id}_{rep_id}/rolling_pearson.png",
        json="results/pearson/front/{sample_id}_{rep_id}/res.json",
        stdout="results/pearson/front/{sample_id}_{rep_id}/stdout.txt"
    log:
        "results/pearson/front/{sample_id}_{rep_id}/stderr.txt"
    shell: """\
        python3 -m pyqcd.pearson.measure_pearson --output-plot {output.plot} --output-json {output.json} \
        -- {input.defended} {input.baseline} {input.dummy_ids} > {output.stdout} 2> {log}
        """

def pearson_front_dummy__all_input(wildcards):
    input_dir = checkpoints.url_dependencies__csv.get(**wildcards).output[0]
    sample_ids = glob_wildcards(input_dir + "/{sample_id}.csv").sample_id

    return expand(rules.pearson_front_dummy.output["plot"], sample_id=sample_ids,
                   rep_id=range(config["collect_reps"]))

rule pearson_front_dummy__all:
    """Determine the number of samples collected and measures the
    Pearson correaltion for the dummy traffic for each."""
    input: pearson_front_dummy__all_input
    message: "rule pearson_front_dummy__all:\n\tMeasure all traces."

rule pearson_front_full:
    """Measures Pearson correlation of defended trace with the full trace of the baseline"""
    input:
        defended="results/collect/front_defended/{sample_id}_{rep_id}/front_traffic.csv",
        baseline="results/collect/front_baseline/{sample_id}_{rep_id}/trace.csv",
        schedule="results/collect/front_defended/{sample_id}_{rep_id}/schedule.csv"
    output:
        stdout="results/pearson/front/{sample_id}_{rep_id}/stdout_test.txt",
        json="results/pearson/front/{sample_id}_{rep_id}/res_full.json",
        plot="results/pearson/front/{sample_id}_{rep_id}/rolling_pearson_all.png"
    log:
        "results/pearson/front/{sample_id}_{rep_id}/stderr_test.txt"
    shell: """\
        python3 -m pyqcd.pearson.measure_pearson_full --output-plot {output.plot} --output-json {output.json} \
        -- {input.defended} {input.baseline} {input.schedule} > {output.stdout} 2> {log}
        """

def pearson_aggregated_input(wildcards):
    collect_dir = rules.collect_front_defended.output[3] # points to the pcap
    s_ids = glob_wildcards(collect_dir).sample_id
    s_rep = glob_wildcards(collect_dir).rep_id
    

    return expand(rules.pearson_front_dummy.output["json"], sample_id=s_ids,
            rep_id=range(max([int(r)+1 for r in s_rep])))


rule pearson_front_dummy_aggregated:
    """Aggregates results from pearson_front_dummy and pearson_front_full into one statistic"""
    input: pearson_aggregated_input
    output: 
        plot = "results/pearson/front/aggregate/pearson_distribution.png",
        stdout = "results/pearson/front/aggregate/stdout.txt"
    log: "results/pearson/front/aggregate/stderr.txt"
    # run:
    # # Must change this to shell because of spawning process
    #     from pyqcd.pearson import aggregate_pearson

    #     aggregate_pearson.plot_dummy_distribution(input, output)
    shell: """\
        python3 -m pyqcd.pearson.aggregate_pearson --output-plot {output.plot} -- {input} 
        > {output.stdout} 2> {log}\
        """

rule test_read:
    run:
        # input_dir = checkpoints.url_dependencies__csv.get().output[0]
        # sample_ids = glob_wildcards(input_dir + "/{sample_id}.csv").sample_id
        
        collect_dir = rules.collect_front_defended.output[3]
        s_ids = glob_wildcards(collect_dir).sample_id
        s_rep = glob_wildcards(collect_dir).rep_id
        

        print(expand(rules.pearson_front_dummy.output["json"], sample_id=s_ids,
                rep_id=range(max([int(r)+1 for r in s_rep]))))
        #print(test_dir)
