rule pearson_front_dummy:
    """Measures Pearson correlation of defended trace with the dummy schedule"""
    input:
        defended="{rootdir}/results/collect/front_defended/{configdir}/{seeddir}/{sample_id}_{rep_id}/front_cover_traffic.csv",
        baseline="{rootdir}/results/collect/front_defended/{configdir}/{seeddir}/{sample_id}_{rep_id}/schedule.csv",
        dummy_ids="{rootdir}/results/collect/front_defended/{configdir}/{seeddir}/{sample_id}_{rep_id}/dummy_streams.txt"
    output:
        plot="{rootdir}/results/pearson/front/{configdir}/{seeddir}/{sample_id}_{rep_id}/rolling_pearson.png",
        json="{rootdir}/results/pearson/front/{configdir}/{seeddir}/{sample_id}_{rep_id}/res.json",
        stdout="{rootdir}/results/pearson/front/{configdir}/{seeddir}/{sample_id}_{rep_id}/stdout.txt"
    params:
        window=125, # int
        rate=50.0, # float
        rootdir=config['rootdir'],
        configdir=config['configdir'],
        seeddir=config['seeddir']
    log:
        "{rootdir}/results/pearson/front/{configdir}/{seeddir}/{sample_id}_{rep_id}/stderr.txt"
    shell: """\
        python3 -m pyqcd.pearson.measure_pearson --output-plot {output.plot} --output-json {output.json} \
        --window-size {params.window} --sample-rate {params.rate} \
        -- {input.defended} {input.baseline} {input.dummy_ids} > {output.stdout} 2> {log}
        """

def pearson_front_dummy__all_input(wildcards):
    import string
    
    collect_dir = rules.collect_front_defended.output["success"] # points to the json
    
    rootdir=config['rootdir']
    configdir=config['configdir']
    seeddir=config['seeddir']
    class FormatDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    
    formatter = string.Formatter()
    mapping = FormatDict(rootdir=rootdir, configdir=configdir, seeddir=seeddir)
    print(formatter.vformat(collect_dir, (), mapping))
    collect_dir = formatter.vformat(collect_dir, (), mapping)
    s_ids = glob_wildcards(collect_dir).sample_id
    s_rep = glob_wildcards(collect_dir).rep_id

    pearson_dir = rules.pearson_front_dummy.output["json"]
    pearson_dir = formatter.vformat(pearson_dir, (), mapping)
    
    return expand(pearson_dir, zip, sample_id=s_ids,
            rep_id=s_rep)

rule pearson_front_dummy__all:
    """Determine the number of samples collected and measures the
    Pearson correaltion for the dummy traffic for each."""
    input: pearson_front_dummy__all_input
    params:
        rootdir=config['rootdir'],
        configdir=config['configdir'],
        seeddir=config['seeddir']
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
    params:
        window=25, # int
        rate=50.0 # float
    log:
        "results/pearson/front/{sample_id}_{rep_id}/stderr_test.txt"
    shell: """\
        python3 -m pyqcd.pearson.measure_pearson_full --output-plot {output.plot} --output-json {output.json} \
        -- {input.defended} {input.baseline} {input.schedule} > {output.stdout} 2> {log}
        """

def pearson_aggregated_input(wildcards):
    collect_dir = rules.pearson_front_dummy.output[1] # points to the json
    s_ids = glob_wildcards(collect_dir).sample_id
    s_rep = glob_wildcards(collect_dir).rep_id
    

    return expand(rules.pearson_front_dummy.output["json"], zip, sample_id=s_ids,
            rep_id=s_rep)

def pearson_aggregated_full_input(wildcards):
    collect_dir = rules.pearson_front_full.output[1] # points to the json
    s_ids = glob_wildcards(collect_dir).sample_id
    s_rep = glob_wildcards(collect_dir).rep_id
    

    return expand(rules.pearson_front_full.output["json"], zip, sample_id=s_ids,
            rep_id=s_rep)

rule pearson_front_dummy_aggregated:
    """Aggregates results from pearson_front_dummy and pearson_front_full into one statistic"""
    input: pearson_aggregated_input
    output: 
        plot = "results/pearson/front/aggregate/pearson_distribution.png",
        json = "results/pearson/front/aggregate/res_dummy.json",
        stdout = "results/pearson/front/aggregate/stdout.txt"
    log: "results/pearson/front/aggregate/stderr.txt"
    shell: """\
        python3 -m pyqcd.pearson.aggregate_pearson --output-plot {output.plot} --output-json {output.json} \
        -- {input} > {output.stdout} 2> {log}
        """

rule pearson_front_full_aggregated:
    """Aggregates results from pearson_front_full into one statistic"""
    input: pearson_aggregated_input
    output: 
        plot = "results/pearson/front/aggregate/pearson_distribution_full.png",
        json = "results/pearson/front/aggregate/res_full.json",
        stdout = "results/pearson/front/aggregate/stdout_full.txt"
    log: "results/pearson/front/aggregate/stderr_full.txt"
    shell: """\
        python3 -m pyqcd.pearson.aggregate_pearson --output-plot {output.plot} --output-json {output.json} \
         -- {input} > {output.stdout} 2> {log}
        """
