improved_tamaraw_config = config["experiment"]["improved_tamaraw"]

rule improved_tamaraw__plot:
    output:
        "results/plots/improved-tamaraw-kfp-goodbad-split.png"
    input:
        "results/improved-tamaraw/defence~tamaraw/classifier~kfp/predictions.csv",
        "results/improved-tamaraw/defence~simulated-tamaraw/classifier~kfp/predictions.csv"
    params:
        layout={
            "$k$-FP": {
                "QCSD": "results/improved-tamaraw/defence~tamaraw/classifier~kfp/predictions.csv",
                "Simulated": "results/improved-tamaraw/defence~simulated-tamaraw/classifier~kfp/predictions.csv"
            }
        },
        line_styles={"QCSD": "solid", "Simulated": "dotted", "Undef.": "dashdot"},
    notebook:
        "../notebooks/result-analysis-curve.ipynb"


rule improved_tamaraw__simulated_dataset:
    output:
        "results/improved-tamaraw/defence~simulated-tamaraw/dataset.h5"
    input:
        "results/improved-tamaraw/defence~tamaraw/dataset/"
    params:
        **improved_tamaraw_config["dataset"],
        simulate="tamaraw",
        simulate_kws={},
    script:
        "../scripts/create_dataset.py"


rule improved_tamaraw__collect__to_binary:
    output:
        "results/improved-tamaraw/defence~{defence}/dataset.h5"
    input:
        "results/improved-tamaraw/defence~{defence}/dataset/"
    params:
        **improved_tamaraw_config["dataset"],
    script:
        "../scripts/create_dataset.py"


rule improved_tamaraw__collect__dataset:
    input:
        monitored="results/improved-tamaraw/defence~{defence}/dataset~monitored/",
        unmonitored="results/improved-tamaraw/defence~{defence}/dataset~unmonitored/"
    output:
        directory("results/improved-tamaraw/defence~{defence}/dataset/")
    shell:
        "mkdir {output} && cp -al {input.monitored}/* {input.unmonitored}/*  {output}/"


rule improved_tamaraw__collect__unmonitored:
    output:
        directory("results/improved-tamaraw/defence~{defence}/dataset~unmonitored/")
    input:
        "results/improved-tamaraw/other-graphs"
    log:
        "results/improved-tamaraw/defence~{defence}/dataset~unmonitored.log"
    threads:
        workflow.cores
    params:
        neqo_args=build_neqo_args(improved_tamaraw_config),
        configfile=workflow.configfiles[0],
        timeout=60,
        max_failures=1,
        n_monitored=0,
        n_instances=0,
        n_unmonitored=improved_tamaraw_config["dataset"]["n_unmonitored"]
    shell:
        "workflow/scripts/run_collectionv2.py --configfile {params.configfile}"
        " --n-monitored {params.n_monitored} --n-instances {params.n_instances}"
        " --n-unmonitored {params.n_unmonitored} --max-failures {params.max_failures}"
        " --timeout {params.timeout} {input} {output} -- {params.neqo_args} 2> {log}"


rule improved_tamaraw__collect__unmonitored_input:
    output:
        directory("results/improved-tamaraw/other-graphs")
    input:
        graphs="results/webpage-graphs/graphs",
        file_list="results/improved-tamaraw/good-behaving.csv"
    shell:
        "cp -r {input.graphs} {output}"
        " && xargs -a {input.file_list} -I@ rm {output}/@.json"


rule improved_tamaraw__collect__monitored:
    output:
        directory("results/improved-tamaraw/defence~{defence}/dataset~monitored/")
    input:
        "results/improved-tamaraw/well-behaved-graphs/"
    log:
        directory("results/improved-tamaraw/defence~{defence}/dataset~monitored.log")
    threads:
        workflow.cores
    params:
        neqo_args=build_neqo_args(improved_tamaraw_config),
        configfile=workflow.configfiles[0],
        timeout=60,
        max_failures=5,
        n_monitored=improved_tamaraw_config["dataset"]["n_monitored"],
        n_instances=improved_tamaraw_config["dataset"]["n_instances"],
        n_unmonitored=0
    shell:
        "workflow/scripts/run_collectionv2.py --configfile {params.configfile}"
        " --n-monitored {params.n_monitored} --n-instances {params.n_instances}"
        " --n-unmonitored {params.n_unmonitored} --max-failures {params.max_failures}"
        " --timeout {params.timeout} {input} {output} -- {params.neqo_args} 2> {log}"
