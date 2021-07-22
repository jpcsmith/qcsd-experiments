rule demo__collect_front:
    input:
        "results/webpage-graphs/graphs/{i}.json"
    output:
        trace="results/demo/{i}/front/trace.csv",
        schedule="results/demo/{i}/front/schedule.csv",
    log:
        "results/demo/{i}/front.log"
    params:
        dir_="results/demo/{i}/front",
        user_agent=config["user_agent"],
    shell:
        "workflow/scripts/neqo-test --relative-to {params.dir_} --"
        " --url-dependencies-from {input} --defence front --defence-packet-size 1200"
        " --defence-seed 99 --defence-event-log {output.schedule}"
        " --header user-agent '{params.user_agent}' > {params.dir_}/stdout.txt 2> {log}"


rule demo__collect_tamaraw:
    input:
        "results/webpage-graphs/graphs/{i}.json"
    output:
        trace="results/demo/{i}/tamaraw/trace.csv",
        schedule="results/demo/{i}/tamaraw/schedule.csv",
    log:
        "results/demo/{i}/tamaraw.log"
    params:
        dir_="results/demo/{i}/tamaraw",
        user_agent=config["user_agent"],
    shell:
        "workflow/scripts/neqo-test --relative-to {params.dir_} --"
        " --url-dependencies-from {input} --defence tamaraw --defence-packet-size 1200"
        " --tamaraw-modulo 300 --defence-event-log {output.schedule}"
        " --header user-agent '{params.user_agent}' > {params.dir_}/stdout.txt 2> {log}"


rule demo__collect_undefended:
    input:
        "results/webpage-graphs/graphs/{i}.json"
    output:
        trace="results/demo/{i}/undefended/trace.csv",
    log:
        "results/demo/{i}/undefended.log"
    params:
        dir_="results/demo/{i}/undefended",
        user_agent=config["user_agent"],
    shell:
        "workflow/scripts/neqo-test --relative-to {params.dir_} --"
        " --url-dependencies-from {input} --defence none"
        " --header user-agent '{params.user_agent}' > {params.dir_}/stdout.txt 2> {log}"


rule demo__plot:
    input:
        undefended="results/demo/{i}/undefended/trace.csv",
        front="results/demo/{i}/front/trace.csv",
        tamaraw="results/demo/{i}/tamaraw/trace.csv",
    output:
        "results/plots/demo-plot-{i}.png",
    params:
        demo_dir="results/demo/{i}/",
    notebook:
        "../notebooks/demo-plot.ipynb"


rule demo__all:
    input:
        "results/plots/demo-plot-5255.png"
