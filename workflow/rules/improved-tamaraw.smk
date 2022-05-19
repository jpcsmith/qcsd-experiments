improved_tamaraw_config = config["experiment"]["improved_tamaraw"]


rule improved_tamaraw__collect__to_binary:
    """Convert a collected file-based dataset to a binary dataset for faster reads
    (pattern rule)."""
    output:
        "results/improved-tamaraw/conn~single/defence~{defence}/dataset.h5"
    input:
        "results/improved-tamaraw/conn~single/defence~{defence}/dataset/"
    params:
        **improved_tamaraw_config["dataset"],
    script:
        "../scripts/create_dataset.py"


rule improved_tamaraw__collect:
    """Collect samples for the single-connection machine-learning evaluation
    (pattern rule)."""
    output:
        directory("results/improved-tamaraw/conn~single/defence~{defence}/dataset/")
    input:
        "results/webpage-graphs/graphs/"
    log:
        directory("results/improved-tamaraw/conn~single/defence~{defence}/dataset.log")
    threads:
        workflow.cores
    params:
        neqo_args=build_neqo_args(improved_tamaraw_config),
        n_monitored=improved_tamaraw_config["dataset"]["n_monitored"],
        n_instances=improved_tamaraw_config["dataset"]["n_instances"],
        n_unmonitored=improved_tamaraw_config["dataset"]["n_unmonitored"],
    script:
        "../scripts/run_collection.py"
