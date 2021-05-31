"""Script to run an experiment with FRONT and collect traces for both
a control sample and a defend sample.
"""
import pandas as pd
from lab.defences import front

import common
from common import neqo


def main(input_, output, log, params, config):
    """Read configuration parameters from the parent snakemake
    process and collect a control sample followed by a sample
    using a generated front defence.
    """
    common.init_logging()

    # Generate the FRONT padding schedule
    exp_config = config["experiment"]["front_single_eval"]
    trace = front.generate_padding(
        **exp_config["front_config"], random_state=params["seed"]
    )
    pd.DataFrame(trace).to_csv(output["schedule"], header=False, index=False)

    args = [
        "--header", "user-agent", config["user_agent"],
        "--url-dependencies-from", input_["url_dep"]
    ]

    # Run the control setting
    neqo.run(
        args,
        stdout=output["control"],
        stderr=log["control"],
        pcap_file=output["control_pcap"],
        env={"CSDEF_NO_SHAPING": "True", "RUST_LOG": config["neqo_log_level"]},
        ignore_errors=True,
    )

    if "drop_unsat_events" in params.keys():
        args += [
            "--drop-unsat-events", str(params["drop_unsat_events"]).lower()
        ]

    # Run the front setting
    neqo.run(
        args + [
            "--target-trace", str(output["schedule"]),
            "--pad-only-mode", "true",
        ],
        stdout=output["front"],
        stderr=log["front"],
        pcap_file=output["front_pcap"],
        env={"RUST_LOG": config["neqo_log_level"]},
        ignore_errors=True,
    )


if __name__ == "__main__":
    snakemake = globals().get("snakemake", None)
    main(
        input_=snakemake.input,
        output=snakemake.output,
        log=snakemake.log,
        params=snakemake.params,
        config=snakemake.config
    )
