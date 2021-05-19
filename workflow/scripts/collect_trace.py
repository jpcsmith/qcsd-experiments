"""Script to run an experiment to collect a control and defended pcap
for a specified trace.
"""
import common
from common import neqo


def main(input_, output, log, params, config):
    """Read configuration parameters from the parent snakemake
    process and collect a control sample followed by a sample
    using a the specified trace.
    """
    common.init_logging()

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

    # Run the defended setting
    args.extend(["--target-trace", str(input_["schedule"])])
    if "pad_only_mode" in params:
        assert params["pad_only_mode"] in ("true", "false"), "invalid parameter"
        args.extend(["--pad-only-mode", params["pad_only_mode"]])
    if "msd_limit_excess" in params:
        args.extend(["--msd-limit-excess", params["msd_limit_excess"]])

    neqo.run(
        args,
        stdout=output["defended"],
        stderr=log["defended"],
        pcap_file=output["defended_pcap"],
        env={"RUST_LOG": config["neqo_log_level"]},
        ignore_errors=True,
    )


if __name__ == "__main__":
    snakemake = globals().get("snakemake", None)
    main(
        input_=snakemake.input,
        output=snakemake.output,
        log=snakemake.log,
        params=dict(snakemake.params),
        config=snakemake.config
    )
