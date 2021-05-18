"""Collect a control PCAP and Tamaraw defended PCAP."""
import io
import sys
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from lab.defences import tamaraw, PACKET_DTYPE

import common
from common import neqo

_LOGGER = logging.getLogger(__name__)


def _load_trace(filename) -> np.ndarray:
    command = [
        "tshark", "-r", str(filename),
        "-T", "fields", "-E", "separator=,",
        "-e", "frame.time_epoch", "-e", "udp.length", "-e", "udp.srcport"
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    data = pd.read_csv(
        io.BytesIO(result.stdout), names=["time", "length", "is_outgoing"]
    )
    data["is_outgoing"] = data["is_outgoing"] != 443
    data.loc[~data["is_outgoing"], "length"] *= -1

    return np.rec.fromarrays(
        data[["time", "length"]].to_numpy().T, dtype=PACKET_DTYPE
    )


def main(input_, output, log, config):
    """Collect a control and Tamaraw defended PCAP. The Tamaraw schedule
    is constructed on the basis of the control PCAP.
    """
    common.init_logging()

    args = [
        "--header", "user-agent", config["user_agent"],
        "--url-dependencies-from", input_["url_dep"]
    ]

    # Run the control setting
    is_success = neqo.run(
        args,
        stdout=output["control"],
        stderr=log["control"],
        pcap_file=output["control_pcap"],
        env={"CSDEF_NO_SHAPING": "True", "RUST_LOG": config["neqo_log_level"]},
        ignore_errors=True,
    )

    if not is_success:
        _LOGGER.error("Exiting due to control setting failure.")
        # Touch the expected output files so that Snakemake doesnt complain
        Path(output["tamaraw"]).touch(exist_ok=False)
        Path(output["tamaraw_pcap"]).touch(exist_ok=False)
        Path(output["schedule"]).touch(exist_ok=False)
        # Exit with a success so that snakemake doesnt complain
        sys.exit(0)

    # Generate the Tamaraw schedule
    control_trace = _load_trace(output["control_pcap"])
    exp_config = config["experiment"]["tamaraw_single_eval"]
    trace = tamaraw.simulate(control_trace, **exp_config["tamaraw_config"])
    pd.DataFrame(trace).to_csv(output["schedule"], header=False, index=False)

    # Run the Tamaraw setting
    neqo.run(
        args + [
            "--target-trace", str(output["schedule"]),
            "--pad-only-mode", "false",  # False to enable shaping
        ],
        stdout=output["tamaraw"],
        stderr=log["tamaraw"],
        pcap_file=output["tamaraw_pcap"],
        env={"RUST_LOG": config["neqo_log_level"]},
        ignore_errors=True,
    )


if __name__ == "__main__":
    snakemake = globals().get("snakemake", None)
    main(
        input_=snakemake.input,
        output=snakemake.output,
        log=snakemake.log,
        config=snakemake.config
    )
