"""Collect a control sample and samples across multiple defences.
"""
import io
import sys
import logging
import pathlib
import subprocess

import numpy as np
import pandas as pd
from lab.defences import front, tamaraw, PACKET_DTYPE

import common
from common import neqo

_LOGGER = logging.getLogger(__name__)


def main(input_, output, log, params, config):
    """Perform the collection, ending early and touching any incomplete
    files on failure.
    """
    common.init_logging()

    assert "url_dep" in input_
    assert (
        # Control output files
        "control" in output and "control_pcap" in output
        # Front output files
        and "front" in output and "front_pcap" in output
        and "front_schedule" in output
        # Tamaraw output files
        and "tamaraw" in output and "tamaraw_pcap" in output
        and "tamaraw_schedule" in output
    )
    assert "control" in log and "front" in log and "tamaraw" in log
    assert "seed" in params

    args = [
        "--header", "user-agent", config["user_agent"],
        "--url-dependencies-from", input_["url_dep"]
    ]
    ml_eval = config["experiment"]["ml_eval"]

    if not neqo.run(
        args,
        stdout=output["control"],
        stderr=log["control"],
        pcap_file=output["control_pcap"],
        env={"CSDEF_NO_SHAPING": "True", "RUST_LOG": config["neqo_log_level"]},
        ignore_errors=True,
        tcpdump_kw=dict(snaplen=ml_eval["snaplen"]),
    ):
        _exit_gracefully("control collection failed", output)

    # Generate the Tamaraw schedule
    control_trace = _load_trace(output["control_pcap"])
    trace = tamaraw.simulate(control_trace, **ml_eval["tamaraw_config"])
    pd.DataFrame(trace).to_csv(
        output["tamaraw_schedule"], header=False, index=False
    )

    if not neqo.run(
        args + [
            "--target-trace", str(output["tamaraw_schedule"]),
            "--pad-only-mode", "false",  # False to enable shaping
        ],
        stdout=output["tamaraw"],
        stderr=log["tamaraw"],
        pcap_file=output["tamaraw_pcap"],
        env={"RUST_LOG": config["neqo_log_level"]},
        ignore_errors=True,
        tcpdump_kw=dict(snaplen=ml_eval["snaplen"]),
    ):
        _exit_gracefully("Tamaraw collection failed", output)

    # Collect FRONT last since it has a tendency to generate tail packets that
    # are spaced very far apart resulting in timeouts even though the majority
    # of the data was collected.
    #
    # Generate the FRONT padding schedule
    trace = front.generate_padding(
        **config["experiment"]["ml_eval"]["front_config"],
        random_state=params["seed"],
    )
    pd.DataFrame(trace).to_csv(
        output["front_schedule"], header=False, index=False
    )

    if not neqo.run(
        args + [
            "--target-trace", str(output["front_schedule"]),
            "--pad-only-mode", "true",
        ],
        stdout=output["front"],
        stderr=log["front"],
        pcap_file=output["front_pcap"],
        env={"RUST_LOG": config["neqo_log_level"]},
        ignore_errors=True,
        tcpdump_kw=dict(snaplen=ml_eval["snaplen"]),
    ):
        _exit_gracefully("FRONT collection failed", output)

    _LOGGER.info("All collections were successful.")


def _exit_gracefully(message: str, output_files) -> None:
    # Touch all output files that dont exist to ensure that Snakemake
    # doesnt complain
    for filename in output_files.values():
        if not pathlib.Path(filename).is_file():
            pathlib.Path(filename).touch()

    _LOGGER.error("Exiting: %s", message)
    sys.exit(0)


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


if __name__ == "__main__":
    snakemake = globals().get("snakemake", None)
    main(
        input_=dict(snakemake.input),
        output=dict(snakemake.output),
        log=dict(snakemake.log),
        params=dict(snakemake.params),
        config=dict(snakemake.config)
    )
