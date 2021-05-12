"""Script to run an experiment with FRONT and collect traces for both
a control sample and a defend sample.
"""
import logging

import numpy as np
import pandas as pd

import common
from common import neqo


_LOGGER = logging.getLogger(__name__)


def _sample_front_timestamps(max_packets, peak_minimum, peak_maximum, rand):
    n_packets = rand.integers(1, max_packets, dtype=int, endpoint=True)
    _LOGGER.info("Sampled n_packets of %d from the interval [1, %d].",
                 n_packets, max_packets)
    peak = (peak_maximum - peak_minimum) * rand.random() + peak_minimum
    _LOGGER.info("Sampled a rayleigh scale of %.3f from the interval"
                 " [%.3f, %.3f).", peak, peak_minimum, peak_maximum)
    return rand.rayleigh(peak, size=n_packets)


def sample_front_trace(
    max_client_packets: int,
    max_server_packets: int,
    packet_size: int,
    peak_minimum: float,
    peak_maximum: float,
    rand: np.random.Generator,
):
    """Return a dataframe with times and packet sampled according to the
    front defence.
    """
    _LOGGER.info("Sampling outgoing timestamps.")
    out_times = _sample_front_timestamps(
        max_client_packets, peak_minimum, peak_maximum, rand)
    _LOGGER.info("Sampling incoming timestamps.")
    in_times = _sample_front_timestamps(
        max_server_packets, peak_minimum, peak_maximum, rand)

    trace = np.zeros((len(out_times) + len(in_times), 2))
    trace[:len(out_times)] = np.column_stack(
        (out_times, np.full_like(out_times, packet_size))
    )
    trace[-len(in_times):] = np.column_stack(
        (in_times, np.full_like(in_times, -packet_size))
    )

    result = pd.DataFrame(trace, columns=["time", "size"])
    result["size"] = result["size"].astype(int)

    return result.sort_values(by="time")


def main(input_, output, log, params, config):
    """Read configuration parameters from the parent snakemake
    process and collect a control sample followed by a sample
    using a generated front defence.
    """
    common.init_logging()

    rand = np.random.default_rng(params["seed"])
    exp_config = config["experiment"]["front_single_eval"]

    trace = sample_front_trace(**exp_config["front_config"], rand=rand)
    trace.to_csv(output["schedule"], header=False, index=False)

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
