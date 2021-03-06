"""Collect a trace with the provided tamaraw arguments."""
# pylint: disable=too-many-arguments
import os
import hashlib
import logging
import functools
import subprocess
from pathlib import Path
from typing import Dict, List

import lab.tracev2 as trace

import common
from common import neqo
from common.collect import Collector

_LOGGER = logging.getLogger(__name__)


def collect_with_args(
    input_file: Path,
    output_dir: Path,
    region_id: int,
    client_id: int,
    neqo_args: List[str],
    config: Dict,
    timeout: float,
) -> bool:
    """Collect a trace using NEQO and return True iff it was successful."""
    #: Copy the args, since it is shared among all method instances
    neqo_args = neqo_args + [
        "--header", "user-agent", config["user_agent"],
        "--url-dependencies-from", str(input_file),
    ]
    if (
        "tamaraw" in neqo_args
        or "front" in neqo_args
        or "schedule" in neqo_args
    ):
        neqo_args += ["--defence-event-log", str(output_dir / "schedule.csv")]

    if "front" in neqo_args:
        # Take a 4 byte integer from the output directory
        dir_bytes = str(output_dir).encode('utf-8')
        seed = int(hashlib.sha256(dir_bytes).hexdigest(), 16) & 0xffffffff
        neqo_args += ["--defence-seed", str(seed)]

    client_port = config["wireguard"]["client_ports"][region_id][client_id]
    interface = config["wireguard"]["interface"]

    try:
        (result, pcap) = neqo.run(
            neqo_args,
            neqo_exe=[
                "workflow/scripts/neqo-client-vpn", str(region_id),
                str(client_id)
            ],
            check=True,
            stdout=str(output_dir / "stdout.txt"),
            stderr=str(output_dir / "stderr.txt"),
            pcap=neqo.PIPE,
            env={"RUST_LOG": config["neqo_log_level"]},
            tcpdump_kw={
                "capture_filter": f"udp port {client_port}", "iface": interface,
            },
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as err:
        _LOGGER.debug("Neqo timed out: %s", err)
    except subprocess.CalledProcessError as err:
        _LOGGER.debug("Neqo failed with error: %s", err)
    else:
        assert result.returncode == 0
        assert pcap is not None
        # (output_dir / "trace.pcapng").write_bytes(result.pcap)
        trace.to_csv((output_dir / "trace.csv"),
                     trace.from_pcap(pcap, client_port=client_port))
        _LOGGER.debug("Neqo succeeded.")
        return True
    return False


def main(
    input_,
    output,
    config: Dict,
    *,
    neqo_args: List[str],
    n_instances: int,
    n_monitored: int,
    n_unmonitored: int = 0,
    max_failures: int = 3,
    timeout: float = 120,
    use_multiple_connections: bool = False,
):
    """Collect all the samples for the speicified arguments."""
    common.init_logging(name_thread=True, verbose=True)

    neqo_args = [str(x) for x in neqo_args]
    n_regions = config["wireguard"]["n_regions"]
    n_clients_per_region = config["wireguard"]["n_clients_per_region"]

    _LOGGER.info("Env variable NEQO_BIN=%s", os.environ["NEQO_BIN"])
    _LOGGER.info("Env variable NEQO_BIN_MP=%s", os.environ["NEQO_BIN_MP"])
    if use_multiple_connections:
        os.environ["NEQO_BIN"] = os.environ["NEQO_BIN_MP"]
        _LOGGER.info("Env variable updated NEQO_BIN=%s", os.environ["NEQO_BIN"])

    Collector(
        functools.partial(
            collect_with_args, neqo_args=neqo_args, config=config,
            timeout=timeout),
        n_regions=n_regions,
        n_clients_per_region=n_clients_per_region,
        n_instances=n_instances,
        n_monitored=n_monitored,
        n_unmonitored=n_unmonitored,
        max_failures=max_failures,
        input_dir=input_,
        output_dir=output,
    ).run()


if __name__ == "__main__":
    snakemake = globals().get("snakemake", None)
    main(input_=str(snakemake.input[0]), output=str(snakemake.output[0]),
         config=snakemake.config, **snakemake.params)
