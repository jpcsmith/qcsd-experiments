"""Collect a defended and control trace."""
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
    common_args = [
        "--url-dependencies-from", str(input_file),
        "--header", "user-agent", config["user_agent"],
    ]
    defended_args = neqo_args + common_args + [
        "--defence-event-log", str(output_dir / "defended" / "schedule.csv"),
    ]

    client_port = config["wireguard"]["client_ports"][region_id][client_id]
    interface = config["wireguard"]["interface"]

    try:
        for setting, args, directory in [
            ("defended", defended_args, (output_dir / "defended")),
            ("undefended", common_args, (output_dir / "undefended")),
        ]:
            directory.mkdir(exist_ok=True)

            # front needs a seed, make one based on the directory and save it
            if setting == "defended" and "front" in args:
                # Take a 4 byte integer from the output directory
                seed = int(
                    hashlib.sha256(str(directory).encode('utf-8')).hexdigest(),
                    16
                ) & 0xffffffff
                args += ["--defence-seed", seed]

            args = [str(arg) for arg in args]

            _LOGGER.debug("Collecting setting %r", setting)
            (result, pcap) = neqo.run(
                args,
                neqo_exe=[
                    "workflow/scripts/neqo-client-vpn", str(region_id),
                    str(client_id)
                ],
                check=True,
                stdout=str(directory / "stdout.txt"),
                stderr=str(directory / "stderr.txt"),
                pcap=neqo.PIPE,
                env={"RUST_LOG": config["neqo_log_level"]},
                tcpdump_kw={
                    "capture_filter": f"udp port {client_port}",
                    "iface": interface,
                },
                timeout=timeout,
            )
            assert result.returncode == 0
            assert pcap is not None
            trace.to_csv((directory / "trace.csv"),
                         trace.from_pcap(pcap, client_port=client_port))
    except subprocess.TimeoutExpired as err:
        _LOGGER.debug("Neqo timed out on setting %r: %s", setting, err)
        return False
    except subprocess.CalledProcessError as err:
        _LOGGER.debug("Neqo failed in setting %r with error: %s", setting, err)
        return False
    return True


def main(
    input_,
    output,
    config: Dict,
    *,
    neqo_args: List[str],
    n_instances: int = 0,
    n_monitored: int = 0,
    n_unmonitored: int = 0,
    max_failures: int = 3,
    timeout: float = 120,
    use_multiple_connections: bool = False,
):
    """Collect all the samples for the speicified arguments."""
    common.init_logging(name_thread=True, verbose=True)

    if (
        "tamaraw" not in neqo_args
        and "front" not in neqo_args
        and "schedule" not in neqo_args
    ):
        raise ValueError("The arguments must correspond to a defence.")

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
