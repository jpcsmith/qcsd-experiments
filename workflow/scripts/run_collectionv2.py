#!/usr/bin/env python3
"""Usage: run_collectionv2.py [options] INPUT_DIR OUTPUT_DIR -- NEQO_ARGS...

Options:
    --configfile <file>
        Load configuration values from <file> [default: config/config.yaml]
    --use-multiple-connections
        Use the binary located at the env variable NEQO_BIN_MP.
    --n-monitored <n>
        Collect <n> monitored domains [default: 3]
    --n-instances <n>
        Collect <n> instances per monitored domain [default: 5]
    --n-unmonitored <n>
        Collect <n> unmonitored domains, 1 instance each [default: 15]
    --max-failures <n>
        Stop collecting a domain after <n> sequential failurs [default: 3]
    --timeout <sec>
        Wait <sec> seconds before aborting a collection [default: 120.0]
"""
# pylint: disable=too-many-arguments
import os
import asyncio
import hashlib
import logging
import functools
import subprocess
from pathlib import Path
from typing import Dict, List

import yaml
import lab.tracev2 as trace

import common
from common import neqo
from common.collectv2 import Collector
from common.doceasy import doceasy, Use, Schema

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

    _check_for_user_error(output_dir / "stderr.txt")
    return False


class MisconfigurationError(RuntimeError):
    """Raised when the experiment is misconfigured."""


def _check_for_user_error(stderr: Path):
    """Check for errors that need to be fixed before the collection
    can continue.
    """
    err_txt = stderr.read_text()

    if "No such file or directory" in err_txt:
        raise MisconfigurationError(
            f"Collection run failed due to files being missing. See {stderr!s}"
            "for more details."
        )

    if (
        "error: Found argument" in err_txt
        or "error: Invalid value for" in err_txt
        or "USAGE:" in err_txt
    ):
        raise MisconfigurationError(
            "Collection run failed due to invalid arguments or parameters."
            f"See {stderr!s} for more details."
        )


async def main(
    input_dir: Path,
    output_dir: Path,
    configfile: Path,
    *,
    neqo_args: List[str],
    n_instances: int,
    n_monitored: int,
    n_unmonitored: int,
    max_failures: int,
    timeout: float,
    use_multiple_connections: bool = False,
):
    """Collect all the samples for the speicified arguments."""
    common.init_logging(name_thread=True, verbose=True)

    config = yaml.safe_load(configfile.read_text())
    n_regions = config["wireguard"]["n_regions"]
    n_clients_per_region = config["wireguard"]["n_clients_per_region"]

    _LOGGER.info("Env variable NEQO_BIN=%s", os.environ["NEQO_BIN"])
    _LOGGER.info("Env variable NEQO_BIN_MP=%s", os.environ["NEQO_BIN_MP"])
    if use_multiple_connections:
        os.environ["NEQO_BIN"] = os.environ["NEQO_BIN_MP"]
        _LOGGER.info("Env variable updated NEQO_BIN=%s", os.environ["NEQO_BIN"])

    await Collector(
        functools.partial(
            collect_with_args, neqo_args=neqo_args, config=config,
            timeout=timeout),
        n_regions=n_regions,
        n_clients_per_region=n_clients_per_region,
        n_instances=n_instances,
        n_monitored=n_monitored,
        n_unmonitored=n_unmonitored,
        max_failures=max_failures,
        input_dir=input_dir,
        output_dir=output_dir,
    ).run()


if __name__ == "__main__":
    asyncio.run(main(**doceasy(__doc__, Schema({
        "INPUT_DIR": Use(Path),
        "OUTPUT_DIR": Use(Path),
        "--configfile": Use(Path),
        "--use-multiple-connections": bool,
        "--n-monitored": Use(int),
        "--n-instances": Use(int),
        "--n-unmonitored": Use(int),
        "--max-failures": Use(int),
        "--timeout": Use(float),
        "NEQO_ARGS": [str],
    }, ignore_extra_keys=True))))
