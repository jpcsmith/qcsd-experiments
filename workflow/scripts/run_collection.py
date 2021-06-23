"""Collect a trace with the provided tamaraw arguments."""
# pylint: disable=too-many-arguments
import logging
import functools
from pathlib import Path
from typing import Dict, List

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
) -> bool:
    """Collect a trace using NEQO and return True iff it was successful."""
    #: Copy the args, since it is shared among all method instances
    neqo_args = neqo_args + [
        "--url-dependencies-from", str(input_file),
        "--defence-event-log", str(output_dir / "schedule.csv"),
    ]
    client_port = config["wg_client_ports"][region_id][client_id]

    result = neqo.run(
        neqo_args,
        neqo_exe=[
            "workflow/scripts/neqo-client-vpn", str(region_id), str(client_id)
        ],
        check=False,
        stdout=str(output_dir / "stdout.txt"),
        stderr=str(output_dir / "stderr.txt"),
        pcap=neqo.PIPE,
        env={"RUST_LOG": config["neqo_log_level"]},
        tcpdump_kw={
            "capture_filter": f"udp port {client_port}", "iface": "docker0",
        },
    )

    if result.returncode == 0:
        assert result.pcap is not None
        (output_dir / "trace.pcapng").write_bytes(result.pcap)

    return result.returncode == 0


def main(
    input_,
    output,
    config: Dict,
    *,
    neqo_args: List[str],
    n_regions: int,
    n_clients_per_region: int,
    n_instances: int,
    n_monitored: int,
    n_unmonitored: int,
    max_failures: int,
):
    """Collect all the samples for the speicified arguments."""
    common.init_logging(name_thread=True, verbose=True)

    Collector(
        functools.partial(
            collect_with_args, neqo_args=neqo_args, config=config),
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
