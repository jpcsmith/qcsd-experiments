import logging
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import lab.tracev2 as trace
from lab.defences import front

import common
from common import neqo

_LOGGER = logging.getLogger(__name__)


class InsufficientSamplesError(Exception):
    """Raised if there are insufficient samples."""


def main(
    input_: List[str],
    output: str,
    defence: str,
    setting: str,
    simulate: bool,
    config: Dict,
):
    common.init_logging()

    n_samples = config["experiment"]["ml_eval"][setting]["samples"]
    assert n_samples >= 1, "must be at least 1 sample required"
    n_instances = config["experiment"]["ml_eval"][setting]["instances"]
    assert n_instances >= 1, "must be at least 1 instance per sample"
    assert isinstance(simulate, bool), "simulate not boolean?"

    directories = np.unique(list(Path(f).parent for f in input_))
    _LOGGER.info("Processing %d directories of %d files.", len(directories),
                 len(input_))

    # Counts of sample ids and the number of instances found
    counts: Dict[int, int] = dict()

    with h5py.File(output, mode="w") as hdf, ThreadPoolExecutor() as executor:
        for (key, dtype) in [
            ("/labels", np.dtype([("sample", "i4"), ("rep", "i4")])),
            ("/sizes", h5py.vlen_dtype(np.dtype("i4"))),
            ("/timestamps", h5py.vlen_dtype(np.dtype(float))),
        ]:
            hdf.create_dataset(key, dtype=dtype, shape=(n_samples*n_instances,))
        _LOGGER.info("Initialised HDF5 file at: %r", output)

        selected = _count_samples(directories, defence, n_samples, n_instances)

        index = 0
        for (sample_id, rep_id, sample_trace) in executor.map(
            lambda directory: _extract_sample(directory, defence, simulate),
            directories
        ):
            case_id = f"{sample_id}_{rep_id}"

            if sample_trace is None:
                _LOGGER.debug("Skipping due to failure: %s", case_id)
            elif sample_id not in selected:
                _LOGGER.debug("Skipping due to unselected: %s", case_id)
            elif counts.get(sample_id, 0) == n_instances:
                _LOGGER.debug("Skipping due to enough instances: %s", case_id)
            else:
                hdf["/labels"][index] = (sample_id, rep_id)
                hdf["/sizes"][index] = sample_trace["size"]
                hdf["/timestamps"][index] = sample_trace["time"]

                counts[sample_id] = counts.get(sample_id, 0) + 1
                index += 1

                if index in np.linspace(0, 10000, 10, dtype=int):
                    complete = index * 100 / (n_samples * n_instances)
                    _LOGGER.info("Progress: %f%% complete", complete)

        _LOGGER.info("Done iterating through %d files.", sum(counts.values()))


def _count_samples(
    directories,
    defence,
    n_samples: int,
    n_instances: int,
):
    _LOGGER.info("Counting the number of successes...")
    counts: Dict[int, int] = {}
    for directory in directories:
        stdout = directory/f"{defence}.stdout.txt"
        sample_id, _ = map(int, str(directory.name).split("_"))
        if (
            neqo.is_run_successful(directory/"control.stdout.txt")
            and neqo.is_run_almost_successful(stdout, 10)
        ):
            counts[sample_id] = counts.get(sample_id, 0) + 1

    n_sufficient = sum(1 for count in counts.values() if count >= n_instances)
    _LOGGER.info(
        "Sample ids with more than %d instances: %d", n_instances, n_sufficient
    )

    if n_sufficient < n_samples:
        insufficient = {
            sample_id: (n_instances - counts[sample_id]) for sample_id in counts
            if counts[sample_id] < n_instances
        }
        raise InsufficientSamplesError(f"Need more samples: {insufficient}")

    sufficient_sample_ids = sorted(
        [id_ for id_, count in counts.items() if count >= n_instances]
    )
    return set(sufficient_sample_ids[:n_samples])


def _extract_sample(directory, defence: str, simulate: bool):
    stdout = directory/f"{defence}.stdout.txt"
    sample_id, rep_id = map(int, str(directory.name).split("_"))

    if (
        not neqo.is_run_successful(directory/"control.stdout.txt")
        or not neqo.is_run_almost_successful(stdout, 10)
    ):
        sample_trace = None
    elif not simulate:
        sample_trace = trace.from_pcap(directory/f"{defence}.pcapng")
    else:
        control = trace.from_pcap(directory/"control.pcapng")
        schedule = trace.from_csv(directory/f"{defence}-schedule.csv")

        if defence == "front":
            sample_trace = front.simulate(control, schedule)
        else:
            raise ValueError(f"Unknown defence {defence!r}")

    return (sample_id, rep_id, sample_trace)


if __name__ == "__main__":
    snakemake = globals().get("snakemake", None)
    main(
        input_=list(snakemake.input),
        output=str(snakemake.output[0]),
        setting=snakemake.params["setting"],
        defence=snakemake.params["defence"],
        simulate=snakemake.params["simulate"],
        config=dict(snakemake.config),
    )
