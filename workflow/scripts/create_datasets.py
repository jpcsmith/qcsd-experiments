"""Create a HDF5 dataset with the tables /labels, /sizes and /timestamps.
"""
import logging
import threading
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
    input_: Dict[str, List[str]],
    output: str,
    defence: str,
    simulate: bool,
    config: Dict,
):
    """Create the dataset.

    When simulate is True, use the simulated control trace as opposed to
    the actual defended trace.
    """
    common.init_logging()
    assert isinstance(simulate, bool), "simulate not boolean?"

    ml_eval = config["experiment"]["ml_eval"]
    paths = {}

    n_total_samples = 0
    for setting in ["monitored", "unmonitored"]:
        dirs = np.unique(list(Path(f).parent for f in input_[setting]))
        _LOGGER.info("%d %s directories on %d files.", len(dirs), setting,
                     len(input_[setting]))

        n_samples = ml_eval[setting]["samples"]
        n_instances = ml_eval[setting]["instances"]
        assert setting != "unmonitored" or n_instances == 1

        paths[setting] = _select_samples(dirs, defence, n_samples, n_instances)
        n_total_samples += (n_samples * n_instances)

    with h5py.File(output, mode="w") as hdf:
        for (key, dtype) in [
            ("/labels", np.dtype([("class", "i4")])),
            ("/sizes", h5py.vlen_dtype(np.dtype("i4"))),
            ("/timestamps", h5py.vlen_dtype(np.dtype(float))),
        ]:
            hdf.create_dataset(key, dtype=dtype, shape=(n_total_samples, ))
        _LOGGER.info("Initialised HDF5 file at: %r", output)

        with ThreadPoolExecutor() as executor:
            lock = threading.Lock()

            row_id = 0
            for class_, rep_dirs in enumerate(paths["monitored"].values()):
                for directory in rep_dirs:
                    executor.submit(
                        _extract_sample, directory, defence, simulate,
                        class_=class_, hdf_details=(hdf, row_id, lock)
                    )
                    row_id += 1
            _LOGGER.info("Generated monitored jobs.")

            for rep_dirs in paths["unmonitored"].values():
                assert len(rep_dirs) == 1, "only 1-sample per unmon supported"
                executor.submit(
                    _extract_sample, rep_dirs[0], defence, simulate,
                    class_=-1, hdf_details=(hdf, row_id, lock)
                )
                row_id += 1
            _LOGGER.info("Generated unmonitored jobs.")

            del paths  # Allow the GC to cleanup the massive lists of paths
            _LOGGER.info("Waiting for all thread tasks to complete...")
    _LOGGER.info("Completed creating the dataset.")


def _select_samples(
    directories,
    defence,
    n_samples: int,
    n_instances: int,
):
    _LOGGER.info("Counting the number of successes...")

    samples: Dict[int, List[str]] = {}
    for directory in directories:
        sample_id, _ = map(int, str(directory.name).split("_"))

        is_valid = neqo.is_run_successful(directory/"control.stdout.txt")
        if is_valid and defence != "control":
            stdout = directory/f"{defence}.stdout.txt"
            is_valid = is_valid and neqo.is_run_almost_successful(stdout, 10)

        if is_valid:
            samples.setdefault(sample_id, []).append(directory)

    n_sufficient = sum(
        len(rep_dirs) >= n_instances for rep_dirs in samples.values()
    )
    _LOGGER.info(
        "Sample ids with more than %d instances: %d", n_instances, n_sufficient
    )

    if n_sufficient < n_samples:
        insufficient = {
            sample_id: (n_instances - len(reps))
            for sample_id, reps in samples.items() if len(reps) < n_instances
        }
        raise InsufficientSamplesError(f"Need more samples: {insufficient}")

    selected_samples = sorted(
        [id_ for id_, reps in samples.items() if len(reps) >= n_instances]
    )[:n_samples]
    return {id_: sorted(samples[id_][:n_instances]) for id_ in selected_samples}


def _extract_sample(
    directory, defence: str, simulate: bool, class_: int, hdf_details
):
    if not simulate:
        sample_trace = trace.from_pcap(directory/f"{defence}.pcapng")
    else:
        control = trace.from_pcap(directory/"control.pcapng")
        if defence == "control":
            sample_trace = control
        else:
            schedule = trace.from_csv(directory/f"{defence}-schedule.csv")

            if defence == "front":
                sample_trace = front.simulate(control, schedule)
            elif defence == "tamaraw":
                sample_trace = schedule
            else:
                raise ValueError(f"Unknown defence {defence!r}")

    hdf, row_id, hdf_lock = hdf_details
    with hdf_lock:
        hdf["/labels"][row_id] = (class_, )
        hdf["/sizes"][row_id] = sample_trace["size"]
        hdf["/timestamps"][row_id] = sample_trace["time"]


if __name__ == "__main__":
    snakemake = globals().get("snakemake", None)
    main(
        input_=dict(snakemake.params["inputs"]),
        output=str(snakemake.output[0]),
        defence=snakemake.params["defence"],
        simulate=snakemake.params["simulate"],
        config=dict(snakemake.config),
    )
