"""Create a dataset of monitored and unmonitored traces."""
from pathlib import Path
from itertools import product, islice
import h5py
import numpy as np
import lab.tracev2 as trace
import common

LABELS_DTYPE = np.dtype([("class", "i4"), ("region", "i4"), ("sample", "i4")])


def main(
    input_: str,
    output: str,
    *,
    n_monitored: int,
    n_instances: int,
    n_unmonitored: int,
    simulate: bool,
    n_regions: int,
):
    """Create an HDF5 dataset at output using the files found in the
    directory input_.  The CSV traces should be found under

        <input_>/<page_id>/<region>_<sample>/

    """
    common.init_logging()

    sample_dirs = [p for p in Path(input_).iterdir() if p.is_dir()]
    sample_dirs.sort(key=lambda p: int(p.stem))

    if len(sample_dirs) < n_monitored + n_unmonitored:
        raise ValueError(f"Insufficient samples: {len(sample_dirs)}")
    sample_dirs = sample_dirs[:(n_monitored + n_unmonitored)]

    # TODO: Need to handle the FRONT defence
    assert isinstance(simulate, bool), "simulate not boolean?"
    trace_file = "schedule.csv" if simulate else "trace.csv"

    n_rows = n_monitored * n_instances + n_unmonitored
    labels = np.recarray(n_rows, dtype=LABELS_DTYPE)
    sizes = np.full(n_rows, None, dtype=object)
    timestamps = np.full(n_rows, None, dtype=object)

    index = 0
    for label, sample_path in enumerate(sample_dirs[:n_monitored]):
        # Tterator that takes one sample from each region before taking
        # the second sample from a region. This is akin to how we
        # generated the samples.
        sample_iter = product(range(n_instances), range(n_regions))

        # Take only the first n_instances from the infinite iterator
        for (id_, region_id) in islice(sample_iter, n_instances):
            index = (label * n_instances) + (id_ * n_regions) + region_id
            labels[index] = (label, region_id, id_)

            sample = trace.from_csv(sample_path/f"{region_id}_{id_}"/trace_file)
            sizes[index] = sample["size"]
            timestamps[index] = sample["time"]

    region_index = 0
    for sample_path in sample_dirs[n_monitored:(n_monitored + n_unmonitored)]:
        paths = [(region_id, sample_path/f"{region_id}_0"/trace_file)
                 for region_id in range(n_regions)]
        paths = [(region_id, p) for (region_id, p) in paths if p.is_file()]
        assert paths, "no samples in directory?"

        # Select a path from the list, cycling among the used region if there
        # are more than one regions, otherwise returning an inde in the paths
        # list
        region_id, path = paths[(region_index % n_regions) % len(paths)]
        region_index += 1

        index = index + 1
        labels[index] = (-1, region_id, 0)

        sample = trace.from_csv(path)
        sizes[index] = sample["size"]
        timestamps[index] = sample["time"]
    assert index == (n_rows - 1), "not enough samples"

    order = labels.argsort(order=["class", "region"])
    with h5py.File(output, mode="w") as h5out:
        h5out.create_dataset("/labels", dtype=LABELS_DTYPE, data=labels[order])
        h5out.create_dataset(
            "/sizes", dtype=h5py.vlen_dtype(np.dtype("i4")), data=sizes[order]
        )
        h5out.create_dataset(
            "/timestamps", dtype=h5py.vlen_dtype(np.dtype(float)),
            data=timestamps[order]
        )


if __name__ == "__main__":
    snakemake = globals().get("snakemake", None)
    main(
        input_=str(snakemake.input[0]),
        output=str(snakemake.output[0]),
        n_monitored=snakemake.params["n_monitored"],
        n_unmonitored=snakemake.params["n_unmonitored"],
        n_instances=snakemake.params["n_instances"],
        simulate=snakemake.params["simulate"],
        n_regions=snakemake.config["wireguard"]["n_regions"],
    )
