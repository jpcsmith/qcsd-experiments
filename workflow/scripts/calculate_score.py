"""Calculate padding-only scores.
"""
import logging
import itertools
import functools
from pathlib import Path
import multiprocessing
import multiprocessing.pool
from typing import Sequence, Dict, Optional

import numpy as np
import pandas as pd
import tslearn.metrics
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean

import common
from common import timeseries

_LOGGER = logging.getLogger(__name__)


def main(
    input_,
    output,
    *,
    defence: str,
    ts_offset: Dict[str, int],
    resample_rates: Sequence["str"],
    lcss_eps: int,
    filter_below: Sequence[int] = (0,),
    jobs: Optional[int] = None,
):
    """Score how close a defended time series is to the theoretical."""
    common.init_logging()
    _LOGGER.info("Using parameters: %s", locals())

    directories = sorted([x.parent for x in Path(input_).glob("**/defended/")])
    _LOGGER.info("Found %d samples", len(directories))

    jobs = jobs or (multiprocessing.cpu_count() or 4)
    func = functools.partial(
        _calculate_score,
        defence=defence,
        ts_offset=ts_offset,
        resample_rates=resample_rates,
        filter_below=filter_below,
        lcss_eps=lcss_eps,
    )

    if jobs > 1:
        chunksize = max(len(directories) // (jobs * 2), 1)
        with multiprocessing.pool.Pool(jobs) as pool:
            scores = list(pool.imap_unordered(func, directories, chunksize=chunksize))
    else:
        # Run in the main process
        scores = list(map(func, directories))
    _LOGGER.info("Score calculation complete")

    pd.DataFrame.from_records(itertools.chain.from_iterable(scores)).to_csv(
        output, header=True, index=False
    )


def _calculate_score(
    dir_,
    *,
    defence: str,
    ts_offset,
    resample_rates,
    filter_below,
    lcss_eps,
):
    """Score how close a padding-only defended time series is to
    padding schedule.
    """
    assert defence in ("front", "tamaraw")
    pad_only = defence == "front"

    schedule_ts = timeseries.from_csv(dir_ / "defended" / "schedule.csv")
    defended_ts = timeseries.from_csv(dir_ / "defended" / "trace.csv")
    control_ts = timeseries.from_csv(dir_ / "undefended" / "trace.csv")

    offsets = range(ts_offset["min"], ts_offset["max"], ts_offset["inc"])
    simulated_ts = simulate_with_lag(
        control_ts, schedule_ts, defended_ts, offsets, pad_only=pad_only
    )

    results = []
    for rate, direction, min_pkt_size in itertools.product(
        resample_rates, ("in", "out"), filter_below
    ):
        series = pd.DataFrame(
            {
                "a": timeseries.resample(
                    _filter(defended_ts[direction], min_pkt_size), rate
                ),
                "b": timeseries.resample(
                    _filter(simulated_ts[direction], min_pkt_size), rate
                ),
                "c": timeseries.resample(
                    _filter(control_ts[direction], min_pkt_size), rate
                ),
            }
        ).fillna(0)

        _LOGGER.debug("Series length at rate %s: %d", rate, len(series))
        _LOGGER.debug("Series summary at rate %s: %s", rate, series.describe())

        results.append(
            {
                "sample": str(dir_),  # Sample name
                "rate": rate,
                "dir": direction,
                "min_pkt_size": min_pkt_size,
                "pearsonr": pearsonr(series["a"], series["b"])[0],
                "spearmanr": spearmanr(series["a"], series["b"])[0],
                "lcss": tslearn.metrics.lcss(series["a"], series["b"], eps=lcss_eps),
                "euclidean": _scaled(euclidean, series["b"], series["a"], series["c"]),
            }
        )
    return results


def _scaled(metric_fn, series_a, series_b, series_c) -> float:
    reference_point = metric_fn(series_a, series_c)
    return (reference_point - metric_fn(series_a, series_b)) / reference_point


def _filter(column, below):
    assert column.min() >= 0
    return column[column >= below]


def simulate_with_lag(
    control,
    schedule,
    defended,
    offsets,
    pad_only: bool,
    rate="5ms",
):
    """Find the best offset such that the simulated trace formed by
    combining the control and schedule where the schedule is lagged
    the offset, has the lowest euclidean distance to the defended trace.

    When pad_only is False, the schedule is taken as the already
    simulated trace
    """
    defended_in = timeseries.resample(defended["in"], rate)
    simulated_out = (
        control["out"].append(schedule["out"]) if pad_only else schedule["out"]
    )

    best_offset = 0
    best_distance = np.inf
    best_simulated = None

    for offset in offsets:
        shifted_schedule = pd.Series(
            schedule["in"].values,
            index=(schedule["in"].index + pd.Timedelta(f"{offset}ms")),
        )
        simulated_in = (
            control["in"].append(shifted_schedule) if pad_only else shifted_schedule
        )

        # Put together in a dataframe to ensure they have the same length and
        # indices
        frame = pd.DataFrame(
            {
                "defended": defended_in,
                "simulated": timeseries.resample(simulated_in, rate),
            }
        ).fillna(0)

        distance = euclidean(frame["defended"], frame["simulated"])
        if distance < best_distance:
            best_offset = offset
            best_distance = distance
            best_simulated = simulated_in

    assert best_simulated is not None

    _LOGGER.debug("Using an incoming offset of %d ms", best_offset)
    return pd.DataFrame(
        {
            # Take the sum of any given time instance to handle rare duplicates
            "in": best_simulated.groupby("time").sum(),
            "out": simulated_out.groupby("time").sum(),
        }
    ).fillna(0)


if __name__ == "__main__":
    main(
        str(snakemake.input[0]),
        str(snakemake.output[0]),
        **snakemake.params,
        jobs=snakemake.threads,
    )
