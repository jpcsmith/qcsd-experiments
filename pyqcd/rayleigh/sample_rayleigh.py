#!/usr/bin/env python3
"""Creates chaff traffic schedules sampled from Rayleigh distribution
"""

import numpy as np
import pandas as pd


def sample_timestamps(n_max, sigma_min, sigma_max):
    """Samples n ~ U(1, n_max) timestamps according to Rayleigh distribution
        with median .
    """
    n = np.random.randint(1, n_max)
    w = np.random.rand() * (sigma_max - sigma_min) + sigma_min

    return np.random.rayleigh(w, n)


def set_seed(seed):
    np.random.seed(seed=seed)


def create_trace(seed, N_TX, N_RX, W_min, W_max, size, outcsv):
    """Creates a dummy schedule from rayleigh distribution"""
    print(f"Seed: {seed}")
    set_seed(seed)

    tx_ts = sample_timestamps(N_TX, W_min, W_max)
    rx_ts = sample_timestamps(N_RX, W_min, W_max)

    tx_trace = [(ts, size) for ts in tx_ts]
    tx_trace.sort(key=lambda t: t[0])
    rx_trace = [(ts, -size) for ts in rx_ts]
    rx_trace.sort(key=lambda t: t[0])

    trace = tx_trace + rx_trace
    pd.DataFrame(trace).to_csv(str(outcsv), index=False, header=False)
