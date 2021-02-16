import numpy as np
from matplotlib import pyplot as plt
import scipy as scp
import json

def plot_dummy_distribution(inputs, output):
    print("inputs: ", inputs)
    print("output: ", output)
    N = len(inputs)
    pearsons_TX = np.array([])
    pearsons_RX = np.array([])
    for path in inputs:
        with open(path, "r") as json_file:
            data = json.load(json_file)
            print(data['TX']['stats'])
            (r, _) = data['TX']['stats']
            print(r)
            pearsons_TX = np.append(pearsons_TX, r)
            print("***")

    print("Pearsons: ", pearsons_TX)
    f, ax = plt.subplot(1, 1, figsize=(12, 6))
    print("MADONNA")
    count, bins, _ = ax[0].hist(pearsons_TX, bins=50, density=True)
    print("CAZZO")
    mu, std = scp.stats.norm.fit(pearsons_TX)
    print("FIGA")
    ax[0].plot(
             bins,
             1/(std * np.sqrt(2 * np.pi)) *
             np.exp(-(bins - mu)**2 / (2 * std**2)),
             linewidth=2, color='r', label=f'$\mu={mu}$\n$stdev={std}$'
            )

    f.savefig(output, dpi=300,  bbox_inches="tight")
