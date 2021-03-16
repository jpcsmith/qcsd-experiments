"""Usage:
    script [--versions ver]... [--sld-domains dom]... <infile> [<outfile>]

Perform filtering of ranked domains with alt_svc entries.

All domains without an alt_svc version specified in versions are
dropped.

A single domain is selected (lowest rank) among those with the same
public suffixes. For example, example.com would be selected from
[(1, example.com), (2, example.de)].

Among the second level domains specified in sld_domains, the top
ranking _DEFAULT_THRESHOLD will be selected. For example, for
sld_domains = [blogspot.com], then the top ranking 50 of
[a.blogspot.com, b.blogspot.com, ...] would be selected.

A csv with the remaining domains, their ranks, and the associated
authority to request QUIC via that domain is written to outfile.

Options:
    --versions versions
        QUIC versions which are deemed acceptable using Alt-svc
        protocol ids.

    --sld-domains domains
        Domains that are above the default number of allowed second
        level domains that are not on the whitelist and that should be
        downsampled.
"""
import logging
from typing import Sequence, Final, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from publicsuffixlist import PublicSuffixList

import common
from common import doceasy, alt_svc

_DEFAULT_THRESHOLD: Final = 50
_LOGGER: Final = logging.getLogger("profile_domains.filter")


def _maybe_plot_counts(
    series, filename: Optional[str], top: int = 20, x_scale: str = None
):
    if filename is None:
        return

    fig, axes = plt.subplots()
    order = series.value_counts().index[:top]
    sns.countplot(y=series, order=order, ax=axes)
    if x_scale is not None:
        axes.set_xscale(x_scale)
    fig.savefig(filename, bbox_inches="tight")


def remove_similar_roots(
    domains: Sequence[str], ranks: Sequence[int], plot_filename=None
) -> Sequence[int]:
    """Remove domains which only differ in the public suffix and return
    the remaining indices, preferring domains with lower ranks.

    We note that mail.ch and mail.co.uk may be entirely different
    domain. However, given the fact that many domains have different
    localisations of their domain with differing public suffixes, we
    opt to simply remove all such occurences.
    """
    frame = (pd.Series(domains)
             .reset_index(drop=True).rename("netloc")
             .str.replace(r":\d+", "")
             .to_frame()
             .assign(ranks=ranks)
             .sort_values(by="ranks"))

    psl = PublicSuffixList()
    frame.loc[:, "public_sfx"] = frame['netloc'].apply(psl.publicsuffix)
    frame.loc[:, "private_part"] = frame.apply(
        lambda x: x["netloc"][:-(len(x["public_sfx"]) + 1)], axis=1)

    _maybe_plot_counts(frame["private_part"], plot_filename)

    frame = frame.groupby('private_part', group_keys=False).head(n=1)
    return np.sort(frame.index)


def reduce_representation(
    domains: Sequence["str"],
    ranks: Sequence[int],
    sld_domains: Sequence[str],
    whitelist: str = r"(com|co|org|ac)\..*",
    threshold: int = 50,
    plot_filename: Optional[str] = None,
) -> Sequence[int]:
    """Reduce the representation of the specified sld_domains
    second-level domains to within the threshold.

    Any domains which are still not within the threshold, but are not
    excluded by the whitelist will result in an error.

    Return the index of the reduced sample, lower ranks take precedence
    when selecting the sample.
    """
    frame = (pd.Series(domains).reset_index(drop=True).rename("netloc")
             .str.replace(r":\d+", "")
             .to_frame()
             .assign(ranks=ranks)
             .sort_values(by="ranks"))

    # Split the domain parts and rejoin everything from the second level domain
    # to the top-level domain
    frame["2LD"] = frame["netloc"].str.split(".").apply(
        lambda x: ".".join(x[-2:]))

    whitelist_mask = frame["2LD"].str.match(whitelist)
    exceptions_mask = whitelist_mask | ~frame["2LD"].isin(sld_domains)

    _maybe_plot_counts(frame["2LD"], plot_filename, x_scale="log")

    # Check that there are no others over the threshold when only considering
    # the whitelist
    non_filtered = (frame[~whitelist_mask & ~frame["2LD"].isin(sld_domains)]
                    .loc[:, "2LD"]
                    .value_counts())
    if (non_filtered > threshold).any():
        unaccounted = non_filtered[non_filtered > threshold].to_dict()
        raise ValueError(f"The provided sld_domains ({sld_domains}) and "
                         f"whitelist ({whitelist}) did not account for all "
                         f"excessive domains: {unaccounted}.")

    # Now perform the downsampling
    samples_idx = (frame[~exceptions_mask]
                   .groupby("2LD", group_keys=False)
                   .head(threshold)
                   .index)

    frame.drop(columns="2LD", inplace=True)
    return np.sort(np.concatenate((frame[exceptions_mask].index, samples_idx)))


# pylint: disable=too-many-arguments
def main(
    infile: str,
    outfile,
    versions: Sequence[str],
    sld_domains: Sequence[str],
    public_root_plot=None,
    sld_plot=None,
):
    """Perform filtering of ranked domains with alt_svc entries.
    """
    common.init_logging()

    data = pd.read_csv(infile, usecols=["rank", "domain", "alt_svc"]).dropna()
    _LOGGER.info("Loaded %d domains with alt-svc records", len(data))

    data["alt_svc"] = data["alt_svc"].apply(alt_svc.parse, errors="log")
    data = data.explode(column="alt_svc", ignore_index=True)
    data[["protocol", "authority"]] = pd.DataFrame(
        data["alt_svc"].tolist(), index=data.index)

    # Drop entries with a non-443 authority
    data = data[data["authority"] == ":443"]

    # Drop entries that do not support a desired version
    data = data[data["protocol"].isin(versions)]
    # Select one entry per domain
    data = data.groupby("rank").head(n=1).set_index("rank")[["domain"]]
    _LOGGER.info("%d domains support the specified versions",
                 data["domain"].nunique())

    filtered_idx = remove_similar_roots(
        data["domain"], data.index, plot_filename=public_root_plot)
    data = data.iloc[filtered_idx]
    _LOGGER.info("Filtered similar private domains to %d domains", len(data))

    data = data.iloc[reduce_representation(
        data["domain"], data.index, sld_domains=sld_domains,
        threshold=_DEFAULT_THRESHOLD, plot_filename=sld_plot
    )]
    _LOGGER.info("Reduced representation of %s in the dataset. Now %d domains",
                 sld_domains, len(data))

    data.sort_index().to_csv(outfile, header=True, index=True)


if __name__ == "__main__":
    try:
        KW_ARGS = {
            "infile": snakemake.input[0],                    # type: ignore
            "outfile": open(snakemake.output[0], mode="w"),  # type: ignore
            "versions": snakemake.params["versions"],        # type: ignore
            "sld_domains": snakemake.params["sld_domains"],  # type: ignore
            "public_root_plot": snakemake.output[1],         # type: ignore
            "sld_plot": snakemake.output[2],                 # type: ignore
        }
    except NameError:
        KW_ARGS = doceasy.doceasy(__doc__, {
            "<infile>": str,
            "<outfile>": doceasy.File(mode="w", default="-"),
            "--versions": [str],
            "--sld-domains": [str],
        })
    main(**KW_ARGS)
