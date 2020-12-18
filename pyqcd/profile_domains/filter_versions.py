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
import json
import logging
from typing import Sequence, Tuple, List, Dict, Final

import numpy as np
import pandas as pd
from publicsuffixlist import PublicSuffixList

import pyqcd
from pyqcd import doceasy
from pyqcd.doceasy import And, Use

_DEFAULT_THRESHOLD: Final = 50
_LOGGER: Final = logging.getLogger("profile_domains.filter")


def split_alt_svc(alt_svc: str) -> List[str]:
    """Split alt-svc records on commas, ignoring quoted commas."""
    in_quote = False
    split_points = [-1]

    for idx, char in enumerate(alt_svc):
        if char == '"':
            in_quote = not in_quote
        if char == "," and not in_quote:
            split_points.append(idx)

    split_points.append(-1)
    return [alt_svc[(split_points[idx] + 1):split_points[idx+1]]
            for idx in range(len(split_points) - 1)]


def parse_alt_svc(alt_svc: str, domain: str) -> List[Tuple[str, str]]:
    """Return a list of protocol ids and authorities from the alt_svc
    record.
    """
    records = []
    # for entry in alt_svc.split(","):
    for entry in split_alt_svc(alt_svc):
        if entry.strip() == "clear":
            continue
        # Entries always start with the protocol
        entry = entry.split(";")[0]
        protocol_id, authority = entry.strip().split("=", maxsplit=1)
        authority = authority.strip('"')

        if authority.startswith(":"):
            authority = domain + authority

        records.append((protocol_id, authority))
    return records


def select_version(row, versions: Sequence[str]) -> Dict[str, str]:
    """Select the first version in the alt_svc field that is in
    versions and return a dict with the new fields.
    """
    domain = row["domain"]
    alt_svc = row["alt_svc"]
    records = parse_alt_svc(alt_svc, domain)

    for protocol_id, authority in records:
        if protocol_id in versions:
            return {
                "domain": domain,
                "protocol_id": protocol_id,
                "authority": authority
            }
    raise ValueError(
        "Alt Svc should contain at least one of the versions: "
        "{alt_svc}, {versions}")


def remove_similar_roots(
    domains: Sequence[str], ranks: Sequence[int]
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
    frame.loc[:, "public_root"] = frame.apply(
        lambda x: x["netloc"][:-(len(x["public_sfx"]) + 1)], axis=1)
    frame = frame.groupby('public_root', group_keys=False).head(n=1)

    return np.sort(frame.index)


def reduce_representation(
    domains: Sequence["str"],
    ranks: Sequence[int],
    sld_domains: Sequence[str],
    whitelist: str = r"(com|co|org|ac)\..*",
    threshold: int = 50,
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


def main(
    infile: str,
    outfile,
    versions: Sequence[str],
    sld_domains: Sequence[str]
):
    """Perform filtering of ranked domains with alt_svc entries.
    """
    pyqcd.init_logging()

    data = pd.read_csv(
        infile, usecols=["rank", "domain", "alt_svc"], index_col="rank"
    ).dropna()
    _LOGGER.info("Loaded %d domains with alt-svc records", len(data))

    # Drop entries that do not support a desired version
    version_mask = pd.Series(False, index=data.index)
    assert versions
    for ver in versions:
        version_mask |= data["alt_svc"].str.contains(ver)
    data = data[version_mask]
    _LOGGER.info("%d domains support the specified versions", len(data))

    # Choose a single version & authority per entry
    data = data.apply(select_version, versions=versions,
                      axis=1, result_type="expand")

    data = data.iloc[remove_similar_roots(data["domain"], data.index)]
    _LOGGER.info("Filtered similar private domains to %d domains", len(data))

    data = data.iloc[reduce_representation(
        data["domain"], data.index, sld_domains=sld_domains,
        threshold=_DEFAULT_THRESHOLD,
    )]
    _LOGGER.info("Reduce representation of %s in the dataset. Now %d domains",
                 sld_domains, len(data))

    data.sort_index().to_csv(outfile, header=True, index=True)


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, {
        "<infile>": str,
        "<outfile>": doceasy.File(mode="w", default="-"),
        "--versions": [str],
        "--sld-domains": [str],
    }))
