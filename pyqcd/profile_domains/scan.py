"""Usage: script [options] DOMAIN_FILE OUTFILE

Profile the list of domain names provided in DOMAIN_FILE, one per line.
The script documents the final HTTP url that is fetched, the alt-svc
fields in the header specifying other services running, as well as the
respone code of the request.  The results are written in CSV format to
OUTFILE.

Options:
    --max-outstanding n
        Only have n outstanding requests pending at a time.  New
        requests must wait until an existing request completes
        [default: 150].

    --timeout sec
        Wait sec seconds for a request to complete, before aborting it
        [default: 30.0].
"""
import time
import asyncio
import socket
import logging
import pathlib
from typing import Sequence, NamedTuple, Optional
from datetime import timedelta

import aiohttp
from aiohttp.resolver import AsyncResolver
import numpy as np
import pandas as pd

import pyqcd
from pyqcd import doceasy
from pyqcd.doceasy import And, Use, AtLeast

_LOGGER = logging.getLogger(pathlib.Path(__file__).name)

NAMESERVERS = [
    # Google
    '8.8.8.8', '8.8.4.4',
    # Cloudflare
    '1.1.1.1', '1.0.0.1',
    # Quad9
    '9.9.9.9', '149.112.112.112',
    # Verisign
    '64.6.64.6', '64.6.65.6',
    # Level-3
    '209.244.0.3', '209.244.0.4',
    # Freenom
    '80.80.80.80', '80.80.81.81',
    # Open DNS
    '208.67.222.222', '208.67.220.220',
    # Yandex
    '77.88.8.8', '77.88.8.7',
    # Comodo
    '8.26.56.26', '8.20.247.20',
]
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36")


class DomainProfile(NamedTuple):
    """Profiling result of a domain.

    status:
        A positive status indicates an HTTP status code.  A negative
        status indicates one of the custom defined error codes.
    """
    domain: str
    fetch_duration: float
    status: Optional[int] = None
    url: Optional[str] = None
    real_url: Optional[str] = None
    alt_svc: Optional[str] = None
    server: Optional[str] = None
    error: Optional[str] = None
    error_str: Optional[str] = None


async def _profile_domain(
    domain: str, connector, timeout: aiohttp.ClientTimeout
) -> DomainProfile:
    headers = {'user-agent': USER_AGENT}
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, connector_owner=False,
        headers=headers,
    ) as client:
        try:
            async with client.get(f'https://{domain}', read_until_eof=False,
                                  allow_redirects=True) as resp:
                return DomainProfile(
                    domain, fetch_duration=np.nan, status=resp.status,
                    url=str(resp.url), real_url=str(resp.real_url),
                    alt_svc=resp.headers.get('alt-svc', None),
                    server=resp.headers.get('server', None))
        except asyncio.TimeoutError:
            return DomainProfile(domain, np.nan, error='timeout')
        except aiohttp.ClientSSLError as err:
            return DomainProfile(domain, np.nan, error='ssl-error',
                                 error_str=str(err))
        except (aiohttp.ClientError, ValueError) as err:
            return DomainProfile(domain, np.nan, error='other-error',
                                 error_str=repr(err))
        except OSError as err:
            return DomainProfile(domain, np.nan, error=f'oserror({err.errno})',
                                 error_str=err.strerror)


async def profile_domain(domain: str, connector, sem: asyncio.Semaphore,
                         timeout: aiohttp.ClientTimeout) -> DomainProfile:
    """Request an https version of the domain and record the resulting
    status code, url, and alternative services.
    """
    async with sem:
        start_time = time.perf_counter()
        result = await _profile_domain(domain, connector, timeout)
        return result._replace(
            fetch_duration=(time.perf_counter() - start_time))


async def run_profiling(domains: Sequence[str], n_outstanding: int,
                        total_timeout: float):
    """Profile the specified domains, with at most n_outstanding
    requests at a time.
    """
    semaphore = asyncio.Semaphore(n_outstanding)
    timeout = aiohttp.ClientTimeout(total=total_timeout)

    # Create a resolver with custom nameservers, and close it manually as it
    # does not support async with
    resolver = AsyncResolver(nameservers=NAMESERVERS, rotate=True, tries=2)
    try:
        # Specifying the family is necessary to avoid network unreachable errs
        async with aiohttp.TCPConnector(
            resolver=resolver, limit=n_outstanding, use_dns_cache=True,
            ttl_dns_cache=(60 * 5), family=socket.AF_INET, force_close=True,
            enable_cleanup_closed=True
        ) as connector:
            return await asyncio.gather(
                *[profile_domain(d, connector, semaphore, timeout)
                  for d in domains])
    finally:
        await resolver.close()


def main(
    domain_file, outfile, max_outstanding: int = 150,
    timeout: float = 30
):
    """Program entry point."""
    pyqcd.init_logging()

    data = pd.read_csv(domain_file, squeeze=True, names=["rank", "domain"])

    _LOGGER.info("Profiling %d domains with timeout=%.2f and "
                 "max-outstanding=%d.", len(data), timeout, max_outstanding)

    start_time = time.perf_counter()
    result = asyncio.run(
        run_profiling(data["domain"], max_outstanding, timeout))
    duration = time.perf_counter() - start_time

    data.merge(pd.DataFrame(result), left_on="domain", right_on="domain",
               how="left").to_csv(
                   outfile, header=True, index=False, errors="backslashreplace"
               )

    _LOGGER.info("Profiling complete in %s (%.2fs)",
                 timedelta(seconds=duration), duration)


if __name__ == '__main__':
    main(**doceasy.doceasy(__doc__, {
        'DOMAIN_FILE': str,
        'OUTFILE': str,
        '--max-outstanding': And(Use(int), AtLeast(1)),
        '--timeout': And(Use(float), AtLeast(0.0)),
    }))
