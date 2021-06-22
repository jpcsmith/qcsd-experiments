import logging
import inspect
from . import pcap, neqo

__all__ = ["init_logging", "pcap", "neqo"]


def init_logging(verbosity: int = 1, filename=None, name_thread: bool = True):
    """Initialise the default logging behaviour.

    Verbosity is a value from 1 to 2 with 1 being INFO, 2 being DEBUG.
    """
    # Introspect the calling module to get the snakemake logfile if present
    snakemake = inspect.stack()[1][0].f_globals.get("snakemake", None)
    if filename is None and snakemake is not None:
        filename = next(iter(snakemake.log), None)

    log_level = {
        1: logging.INFO,
        2: logging.DEBUG
    }[verbosity]

    thread = "@%(threadName)s" if name_thread else ""
    logging.basicConfig(
        format=f"[%(asctime)s] %(name)s{thread} [%(levelname)s] %(message)s",
        level=log_level, filename=filename)
