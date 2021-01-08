import logging

__all__ = ["init_logging"]


def init_logging(verbosity: int = 1):
    """Initialise the default logging behaviour.

    Verbosity is a value from 1 to 2 with 1 being INFO, 2 being DEBUG.
    """
    log_level = {
        1: logging.INFO,
        2: logging.DEBUG
    }[verbosity]

    logging.basicConfig(
        format='[%(asctime)s] %(name)s [%(levelname)s] %(message)s',
        level=log_level)
