import logging

__all__ = ["init_logging"]


def init_logging():
    """Initialise the default logging behaviour."""
    logging.basicConfig(
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)
