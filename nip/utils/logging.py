"""Utility functions for logging."""

from logging import Logger, Handler


def set_logger_handler(logger: Logger, handler: Handler):
    """Set the handler for the logger.

    Parameters
    ----------
    logger : Logger
        The logger to set the handler for.
    handler : Handler
        The handler to set for the logger.
    """

    for other_handler in logger.handlers:
        if other_handler is handler:
            continue
        logger.removeHandler(other_handler)
    logger.addHandler(handler)
