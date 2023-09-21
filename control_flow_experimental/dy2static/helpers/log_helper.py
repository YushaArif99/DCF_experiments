import logging

__all__ = ['get_logger']


def get_logger(name, level, fmt=None):
    """
    Get logger from logging with given name, level and format without
    setting logging basicConfig. For setting basicConfig in paddle
    will disable basicConfig setting after import paddle.

    Args:
        name (str): The logger name.
        level (logging.LEVEL): The base level of the logger
        fmt (str): Format of logger output

    Returns:
        logging.Logger: logging logger with given settings

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> import logging
            >>> from paddle.base import log_helper

            >>> logger = log_helper.get_logger(__name__, logging.INFO,
            ...                 fmt='%(asctime)s-%(levelname)s: %(message)s')
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()

    if fmt:
        formatter = logging.Formatter(fmt=fmt, datefmt='%a %b %d %H:%M:%S')
        handler.setFormatter(formatter)

    logger.addHandler(handler)

    # stop propagate for propagating may print
    # log multiple times
    logger.propagate = False
    return logger
