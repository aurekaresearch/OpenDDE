# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import logging
from typing import Optional


class WarningFilter(logging.Filter):
    def filter(self, record):
        # Filter out specific warnings by their message or other criteria
        if "simtk.openmm" in record.getMessage():
            return False
        return True


def get_logger(
    name: str = "", loglevel: str = "INFO", log_file_path: Optional[str] = None
) -> logging.Logger:
    """
    Get a logger with the specified name and add handlers to the root logger.

    Args:
        name (str): The name of the logger. Defaults to an empty string.
        loglevel (str): The logging level. Defaults to "INFO".
        log_file_path (Optional[str]): The path to the log file. Defaults to None.

    Returns:
        logging.Logger: The configured logger.
    """
    root_logger = logging.getLogger()
    logger = logging.getLogger(name)
    # we only add handlers to the root logger! Let the propogation handle the rest.
    add_handlers(root_logger, loglevel, log_file_path)
    return logger


def add_handlers(
    logger: logging.Logger, loglevel: str, log_file_path: Optional[str] = None
) -> logging.Logger:
    """
    Add handlers to the specified logger.

    Args:
        logger (logging.Logger): The logger to which handlers will be added.
        loglevel (str): The logging level.
        log_file_path (Optional[str]): The path to the log file. Defaults to None.

    Returns:
        logging.Logger: The logger with added handlers.
    """
    fmt = "%(asctime)-15s [%(pathname)s:%(lineno)d] %(levelname)s %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)
    numeric_loglevel = getattr(logging, loglevel.upper(), logging.INFO)
    logger.setLevel(numeric_loglevel)

    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    else:
        handler = logger.handlers[0]
    handler.setFormatter(formatter)
    warning_filter = WarningFilter()
    handler.addFilter(warning_filter)

    # we output to at most two streams: one stdout and one file
    if log_file_path is not None and len(logger.handlers) == 1:
        handler = logging.FileHandler(log_file_path, mode="a")
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler.addFilter(warning_filter)

    return logger
