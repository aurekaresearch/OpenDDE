# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Logging utilities for OpenDDE."""

import logging


def init_logging() -> None:
    """
    Initialize logging configuration for OpenDDE.

    Sets up a consistent logging format across all modules with timestamp,
    log level, file location, and message.
    """
    log_format = (
        "%(asctime)s,%(msecs)-3d %(levelname)-8s "
        "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    )
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
