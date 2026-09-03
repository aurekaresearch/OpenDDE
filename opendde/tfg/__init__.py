# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""
This code provides a modular guidance engine that can be plugged into
`opendde.model.generator.sample_diffusion`.
"""

from .config import Schedule, TFGConfig, parse_tfg_config, schedule_from_cfg
from .engine import TFGEngine
