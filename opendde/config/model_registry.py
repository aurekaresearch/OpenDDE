# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Model registry for released inference checkpoints.

# Currently, only the following inference model is supported. It uses the
# defaults in model_base.py and a 2021-09-30 wwPDB data cutoff.

| Model Name  |  MSA/Constraint/RNA MSA/Template | Model Parameters (M) |
|-------------|----------------------------------|----------------------|
| `opendde_v1` |      ✓ / × / ✓ / ✓              |       656            |


"""

from typing import Any

DEFAULT_MODEL_NAME = "opendde_v1"

model_configs: dict[str, dict[str, Any]] = {
    # opendde_v1 is the base config in model_base.py. Keep this empty while it is
    # the only supported model to avoid duplicating defaults in two places.
    DEFAULT_MODEL_NAME: {},
}
