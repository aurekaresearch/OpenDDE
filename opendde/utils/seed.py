# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These are process-wide switches. Set both branches explicitly so a
    # non-deterministic run cannot inherit True from an earlier deterministic
    # run in a long-lived Python process.
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))
    if deterministic:
        torch.backends.cudnn.benchmark = False
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
