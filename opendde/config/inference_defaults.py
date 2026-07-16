# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# pylint: disable=C0114
import os
from typing import Any

from opendde.config.data import default_root_dir
from opendde.config.extend_types import ListValue, RequiredValue
from opendde.config.model_registry import DEFAULT_MODEL_NAME

OPENDDE_ROOT_DIR = os.environ.get("OPENDDE_ROOT_DIR", default_root_dir())
inference_configs: dict[str, Any] = {
    "model_name": DEFAULT_MODEL_NAME,  # inference model selection
    # Empty = "unset"; resolved at run time to CLI > JSON modelSeeds > random seed.
    "seeds": ListValue([], dtype=int),
    "dump_dir": "./output",
    "need_atom_confidence": False,
    "sorted_by_ranking_score": True,
    "device": "auto",
    "input_json_path": RequiredValue(str),
    "load_checkpoint_dir": os.path.join(OPENDDE_ROOT_DIR, "checkpoint"),
    "num_workers": 0,
    "use_msa": True,
    "enable_tf32": True,
    "enable_efficient_fusion": True,
    "enable_diffusion_shared_vars_cache": True,
    "msa_pair_as_unpair": True,
    "use_template": False,
    "use_rna_msa": False,
    # Fold-CP migration switch.  "single" keeps the original single-card path;
    # "distributed" enables the future 2D context-parallel path.
    "foldcp_mode": "single",
    "foldcp_size_dp": 1,
    "foldcp_size_cp": 1,
    "foldcp_devices": "",
    "foldcp_metrics_jsonl": "",
}
