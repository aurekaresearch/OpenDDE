# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Golden tests for the typed config view (opendde.config.schema).

These guard the behaviour-preserving migration: the typed ``OpenDDEConfig`` must
reproduce the resolved tree produced by the existing merge/CLI engine exactly,
and must keep the ``ml_collections.ConfigDict`` access patterns used across the
~170 consumption sites.
"""

import copy

from opendde.config.config import parse_configs
from opendde.config.inference import build_inference_config, make_base_inference_config
from opendde.config.schema import BaseConfig, OpenDDEConfig


def _engine_resolved_dict(arg_str=None) -> dict:
    """Resolved plain dict straight from the (untouched) merge engine."""
    return parse_configs(
        configs=make_base_inference_config(),
        arg_str=arg_str,
        fill_required_with_null=True,
    ).to_dict()


def _find_extra(model, path="") -> list:
    """Any field the engine produced but the schema failed to model lands in
    ``__pydantic_extra__``; the schema is complete iff this is empty everywhere."""
    leaked = []
    extra = getattr(model, "__pydantic_extra__", None) or {}
    leaked += [f"{path}.{k}" for k in extra]
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseConfig):
            leaked += _find_extra(value, f"{path}.{name}")
    return leaked


def test_typed_config_reproduces_engine_tree_exactly():
    resolved = _engine_resolved_dict()
    typed = OpenDDEConfig.model_validate(copy.deepcopy(resolved))
    # Semantic equality: every key/value round-trips (int->float on bin params is
    # numerically identical and accepted by ==).
    assert typed.model_dump() == resolved


def test_schema_models_every_engine_key():
    typed = OpenDDEConfig.model_validate(_engine_resolved_dict())
    assert _find_extra(typed) == []


def test_cli_overrides_round_trip():
    arg_str = "--model.N_cycle 3 --sample_diffusion.N_step 7 --triangle_attention torch"
    resolved = _engine_resolved_dict(arg_str=arg_str)
    typed = OpenDDEConfig.model_validate(copy.deepcopy(resolved))
    assert typed.model_dump() == resolved
    assert typed.model.N_cycle == 3
    assert typed.sample_diffusion.N_step == 7
    assert typed.triangle_attention == "torch"


def test_build_inference_config_returns_typed_model():
    cfg = build_inference_config(fill_required_with_null=True)
    assert isinstance(cfg, OpenDDEConfig)
    # representative leaf types
    assert isinstance(cfg.c_z, int) and cfg.c_z == 384
    assert isinstance(cfg.model.pairformer.c_z, int) and cfg.model.pairformer.c_z == 384
    assert cfg.dtype in ("bf16", "fp32")
    # Default is empty ("unset"); resolved at run time to CLI > JSON > random seed.
    assert isinstance(cfg.seeds, list) and cfg.seeds == []
    assert isinstance(cfg.infer_setting.chunk_size_thresholds, dict)
    assert cfg.infer_setting.chunk_size_thresholds["2048"] == 256


def test_configdict_access_compatibility():
    cfg = build_inference_config(fill_required_with_null=True)
    # attribute and item access are interchangeable
    assert cfg["c_z"] == cfg.c_z
    assert cfg.sample_diffusion["N_step"] == cfg.sample_diffusion.N_step == 200
    assert cfg.data["msa"]["msa_depth"] == cfg.data.msa.msa_depth == 1280
    # `.get` with default
    assert cfg.model.structural_token_expansion.get("enable", False) is True
    assert cfg.model.structural_token_expansion.get("missing", "fallback") == "fallback"
    # `**section` unpacking (leaf sections feed nn.Module kwargs)
    assert dict(**cfg.model.input_embedder) == {
        "c_atom": 128,
        "c_atompair": 16,
        "c_token": 384,
    }
    # `.to_dict()` parity with `.model_dump()`
    assert (
        cfg.model.diffusion_module.to_dict() == cfg.model.diffusion_module.model_dump()
    )


def test_runtime_mutation_is_preserved():
    cfg = build_inference_config(fill_required_with_null=True)
    cfg.skip_amp.confidence_head = False
    cfg.triangle_multiplicative = "torch"
    cfg["input_json_path"] = "/tmp/x.json"
    assert cfg.skip_amp.confidence_head is False
    assert cfg.triangle_multiplicative == "torch"
    assert cfg.input_json_path == "/tmp/x.json"
