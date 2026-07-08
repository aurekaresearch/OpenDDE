from __future__ import annotations

from types import SimpleNamespace

import torch

from opendde.distributed.foldcp.pair_sharding import FoldCPPairShardSpec
from opendde.model.modules.confidence import ConfidenceHead
from opendde.model.opendde import OpenDDE


def test_foldcp_confidence_non_output_rank_processes_every_sample(monkeypatch):
    head = ConfidenceHead.__new__(ConfidenceHead)
    torch.nn.Module.__init__(head)
    head.input_strunk_ln = torch.nn.Identity()
    head.pairformer_stack = SimpleNamespace(blocks=[SimpleNamespace(c_s=1)])
    mesh = SimpleNamespace()
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 1),
        padded_shape=(4, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(2, 2),
        mesh_coord=(0, 0),
    )

    monkeypatch.setattr(head, "_maybe_create_foldcp_mesh", lambda: mesh)
    monkeypatch.setattr(head, "_foldcp_is_non_output_rank", lambda: True)
    monkeypatch.setattr(
        head,
        "_build_confidence_z_init_local",
        lambda **kwargs: torch.zeros_like(kwargs["reference"]),
    )
    monkeypatch.setattr(
        head,
        "_select_distogram_rep_atom_mask",
        lambda input_feature_dict, n_token: torch.ones(n_token, dtype=torch.bool),
    )

    calls = []

    def fake_foldcp_local(**kwargs):
        calls.append(tuple(kwargs["x_pred_rep_coords"].shape))
        return None, None, None, None

    monkeypatch.setattr(head, "memory_efficient_forward_foldcp_local", fake_foldcp_local)

    result = head.forward(
        input_feature_dict={},
        s_inputs=torch.zeros(4, 1),
        s_trunk=torch.zeros(4, 1),
        z_trunk=torch.zeros(2, 2, 1),
        pair_mask=torch.ones(4, 4),
        x_pred_coords=torch.zeros(3, 4, 3),
        z_trunk_spec=spec,
    )

    assert result == (None, None, None, None)
    assert calls == [(4, 3), (4, 3), (4, 3)]


def test_foldcp_model_seed_non_output_rank_skips_rank0_output_merge(monkeypatch):
    model = OpenDDE.__new__(OpenDDE)
    torch.nn.Module.__init__(model)
    monkeypatch.setattr(model, "_foldcp_is_non_output_rank", lambda: True)

    calls = []

    def fake_main_inference_loop(**_kwargs):
        calls.append(len(calls) + 1)
        value = calls[-1]
        return (
            {"coordinate": torch.tensor([[value]], dtype=torch.float32)},
            {"call": value},
            {"elapsed": float(value)},
        )

    monkeypatch.setattr(model, "_main_inference_loop", fake_main_inference_loop)

    pred_dict, log_dict, time_tracker = model.main_inference_loop(
        input_feature_dict={},
        N_cycle=1,
        N_model_seed=3,
    )

    assert calls == [1, 2, 3]
    assert pred_dict["coordinate"].item() == 3
    assert log_dict["call"].tolist() == [1, 2, 3]
    assert time_tracker["elapsed"].tolist() == [1.0, 2.0, 3.0]
