from __future__ import annotations

from contextlib import nullcontext
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from opendde.distributed.foldcp.pair_sharding import FoldCPPairShardSpec
from opendde.model.modules.confidence import ConfidenceHead
from opendde.model.opendde import OpenDDE, _offload_prediction_tree_to_cpu


def test_foldcp_confidence_non_output_rank_processes_every_sample(monkeypatch):
    head = ConfidenceHead.__new__(ConfidenceHead)
    torch.nn.Module.__init__(head)
    head.input_strunk_ln = torch.nn.Identity()
    head.pairformer_stack = SimpleNamespace(blocks=[SimpleNamespace(c_s=1)])
    mesh = SimpleNamespace(group_2d=object())
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
        calls.append(
            (
                tuple(kwargs["x_pred_rep_coords"].shape),
                kwargs["pair_output_device"],
            )
        )
        return None, None, None, None

    monkeypatch.setattr(
        head, "memory_efficient_forward_foldcp_local", fake_foldcp_local
    )
    monkeypatch.setattr(
        "opendde.model.modules.confidence.run_group_rank_action_synchronized",
        lambda action, **kwargs: action(),
    )

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
    assert calls == [
        ((4, 3), torch.device("cpu")),
        ((4, 3), torch.device("cpu")),
        ((4, 3), torch.device("cpu")),
    ]


def test_confidence_preallocates_multi_sample_outputs(monkeypatch):
    head = ConfidenceHead.__new__(ConfidenceHead)
    torch.nn.Module.__init__(head)
    head.input_strunk_ln = torch.nn.Identity()
    head.linear_no_bias_s1 = torch.nn.Linear(1, 1, bias=False)
    head.linear_no_bias_s2 = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(head.linear_no_bias_s1.weight)
    torch.nn.init.zeros_(head.linear_no_bias_s2.weight)
    head.pairformer_stack = SimpleNamespace(blocks=[])

    monkeypatch.setattr(head, "_maybe_create_foldcp_mesh", lambda: None)
    monkeypatch.setattr(head, "_foldcp_is_non_output_rank", lambda: False)
    monkeypatch.setattr(
        head,
        "_select_distogram_rep_atom_mask",
        lambda input_feature_dict, n_token: torch.ones(n_token, dtype=torch.bool),
    )

    calls = []

    def fake_forward(**_kwargs):
        value = float(len(calls) + 1)
        calls.append(value)
        return (
            torch.full((4, 2), value),
            torch.full((4, 4, 3), value),
            torch.full((4, 4, 3), value),
            torch.full((4, 2), value),
        )

    monkeypatch.setattr(head, "memory_efficient_forward", fake_forward)
    plddt, pae, pde, resolved = head.forward(
        input_feature_dict={},
        s_inputs=torch.zeros(4, 1),
        s_trunk=torch.zeros(4, 1),
        z_trunk=torch.zeros(4, 4, 1),
        pair_mask=torch.ones(4, 4),
        x_pred_coords=torch.zeros(3, 4, 3),
    )

    assert calls == [1.0, 2.0, 3.0]
    assert plddt.shape == (3, 4, 2)
    assert pae.shape == (3, 4, 4, 3)
    assert pde.shape == (3, 4, 4, 3)
    assert resolved.shape == (3, 4, 2)
    assert plddt[:, 0, 0].tolist() == calls
    assert pae[:, 0, 0, 0].tolist() == calls


def test_post_confidence_preserves_pair_storage_device(monkeypatch):
    model = OpenDDE.__new__(OpenDDE)
    torch.nn.Module.__init__(model)
    model.configs = SimpleNamespace(
        need_atom_confidence=False,
        confidence=SimpleNamespace(),
    )
    observed = {}

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        model, "add_shape_complementarity_predictions", lambda **_: None
    )

    def fake_residue_inputs(*, target_device, **_kwargs):
        observed["target_device"] = target_device
        return {
            "pae_logits": torch.zeros(2, 2, 2, 1),
            "pde_logits": torch.zeros(2, 2, 2, 1),
            "contact_probs": torch.zeros(2, 2, 2),
            "token_asym_id": torch.zeros(2, dtype=torch.long),
            "token_has_frame": torch.ones(2, dtype=torch.bool),
            "atom_to_token_idx": torch.arange(2),
        }

    monkeypatch.setattr(
        model, "get_residue_level_confidence_inputs", fake_residue_inputs
    )
    monkeypatch.setattr(
        model, "replace_public_pair_logits_with_residue_level", lambda **_: None
    )
    monkeypatch.setattr(
        "opendde.model.sample_confidence.compute_full_data_and_summary",
        lambda **_: ([], []),
    )

    pred_dict = {
        "coordinate": torch.zeros(2, 2, 3),
        "plddt": torch.zeros(2, 2, 1),
        "pae": torch.empty(2, 2, 2, 1, device="meta"),
        "pde": torch.empty(2, 2, 2, 1, device="meta"),
        "contact_probs": torch.zeros(2, 2),
    }
    model.run_post_confidence_outputs_stage(
        pred_dict=pred_dict,
        input_feature_dict={"is_ligand": torch.zeros(2)},
        pair_input_feature_dict={},
        N_cycle=1,
    )

    assert observed["target_device"] == torch.device("meta")


def test_post_confidence_releases_trunk_tensor_aliases_before_summary():
    model = OpenDDE.__new__(OpenDDE)
    torch.nn.Module.__init__(model)
    model.configs = SimpleNamespace(
        infer_setting=SimpleNamespace(
            dynamic_chunk_size=False,
            sample_diffusion_chunk_size=None,
        ),
        sample_diffusion={"N_sample": 1, "N_step": 1},
        triangle_multiplicative="torch",
        triangle_attention="torch",
    )
    model.enable_structural_token_expansion = False
    source_refs = []

    def get_pairformer_output(**_kwargs):
        sources = (
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.zeros(2, 2, 1),
        )
        source_refs.extend(weakref.ref(source) for source in sources)
        return sources

    model.get_pairformer_output = get_pairformer_output
    model._bound_pairformer_chunk_size = lambda _n, chunk: chunk
    model._foldcp_stage_context = lambda *_args: nullcontext()
    model._maybe_foldcp_mesh = lambda: None
    model._run_foldcp_local_action_synchronized = lambda _mesh, action, **_kwargs: (
        action()
    )
    model.expand_to_structural_tokens = lambda **kwargs: (
        kwargs["input_feature_dict"],
        kwargs["s_inputs"],
        kwargs["s"],
        kwargs["z"],
    )
    model.select_pair_output_branch = lambda **kwargs: (
        kwargs["structural_feature_dict"],
        kwargs["structural_s_inputs"],
        kwargs["structural_s"],
        kwargs["structural_z"],
    )
    model._foldcp_cleanup_before_p2p_warmup = lambda: None
    model.inference_noise_scheduler = lambda **_kwargs: torch.zeros(1)
    model.prepare_diffusion_cache_for_sampling = lambda **_kwargs: {}

    def sample_diffusion_stage(*, pred_dict, **_kwargs):
        pred_dict["coordinate"] = torch.zeros(1, 2, 3)

    def distogram_stage(*, pred_dict, **_kwargs):
        pred_dict["contact_probs"] = torch.zeros(2, 2)

    def confidence_stage(*, pred_dict, **_kwargs):
        pred_dict.update(
            {
                "plddt": torch.zeros(1, 2, 1),
                "pae": torch.zeros(1, 2, 2, 1),
                "pde": torch.zeros(1, 2, 2, 1),
                "resolved": torch.zeros(1, 2, 1),
            }
        )

    model.run_sample_diffusion_stage = sample_diffusion_stage
    model.run_distogram_contact_stage = distogram_stage
    model.run_confidence_head_stage = confidence_stage
    model._foldcp_is_non_output_rank = lambda: False

    def post_confidence_stage(**_kwargs):
        gc.collect()
        assert len(source_refs) == 3
        assert all(reference() is None for reference in source_refs)

    model.run_post_confidence_outputs_stage = post_confidence_stage

    OpenDDE._main_inference_loop(
        model,
        input_feature_dict={"residue_index": torch.zeros(2)},
        N_cycle=1,
        chunk_size=1,
    )


def test_confidence_summary_stages_one_pair_sample_to_compute_device(monkeypatch):
    from opendde.model import sample_confidence

    observed = []

    def fake_compute(**kwargs):
        observed.append(
            {
                key: value.device
                for key, value in kwargs.items()
                if isinstance(value, torch.Tensor)
            }
        )
        return [{}], []

    monkeypatch.setattr(
        sample_confidence, "_compute_full_data_and_summary", fake_compute
    )
    sample_confidence.compute_full_data_and_summary(
        configs=SimpleNamespace(),
        pae_logits=torch.zeros(2, 3, 3, 4),
        plddt_logits=torch.empty(2, 5, 4, device="meta"),
        pde_logits=torch.zeros(2, 3, 3, 4),
        contact_probs=torch.zeros(2, 3, 3),
        token_asym_id=torch.zeros(3, dtype=torch.long),
        token_has_frame=torch.ones(3, dtype=torch.bool),
        atom_coordinate=torch.zeros(2, 5, 3),
        atom_to_token_idx=torch.zeros(5, dtype=torch.long),
        atom_is_polymer=torch.ones(5),
        N_recycle=1,
    )

    assert len(observed) == 2
    assert all(
        all(device == torch.device("meta") for device in devices.values())
        for devices in observed
    )


def test_confidence_summary_offloads_each_completed_sample(monkeypatch):
    from opendde.model import sample_confidence

    offloaded = []

    monkeypatch.setattr(
        sample_confidence,
        "_compute_full_data_and_summary",
        lambda **_: ([{"summary": torch.ones(1)}], [{"pair": torch.ones(1)}]),
    )

    def fake_offload(value):
        offloaded.append(value)
        return value

    monkeypatch.setattr(
        sample_confidence,
        "_offload_confidence_tree_to_cpu",
        fake_offload,
    )
    summary, full_data = sample_confidence.compute_full_data_and_summary(
        configs=SimpleNamespace(),
        pae_logits=torch.zeros(3, 2, 2, 1),
        plddt_logits=torch.zeros(3, 2, 1),
        pde_logits=torch.zeros(3, 2, 2, 1),
        contact_probs=torch.zeros(3, 2, 2),
        token_asym_id=torch.zeros(2, dtype=torch.long),
        token_has_frame=torch.ones(2, dtype=torch.bool),
        atom_coordinate=torch.zeros(3, 2, 3),
        atom_to_token_idx=torch.arange(2),
        atom_is_polymer=torch.ones(2),
        N_recycle=1,
        return_full_data=True,
    )

    assert len(summary) == 3
    assert len(full_data) == 3
    assert len(offloaded) == 6


def test_foldcp_confidence_output_failure_stops_before_next_sample(monkeypatch):
    head = ConfidenceHead.__new__(ConfidenceHead)
    torch.nn.Module.__init__(head)
    head.input_strunk_ln = torch.nn.Identity()
    head.pairformer_stack = SimpleNamespace(blocks=[SimpleNamespace(c_s=1)])
    mesh = SimpleNamespace(group_2d=object())
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
    monkeypatch.setattr(head, "_foldcp_is_non_output_rank", lambda: False)
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

    def fake_foldcp_local(**_kwargs):
        calls.append(len(calls))
        return (
            torch.zeros(4, 2),
            torch.zeros(4, 4, 3),
            torch.zeros(4, 4, 3),
            torch.zeros(4, 2),
        )

    def synchronized(action, *, description, **_kwargs):
        if description == "Fold-CP confidence sample 0 output retention":
            raise RuntimeError("remote confidence output OOM")
        return action()

    monkeypatch.setattr(
        head, "memory_efficient_forward_foldcp_local", fake_foldcp_local
    )
    monkeypatch.setattr(
        "opendde.model.modules.confidence.run_group_rank_action_synchronized",
        synchronized,
    )

    with pytest.raises(RuntimeError, match="remote confidence output OOM"):
        head.forward(
            input_feature_dict={},
            s_inputs=torch.zeros(4, 1),
            s_trunk=torch.zeros(4, 1),
            z_trunk=torch.zeros(2, 2, 1),
            pair_mask=torch.ones(4, 4),
            x_pred_coords=torch.zeros(3, 4, 3),
            z_trunk_spec=spec,
        )

    assert calls == [0]


def test_foldcp_confidence_distance_failure_stops_before_pairformer(monkeypatch):
    from opendde.model.modules import confidence

    pairformer_calls = []
    monkeypatch.setattr(
        confidence,
        "run_group_rank_action_synchronized",
        lambda _action, *, description, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("remote confidence distance OOM")
        ),
    )
    monkeypatch.setattr(
        confidence,
        "distributed_pairformer_stack_single_bridge_update",
        lambda *args, **kwargs: pairformer_calls.append("pairformer"),
    )

    with pytest.raises(RuntimeError, match="remote confidence distance OOM"):
        ConfidenceHead.memory_efficient_forward_foldcp_local(
            SimpleNamespace(),
            input_feature_dict={},
            s_trunk=torch.zeros(1, 1),
            z_pair_local=torch.zeros(1, 1, 1),
            z_pair_spec=SimpleNamespace(),
            foldcp_mesh=SimpleNamespace(group_2d=object()),
            pair_mask=None,
            x_pred_rep_coords=torch.zeros(1, 3),
        )

    assert pairformer_calls == []


def test_foldcp_confidence_releases_caller_pair_owner_before_transpose(monkeypatch):
    from opendde.model.modules import confidence

    released = []
    source_refs = []
    mesh = SimpleNamespace(group_2d=object())
    module = SimpleNamespace(
        lower_bins=torch.zeros(1),
        upper_bins=torch.ones(1),
        linear_no_bias_d=torch.nn.Identity(),
        linear_no_bias_d_wo_onehot=torch.nn.Identity(),
        pairformer_stack=SimpleNamespace(),
        pae_ln=torch.nn.Identity(),
        linear_no_bias_pae=torch.nn.Identity(),
        pde_ln=torch.nn.Identity(),
        linear_no_bias_pde=torch.nn.Identity(),
    )

    def add_distance(*, z_pair_local, **_kwargs):
        source_refs.append(weakref.ref(z_pair_local))
        return z_pair_local

    def pairformer(_stack, s, z, *_args, **kwargs):
        return s, z, kwargs["z_spec"]

    def pair_logits(*, z_pair_local, release_pair_source, **_kwargs):
        source_ref = weakref.ref(z_pair_local)
        release_pair_source()
        z_pair_local = None
        gc.collect()
        released.append(source_ref() is None)
        return None, None

    monkeypatch.setattr(
        confidence,
        "add_confidence_distance_embedding_local",
        add_distance,
    )
    monkeypatch.setattr(
        confidence,
        "distributed_pairformer_stack_single_bridge_update",
        pairformer,
    )
    monkeypatch.setattr(confidence, "distributed_confidence_pair_logits", pair_logits)
    monkeypatch.setattr(
        confidence,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(confidence.dist, "get_rank", lambda _group: 0)

    outputs = ConfidenceHead.memory_efficient_forward_foldcp_local(
        module,
        input_feature_dict={
            "atom_to_token_idx": torch.zeros(1, dtype=torch.long),
            "atom_to_tokatom_idx": torch.zeros(1, dtype=torch.long),
        },
        s_trunk=torch.zeros(1, 1),
        z_pair_local=torch.zeros(1, 1, 1),
        z_pair_spec=SimpleNamespace(),
        foldcp_mesh=mesh,
        pair_mask=None,
        x_pred_rep_coords=torch.zeros(1, 3),
        compute_plddt=False,
        compute_pae=False,
        compute_pde=True,
        compute_resolved=False,
    )

    gc.collect()
    assert outputs == (None, None, None, None)
    assert released == [True]
    assert source_refs[0]() is None


def test_foldcp_confidence_direct_shard_failure_stops_before_local_head(monkeypatch):
    from opendde.model.modules import confidence

    local_head_calls = []
    module = SimpleNamespace(
        _maybe_create_foldcp_mesh=lambda: SimpleNamespace(group_2d=object()),
        pairformer_stack=SimpleNamespace(blocks=[SimpleNamespace(c_s=1)]),
        memory_efficient_forward_foldcp_local=lambda **_kwargs: local_head_calls.append(
            "local"
        ),
    )
    monkeypatch.setattr(
        confidence,
        "run_group_rank_action_synchronized",
        lambda _action, *, description, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("remote direct-shard OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote direct-shard OOM"):
        ConfidenceHead.memory_efficient_forward(
            module,
            input_feature_dict={},
            s_trunk=torch.zeros(1, 1),
            z_pair=torch.zeros(1, 1, 1),
            pair_mask=torch.ones(1, 1),
            x_pred_rep_coords=torch.zeros(1, 3),
        )

    assert local_head_calls == []


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


def test_foldcp_model_seed_non_output_releases_previous_prediction(monkeypatch):
    model = OpenDDE.__new__(OpenDDE)
    torch.nn.Module.__init__(model)
    monkeypatch.setattr(model, "_foldcp_is_non_output_rank", lambda: True)

    previous_tensor = None
    calls = 0

    def fake_main_inference_loop(**_kwargs):
        nonlocal calls, previous_tensor
        calls += 1
        if calls == 2:
            gc.collect()
            assert previous_tensor() is None
        tensor = torch.tensor([[calls]], dtype=torch.float32)
        if calls == 1:
            previous_tensor = weakref.ref(tensor)
        return {"coordinate": tensor}, {"call": calls}, {"elapsed": float(calls)}

    monkeypatch.setattr(model, "_main_inference_loop", fake_main_inference_loop)

    pred_dict, _, _ = model.main_inference_loop(
        input_feature_dict={},
        N_cycle=1,
        N_model_seed=2,
    )

    assert calls == 2
    assert pred_dict["coordinate"].item() == 2


def test_foldcp_model_seed_retention_failure_stops_before_next_seed(monkeypatch):
    model = OpenDDE.__new__(OpenDDE)
    torch.nn.Module.__init__(model)
    monkeypatch.setattr(model, "_foldcp_is_non_output_rank", lambda: False)
    monkeypatch.setattr(model, "_maybe_foldcp_mesh", lambda: object())

    calls = 0

    def fake_main_inference_loop(**_kwargs):
        nonlocal calls
        calls += 1
        tensor = torch.tensor([[calls]], dtype=torch.float32)
        return {"coordinate": tensor}, {}, {}

    def fake_synchronized_action(_mesh, action, *, description):
        if "prediction retention" in description:
            raise RuntimeError("remote prediction retention failed")
        return action()

    monkeypatch.setattr(model, "_main_inference_loop", fake_main_inference_loop)
    monkeypatch.setattr(
        model,
        "_run_foldcp_local_action_synchronized",
        fake_synchronized_action,
    )

    with pytest.raises(RuntimeError, match="remote prediction retention failed"):
        model.main_inference_loop(
            input_feature_dict={},
            N_cycle=1,
            N_model_seed=2,
        )

    assert calls == 1


def test_model_seed_output_tree_is_offloaded_before_merge(monkeypatch):
    model = OpenDDE.__new__(OpenDDE)
    torch.nn.Module.__init__(model)
    monkeypatch.setattr(model, "_foldcp_is_non_output_rank", lambda: False)

    calls = []

    def fake_main_inference_loop(**_kwargs):
        value = len(calls) + 1
        calls.append(value)
        tensor = torch.tensor([[value]], dtype=torch.float32)
        return (
            {
                "coordinate": tensor,
                "summary_confidence": [{"ranking_score": float(value)}],
                "full_data": [{"atom_plddt": tensor}],
                "plddt": tensor,
                "pae": tensor,
                "pde": tensor,
                "resolved": tensor,
            },
            {"call": value},
            {"elapsed": float(value)},
        )

    monkeypatch.setattr(model, "_main_inference_loop", fake_main_inference_loop)
    pred_dict, _, _ = model.main_inference_loop(
        input_feature_dict={},
        N_cycle=1,
        N_model_seed=3,
    )

    assert calls == [1, 2, 3]
    assert pred_dict["coordinate"].flatten().tolist() == [1.0, 2.0, 3.0]
    assert pred_dict["coordinate"].device.type == "cpu"
    assert pred_dict["full_data"][0]["atom_plddt"].device.type == "cpu"


def test_prediction_tree_cpu_offload_preserves_container_shape():
    tensor = torch.tensor([1.0])
    result = _offload_prediction_tree_to_cpu(
        {"list": [tensor], "tuple": (tensor,), "scalar": 7}
    )

    assert result["list"][0].device.type == "cpu"
    assert result["tuple"][0].device.type == "cpu"
    assert result["scalar"] == 7


def test_confidence_stream_output_falls_back_to_cpu_after_cuda_oom(monkeypatch):
    from opendde.distributed.foldcp import confidence

    allocations = []
    cache_clears = []
    original_empty = torch.empty

    def fake_empty(shape, *, dtype, device):
        device = torch.device(device)
        allocations.append(device.type)
        if device.type == "cuda":
            raise torch.OutOfMemoryError("CUDA out of memory")
        return original_empty(shape, dtype=dtype)

    monkeypatch.setattr(confidence.torch, "empty", fake_empty)
    monkeypatch.setattr(
        confidence.torch.cuda,
        "empty_cache",
        lambda: cache_clears.append(True),
    )

    output = confidence._allocate_confidence_stream_output(
        (4, 4, 2),
        dtype=torch.float32,
        source_device=torch.device("cuda", 0),
        output_bytes=4 * 4 * 2 * 4,
        gpu_output_max_bytes=-1,
    )

    assert allocations == ["cuda", "cpu"]
    assert cache_clears == [True]
    assert output.device.type == "cpu"
    assert output.shape == (4, 4, 2)


def test_confidence_stream_honors_explicit_cpu_output_device(monkeypatch):
    from opendde.distributed.foldcp import confidence

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        cp_global_ranks=(0,),
        coord=(0, 0),
        layout=SimpleNamespace(shape=(1, 1), numel=1, to_coord=lambda _rank: (0, 0)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(3, 3, 1),
        padded_shape=(3, 3, 1),
        pair_dims=(0, 1),
        row_range=(0, 3),
        col_range=(0, 3),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    expected = torch.arange(9, dtype=torch.float32).reshape(3, 3, 1)

    monkeypatch.setattr(confidence.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(
        confidence.dist,
        "gather",
        lambda local_chunk, *, gather_list, **_kwargs: gather_list[0].copy_(
            local_chunk
        ),
    )
    monkeypatch.setattr(
        confidence,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action() if action is not None else None,
    )
    monkeypatch.setattr(
        confidence,
        "_confidence_pair_logits_local_rowslab",
        lambda **_kwargs: expected,
    )

    output = confidence._stream_pair_logits_to_rank0(
        z_pair_local=torch.zeros(3, 3, 1),
        z_pair_spec=spec,
        mesh=mesh,
        layer_norm=torch.nn.Identity(),
        linear=SimpleNamespace(weight=torch.zeros(1, 1)),
        output_device=torch.device("cpu"),
    )

    assert output is not None
    assert output.device.type == "cpu"
    assert torch.equal(output, expected)


def test_confidence_stream_drains_gathers_before_raising_assembly_oom(monkeypatch):
    from opendde.distributed.foldcp import confidence

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        cp_global_ranks=(0,),
        coord=(0, 0),
        layout=SimpleNamespace(shape=(1, 1), numel=1, to_coord=lambda _rank: (0, 0)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(260, 260, 1),
        padded_shape=(260, 260, 1),
        pair_dims=(0, 1),
        row_range=(0, 260),
        col_range=(0, 260),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    gathered_chunks = []

    monkeypatch.setattr(confidence.dist, "get_rank", lambda _group: 0)

    def gather(local_chunk, *, gather_list, dst, group):
        gathered_chunks.append(tuple(local_chunk.shape))
        gather_list[0].copy_(local_chunk)

    monkeypatch.setattr(confidence.dist, "gather", gather)
    monkeypatch.setattr(
        confidence,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action() if action is not None else None,
    )
    monkeypatch.setattr(
        confidence,
        "_confidence_pair_logits_local_rowslab",
        lambda **_kwargs: torch.zeros(260, 260, 1),
    )
    monkeypatch.setattr(
        confidence,
        "_copy_pair_shard_into_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("host OOM during confidence assembly")
        ),
    )

    with pytest.raises(RuntimeError, match="host OOM during confidence assembly"):
        confidence._stream_pair_logits_to_rank0(
            z_pair_local=torch.zeros(260, 260, 1),
            z_pair_spec=spec,
            mesh=mesh,
            layer_norm=torch.nn.Identity(),
            linear=SimpleNamespace(weight=torch.zeros(1, 1)),
        )

    # 260 rows at a fixed 128-row transfer size. Even though assembly fails on
    # the first chunk, rank 0 participates in all gathers before synchronizing
    # the error, so peers cannot be stranded in a later collective.
    assert gathered_chunks == [(128, 260, 1)] * 3


def test_confidence_stream_gathers_placeholder_after_local_chunk_failure(monkeypatch):
    from opendde.distributed.foldcp import confidence

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        cp_global_ranks=(0,),
        coord=(0, 0),
        layout=SimpleNamespace(shape=(1, 1), numel=1, to_coord=lambda _rank: (0, 0)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(130, 130, 1),
        padded_shape=(130, 130, 1),
        pair_dims=(0, 1),
        row_range=(0, 130),
        col_range=(0, 130),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    gathered_chunks = []

    monkeypatch.setattr(confidence.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(
        confidence.dist,
        "gather",
        lambda local_chunk, **_kwargs: gathered_chunks.append(tuple(local_chunk.shape)),
    )
    monkeypatch.setattr(
        confidence,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action() if action is not None else None,
    )
    # The projected channel count intentionally disagrees with linear.weight,
    # making the final fixed-size chunk copy fail before the gather.
    monkeypatch.setattr(
        confidence,
        "_confidence_pair_logits_local_rowslab",
        lambda **_kwargs: torch.zeros(130, 130, 2),
    )
    monkeypatch.setattr(confidence, "_copy_pair_shard_into_output", lambda *args: None)

    with pytest.raises(
        RuntimeError, match="confidence local gather-chunk preparation failed"
    ):
        confidence._stream_pair_logits_to_rank0(
            z_pair_local=torch.zeros(130, 130, 1),
            z_pair_spec=spec,
            mesh=mesh,
            layer_norm=torch.nn.Identity(),
            linear=SimpleNamespace(weight=torch.zeros(1, 1)),
        )

    # The malformed first full-size block is staged as well, so its local
    # preparation failure cannot escape immediately before the first gather.
    assert gathered_chunks == [(128, 130, 1), (128, 130, 1)]
