# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

_FAST_LAYER_NORM_SYMBOLS = (
    "forward_none_affine",
    "forward_with_bias_affine",
    "forward_with_weight_affine",
    "forward_with_both_affine",
    "backward_none_affine",
    "backward_with_bias_affine",
    "backward_with_weight_affine",
    "backward_with_both_affine",
)


def _fake_fast_layer_norm_extension():
    return SimpleNamespace(
        **{
            symbol: lambda *_args, **_kwargs: None
            for symbol in _FAST_LAYER_NORM_SYMBOLS
        }
    )


@pytest.fixture
def reset_fast_layer_norm_loader():
    from opendde.model.layer_norm import layer_norm

    original_values = (
        layer_norm.fast_layer_norm_cuda_v2,
        layer_norm._fast_layer_norm_load_attempted,
    )
    with layer_norm._fast_layer_norm_load_lock:
        layer_norm.fast_layer_norm_cuda_v2 = None
        layer_norm._fast_layer_norm_load_attempted = False
    try:
        yield layer_norm
    finally:
        with layer_norm._fast_layer_norm_load_lock:
            (
                layer_norm.fast_layer_norm_cuda_v2,
                layer_norm._fast_layer_norm_load_attempted,
            ) = original_values


def test_cpu_triangle_attention_cuequivariance_falls_back_to_torch(monkeypatch):
    from opendde.model.triangular import layers

    monkeypatch.setattr(
        layers,
        "cuequivariance_triangular_attn",
        lambda *args, **kwargs: pytest.fail("CPU tensor reached cuEquivariance"),
    )
    attention = layers.Attention(c_q=4, c_k=4, c_v=4, c_hidden=4, no_heads=1)
    q_x = torch.randn(2, 17, 4)
    biases = [torch.zeros(2, 1, 17, 17), torch.zeros(2, 1, 17, 17)]

    expected = attention(q_x, q_x, biases=biases, triangle_attention="torch")
    output = attention(
        q_x,
        q_x,
        biases=biases,
        triangle_attention="cuequivariance",
    )

    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("c_hidden", [2, 4])
def test_cpu_triangle_multiplication_cuequivariance_falls_back_to_torch(
    monkeypatch,
    c_hidden,
):
    from opendde.model.triangular import triangular

    monkeypatch.setattr(
        triangular,
        "kernel_triangular_mult",
        lambda *args, **kwargs: pytest.fail("CPU tensor reached cuEquivariance"),
    )
    layer = triangular.TriangleMultiplicationOutgoing(c_z=4, c_hidden=c_hidden)
    z = torch.randn(3, 3, 4)

    expected = layer(z, triangle_multiplicative="torch")
    output = layer(z, triangle_multiplicative="cuequivariance")

    torch.testing.assert_close(output, expected)


def test_cpu_triangle_multiplication_rejects_invalid_backend():
    from opendde.model.triangular import triangular

    layer = triangular.TriangleMultiplicationOutgoing(c_z=4, c_hidden=2)

    with pytest.raises(ValueError, match="triangle_multiplicative"):
        layer(torch.randn(3, 3, 4), triangle_multiplicative="invalid")


def test_foldcp_confidence_distance_embedding_matches_serial_slice(monkeypatch):
    from opendde.distributed.foldcp.confidence import (
        add_confidence_distance_embedding_local,
    )
    from opendde.distributed.foldcp.pair_sharding import FoldCPPairShardSpec
    from opendde.model.utils import one_hot

    monkeypatch.setenv("OPENDDE_FOLDCP_CONFIDENCE_DISTANCE_ROW_CHUNK", "1")
    generator = torch.Generator().manual_seed(11)
    n_token, padded_n, c_z, n_bin = 5, 6, 4, 3
    coords = torch.randn(n_token, 3, generator=generator)
    lower_bins = torch.tensor([-1.0, 1.0, 2.0])
    upper_bins = torch.tensor([1.0, 2.0, float("inf")])
    linear_onehot = torch.nn.Linear(n_bin, c_z, bias=False)
    linear_distance = torch.nn.Linear(1, c_z, bias=False)
    z_pair = torch.randn(3, 3, c_z, generator=generator)
    spec = FoldCPPairShardSpec(
        original_shape=(n_token, n_token, c_z),
        padded_shape=(padded_n, padded_n, c_z),
        pair_dims=(0, 1),
        row_range=(3, 6),
        col_range=(3, 6),
        mesh_shape=(2, 2),
        mesh_coord=(1, 1),
    )

    distances = torch.linalg.vector_norm(
        coords[:, None, :] - coords[None, :, :], dim=-1
    )
    full_update = linear_onehot(one_hot(distances, lower_bins, upper_bins))
    full_update = full_update + linear_distance(distances.unsqueeze(-1))
    expected = z_pair.clone()
    expected[:2, :2] += full_update[3:5, 3:5]
    actual = add_confidence_distance_embedding_local(
        z_pair_local=z_pair.clone(),
        z_pair_spec=spec,
        x_pred_rep_coords=coords,
        lower_bins=lower_bins,
        upper_bins=upper_bins,
        linear_onehot=linear_onehot,
        linear_distance=linear_distance,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[2], z_pair[2])
    torch.testing.assert_close(actual[:, 2], z_pair[:, 2])


def test_cpu_cleanup_does_not_touch_visible_cuda(monkeypatch):
    from opendde.utils import torch_utils

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: pytest.fail("CPU cleanup touched the CUDA allocator"),
    )

    torch_utils.cleanup_device_memory("cpu", collect_garbage=False)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an Apple silicon Mac with a usable MPS backend",
)
def test_mps_attention_matches_cpu_for_batched_local_trunks():
    """MPS SDPA folds high ranks with a layout-sensitive view.

    Sampled local attention feeds it rank-5 trunks with a bias that broadcasts
    over samples and heads, which requires explicit materialization.
    """
    from opendde.model.modules.primitives import _attention

    torch.manual_seed(0)
    n_sample, n_trunks, n_heads, n_queries, n_keys, c_hidden = 5, 3, 4, 8, 16, 32
    q = torch.randn(1, n_trunks, n_heads, n_queries, c_hidden).expand(
        n_sample, n_trunks, n_heads, n_queries, c_hidden
    )
    k = torch.randn(n_sample, n_trunks, n_heads, n_keys, c_hidden)
    v = torch.randn_like(k)
    attn_bias = torch.randn(1, n_trunks, 1, n_queries, n_keys)

    expected = _attention(q, k, v, attn_bias)
    actual = _attention(
        q.to("mps"), k.to("mps"), v.to("mps"), attn_bias.to("mps")
    ).cpu()

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    # Rank-4 inputs take the native kernel path, broadcast bias included.
    q4, k4, v4 = q[0], k[0], v[0]
    bias4 = attn_bias[0]
    expected4 = _attention(q4, k4, v4, bias4)
    actual4 = _attention(
        q4.to("mps"), k4.to("mps"), v4.to("mps"), bias4.to("mps")
    ).cpu()
    torch.testing.assert_close(actual4, expected4, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an Apple silicon Mac with a usable MPS backend",
)
def test_mps_attention_broadcasts_rank4_qkv_batch_dimensions():
    """MPS SDPA silently skips Q/K/V batch broadcasting at rank 4."""
    from opendde.model.modules.primitives import _attention

    torch.manual_seed(1)
    q = torch.randn(1, 3, 4, 8)
    k = torch.randn(2, 3, 5, 8)
    v = torch.randn_like(k)
    attn_bias = torch.randn(1, 3, 4, 5)

    expected = _attention(q, k, v, attn_bias)
    actual = _attention(
        q.to("mps"), k.to("mps"), v.to("mps"), attn_bias.to("mps")
    ).cpu()

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an Apple silicon Mac with a usable MPS backend",
)
def test_fp32_guard_clears_mps_autocast():
    """MPS keeps an autocast state of its own, separate from CUDA's."""
    from opendde.utils.torch_utils import (
        autocasting_disable_decorator,
        disabled_autocast,
    )

    a = torch.randn(8, 8, device="mps")
    with torch.autocast("mps", dtype=torch.bfloat16):
        assert (a @ a).dtype == torch.bfloat16
        with disabled_autocast():
            assert (a @ a).dtype == torch.float32
        matmul = autocasting_disable_decorator(True)(lambda x: x @ x)
        assert matmul(a).dtype == torch.float32


def test_mps_cleanup_uses_the_mps_allocator(monkeypatch):
    from opendde.utils import torch_utils

    calls = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: pytest.fail("MPS cleanup touched the CUDA allocator"),
    )
    monkeypatch.setattr(
        torch.mps, "synchronize", lambda: calls.append(("synchronize", None))
    )
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append(("empty", None)))

    torch_utils.cleanup_device_memory("mps", collect_garbage=False)
    torch_utils.cleanup_device_memory("mps", collect_garbage=False, synchronize=True)

    assert calls == [
        ("empty", None),
        ("synchronize", None),
        ("empty", None),
    ]

    def failing_synchronize():
        raise RuntimeError("previous MPS failure")

    monkeypatch.setattr(torch.mps, "synchronize", failing_synchronize)
    torch_utils.cleanup_device_memory(
        "mps",
        collect_garbage=False,
        synchronize=True,
        suppress_errors=True,
    )
    assert calls[-1] == ("empty", None)


def test_cpu_cleanup_does_not_touch_visible_mps(monkeypatch):
    from opendde.utils import torch_utils

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.mps,
        "empty_cache",
        lambda: pytest.fail("CPU cleanup touched the MPS allocator"),
    )

    torch_utils.cleanup_device_memory("cpu", collect_garbage=False)


def test_cuda_cleanup_synchronizes_only_when_requested(monkeypatch):
    from opendde.utils import torch_utils

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *, device: calls.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: calls.append(("empty", None))
    )

    torch_utils.cleanup_device_memory("cuda:0", collect_garbage=False)
    torch_utils.cleanup_device_memory("cuda:0", collect_garbage=False, synchronize=True)

    assert calls == [
        ("empty", None),
        ("synchronize", torch.device("cuda:0")),
        ("empty", None),
    ]


def test_cuda_cleanup_suppresses_errors_during_failure_recovery(monkeypatch):
    from opendde.utils import torch_utils

    def failing_synchronize(*, device):
        raise RuntimeError("previous CUDA failure")

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", failing_synchronize)
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: calls.append(("empty", None))
    )

    torch_utils.cleanup_device_memory(
        "cuda:0",
        collect_garbage=False,
        synchronize=True,
        suppress_errors=True,
    )

    # A failed synchronize must not stop the allocator from releasing blocks.
    assert calls == [("empty", None)]


def test_cuda_cleanup_propagates_synchronize_error_after_releasing_cache(monkeypatch):
    from opendde.utils import torch_utils

    error = RuntimeError("asynchronous CUDA failure")
    calls = []

    def failing_synchronize(*, device):
        assert device == torch.device("cuda:0")
        raise error

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", failing_synchronize)
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: calls.append(("empty", None))
    )

    with pytest.raises(RuntimeError) as exc_info:
        torch_utils.cleanup_device_memory(
            "cuda:0", collect_garbage=False, synchronize=True
        )

    assert exc_info.value is error
    assert calls == [("empty", None)]


def test_cuda_cleanup_suppresses_failing_empty_cache_during_recovery(monkeypatch):
    from opendde.utils import torch_utils

    def failing_empty_cache():
        raise RuntimeError("previous CUDA failure")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", failing_empty_cache)

    # A sticky CUDA context re-reports the original error from every CUDA call,
    # so the non-synchronizing cleanup must swallow it too.
    torch_utils.cleanup_device_memory(
        "cuda:0", collect_garbage=False, suppress_errors=True
    )


def test_fused_layer_norm_uses_torch_fallback_for_cpu(monkeypatch):
    from opendde.model.layer_norm import layer_norm

    monkeypatch.setattr(
        layer_norm,
        "_load_fast_layer_norm_cuda_v2",
        lambda: pytest.fail("CPU LayerNorm loaded a CUDA extension"),
    )
    module = layer_norm.FusedLayerNorm(4)
    x = torch.randn(2, 4)

    torch.testing.assert_close(
        module(x),
        torch.nn.functional.layer_norm(
            x,
            module.normalized_shape,
            module.weight,
            module.bias,
            module.eps,
        ),
    )


def test_fast_layer_norm_prebuilt_extension_is_cached(
    monkeypatch,
    reset_fast_layer_norm_loader,
):
    layer_norm = reset_fast_layer_norm_loader
    extension = _fake_fast_layer_norm_extension()
    imports = []

    def fake_import(name, package=None):
        imports.append((name, package))
        return extension

    monkeypatch.setattr(layer_norm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(layer_norm.importlib, "import_module", fake_import)

    assert layer_norm._load_fast_layer_norm_cuda_v2() is extension
    assert layer_norm._load_fast_layer_norm_cuda_v2() is extension
    assert imports == [(".fast_layer_norm_cuda_v2", layer_norm.__package__)]


def test_fast_layer_norm_cpu_does_not_mark_load_attempted(
    monkeypatch,
    reset_fast_layer_norm_loader,
):
    layer_norm = reset_fast_layer_norm_loader
    monkeypatch.setattr(layer_norm.torch.cuda, "is_available", lambda: False)

    assert layer_norm._load_fast_layer_norm_cuda_v2() is None
    assert not layer_norm._fast_layer_norm_load_attempted


def test_fast_layer_norm_failure_is_cached_and_warned_once(
    caplog,
    monkeypatch,
    reset_fast_layer_norm_loader,
):
    layer_norm = reset_fast_layer_norm_loader
    compile_error = RuntimeError("compilation failed")
    compile_calls = []

    def fake_compile(**kwargs):
        compile_calls.append(kwargs)
        raise compile_error

    compile_module = ModuleType("opendde.model.layer_norm.torch_ext_compile")
    setattr(compile_module, "compile", fake_compile)
    monkeypatch.setitem(
        sys.modules,
        "opendde.model.layer_norm.torch_ext_compile",
        compile_module,
    )
    monkeypatch.setattr(layer_norm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        layer_norm.importlib,
        "import_module",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    caplog.set_level(logging.WARNING, logger=layer_norm.__name__)

    assert layer_norm._load_fast_layer_norm_cuda_v2() is None
    assert layer_norm._load_fast_layer_norm_cuda_v2() is None

    assert len(compile_calls) == 1
    assert layer_norm._fast_layer_norm_load_attempted
    assert layer_norm.fast_layer_norm_cuda_v2 is None
    assert (
        sum(
            "Fast LayerNorm CUDA extension is unavailable" in record.message
            for record in caplog.records
        )
        == 1
    )


def test_fused_layer_norm_backward_returns_one_gradient_per_input(monkeypatch):
    from opendde.model.layer_norm import layer_norm

    extension = SimpleNamespace(
        forward_with_both_affine=lambda input_, shape, weight, bias, eps: (
            input_.clone(),
            input_.new_zeros(input_.shape[:-1]),
            input_.new_ones(input_.shape[:-1]),
        ),
        backward_with_both_affine=lambda grad, mean, invvar, input_, shape, weight, bias, eps: (
            grad,
            torch.zeros_like(weight),
            torch.zeros_like(bias),
        ),
    )
    monkeypatch.setattr(layer_norm, "fast_layer_norm_cuda_v2", extension)
    input_ = torch.randn(2, 4, requires_grad=True)
    weight = torch.ones(4, requires_grad=True)
    bias = torch.zeros(4, requires_grad=True)

    output = layer_norm.FusedLayerNormAffineFunction.apply(
        input_, weight, bias, torch.Size([4]), 1e-5
    )
    output.sum().backward()

    torch.testing.assert_close(input_.grad, torch.ones_like(input_))
