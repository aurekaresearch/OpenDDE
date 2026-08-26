from types import SimpleNamespace

import pytest
import torch

from opendde.distributed.foldcp.config import FoldCPConfig
from opendde.distributed.foldcp.layout import FoldCP2DLayout
from opendde.distributed.foldcp.pair_sharding import shard_pair_tensor


@pytest.mark.parametrize(
    ("mode", "size_dp", "size_cp"),
    [
        ("single", 1, 1),
        ("single", 2, 1),
        ("single", 3, 1),
        ("single", 4, 1),
        ("distributed", 1, 2),
        ("distributed", 1, 3),
        ("distributed", 1, 4),
    ],
)
def test_supported_inference_topologies(mode, size_dp, size_cp):
    config = FoldCPConfig.from_runtime_args(
        mode=mode,
        size_dp=size_dp,
        size_cp=size_cp,
    )

    assert (config.size_dp, config.size_cp) == (size_dp, size_cp)


def test_seed_parallel_launch_hint_uses_torchrun():
    config = FoldCPConfig.from_runtime_args(mode="single", size_dp=4, size_cp=1)

    assert config.launch_hint() == (
        "torchrun --standalone --nproc_per_node 4 "
        "-m runner.batch_inference pred --foldcp_mode single "
        "--foldcp_size_dp 4 --foldcp_size_cp 1"
    )
    assert config.enabled is False


@pytest.mark.parametrize(
    ("mode", "size_dp", "size_cp", "message"),
    [
        ("single", 2, 2, "Hybrid seed-parallel and Fold-CP"),
        ("distributed", 2, 2, "Hybrid seed-parallel and Fold-CP"),
        ("single", 1, 2, "foldcp_mode='single' requires"),
        ("distributed", 2, 1, "foldcp_mode='distributed' requires foldcp_size_dp=1"),
        ("distributed", 1, 1, "foldcp_mode='distributed' requires foldcp_size_cp > 1"),
    ],
)
def test_rejected_inference_topologies(mode, size_dp, size_cp, message):
    with pytest.raises(ValueError, match=message):
        FoldCPConfig.from_runtime_args(
            mode=mode,
            size_dp=size_dp,
            size_cp=size_cp,
        )


@pytest.mark.parametrize("p", range(2, 9))
def test_distributed_config_uses_one_by_p_for_any_cp_size(monkeypatch, p):
    """Catch reintroducing the perfect-square-only CP launch restriction."""

    monkeypatch.setenv("WORLD_SIZE", str(p))

    config = FoldCPConfig.from_runtime_args(
        mode="distributed",
        size_dp=1,
        size_cp=p,
    )

    assert config.cp_mesh_shape == (1, p)


@pytest.mark.parametrize("p", range(2, 9))
def test_one_by_p_layout_keeps_transpose_on_the_same_rank(p):
    """Catch treating a 1 x P column shard as a square-mesh transpose."""

    layout = FoldCP2DLayout((1, p))

    assert [layout.transpose_rank(layout.to_coord(rank)) for rank in range(p)] == list(
        range(p)
    )


def test_square_layout_remains_compatible_with_public_cp4():
    """Catch breaking the original public 2 x 2 layout contract."""

    layout = FoldCP2DLayout((2, 2))

    assert layout.transpose_rank((0, 1)) == layout.to_linear((1, 0))


@pytest.mark.parametrize("p", range(1, 9))
def test_foldcp_does_not_limit_the_cuda_allocator_by_default(p):
    from runner.inference import _default_foldcp_cuda_memory_fraction

    assert _default_foldcp_cuda_memory_fraction(p) == 0.0


@pytest.mark.parametrize("n,p", [(7, 2), (7, 3), (10, 5), (17, 8)])
def test_one_by_p_pair_sharding_splits_only_the_column_axis(n, p):
    """Each rank keeps all rows and receives one padded column partition."""

    full = torch.arange(n * n).reshape(n, n)
    layout = FoldCP2DLayout((1, p))
    local_width = (n + p - 1) // p

    for rank in range(p):
        mesh = SimpleNamespace(layout=layout, coord=layout.to_coord(rank))
        local, spec = shard_pair_tensor(full, mesh, pair_dims=(0, 1), pad_value=-1)

        assert local.shape == (n, local_width)
        assert spec.row_range == (0, n)
        assert spec.col_range == (rank * local_width, (rank + 1) * local_width)
        valid_width = max(0, min(local_width, n - rank * local_width))
        if valid_width:
            assert torch.equal(
                local[:, :valid_width],
                full[:, rank * local_width : rank * local_width + valid_width],
            )
        if valid_width < local_width:
            assert torch.equal(
                local[:, valid_width:],
                torch.full_like(local[:, valid_width:], -1),
            )
