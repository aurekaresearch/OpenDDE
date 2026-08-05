from types import SimpleNamespace

import pytest
import torch

from opendde.distributed.foldcp.config import FoldCPConfig
from opendde.distributed.foldcp.layout import FoldCP2DLayout
from opendde.distributed.foldcp.pair_sharding import shard_pair_tensor


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
