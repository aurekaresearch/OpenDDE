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


@pytest.mark.parametrize("p", [2, 3, 5, 8])
def test_distributed_environment_infers_cp_size_from_initialized_world(
    monkeypatch,
    p,
):
    """Direct model integration must not silently fall back to historical CP=4."""

    from opendde.distributed.foldcp import config as config_module

    monkeypatch.setenv("OPENDDE_FOLDCP_MODE", "distributed")
    monkeypatch.delenv("OPENDDE_FOLDCP_SIZE_DP", raising=False)
    monkeypatch.delenv("OPENDDE_FOLDCP_SIZE_CP", raising=False)
    monkeypatch.setattr(config_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(config_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(config_module.dist, "get_world_size", lambda: p)

    config = FoldCPConfig.from_environment()

    assert config.size_dp == 1
    assert config.size_cp == p
    assert config.cp_mesh_shape == (1, p)


def test_explicit_environment_cp_size_wins_over_world_size(monkeypatch):
    from opendde.distributed.foldcp import config as config_module

    monkeypatch.setenv("OPENDDE_FOLDCP_MODE", "distributed")
    monkeypatch.setenv("OPENDDE_FOLDCP_SIZE_CP", "3")
    monkeypatch.setattr(config_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(config_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(config_module.dist, "get_world_size", lambda: 8)

    assert FoldCPConfig.from_environment().size_cp == 3


@pytest.mark.parametrize("rank,expected", [(0, False), (1, True), (4, True)])
def test_output_rank_uses_inferred_arbitrary_cp_size(monkeypatch, rank, expected):
    from opendde.distributed.foldcp import config as config_module
    from opendde.model import opendde as model_module

    monkeypatch.setenv("OPENDDE_FOLDCP_MODE", "distributed")
    monkeypatch.delenv("OPENDDE_FOLDCP_SIZE_CP", raising=False)
    monkeypatch.setattr(config_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(config_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(config_module.dist, "get_world_size", lambda: 5)
    monkeypatch.setattr(config_module.dist, "get_rank", lambda: rank)

    assert model_module.OpenDDE._foldcp_is_non_output_rank() is expected


@pytest.mark.parametrize("p", range(2, 9))
def test_one_by_p_layout_keeps_transpose_on_the_same_rank(p):
    """Catch treating a 1 x P column shard as a square-mesh transpose."""

    layout = FoldCP2DLayout((1, p))

    assert [layout.transpose_rank(layout.to_coord(rank)) for rank in range(p)] == list(
        range(p)
    )


@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (4, 4)])
def test_layout_rejects_removed_multirow_topologies(shape):
    """Direct library callers cannot bypass the runtime 1 x P boundary."""

    with pytest.raises(ValueError, match="1 x P"):
        FoldCP2DLayout(shape)


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
