# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import os
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from opendde.distributed.foldcp.config import FoldCPConfig
from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.launch import (
    foldcp_linear_with_source_launch_shape,
    foldcp_pair_row_slab_linear_with_source_launch_policy,
)
from opendde.distributed.foldcp.pair_sharding import (
    FoldCPPairShardSpec,
    gather_pair_tensor,
    shard_pair_tensor,
)
from opendde.model.modules.embedders import (
    FourierEmbedding,
    LazyRelativePositionEncodingFeatures,
    RelativePositionEncoding,
)
from opendde.model.modules.primitives import LinearNoBias, Transition
from opendde.model.modules.transformer import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionTransformer,
)
from opendde.model.triangular.layers import LayerNorm
from opendde.model.utils import expand_at_dim, get_checkpoint_fn, permute_final_dims


def _foldcp_diffusion_cache_trunk_row_chunk_size(valid_rows: int) -> int:
    value = os.environ.get("OPENDDE_FOLDCP_DIFFUSION_CACHE_TRUNK_ROW_CHUNK")
    row_chunk_size = int("128" if value is None else value)
    if row_chunk_size <= 0:
        return int(valid_rows)
    return min(int(valid_rows), row_chunk_size)


def _foldcp_diffusion_cache_relp_row_chunk_size(valid_rows: int) -> int:
    value = os.environ.get("OPENDDE_FOLDCP_DIFFUSION_CACHE_RELP_ROW_CHUNK")
    row_chunk_size = int("128" if value is None else value)
    if row_chunk_size <= 0:
        return int(valid_rows)
    return min(int(valid_rows), row_chunk_size)


def _foldcp_diffusion_cache_relp_slab_max_bytes() -> int:
    value = os.environ.get("OPENDDE_FOLDCP_DIFFUSION_CACHE_RELP_SLAB_MAX_BYTES")
    if value is None:
        return 3 * 1024**3
    return int(value)


def _foldcp_diffusion_cache_relp_slab_fits(
    *,
    valid_rows: int,
    n_token: int,
    feature_dim: int,
) -> bool:
    max_bytes = _foldcp_diffusion_cache_relp_slab_max_bytes()
    if max_bytes <= 0:
        return False
    # Lazy relative-position features are assembled from int64 one-hot blocks.
    materialize_bytes = int(valid_rows) * int(n_token) * int(feature_dim) * 8
    return materialize_bytes <= max_bytes


def _foldcp_diffusion_cache_pair_z_row_chunk_size(valid_rows: int) -> int:
    value = os.environ.get("OPENDDE_FOLDCP_DIFFUSION_CACHE_PAIR_Z_ROW_CHUNK")
    row_chunk_size = int("256" if value is None else value)
    if row_chunk_size <= 0:
        return int(valid_rows)
    return min(int(valid_rows), row_chunk_size)


def _foldcp_diffusion_cache_pair_z_projection_max_bytes() -> int:
    value = os.environ.get("OPENDDE_FOLDCP_DIFFUSION_CACHE_PAIR_Z_PROJECTION_MAX_BYTES")
    if value is None:
        return 8 * 1024**3
    return int(value)


def _foldcp_diffusion_cache_pair_z_projection_fits(
    *,
    valid_rows: int,
    n_token: int,
    feature_dim: int,
) -> bool:
    max_bytes = _foldcp_diffusion_cache_pair_z_projection_max_bytes()
    if max_bytes <= 0:
        return False
    layernorm_workspace_bytes = int(valid_rows) * int(n_token) * int(feature_dim) * 4 * 3
    return layernorm_workspace_bytes <= max_bytes


def _foldcp_diffusion_cache_source_grid_launch_max_bytes() -> int:
    value = os.environ.get("OPENDDE_FOLDCP_DIFFUSION_CACHE_SOURCE_GRID_MAX_BYTES")
    if value is None:
        return 24 * 1024**3
    return int(value)


def _foldcp_diffusion_cache_source_grid_launch_fits(
    *,
    source_rows: int,
    in_features: int,
    element_size: int,
) -> bool:
    max_bytes = _foldcp_diffusion_cache_source_grid_launch_max_bytes()
    if max_bytes <= 0:
        return False
    launch_bytes = int(source_rows) * int(in_features) * int(element_size)
    return launch_bytes <= max_bytes


def _foldcp_diffusion_cache_source_grid_max_rows() -> int:
    value = os.environ.get("OPENDDE_FOLDCP_DIFFUSION_SOURCE_GRID_MAX_ROWS")
    if value is None:
        return 2_250_000
    return int(value)


class DiffusionConditioning(nn.Module):
    """
    Implements Algorithm 21 in AF3

    Args:
        sigma_data (float, optional): the standard deviation of the data. Defaults to 16.0.
        c_z (int, optional): hidden dim [for trunk pair embedding]. Defaults to 128.
        c_z_pair_diffusion (int, optional): hidden dim [for diffusion pair embedding].
            Defaults to c_z.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_s_inputs (int, optional): input embedding dim from InputEmbedder. Defaults to 449.
        c_noise_embedding (int, optional): noise embedding dim. Defaults to 256.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_z: int = 128,
        c_z_pair_diffusion: Optional[int] = None,
        c_s: int = 384,
        c_s_inputs: int = 449,
        c_noise_embedding: int = 256,
    ) -> None:
        super(DiffusionConditioning, self).__init__()
        self.sigma_data = sigma_data
        self.c_z = c_z
        self.c_z_pair_diffusion = (
            c_z if c_z_pair_diffusion is None else c_z_pair_diffusion
        )
        self.compress_pair_z = self.c_z_pair_diffusion != self.c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs
        # Line1-Line3:
        self.relpe = RelativePositionEncoding(c_z=self.c_z_pair_diffusion)
        if self.compress_pair_z:
            self.layernorm_z_trunk = LayerNorm(self.c_z, create_offset=False)
            self.linear_no_bias_z_trunk = LinearNoBias(
                in_features=self.c_z,
                out_features=self.c_z_pair_diffusion,
                precision=torch.float32,
            )
        self.layernorm_z = LayerNorm(2 * self.c_z_pair_diffusion, create_offset=False)
        self.linear_no_bias_z = LinearNoBias(
            in_features=2 * self.c_z_pair_diffusion,
            out_features=self.c_z_pair_diffusion,
            precision=torch.float32,
        )
        # Line3-Line5:
        self.transition_z1 = Transition(c_in=self.c_z_pair_diffusion, n=2)
        self.transition_z2 = Transition(c_in=self.c_z_pair_diffusion, n=2)

        # Line6-Line7
        self.layernorm_s = LayerNorm(self.c_s + self.c_s_inputs, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s + self.c_s_inputs,
            out_features=self.c_s,
            precision=torch.float32,
        )
        # Line8-Line9
        self.fourier_embedding = FourierEmbedding(c=c_noise_embedding)
        self.layernorm_n = LayerNorm(c_noise_embedding, create_offset=False)
        self.linear_no_bias_n = LinearNoBias(
            in_features=c_noise_embedding,
            out_features=self.c_s,
            precision=torch.float32,
        )
        # Line10-Line12
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)

    def _project_z_trunk(self, z_trunk: torch.Tensor) -> torch.Tensor:
        z_norm = self.layernorm_z_trunk(z_trunk)
        return self.linear_no_bias_z_trunk(z_norm)

    def _project_pair_z(self, pair_z: torch.Tensor) -> torch.Tensor:
        pair_z = self.layernorm_z(pair_z)
        return self.linear_no_bias_z(pair_z)

    @staticmethod
    def _collect_pair_row_slab(
        z_pair_local: torch.Tensor,
        mesh: FoldCPProcessMesh,
        n_token: int,
    ) -> torch.Tensor:
        side = mesh.layout.shape[1]
        if side == 1:
            return z_pair_local[..., :n_token, :].contiguous()
        ring = mesh.ring_comm()
        local_cols = z_pair_local.shape[-2]
        out = z_pair_local.new_empty(
            *z_pair_local.shape[:-2],
            int(n_token),
            z_pair_local.shape[-1],
        )

        def copy_source_tile(tile: torch.Tensor, source_col: int) -> None:
            target_col_start = int(source_col) * local_cols
            target_col_end = min(target_col_start + local_cols, int(n_token))
            if target_col_start >= target_col_end:
                return
            source_width = target_col_end - target_col_start
            out[..., target_col_start:target_col_end, :] = tile[
                ...,
                :source_width,
                :,
            ]

        ready = z_pair_local.contiguous()
        copy_source_tile(ready, mesh.coord[1])
        for step in range(1, side):
            ready = ring.comm_row.exchange(ready.contiguous())
            source_col = (mesh.coord[1] + step) % side
            copy_source_tile(ready, source_col)
        return out.contiguous()

    @staticmethod
    def _linear_pair_row_slab_source_grid_launch(
        linear: nn.Module,
        x: torch.Tensor,
        *,
        original_n: int,
        row_start: int,
        valid_rows: int,
    ) -> torch.Tensor:
        if valid_rows <= 0:
            return linear(x)
        flat = x.contiguous().reshape(-1, x.shape[-1])
        source_rows = int(original_n) * int(original_n)
        launch = flat.new_zeros(source_rows, flat.shape[-1])
        row_offsets = (
            (torch.arange(valid_rows, device=x.device) + int(row_start))
            * int(original_n)
        )
        source_index = (
            row_offsets[:, None]
            + torch.arange(int(original_n), device=x.device)[None, :]
        ).reshape(-1)
        launch.index_copy_(0, source_index, flat[: source_index.numel()])
        projected = linear(launch).index_select(0, source_index)
        return projected.reshape(*x.shape[:-3], valid_rows, int(original_n), -1)

    @staticmethod
    def _project_pair_tile_bounded_source_launch(
        linear: nn.Module,
        x: torch.Tensor,
        *,
        source_rows: int,
        original_n: int,
        row_start: int,
        valid_rows: int,
        col_start: int = 0,
        valid_cols: int | None = None,
    ) -> torch.Tensor:
        if valid_cols is None:
            valid_cols = int(original_n)
        max_source_grid_rows = _foldcp_diffusion_cache_source_grid_max_rows()
        if max_source_grid_rows < 0 or int(source_rows) <= max_source_grid_rows:
            return foldcp_pair_row_slab_linear_with_source_launch_policy(
                linear,
                x,
                original_n=original_n,
                row_start=row_start,
                col_start=col_start,
                valid_rows=valid_rows,
                valid_cols=valid_cols,
            )
        return foldcp_linear_with_source_launch_shape(
            linear,
            x,
            source_rows=source_rows,
        )

    def _project_z_trunk_source_launch(
        self,
        z_trunk: torch.Tensor,
        *,
        source_rows: int,
        original_n: int,
        row_start: int,
        valid_rows: int,
        col_start: int = 0,
        valid_cols: int | None = None,
    ) -> torch.Tensor:
        z_norm = self.layernorm_z_trunk(z_trunk)
        if z_norm.ndim == 3:
            return self._project_pair_tile_bounded_source_launch(
                self.linear_no_bias_z_trunk,
                z_norm,
                source_rows=source_rows,
                original_n=original_n,
                row_start=row_start,
                valid_rows=valid_rows,
                col_start=col_start,
                valid_cols=valid_cols,
            )
        if original_n <= 2048 and valid_rows > 0:
            return self._linear_pair_row_slab_source_grid_launch(
                self.linear_no_bias_z_trunk,
                z_norm[..., :valid_rows, :original_n, :],
                original_n=original_n,
                row_start=row_start,
                valid_rows=valid_rows,
            )
        return foldcp_pair_row_slab_linear_with_source_launch_policy(
            self.linear_no_bias_z_trunk,
            z_norm,
            original_n=original_n,
            row_start=row_start,
            valid_rows=valid_rows,
            valid_cols=original_n,
        )

    def _project_z_trunk_row_slab_from_local_chunks(
        self,
        z_trunk_local: torch.Tensor,
        mesh: FoldCPProcessMesh,
        *,
        n_token: int,
        row_start: int,
        valid_rows: int,
    ) -> torch.Tensor:
        out = z_trunk_local.new_empty(
            *z_trunk_local.shape[:-3],
            int(valid_rows),
            int(n_token),
            self.c_z_pair_diffusion,
        )
        if valid_rows <= 0:
            return out.contiguous()

        row_chunk_size = _foldcp_diffusion_cache_trunk_row_chunk_size(valid_rows)
        for chunk_row_start in range(0, valid_rows, row_chunk_size):
            chunk_row_end = min(chunk_row_start + row_chunk_size, valid_rows)
            chunk_row_slice = slice(chunk_row_start, chunk_row_end)
            chunk_valid_rows = chunk_row_end - chunk_row_start
            chunk_global_row_start = row_start + chunk_row_start
            z_trunk_row_slab = self._collect_pair_row_slab(
                z_trunk_local[..., chunk_row_slice, :, :],
                mesh,
                n_token,
            )
            z_norm = self.layernorm_z_trunk(z_trunk_row_slab)
            z_norm = z_norm[..., :chunk_valid_rows, : int(n_token), :]
            if _foldcp_diffusion_cache_source_grid_launch_fits(
                source_rows=int(n_token) * int(n_token),
                in_features=z_norm.shape[-1],
                element_size=z_norm.element_size(),
            ):
                projected = self._linear_pair_row_slab_source_grid_launch(
                    self.linear_no_bias_z_trunk,
                    z_norm,
                    original_n=n_token,
                    row_start=chunk_global_row_start,
                    valid_rows=chunk_valid_rows,
                )
            else:
                projected = foldcp_linear_with_source_launch_shape(
                    self.linear_no_bias_z_trunk,
                    z_norm,
                    source_rows=int(n_token) * int(n_token),
                )
            out[..., chunk_row_slice, :, :] = projected[
                ...,
                :chunk_valid_rows,
                : int(n_token),
                :,
            ]
            del z_trunk_row_slab, z_norm, projected
        return out.contiguous()

    def _project_pair_z_source_launch(
        self,
        pair_z: torch.Tensor,
        *,
        source_rows: int,
        original_n: int,
        row_start: int,
        valid_rows: int,
        col_start: int = 0,
        valid_cols: int | None = None,
    ) -> torch.Tensor:
        pair_z = self.layernorm_z(pair_z)
        if pair_z.ndim == 3:
            return self._project_pair_tile_bounded_source_launch(
                self.linear_no_bias_z,
                pair_z,
                source_rows=source_rows,
                original_n=original_n,
                row_start=row_start,
                valid_rows=valid_rows,
                col_start=col_start,
                valid_cols=valid_cols,
            )
        if original_n <= 2048 and valid_rows > 0:
            return self._linear_pair_row_slab_source_grid_launch(
                self.linear_no_bias_z,
                pair_z[..., :valid_rows, :original_n, :],
                original_n=original_n,
                row_start=row_start,
                valid_rows=valid_rows,
            )
        return foldcp_pair_row_slab_linear_with_source_launch_policy(
            self.linear_no_bias_z,
            pair_z,
            original_n=original_n,
            row_start=row_start,
            valid_rows=valid_rows,
            valid_cols=original_n,
        )

    def _project_pair_z_row_slab_chunks(
        self,
        z_pair_trunk: torch.Tensor,
        relp_pair_slab: torch.Tensor,
        *,
        source_rows: int,
        original_n: int,
        row_start: int,
        valid_rows: int,
    ) -> torch.Tensor:
        out = z_pair_trunk.new_empty(
            *z_pair_trunk.shape[:-1],
            self.c_z_pair_diffusion,
        )
        if valid_rows <= 0:
            return out.contiguous()

        row_chunk_size = _foldcp_diffusion_cache_pair_z_row_chunk_size(valid_rows)
        for chunk_row_start in range(0, valid_rows, row_chunk_size):
            chunk_row_end = min(chunk_row_start + row_chunk_size, valid_rows)
            chunk_row_slice = slice(chunk_row_start, chunk_row_end)
            chunk_valid_rows = chunk_row_end - chunk_row_start
            chunk_global_row_start = row_start + chunk_row_start
            pair_z_chunk = torch.cat(
                tensors=[
                    z_pair_trunk[..., chunk_row_slice, :, :],
                    relp_pair_slab[..., chunk_row_slice, :, :],
                ],
                dim=-1,
            )
            pair_z_norm = self.layernorm_z(pair_z_chunk)
            pair_z_norm = pair_z_norm[..., :chunk_valid_rows, : int(original_n), :]
            if _foldcp_diffusion_cache_source_grid_launch_fits(
                source_rows=source_rows,
                in_features=pair_z_norm.shape[-1],
                element_size=pair_z_norm.element_size(),
            ):
                projected = self._linear_pair_row_slab_source_grid_launch(
                    self.linear_no_bias_z,
                    pair_z_norm,
                    original_n=original_n,
                    row_start=chunk_global_row_start,
                    valid_rows=chunk_valid_rows,
                )
            else:
                projected = foldcp_linear_with_source_launch_shape(
                    self.linear_no_bias_z,
                    pair_z_norm,
                    source_rows=source_rows,
                )
            out[..., chunk_row_slice, :, :] = projected[
                ...,
                :chunk_valid_rows,
                : int(original_n),
                :,
            ]
            del pair_z_chunk, pair_z_norm, projected
        return out.contiguous()

    def _project_pair_z_from_component_row_slab(
        self,
        z_pair_trunk: torch.Tensor,
        relp_pair_slab: torch.Tensor,
        *,
        source_rows: int,
        original_n: int,
        row_start: int,
        valid_rows: int,
    ) -> torch.Tensor:
        pair_z_feature_dim = z_pair_trunk.shape[-1] + relp_pair_slab.shape[-1]
        if _foldcp_diffusion_cache_pair_z_projection_fits(
            valid_rows=valid_rows,
            n_token=original_n,
            feature_dim=pair_z_feature_dim,
        ):
            pair_z_row_slab = torch.cat(
                tensors=[
                    z_pair_trunk,
                    relp_pair_slab,
                ],
                dim=-1,
            )
            return self._project_pair_z_source_launch(
                pair_z_row_slab,
                source_rows=source_rows,
                original_n=original_n,
                row_start=row_start,
                valid_rows=valid_rows,
            )
        return self._project_pair_z_row_slab_chunks(
            z_pair_trunk,
            relp_pair_slab,
            source_rows=source_rows,
            original_n=original_n,
            row_start=row_start,
            valid_rows=valid_rows,
        )

    @staticmethod
    def _row_slab_relp_from_spec(
        relp_feature: Union[torch.Tensor, LazyRelativePositionEncodingFeatures],
        spec: FoldCPPairShardSpec,
        reference: torch.Tensor,
        feature_dim: int,
        n_token: int,
    ) -> torch.Tensor:
        row_start, row_end = spec.row_range
        valid_row_end = min(row_end, n_token)
        valid_rows = max(0, valid_row_end - row_start)
        local = reference.new_zeros(
            *reference.shape[:-3],
            reference.shape[-3],
            n_token,
            feature_dim,
        )
        if valid_rows == 0 or n_token == 0:
            return local
        if isinstance(relp_feature, LazyRelativePositionEncodingFeatures):
            relp_valid = relp_feature.materialize(
                row_slice=slice(row_start, valid_row_end),
                col_slice=slice(0, n_token),
            ).to(device=reference.device, dtype=reference.dtype)
        else:
            relp_valid = relp_feature[
                ...,
                row_start:valid_row_end,
                :n_token,
                :,
            ].to(device=reference.device, dtype=reference.dtype)
        local[..., :valid_rows, :, :] = relp_valid
        return local.contiguous()

    def _project_relp_row_slab_source_launch(
        self,
        relp_row_slab: torch.Tensor,
        *,
        original_n: int,
        row_start: int,
        valid_rows: int,
    ) -> torch.Tensor:
        relp_row_slab = relp_row_slab[..., :valid_rows, : int(original_n), :]
        if valid_rows <= 0:
            return self.relpe.linear_no_bias(relp_row_slab)
        if _foldcp_diffusion_cache_source_grid_launch_fits(
            source_rows=int(original_n) * int(original_n),
            in_features=relp_row_slab.shape[-1],
            element_size=relp_row_slab.element_size(),
        ):
            return self._linear_pair_row_slab_source_grid_launch(
                self.relpe.linear_no_bias,
                relp_row_slab,
                original_n=original_n,
                row_start=row_start,
                valid_rows=valid_rows,
            )
        return foldcp_linear_with_source_launch_shape(
            self.relpe.linear_no_bias,
            relp_row_slab,
            source_rows=int(original_n) * int(original_n),
        )

    def _relpe_row_slab_from_spec_chunks(
        self,
        relp_feature: Union[torch.Tensor, LazyRelativePositionEncodingFeatures],
        spec: FoldCPPairShardSpec,
        reference: torch.Tensor,
        *,
        n_token: int,
        valid_rows: int,
    ) -> torch.Tensor:
        row_start, row_end = spec.row_range
        out = reference.new_empty(
            *reference.shape[:-1],
            self.c_z_pair_diffusion,
        )
        if valid_rows <= 0:
            return out.contiguous()
        row_chunk_size = _foldcp_diffusion_cache_relp_row_chunk_size(valid_rows)
        for chunk_row_start in range(0, valid_rows, row_chunk_size):
            chunk_row_end = min(chunk_row_start + row_chunk_size, valid_rows)
            chunk_row_slice = slice(chunk_row_start, chunk_row_end)
            chunk_global_row_start = row_start + chunk_row_start
            chunk_global_row_end = min(row_start + chunk_row_end, row_end)
            chunk_valid_rows = chunk_row_end - chunk_row_start
            chunk_spec = FoldCPPairShardSpec(
                original_shape=spec.original_shape,
                padded_shape=spec.padded_shape,
                pair_dims=spec.pair_dims,
                row_range=(chunk_global_row_start, chunk_global_row_end),
                col_range=spec.col_range,
                mesh_shape=spec.mesh_shape,
                mesh_coord=spec.mesh_coord,
            )
            relp_chunk = self._row_slab_relp_from_spec(
                relp_feature=relp_feature,
                spec=chunk_spec,
                reference=reference[..., chunk_row_slice, :, :],
                feature_dim=self.relpe.linear_no_bias.in_features,
                n_token=n_token,
            )
            relp_projected = self._project_relp_row_slab_source_launch(
                relp_chunk,
                original_n=n_token,
                row_start=chunk_global_row_start,
                valid_rows=chunk_valid_rows,
            )
            out[..., chunk_row_slice, :, :] = relp_projected[
                ...,
                :chunk_valid_rows,
                : int(n_token),
                :,
            ]
            del relp_chunk, relp_projected
        return out.contiguous()

    def _project_relp_tile_source_launch(
        self,
        relp_tile: torch.Tensor,
        *,
        source_rows: int,
        original_n: int,
        row_start: int,
        valid_rows: int,
        col_start: int,
        valid_cols: int,
    ) -> torch.Tensor:
        return self._project_pair_tile_bounded_source_launch(
            self.relpe.linear_no_bias,
            relp_tile,
            source_rows=source_rows,
            original_n=original_n,
            row_start=row_start,
            valid_rows=valid_rows,
            col_start=col_start,
            valid_cols=valid_cols,
        )

    @staticmethod
    def _apply_transition_source_flat_chunks(
        flat: torch.Tensor,
        transition: Transition,
        *,
        global_flat_start: int,
        source_rows: int,
        flat_chunk_size: int = 262144,
    ) -> None:
        local_rows = flat.shape[0]
        global_flat_end = global_flat_start + local_rows
        for chunk_start in range(0, source_rows, flat_chunk_size):
            chunk_end = min(chunk_start + flat_chunk_size, source_rows)
            overlap_start = max(global_flat_start, chunk_start)
            overlap_end = min(global_flat_end, chunk_end)
            if overlap_start >= overlap_end:
                continue
            local_start = overlap_start - global_flat_start
            local_end = overlap_end - global_flat_start
            offset = overlap_start - chunk_start
            chunk_len = chunk_end - chunk_start
            launch = flat.new_zeros(chunk_len, flat.shape[-1])
            launch[offset : offset + (local_end - local_start)].copy_(
                flat[local_start:local_end]
            )
            update = transition(launch)
            flat[local_start:local_end] += update[
                offset : offset + (local_end - local_start)
            ]
            del launch, update

    def _apply_pair_z_transitions_foldcp_row_slab(
        self,
        pair_z_row_slab: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
    ) -> torch.Tensor:
        n_token = z_spec.original_shape[z_spec.pair_dims[0]]
        row_start, row_end = z_spec.row_range
        valid_rows = max(0, min(row_end, n_token) - row_start)
        if valid_rows == 0:
            return pair_z_row_slab.contiguous()
        flat = pair_z_row_slab[..., :valid_rows, :n_token, :].contiguous().reshape(
            valid_rows * n_token,
            pair_z_row_slab.shape[-1],
        )
        global_flat_start = row_start * n_token
        source_rows = n_token * n_token
        self._apply_transition_source_flat_chunks(
            flat,
            self.transition_z1,
            global_flat_start=global_flat_start,
            source_rows=source_rows,
        )
        self._apply_transition_source_flat_chunks(
            flat,
            self.transition_z2,
            global_flat_start=global_flat_start,
            source_rows=source_rows,
        )
        pair_z_row_slab[..., :valid_rows, :n_token, :] = flat.reshape(
            *pair_z_row_slab.shape[:-3],
            valid_rows,
            n_token,
            pair_z_row_slab.shape[-1],
        )
        return pair_z_row_slab.contiguous()

    @staticmethod
    def _apply_transition_source_tile_chunks(
        tile: torch.Tensor,
        transition: Transition,
        *,
        row_start: int,
        col_start: int,
        valid_rows: int,
        valid_cols: int,
        original_n: int,
        flat_chunk_size: int = 262144,
    ) -> None:
        if valid_rows <= 0 or valid_cols <= 0:
            return
        tile_cols = tile.shape[-2]
        flat = tile.reshape(-1, tile.shape[-1])
        source_rows = int(original_n) * int(original_n)
        valid_row_end = int(row_start) + int(valid_rows)
        col_offsets = torch.arange(int(valid_cols), device=tile.device)
        local_col_offsets = torch.arange(int(valid_cols), device=tile.device)
        for chunk_start in range(0, source_rows, flat_chunk_size):
            chunk_end = min(chunk_start + flat_chunk_size, source_rows)
            global_row_start = max(int(row_start), chunk_start // int(original_n))
            global_row_end = min(
                valid_row_end,
                ((chunk_end - 1) // int(original_n)) + 1,
            )
            if global_row_start >= global_row_end:
                continue
            global_rows = torch.arange(
                global_row_start,
                global_row_end,
                device=tile.device,
            )
            source_index = (
                global_rows[:, None] * int(original_n)
                + int(col_start)
                + col_offsets[None, :]
            )
            mask = (source_index >= chunk_start) & (source_index < chunk_end)
            if not bool(mask.any()):
                continue
            source_offsets = (source_index[mask] - chunk_start).to(torch.long)
            local_rows = global_rows - int(row_start)
            tile_index = (
                local_rows[:, None] * int(tile_cols) + local_col_offsets[None, :]
            )[mask].to(torch.long)
            launch = flat.new_zeros(chunk_end - chunk_start, flat.shape[-1])
            launch.index_copy_(0, source_offsets, flat.index_select(0, tile_index))
            update = transition(launch).index_select(0, source_offsets)
            flat.index_copy_(0, tile_index, flat.index_select(0, tile_index) + update)
            del launch, update, source_offsets, tile_index

    def _apply_pair_z_transitions_foldcp_tile(
        self,
        pair_z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
    ) -> torch.Tensor:
        n_token = z_spec.original_shape[z_spec.pair_dims[0]]
        row_start, row_end = z_spec.row_range
        col_start, col_end = z_spec.col_range
        valid_rows = max(0, min(row_end, n_token) - row_start)
        valid_cols = max(0, min(col_end, n_token) - col_start)
        if valid_rows == 0 or valid_cols == 0:
            return pair_z_local.contiguous()
        pair_z_local = pair_z_local.contiguous()
        self._apply_transition_source_tile_chunks(
            pair_z_local,
            self.transition_z1,
            row_start=row_start,
            col_start=col_start,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
            original_n=n_token,
        )
        self._apply_transition_source_tile_chunks(
            pair_z_local,
            self.transition_z2,
            row_start=row_start,
            col_start=col_start,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
            original_n=n_token,
        )
        return pair_z_local.contiguous()

    def prepare_cache(
        self,
        relp_feature: Union[torch.Tensor, LazyRelativePositionEncodingFeatures],
        z_trunk: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        # Pair conditioning
        z_pair_trunk = z_trunk
        if self.compress_pair_z:
            z_pair_trunk = self._project_z_trunk(z_trunk)
        pair_z = torch.cat(
            tensors=[
                z_pair_trunk,
                self.relpe(relp_feature),
            ],
            dim=-1,
        )  # [..., N_tokens, N_tokens, 2*c_z_pair_diffusion]
        pair_z = self._project_pair_z(pair_z)
        return self._apply_pair_z_transitions(pair_z, inplace_safe=inplace_safe)

    @staticmethod
    def _local_relp_from_spec(
        relp_feature: Union[torch.Tensor, LazyRelativePositionEncodingFeatures],
        spec: FoldCPPairShardSpec,
        reference: torch.Tensor,
        feature_dim: int,
    ) -> torch.Tensor:
        return DiffusionConditioning._local_relp_chunk_from_spec(
            relp_feature=relp_feature,
            spec=spec,
            reference=reference,
            feature_dim=feature_dim,
            local_row_start=0,
            local_row_end=spec.local_shape[-3],
        )

    @staticmethod
    def _local_relp_chunk_from_spec(
        relp_feature: Union[torch.Tensor, LazyRelativePositionEncodingFeatures],
        spec: FoldCPPairShardSpec,
        reference: torch.Tensor,
        feature_dim: int,
        local_row_start: int,
        local_row_end: int,
    ) -> torch.Tensor:
        row_start, row_end = spec.row_range
        col_start, col_end = spec.col_range
        n_token = spec.original_shape[spec.pair_dims[0]]
        chunk_row_start = row_start + local_row_start
        chunk_row_end = min(row_start + local_row_end, row_end)
        valid_row_end = min(chunk_row_end, n_token)
        valid_col_end = min(col_end, n_token)
        valid_rows = max(0, valid_row_end - chunk_row_start)
        valid_cols = max(0, valid_col_end - col_start)
        local_shape = (
            *reference.shape[:-3],
            local_row_end - local_row_start,
            spec.local_shape[-2],
            feature_dim,
        )
        local = reference.new_zeros(local_shape)
        if valid_rows == 0 or valid_cols == 0:
            return local
        if isinstance(relp_feature, LazyRelativePositionEncodingFeatures):
            relp_valid = relp_feature.materialize(
                row_slice=slice(chunk_row_start, valid_row_end),
                col_slice=slice(col_start, valid_col_end),
            ).to(device=reference.device, dtype=reference.dtype)
        else:
            relp_valid = relp_feature[
                ...,
                chunk_row_start:valid_row_end,
                col_start:valid_col_end,
                :,
            ].to(device=reference.device, dtype=reference.dtype)
        local[..., :valid_rows, :valid_cols, :] = relp_valid
        return local.contiguous()

    def prepare_cache_foldcp_local(
        self,
        relp_feature: Union[torch.Tensor, LazyRelativePositionEncodingFeatures],
        z_trunk_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        mesh: FoldCPProcessMesh,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, FoldCPPairShardSpec]:
        del inplace_safe
        n_token = z_spec.original_shape[z_spec.pair_dims[0]]
        row_start, row_end = z_spec.row_range
        col_start, col_end = z_spec.col_range
        valid_rows = max(0, min(row_end, n_token) - row_start)
        valid_cols = max(0, min(col_end, n_token) - col_start)
        local_shape = z_trunk_local.shape[:-1] + (self.c_z_pair_diffusion,)
        pair_z_local = z_trunk_local.new_zeros(local_shape)
        pair_spec = FoldCPPairShardSpec(
            original_shape=z_spec.original_shape[:-1] + (self.c_z_pair_diffusion,),
            padded_shape=z_spec.padded_shape[:-1] + (self.c_z_pair_diffusion,),
            pair_dims=z_spec.pair_dims,
            row_range=z_spec.row_range,
            col_range=z_spec.col_range,
            mesh_shape=z_spec.mesh_shape,
            mesh_coord=z_spec.mesh_coord,
        )
        if valid_rows == 0 or valid_cols == 0:
            return pair_z_local.contiguous(), pair_spec

        del mesh
        source_rows = n_token * n_token
        row_chunk_size = _foldcp_diffusion_cache_trunk_row_chunk_size(valid_rows)
        for local_row_start in range(0, valid_rows, row_chunk_size):
            local_row_end = min(local_row_start + row_chunk_size, valid_rows)
            chunk_rows = local_row_end - local_row_start
            global_row_start = row_start + local_row_start
            z_trunk_chunk = z_trunk_local[
                ...,
                local_row_start:local_row_end,
                :,
                :,
            ]
            z_pair_trunk = z_trunk_chunk
            if self.compress_pair_z:
                z_pair_trunk = self._project_z_trunk_source_launch(
                    z_trunk_chunk,
                    source_rows=source_rows,
                    original_n=n_token,
                    row_start=global_row_start,
                    valid_rows=chunk_rows,
                    col_start=col_start,
                    valid_cols=valid_cols,
                )
            relp_local = self._local_relp_chunk_from_spec(
                relp_feature=relp_feature,
                spec=z_spec,
                reference=z_pair_trunk,
                feature_dim=self.relpe.linear_no_bias.in_features,
                local_row_start=local_row_start,
                local_row_end=local_row_end,
            )
            relp_tile = self._project_relp_tile_source_launch(
                relp_local,
                source_rows=source_rows,
                original_n=n_token,
                row_start=global_row_start,
                valid_rows=chunk_rows,
                col_start=col_start,
                valid_cols=valid_cols,
            )
            pair_z_tile = torch.cat(
                tensors=[
                    z_pair_trunk,
                    relp_tile,
                ],
                dim=-1,
            )
            pair_z_tile = self._project_pair_z_source_launch(
                pair_z_tile,
                source_rows=source_rows,
                original_n=n_token,
                row_start=global_row_start,
                valid_rows=chunk_rows,
                col_start=col_start,
                valid_cols=valid_cols,
            )
            del z_pair_trunk, relp_local, relp_tile
            chunk_spec = FoldCPPairShardSpec(
                original_shape=pair_spec.original_shape,
                padded_shape=pair_spec.padded_shape,
                pair_dims=pair_spec.pair_dims,
                row_range=(global_row_start, global_row_start + chunk_rows),
                col_range=pair_spec.col_range,
                mesh_shape=pair_spec.mesh_shape,
                mesh_coord=pair_spec.mesh_coord,
            )
            pair_z_tile = self._apply_pair_z_transitions_foldcp_tile(
                pair_z_tile,
                chunk_spec,
            )
            pair_z_local[
                ...,
                local_row_start:local_row_end,
                :valid_cols,
                :,
            ] = pair_z_tile[
                ...,
                :chunk_rows,
                :valid_cols,
                :,
            ]
            del pair_z_tile, z_trunk_chunk
        return pair_z_local.contiguous(), pair_spec

    def _apply_pair_z_transitions(
        self,
        pair_z: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        if torch.is_grad_enabled():
            if inplace_safe:
                pair_z = pair_z + self.transition_z1(pair_z)
                pair_z = pair_z + self.transition_z2(pair_z)
                return pair_z
            pair_z = pair_z + self.transition_z1(pair_z)
            return pair_z + self.transition_z2(pair_z)

        flat_chunk_size = 262144
        flat = pair_z.reshape(-1, pair_z.shape[-1])
        for start in range(0, flat.shape[0], flat_chunk_size):
            end = min(start + flat_chunk_size, flat.shape[0])
            flat[start:end] += self.transition_z1(flat[start:end])
        for start in range(0, flat.shape[0], flat_chunk_size):
            end = min(start + flat_chunk_size, flat.shape[0])
            flat[start:end] += self.transition_z2(flat[start:end])
        return pair_z.contiguous()

    def forward(
        self,
        t_hat_noise_level: torch.Tensor,
        relp_feature: Union[torch.Tensor, LazyRelativePositionEncodingFeatures],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        inplace_safe: bool = False,
        use_conditioning: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]
            asym_id (torch.Tensor): asym_id
            residue_index (torch.Tensor): residue_index
            entity_id (torch.Tensor): entity_id
            token_index (torch.Tensor): token_index
            sym_id (torch.Tensor): sym_id
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations.
            use_conditioning (bool): Whether to drop the s/z embeddings.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: embeddings s and z
                - s (torch.Tensor): [..., N_sample, N_tokens, c_s]
                - z (torch.Tensor): [..., N_tokens, N_tokens, c_z_pair_diffusion]
        """
        if pair_z is None:
            if not use_conditioning:
                if inplace_safe:
                    s_trunk *= 0
                    z_trunk *= 0
                else:
                    s_trunk = 0 * s_trunk
                    z_trunk = 0 * z_trunk
            pair_z = self.prepare_cache(relp_feature, z_trunk, inplace_safe)
        else:
            # Pair conditioning
            if inplace_safe:
                pair_z_clone = pair_z.clone()
                pair_z = pair_z_clone
        # Single conditioning
        single_s = torch.cat(
            tensors=[s_trunk, s_inputs], dim=-1
        )  # [..., N_tokens, c_s + c_s_inputs]
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
        noise_ratio = (t_hat_noise_level / self.sigma_data).clamp(min=1e-10)
        noise_n = self.fourier_embedding(
            t_hat_noise_level=torch.log(input=noise_ratio) / 4
        ).to(single_s.dtype)  # [..., N_sample, c_in]
        single_s = single_s.unsqueeze(dim=-3) + self.linear_no_bias_n(
            self.layernorm_n(noise_n)
        ).unsqueeze(dim=-2)  # [..., N_sample, N_tokens, c_s]
        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)
        return single_s, pair_z


class DiffusionModule(nn.Module):
    """
    Implements Algorithm 20 in AF3

    Args:
        sigma_data (torch.float, optional): the standard deviation of the data. Defaults to 16.0.
        c_atom (int, optional): embedding dim for atom feature. Defaults to 128.
        c_atompair (int, optional): embedding dim for atompair feature. Defaults to 16.
        c_token (int, optional): feature channel of token (single a). Defaults to 768.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_z_pair_diffusion (int, optional): hidden dim [for diffusion pair embedding].
            Defaults to c_z.
        c_s_inputs (int, optional): hidden dim [for single input embedding]. Defaults to 449.
        atom_encoder (dict[str, int], optional): configs in AtomAttentionEncoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
        transformer (dict[str, int], optional): configs in DiffusionTransformer. Defaults to {"n_blocks": 24, "n_heads": 16}.
        atom_decoder (dict[str, int], optional): configs in AtomAttentionDecoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
        blocks_per_ckpt: number of atom_encoder/transformer/atom_decoder blocks in each activation checkpoint
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing is performed.
        use_fine_grained_checkpoint: whether use fine-gained checkpoint for finetuning stage 2
            only effective if blocks_per_ckpt is not None.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_z_pair_diffusion: Optional[int] = None,
        c_s_inputs: int = 449,
        atom_encoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        transformer: dict[str, Any] = {
            "n_blocks": 24,
            "n_heads": 16,
        },
        atom_decoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        blocks_per_ckpt: Optional[int] = None,
        use_fine_grained_checkpoint: bool = False,
    ) -> None:
        super(DiffusionModule, self).__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s_inputs = c_s_inputs
        self.c_s = c_s
        self.c_z = c_z
        self.c_z_pair_diffusion = (
            c_z if c_z_pair_diffusion is None else c_z_pair_diffusion
        )

        # Grad checkpoint setting
        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_fine_grained_checkpoint = use_fine_grained_checkpoint

        self.diffusion_conditioning = DiffusionConditioning(
            sigma_data=self.sigma_data,
            c_z=c_z,
            c_z_pair_diffusion=self.c_z_pair_diffusion,
            c_s=c_s,
            c_s_inputs=c_s_inputs,
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            **atom_encoder,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=True,
            c_s=c_s,
            c_z=self.c_z_pair_diffusion,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        # Alg20: line4
        self.layernorm_s = LayerNorm(c_s, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(
            in_features=c_s,
            out_features=c_token,
            precision=torch.float32,
            initializer="zeros",
        )
        self.diffusion_transformer = DiffusionTransformer(
            **transformer,
            c_a=c_token,
            c_s=c_s,
            c_z=self.c_z_pair_diffusion,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_a = LayerNorm(c_token, create_offset=False)
        self.atom_attention_decoder = AtomAttentionDecoder(
            **atom_decoder,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.normalize = LayerNorm(
            self.c_z_pair_diffusion, create_offset=False, create_scale=False
        )
        self._foldcp_mesh: Optional[FoldCPProcessMesh] = None

    def _maybe_foldcp_mesh(self) -> Optional[FoldCPProcessMesh]:
        if os.environ.get("OPENDDE_FOLDCP_MODE", "single") != "distributed":
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None
        if torch.is_grad_enabled():
            return None
        if self._foldcp_mesh is not None:
            return self._foldcp_mesh
        foldcp = FoldCPConfig.from_runtime_args(
            mode="distributed",
            size_dp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_DP", "1")),
            size_cp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_CP", "4")),
            devices=os.environ.get("OPENDDE_FOLDCP_DEVICES", ""),
            metrics_jsonl=os.environ.get("OPENDDE_FOLDCP_METRICS_JSONL", ""),
        )
        self._foldcp_mesh = FoldCPProcessMesh.create(foldcp)
        return self._foldcp_mesh

    def f_forward(
        self,
        x_noisy: torch.Tensor,
        r_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        p_lm: torch.Tensor,
        c_l: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        use_conditioning: bool = True,
        enable_efficient_fusion: bool = False,
        pair_z_spec: Optional[FoldCPPairShardSpec] = None,
    ) -> torch.Tensor:
        """The denoising network used by diffusion sampling.
        As in EDM equation (7), this is F_theta(c_in * x, c_noise(sigma)).
        Here, c_noise(sigma) is computed in Conditioning module.

        Args:
            x_noisy (torch.Tensor): current noisy coordinates in x space
                [..., N_sample, N_atom, 3]
            r_noisy (torch.Tensor): scaled x_noisy (i.e., c_in * x)
                [..., N_sample, N_atom, 3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input feature
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_z (torch.Tensor): diffusion pair embedding
                [..., N_tokens, N_tokens, c_z_pair_diffusion]
            p_lm (torch.Tensor): MSA embedding
                [..., N_tokens, c_p_lm]
            c_l (torch.Tensor): ligand embedding
                [..., N_tokens, c_c_l]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            use_conditioning (bool): Whether to drop the s/z embeddings in DiffusionConditioning.
            enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.

        Returns:
            torch.Tensor: coordinates update
                [..., N_sample, N_atom, 3]
        """
        N_sample = r_noisy.size(-3)
        assert t_hat_noise_level.size(-1) == N_sample

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        foldcp_mesh = self._maybe_foldcp_mesh()

        atom_window_spec = None
        if foldcp_mesh is not None and pair_z_spec is not None and r_noisy.is_cuda:
            self.atom_attention_encoder._warmup_foldcp_atom_window_p2p(
                mesh=foldcp_mesh,
                device=r_noisy.device,
                dtype=r_noisy.dtype,
            )
        if pair_z is None and foldcp_mesh is not None and pair_z_spec is not None:
            z_trunk_for_cache = z_trunk
            if not use_conditioning:
                if inplace_safe:
                    s_trunk *= 0
                    z_trunk *= 0
                else:
                    s_trunk = 0 * s_trunk
                    z_trunk_for_cache = 0 * z_trunk
            pair_z, pair_z_spec = self.diffusion_conditioning.prepare_cache_foldcp_local(
                input_feature_dict["relp"],
                z_trunk_for_cache,
                pair_z_spec,
                foldcp_mesh,
                inplace_safe,
            )
        # Conditioning, shared across difference samples
        # Diffusion_conditioning consumes 7-8G when token num is 768,
        # use checkpoint here if blocks_per_ckpt is not None.
        if blocks_per_ckpt:
            checkpoint_fn = get_checkpoint_fn()
            s_single, z_pair = checkpoint_fn(
                self.diffusion_conditioning,
                t_hat_noise_level,
                input_feature_dict["relp"],
                s_inputs,
                s_trunk,
                z_trunk,
                pair_z,
                inplace_safe,
                use_conditioning,
            )
        else:
            s_single, z_pair = self.diffusion_conditioning(
                t_hat_noise_level,
                input_feature_dict["relp"],
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                inplace_safe=inplace_safe,
                use_conditioning=use_conditioning,
            )  # [..., N_sample, N_token, c_s], [..., N_token, N_token, c_z]

        # Expand single embeddings to match N_sample. Pair conditioning is shared
        # across diffusion samples and is broadcast inside attention/local atom pair
        # paths to avoid materializing [N_sample, N_token, N_token, c_z].
        s_trunk = expand_at_dim(s_trunk, dim=-3, n=1)  # [..., N_sample, N_token, c_s]
        # Fine-grained checkpoint for finetuning stage 2 (token num: 768) for avoiding OOM
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            a_token, q_skip, c_skip, p_skip = checkpoint_fn(
                self.atom_attention_encoder,
                input_feature_dict["atom_to_token_idx"],
                input_feature_dict["ref_pos"],
                input_feature_dict["ref_charge"],
                input_feature_dict["ref_mask"],
                input_feature_dict["ref_atom_name_chars"],
                input_feature_dict["ref_element"],
                input_feature_dict["d_lm"],
                input_feature_dict["v_lm"],
                input_feature_dict["pad_info"],
                r_noisy,
                s_trunk,
                z_pair,
                p_lm,
                c_l,
                inplace_safe,
                chunk_size,
            )
        else:
            # Sequence-local Atom Attention and aggregation to coarse-grained tokens
            if foldcp_mesh is not None:
                (
                    a_token,
                    q_skip,
                    c_skip,
                    p_skip,
                    atom_window_spec,
                ) = self.atom_attention_encoder.forward_foldcp_window(
                    input_feature_dict["atom_to_token_idx"],
                    input_feature_dict["ref_pos"],
                    input_feature_dict["ref_charge"],
                    input_feature_dict["ref_mask"],
                    input_feature_dict["ref_atom_name_chars"],
                    input_feature_dict["ref_element"],
                    input_feature_dict["d_lm"],
                    input_feature_dict["v_lm"],
                    input_feature_dict["pad_info"],
                    mesh=foldcp_mesh,
                    r_l=r_noisy,
                    s=s_trunk,
                    z=z_pair,
                    z_spec=pair_z_spec,
                    p_lm=p_lm,
                    c_l=c_l,
                    inplace_safe=inplace_safe,
                )
            else:
                a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
                    input_feature_dict["atom_to_token_idx"],
                    input_feature_dict["ref_pos"],
                    input_feature_dict["ref_charge"],
                    input_feature_dict["ref_mask"],
                    input_feature_dict["ref_atom_name_chars"],
                    input_feature_dict["ref_element"],
                    input_feature_dict["d_lm"],
                    input_feature_dict["v_lm"],
                    input_feature_dict["pad_info"],
                    r_l=r_noisy,
                    s=s_trunk,
                    z=z_pair,
                    p_lm=p_lm,
                    c_l=c_l,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
        # Upcast
        a_token = a_token.to(dtype=torch.float32)
        # Full self-attention on token level.
        if inplace_safe:
            a_token += self.linear_no_bias_s(
                self.layernorm_s(s_single)
            )  # [..., N_sample, N_token, c_token]
        else:
            a_token = a_token + self.linear_no_bias_s(
                self.layernorm_s(s_single)
            )  # [..., N_sample, N_token, c_token]
        if foldcp_mesh is not None and pair_z_spec is not None:
            if enable_efficient_fusion:
                z = self.normalize(z_pair.to(dtype=torch.float32))
            else:
                z = z_pair.to(dtype=torch.float32)
            a_token = self.diffusion_transformer.forward_foldcp_local_z(
                a=a_token.to(dtype=torch.float32),
                s=s_single.to(dtype=torch.float32),
                z_local=z,
                z_spec=pair_z_spec,
                mesh=foldcp_mesh,
                inplace_safe=inplace_safe,
                extra_attn_bias=input_feature_dict.get("structural_pair_attn_bias", None),
                enable_efficient_fusion=enable_efficient_fusion,
            )
        elif enable_efficient_fusion:
            z = self.normalize(z_pair.to(dtype=torch.float32))
            z = permute_final_dims(z, [2, 0, 1]).contiguous()
            a_token = self.diffusion_transformer(
                a=a_token.to(dtype=torch.float32),  # Upcast all inputs
                s=s_single.to(dtype=torch.float32),
                z=z,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                enable_efficient_fusion=enable_efficient_fusion,
                extra_attn_bias=input_feature_dict.get("structural_pair_attn_bias", None),
            )
        else:
            z = z_pair.to(dtype=torch.float32)
            a_token = self.diffusion_transformer(
                a=a_token.to(dtype=torch.float32),  # Upcast all inputs
                s=s_single.to(dtype=torch.float32),
                z=z,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                enable_efficient_fusion=enable_efficient_fusion,
                extra_attn_bias=input_feature_dict.get("structural_pair_attn_bias", None),
            )

        a_token = self.layernorm_a(a_token)
        # Fine-grained checkpoint for finetuning stage 2 (token num: 768) for avoiding OOM
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            r_update = checkpoint_fn(
                self.atom_attention_decoder,
                input_feature_dict["atom_to_token_idx"],
                a_token,
                q_skip,
                c_skip,
                p_skip,
                inplace_safe,
                chunk_size,
            )
        else:
            # Broadcast token activations to atoms and run Sequence-local Atom Attention
            if foldcp_mesh is not None and atom_window_spec is not None:
                r_update = self.atom_attention_decoder.forward_foldcp_window(
                    atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                    a=a_token,
                    q_skip=q_skip,
                    c_skip=c_skip,
                    p_skip_local=p_skip,
                    window_spec=atom_window_spec,
                    mesh=foldcp_mesh,
                    inplace_safe=inplace_safe,
                )
            else:
                r_update = self.atom_attention_decoder(
                    atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                    a=a_token,
                    q_skip=q_skip,
                    c_skip=c_skip,
                    p_skip=p_skip,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        return r_update

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        p_lm: torch.Tensor,
        c_l: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        use_conditioning: bool = True,
        enable_efficient_fusion: bool = False,
        pair_z_spec: Optional[FoldCPPairShardSpec] = None,
    ) -> torch.Tensor:
        """One step denoise: x_noisy, noise_level -> x_denoised

        Args:
            x_noisy (torch.Tensor): the noisy version of the input atom coords
                [..., N_sample, N_atom,3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_z (torch.Tensor): diffusion pair embedding
                [..., N_tokens, N_tokens, c_z]
            p_lm (torch.Tensor): MSA embedding
                [..., N_tokens, c_p_lm]
            c_l (torch.Tensor): ligand embedding
                [..., N_tokens, c_c_l]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            use_conditioning (bool): Whether to drop the s/z embeddings in DiffusionConditioning.
            enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.

        Returns:
            torch.Tensor: the denoised coordinates of x
                [..., N_sample, N_atom,3]
        """
        # Scale positions to dimensionless vectors with approximately unit variance
        # As in EDM:
        #     r_noisy = (c_in * x_noisy)
        #     where c_in = 1 / sqrt(sigma_data^2 + sigma^2)
        r_noisy = (
            x_noisy
            / torch.sqrt(self.sigma_data**2 + t_hat_noise_level**2)[..., None, None]
        )

        # Compute the update given r_noisy (the scaled x_noisy)
        # As in EDM:
        #     r_update = F(r_noisy, c_noise(sigma))
        r_update = self.f_forward(
            x_noisy=x_noisy,
            r_noisy=r_noisy,
            t_hat_noise_level=t_hat_noise_level,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=enable_efficient_fusion,
            pair_z_spec=pair_z_spec,
        )

        # Rescale updates to positions and combine with input positions
        # As in EDM:
        #     D = c_skip * x_noisy + c_out * r_update
        #     c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
        #     c_out = (sigma_data * sigma) / sqrt(sigma_data^2 + sigma^2)
        #     s_ratio = sigma / sigma_data
        #     c_skip = 1 / (1 + s_ratio^2)
        #     c_out = sigma / sqrt(1 + s_ratio^2)

        s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].to(
            r_update.dtype
        )
        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy
            + t_hat_noise_level[..., None, None] / torch.sqrt(1 + s_ratio**2) * r_update
        ).to(r_update.dtype)

        return x_denoised
