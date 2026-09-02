# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.comm import (
    detach_rank_local_error_traceback,
    dispatch_p2p_batch_and_wait,
    gather_tensor_by_ring,
    run_group_rank_action_synchronized,
)
from opendde.distributed.foldcp.atom_window import (
    FoldCPWindowShardSpec,
    atom_window_token_indices,
    gather_window_blocks,
    window_block_range,
)
from opendde.distributed.foldcp.pair_sharding import FoldCPPairShardSpec
from opendde.model.modules.primitives import (
    AdaptiveLayerNorm,
    Attention,
    BiasInitLinear,
    LinearNoBias,
    _attention,
    broadcast_token_to_local_atom_pair,
    gather_pair_embedding_in_dense_trunk,
    rearrange_qk_to_dense_trunk,
)
from opendde.model.triangular.layers import LayerNorm
from opendde.model.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
    checkpoint_blocks,
    permute_final_dims,
)
from opendde.utils.logger import get_logger

logger = get_logger(__name__)


def _zero_foldcp_drain_buffer(tensor: torch.Tensor) -> None:
    """Reset a reusable CUDA buffer at a drainable failure boundary."""

    tensor.zero_()


# This is a resident-cache budget, not a total construction or device-memory
# limit. The resident cache costs
# 24 blocks * 16 heads * ceil(N/P) * N * 4 B; the cap prevents this optional
# speed cache from consuming an unbounded fraction of a device by itself.
#
# A 1xP cache build also temporarily materializes the query-owned
# N*ceil(N/P)*c_z(+1) fp32 gathered source and its projection workspaces. Those
# temporaries are not included in this resident budget, so this gate is not a
# total-model OOM predictor. It does run before the gather, allowing a lower
# budget to avoid both the resident cache and its construction workspace.
_FOLDCP_DIFFUSION_BIAS_CACHE_MAX_BYTES = 16 * 1024**3


def _foldcp_diffusion_bias_cache_max_bytes() -> int:
    """Return the resident budget for the per-block diffusion pair-bias cache."""

    value = os.environ.get("OPENDDE_FOLDCP_DIFFUSION_BIAS_CACHE_MAX_BYTES")
    if value is None or value == "":
        return _FOLDCP_DIFFUSION_BIAS_CACHE_MAX_BYTES
    return max(0, int(value))


def foldcp_diffusion_bias_cache_is_safe(
    *,
    n_blocks: int,
    n_heads: int,
    bias_rows: int,
    bias_cols: int,
    element_size: int,
) -> bool:
    """Check the resident cost of caching one pair bias per diffusion block.

    Every block retains a [n_heads, bias_rows, bias_cols] tile for the whole
    sampling loop, so the total grows as n_token**2 / P. Callers recompute the
    bias per denoise step when the estimate does not fit. This check does not
    include temporary cache-construction workspaces or predict total device
    peak memory.

    All inputs are globally consistent quantities, so every rank reaches the
    same verdict and none of them skips the collectives in the cache build.
    """

    resident_bytes = (
        int(n_blocks)
        * int(n_heads)
        * int(bias_rows)
        * int(bias_cols)
        * int(element_size)
    )
    return resident_bytes <= _foldcp_diffusion_bias_cache_max_bytes()


def _prepare_foldcp_diffusion_bias_cache_source(
    z_local: torch.Tensor,
    extra_attn_bias: Optional[torch.Tensor],
    *,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    valid_rows: int,
    valid_cols: int,
    tile_cols: int,
    mesh_cols: int,
) -> tuple[torch.Tensor, torch.Tensor, bool, Optional[torch.Tensor]]:
    """Prepare query-row all-to-all buffers in one OOM boundary."""

    extra_local = None
    if extra_attn_bias is not None:
        if extra_attn_bias.shape[-2:] == z_local.shape[-3:-1]:
            extra_local = extra_attn_bias[..., :valid_rows, :valid_cols]
        else:
            extra_local = extra_attn_bias[..., row_start:row_end, col_start:col_end]

    packed_extra_bias = (
        extra_local is not None
        and extra_local.shape == z_local[..., :valid_rows, :valid_cols, :].shape[:-1]
    )
    packed_features = z_local.shape[-1] + int(packed_extra_bias)
    local_cols_front = z_local.new_zeros(
        mesh_cols * tile_cols,
        *z_local.shape[:-3],
        tile_cols,
        packed_features,
    )
    for destination in range(mesh_cols):
        query_start, query_end = _foldcp_diffusion_query_range(
            n_token=valid_rows,
            cp_size=mesh_cols,
            cp_rank=destination,
        )
        valid_query_rows = query_end - query_start
        destination_chunk = local_cols_front[
            destination * tile_cols : (destination + 1) * tile_cols
        ]
        z_query_tile = z_local[..., query_start:query_end, :valid_cols, :].movedim(
            -2, 0
        )
        if packed_extra_bias:
            destination_chunk[:valid_cols, ..., :valid_query_rows, :-1].copy_(
                z_query_tile
            )
            extra_query_tile = extra_local[
                ..., query_start:query_end, :valid_cols
            ].movedim(-1, 0)
            destination_chunk[:valid_cols, ..., :valid_query_rows, -1].copy_(
                extra_query_tile.to(dtype=z_local.dtype, device=z_local.device)
            )
            del extra_query_tile
        else:
            destination_chunk[:valid_cols, ..., :valid_query_rows, :].copy_(
                z_query_tile
            )
        del destination_chunk, z_query_tile
    gathered_cols_front = local_cols_front.new_empty(
        mesh_cols * tile_cols, *local_cols_front.shape[1:]
    )
    return local_cols_front, gathered_cols_front, packed_extra_bias, extra_local


def _attention_pair_bias_row_chunk_size(n_token: int) -> int:
    """Return the P-independent row launch size used by distributed FoldCP."""

    n_token = int(n_token)
    if n_token <= 512:
        return n_token
    value = int(os.environ.get("OPENDDE_PAIR_BIAS_ROW_CHUNK", "112"))
    if value <= 0:
        return n_token
    return min(n_token, value)


def _foldcp_diffusion_query_range(
    *,
    n_token: int,
    cp_size: int,
    cp_rank: int,
) -> tuple[int, int]:
    """Return a balanced contiguous query-row range for one 1xP rank."""

    n_token = int(n_token)
    cp_size = int(cp_size)
    cp_rank = int(cp_rank)
    if n_token <= 0:
        raise ValueError("n_token must be positive")
    if cp_size <= 0:
        raise ValueError("cp_size must be positive")
    if not 0 <= cp_rank < cp_size:
        raise ValueError("cp_rank must be in [0, cp_size)")
    return (
        n_token * cp_rank // cp_size,
        n_token * (cp_rank + 1) // cp_size,
    )


@dataclass(frozen=True)
class FoldCPQueryOwnedAttentionBias:
    """Static diffusion bias after the 1xP column-to-query transpose."""

    tensor: torch.Tensor


@dataclass
class _FoldCPAttentionWorkspace:
    """Reusable 1xP diffusion-attention collective buffers for one rollout."""

    output_key: tuple[Any, ...] | None = None
    local_raw_front: torch.Tensor | None = None
    gathered_raw_front: torch.Tensor | None = None
    full_raw: torch.Tensor | None = None
    bias_key: tuple[Any, ...] | None = None
    send_bias: torch.Tensor | None = None
    received_bias: torch.Tensor | None = None
    row_bias: torch.Tensor | None = None

    def output_buffers(
        self,
        q_proj: torch.Tensor,
        *,
        n_token: int,
        mesh: FoldCPProcessMesh,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        side = int(mesh.layout.shape[1])
        query_tile_rows = int(math.ceil(n_token / side))
        transfer_rows = query_tile_rows + 1
        key = (
            tuple(q_proj.shape),
            q_proj.dtype,
            q_proj.device,
            int(n_token),
            side,
        )
        if self.output_key != key:
            self.output_key = None
            self.local_raw_front = None
            self.gathered_raw_front = None
            self.full_raw = None

            def _allocate_output_workspace():
                local_shape = (
                    transfer_rows,
                    *q_proj.shape[:-3],
                    q_proj.shape[-3],
                    q_proj.shape[-1],
                )
                local = q_proj.new_empty(local_shape)
                gathered = q_proj.new_empty((side * transfer_rows, *local_shape[1:]))
                full = q_proj.new_empty(
                    *q_proj.shape[:-3],
                    n_token,
                    q_proj.shape[-3],
                    q_proj.shape[-1],
                )
                return local, gathered, full

            buffers = run_group_rank_action_synchronized(
                _allocate_output_workspace,
                group=mesh.group_2d,
                description="Fold-CP diffusion output-workspace allocation",
            )
            if buffers is None:  # pragma: no cover
                raise RuntimeError(
                    "Fold-CP diffusion output workspace returned no buffers."
                )
            self.local_raw_front, self.gathered_raw_front, self.full_raw = buffers
            self.output_key = key
        if (
            self.local_raw_front is None
            or self.gathered_raw_front is None
            or self.full_raw is None
        ):  # pragma: no cover
            raise RuntimeError("Fold-CP diffusion output workspace is incomplete.")
        return self.local_raw_front, self.gathered_raw_front, self.full_raw

    def bias_buffers(
        self,
        bias_tile: torch.Tensor,
        *,
        n_token: int,
        mesh: FoldCPProcessMesh,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        side = int(mesh.layout.shape[1])
        query_tile_rows = int(math.ceil(n_token / side))
        transfer_rows = query_tile_rows + 1
        tile_cols = int(bias_tile.shape[-1])
        key = (
            tuple(bias_tile.shape),
            bias_tile.dtype,
            bias_tile.device,
            int(n_token),
            side,
        )
        if self.bias_key != key:
            self.bias_key = None
            self.send_bias = None
            self.received_bias = None
            self.row_bias = None

            def _allocate_bias_workspace():
                send = bias_tile.new_empty(
                    side,
                    *bias_tile.shape[:-2],
                    transfer_rows,
                    tile_cols,
                )
                return (
                    send,
                    torch.empty_like(send),
                    bias_tile.new_empty(
                        *bias_tile.shape[:-2], query_tile_rows, n_token
                    ),
                )

            buffers = run_group_rank_action_synchronized(
                _allocate_bias_workspace,
                group=mesh.group_2d,
                description="Fold-CP diffusion bias-workspace allocation",
            )
            if buffers is None:  # pragma: no cover
                raise RuntimeError(
                    "Fold-CP diffusion bias workspace returned no buffers."
                )
            self.send_bias, self.received_bias, self.row_bias = buffers
            self.bias_key = key
        if (
            self.send_bias is None
            or self.received_bias is None
            or self.row_bias is None
        ):  # pragma: no cover
            raise RuntimeError("Fold-CP diffusion bias workspace is incomplete.")
        return self.send_bias, self.received_bias, self.row_bias


class AttentionPairBias(nn.Module):
    """
    Implements Algorithm 24 in AF3

    Args:
        has_s (bool, optional):  whether s is None as stated in Algorithm 24 Line1. Defaults to True.
        create_offset_ln_z (bool, optional): the value of create_offset for the LayerNorm applied to z. Defaults to False.
        n_heads (int, optional): number of attention-like head in AttentionPairBias. Defaults to 16.
        c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        biasinit (float, optional): biasinit for BiasInitLinear. Defaults to -2.0.
        cross_attention_mode (bool, optional): If cross_attention_model = True, the adaptive layernorm will be applied
            to query and key/value seperately. Defaults to False.
    """

    def __init__(
        self,
        has_s: bool = True,
        create_offset_ln_z: bool = False,
        n_heads: int = 16,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        biasinit: float = -2.0,
        cross_attention_mode: bool = False,
    ) -> None:
        super(AttentionPairBias, self).__init__()
        assert c_a % n_heads == 0
        self.n_heads = n_heads
        self.has_s = has_s
        self.create_offset_ln_z = create_offset_ln_z
        self.cross_attention_mode = cross_attention_mode
        if has_s:
            # Line2
            self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
            if self.cross_attention_mode:
                self.layernorm_kv = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        else:
            self.layernorm_a = LayerNorm(c_a)
            if self.cross_attention_mode:
                self.layernorm_kv = LayerNorm(c_a)

        # Line 6-11
        self.attention = Attention(
            c_q=c_a,
            c_k=c_a,
            c_v=c_a,
            c_hidden=c_a // n_heads,
            num_heads=n_heads,
            gating=True,
            q_linear_bias=True,
            zero_init=not self.has_s,  # Adaptive zero init
        )
        self.layernorm_z = LayerNorm(c_z, create_offset=self.create_offset_ln_z)
        # Alg24. Line8 is scalar, but this is different for different heads
        self.linear_nobias_z = LinearNoBias(in_features=c_z, out_features=n_heads)

        # Line 13
        if self.has_s:
            self.linear_a_last = BiasInitLinear(
                in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
            )

    @staticmethod
    def _align_bias_to_query(
        bias: torch.Tensor, q: torch.Tensor, n_pair_dims: int
    ) -> torch.Tensor:
        """Insert missing sample/batch dims before the head dim.

        Pair features may be shared across diffusion samples, e.g. [H, N, N],
        while q carries a sample dimension, e.g. [N_sample, N, C]. The head dim
        is the first trailing non-pair dim, so missing broadcast dims must be
        inserted before it.
        """

        target_ndim = len(q.shape[:-2]) + 1 + n_pair_dims
        while bias.dim() < target_ndim:
            bias = bias.unsqueeze(dim=bias.dim() - (1 + n_pair_dims))
        return bias

    @staticmethod
    def _foldcp_diffusion_bias_row_chunk_size() -> int:
        """Return the row chunk size for Fold-CP diffusion pair-bias streaming."""

        if (
            os.environ.get("OPENDDE_FOLDCP_MODE") != "distributed"
            or not dist.is_available()
            or not dist.is_initialized()
            or dist.get_world_size() <= 1
        ):
            return 0
        value = os.environ.get("OPENDDE_FOLDCP_DIFFUSION_BIAS_ROW_CHUNK", "128")
        return max(int(value or "0"), 0)

    @staticmethod
    def _slice_extra_attn_bias_rows(
        extra_attn_bias: torch.Tensor,
        row_start: int,
        row_end: int,
    ) -> torch.Tensor:
        slices = [slice(None)] * extra_attn_bias.dim()
        slices[-2] = slice(row_start, row_end)
        return extra_attn_bias[tuple(slices)]

    def _add_extra_attn_bias_to_chunk(
        self,
        bias: torch.Tensor,
        extra_attn_bias: Optional[torch.Tensor],
        row_start: int,
        row_end: int,
    ) -> torch.Tensor:
        if extra_attn_bias is None:
            return bias
        extra_chunk = self._slice_extra_attn_bias_rows(
            extra_attn_bias,
            row_start,
            row_end,
        )
        while len(extra_chunk.shape) < len(bias.shape) - 1:
            extra_chunk = extra_chunk.unsqueeze(dim=0)
        if len(extra_chunk.shape) == len(bias.shape) - 1:
            extra_chunk = extra_chunk.unsqueeze(dim=-3)
        return bias + extra_chunk.to(dtype=bias.dtype, device=bias.device)

    def _standard_multihead_attention_stream_pair_bias(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
        extra_attn_bias: Optional[torch.Tensor],
        inplace_safe: bool,
        row_chunk_size: int,
    ) -> torch.Tensor:
        """Compute full token attention while materializing pair bias by row chunks.

        The attention softmax is independent for each query row. This keeps the
        serial formula intact while avoiding a full [H, N, N] pair-bias tensor in
        every diffusion transformer block.
        """

        q_proj, k_proj, v_proj = self.attention._prep_qkv(
            q_x=q,
            kv_x=kv,
            apply_scale=True,
        )
        n_token = z.shape[-3]
        out = q.new_empty(q.shape)
        for row_start in range(0, n_token, row_chunk_size):
            row_end = min(row_start + row_chunk_size, n_token)
            q_chunk = q[..., row_start:row_end, :]
            q_proj_chunk = q_proj[..., row_start:row_end, :]
            z_chunk = z[..., row_start:row_end, :, :]
            bias = self.linear_nobias_z(self.layernorm_z(z_chunk))
            bias = permute_final_dims(bias, [2, 0, 1])
            bias = self._add_extra_attn_bias_to_chunk(
                bias,
                extra_attn_bias,
                row_start,
                row_end,
            )
            bias = self._align_bias_to_query(bias, q_chunk, n_pair_dims=2)
            o_chunk = _attention(
                q=q_proj_chunk,
                k=k_proj,
                v=v_proj,
                attn_bias=bias,
                use_efficient_implementation=self.attention.use_efficient_implementation,
                inplace_safe=inplace_safe,
            )
            o_chunk = o_chunk.transpose(-2, -3)
            out[..., row_start:row_end, :] = self.attention._wrap_up(
                o_chunk,
                q_chunk,
            )
            del bias, z_chunk, q_proj_chunk, q_chunk, o_chunk
        return out

    @staticmethod
    def _foldcp_valid_ranges(
        spec: FoldCPPairShardSpec,
    ) -> tuple[int, int, int, int, int, int]:
        row_start, row_end = spec.row_range
        col_start, col_end = spec.col_range
        n_token = spec.original_shape[spec.pair_dims[0]]
        valid_row_end = min(row_end, n_token)
        valid_col_end = min(col_end, n_token)
        return (
            row_start,
            valid_row_end,
            col_start,
            valid_col_end,
            max(0, valid_row_end - row_start),
            max(0, valid_col_end - col_start),
        )

    @staticmethod
    def _foldcp_gather_row_outputs_by_col_ring(
        local_out: torch.Tensor,
        n_token: int,
        mesh: FoldCPProcessMesh,
    ) -> torch.Tensor:
        """Gather row-sharded attention outputs without a column all-gather."""

        return AttentionPairBias._foldcp_gather_rows_by_col_ring(
            local_out,
            n_token=n_token,
            mesh=mesh,
            row_dim=-2,
        )

    @staticmethod
    def _foldcp_gather_rows_by_col_ring(
        local_rows: torch.Tensor,
        *,
        n_token: int,
        mesh: FoldCPProcessMesh,
        row_dim: int,
    ) -> torch.Tensor:
        """Gather row-sharded tensors without changing non-row dimensions."""

        row_dim = row_dim % local_rows.dim()
        local_out = local_rows.contiguous()
        side = mesh.layout.shape[0]
        if side == 1:
            slices = [slice(None)] * local_out.dim()
            slices[row_dim] = slice(0, n_token)
            return local_out[tuple(slices)].contiguous()

        ring = mesh.ring_comm()
        return gather_tensor_by_ring(
            local_out,
            comm=ring.comm_col,
            group=mesh.group_col,
            local_index=mesh.coord[0],
            side=side,
            dim=row_dim,
            length=n_token,
            description="diffusion attention output-row ring",
        )

    def standard_multihead_attention_foldcp_local_z(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        mesh: FoldCPProcessMesh,
        extra_attn_bias: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        enable_efficient_fusion: bool = False,
        projected_bias_local: Optional[
            torch.Tensor | FoldCPQueryOwnedAttentionBias
        ] = None,
        workspace: Optional[_FoldCPAttentionWorkspace] = None,
    ) -> torch.Tensor:
        def _prepare_attention_inputs():
            q_proj, k_proj, v_proj = self.attention._prep_qkv(
                q_x=q,
                kv_x=kv,
                apply_scale=True,
            )
            ranges = self._foldcp_valid_ranges(z_spec)
            row_start, row_end, _col_start, _col_end, valid_rows, valid_cols = ranges
            tile_rows = z_spec.local_shape[z_spec.pair_dims[0]]
            tile_cols = z_spec.local_shape[z_spec.pair_dims[1]]
            side = mesh.layout.shape[1]
            is_one_by_p = mesh.layout.shape[0] == 1 and side > 1
            # A 1xP rank never loses rows, but its columns can be pure padding
            # when N is small relative to P. It still owns query rows and joins
            # every collective, so build its all-zero bias tile as needed.
            query_owned_bias = None
            bias_tile = None
            if is_one_by_p or (valid_rows > 0 and valid_cols > 0):
                q_chunk = q[..., row_start:row_end, :]
                if isinstance(projected_bias_local, FoldCPQueryOwnedAttentionBias):
                    query_owned_bias = projected_bias_local.tensor
                    bias_local = None
                elif projected_bias_local is None:
                    bias_local = self.project_foldcp_attention_bias_local(
                        z_local=z_local,
                        z_spec=z_spec,
                        extra_attn_bias=extra_attn_bias,
                        enable_efficient_fusion=enable_efficient_fusion,
                    )
                else:
                    bias_local = projected_bias_local
                if bias_local is not None:
                    bias_local = self._align_bias_to_query(
                        bias_local,
                        q_chunk,
                        n_pair_dims=2,
                    ).contiguous()

                    if bias_local.shape[-1] != tile_cols:
                        bias_tile = bias_local.new_zeros(
                            *bias_local.shape[:-1], tile_cols
                        )
                        bias_tile[..., : bias_local.shape[-1]] = bias_local
                    else:
                        bias_tile = bias_local
            return (
                q_proj,
                k_proj,
                v_proj,
                ranges,
                tile_rows,
                tile_cols,
                side,
                is_one_by_p,
                query_owned_bias,
                bias_tile,
            )

        prepared = (
            run_group_rank_action_synchronized(
                _prepare_attention_inputs,
                group=mesh.group_2d,
                description="Fold-CP diffusion attention input preparation",
            )
            if int(mesh.layout.shape[0]) == 1 and int(mesh.layout.shape[1]) > 1
            else _prepare_attention_inputs()
        )
        if prepared is None:  # pragma: no cover
            raise RuntimeError("Fold-CP diffusion attention inputs were not prepared.")
        (
            q_proj,
            k_proj,
            v_proj,
            ranges,
            tile_rows,
            tile_cols,
            side,
            is_one_by_p,
            query_owned_bias,
            bias_tile,
        ) = prepared
        row_start, row_end, col_start, col_end, valid_rows, valid_cols = ranges

        if is_one_by_p:
            n_token = q.shape[-2]
            query_tile_rows = int(math.ceil(n_token / side))
            transfer_rows = query_tile_rows + 1
            group_rank = mesh.coord[1]
            query_start, query_end = _foldcp_diffusion_query_range(
                n_token=n_token,
                cp_size=side,
                cp_rank=group_rank,
            )
            valid_query_rows = query_end - query_start
            local_compute_error: Exception | None = None
            if query_owned_bias is not None:
                row_bias = query_owned_bias
            elif bias_tile is not None:
                row_bias = self._foldcp_transpose_bias_to_query_rows(
                    bias_tile=bias_tile,
                    n_token=n_token,
                    mesh=mesh,
                    workspace=workspace,
                )
            else:
                raise RuntimeError(
                    "Fold-CP 1xP diffusion attention needs either a query-owned "
                    "bias cache or a local pair-bias tile."
                )
            if workspace is None:

                def _allocate_output_buffers():
                    local_shape = (
                        transfer_rows,
                        *q_proj.shape[:-3],
                        q_proj.shape[-3],
                        q_proj.shape[-1],
                    )
                    local_front = q_proj.new_zeros(local_shape)
                    gathered_front = q_proj.new_empty(
                        side * transfer_rows, *local_shape[1:]
                    )
                    full = q_proj.new_empty(
                        *q_proj.shape[:-3],
                        n_token,
                        q_proj.shape[-3],
                        q_proj.shape[-1],
                    )
                    # Keep the piggybacked failure flag armed until local
                    # attention has completed successfully. An OOM path then
                    # does not need another CUDA write before draining the
                    # required all-gather.
                    local_front[-1].reshape(-1)[0].fill_(1)
                    return local_front, gathered_front, full

                output_buffers = run_group_rank_action_synchronized(
                    _allocate_output_buffers,
                    group=mesh.group_2d,
                    description="Fold-CP diffusion output allocation",
                )
                if output_buffers is None:  # pragma: no cover
                    raise RuntimeError(
                        "Fold-CP diffusion output allocation returned no buffers."
                    )
                local_raw_front, gathered_raw_front, full_raw = output_buffers
                local_raw = local_raw_front.movedim(0, -3)
            else:
                (
                    local_raw_front,
                    gathered_raw_front,
                    full_raw,
                ) = workspace.output_buffers(q_proj, n_token=n_token, mesh=mesh)
                try:
                    local_raw_front[-1].reshape(-1)[0].fill_(1)
                    _zero_foldcp_drain_buffer(local_raw_front[:-1])
                except Exception as exc:
                    local_compute_error = detach_rank_local_error_traceback(exc)
                local_raw = local_raw_front.movedim(0, -3)
            if local_compute_error is None:
                try:
                    q_local = q[..., query_start:query_end, :]
                    row_bias = self._align_bias_to_query(
                        row_bias[..., :valid_query_rows, :],
                        q_local,
                        n_pair_dims=2,
                    )
                    raw_chunk = _attention(
                        q=q_proj[..., query_start:query_end, :],
                        k=k_proj,
                        v=v_proj,
                        attn_bias=row_bias,
                        use_efficient_implementation=(
                            self.attention.use_efficient_implementation
                        ),
                        inplace_safe=inplace_safe,
                    )
                    local_raw[..., :valid_query_rows, :, :].copy_(
                        raw_chunk.transpose(-2, -3)
                    )
                except Exception as exc:
                    # The output/transfer buffers already exist. Mark this rank
                    # and still drain the scheduled all-gather so no healthy
                    # peer waits forever after a rank-local CUDA failure.
                    local_compute_error = detach_rank_local_error_traceback(exc)
            if local_compute_error is None:
                try:
                    local_raw_front[-1].reshape(-1)[0].zero_()
                except Exception as exc:
                    # The flag was armed before local work. If clearing it
                    # fails, retain the failure and leave it armed while every
                    # rank drains the same gather.
                    local_compute_error = detach_rank_local_error_traceback(exc)

            if gathered_raw_front is None:  # pragma: no cover
                raise RuntimeError("Fold-CP diffusion gather workspace is missing.")
            dist.all_gather_into_tensor(
                gathered_raw_front,
                local_raw_front,
                group=mesh.group_2d,
            )
            del local_raw_front, local_raw
            failure_flags = gathered_raw_front.reshape(
                side,
                transfer_rows,
                -1,
            )[:, query_tile_rows, 0]
            failure_message = "Fold-CP diffusion local attention failed on a CP rank."
            if failure_flags.is_cuda:
                # Every rank gathered identical flags, so they enqueue the same
                # device-side assertion. Avoiding ``.item()`` here is critical:
                # this path runs in every denoise attention and a host sync would
                # serialize the whole rollout.
                torch._assert_async(
                    torch.all(failure_flags == 0),
                    failure_message,
                )
            elif bool(torch.any(failure_flags != 0).item()):
                if local_compute_error is not None:
                    raise RuntimeError(failure_message) from local_compute_error
                raise RuntimeError(failure_message)
            del failure_flags, local_compute_error
            for source_col in range(side):
                source_start, source_end = _foldcp_diffusion_query_range(
                    n_token=n_token,
                    cp_size=side,
                    cp_rank=source_col,
                )
                source_front = gathered_raw_front[
                    source_col * transfer_rows : source_col * transfer_rows
                    + query_tile_rows
                ]
                full_raw[..., source_start:source_end, :, :].copy_(
                    source_front[: source_end - source_start].movedim(0, -3)
                )
            del gathered_raw_front
            return self.attention._wrap_up(full_raw, q)

        local_raw = q_proj.new_zeros(
            *q_proj.shape[:-3],
            tile_rows,
            q_proj.shape[-3],
            q_proj.shape[-1],
        )
        if valid_rows > 0 and valid_cols > 0:
            if bias_tile is None:
                raise RuntimeError(
                    "Fold-CP diffusion attention needs a local pair-bias tile."
                )
            ring = mesh.ring_comm()
            row_bias = gather_tensor_by_ring(
                bias_tile,
                comm=ring.comm_row,
                group=mesh.group_row,
                local_index=mesh.coord[1],
                side=side,
                dim=-1,
                length=q.shape[-2],
                description="diffusion attention bias-column ring",
            )
            raw_chunk = _attention(
                q=q_proj[..., row_start:row_end, :],
                k=k_proj,
                v=v_proj,
                attn_bias=row_bias,
                use_efficient_implementation=self.attention.use_efficient_implementation,
                inplace_safe=inplace_safe,
            )
            local_raw[..., :valid_rows, :, :] = raw_chunk.transpose(-2, -3)

        full_raw = self._foldcp_gather_rows_by_col_ring(
            local_raw,
            n_token=q.shape[-2],
            mesh=mesh,
            row_dim=-3,
        )
        return self.attention._wrap_up(full_raw, q)

    @staticmethod
    def _foldcp_transpose_bias_to_query_rows(
        *,
        bias_tile: torch.Tensor,
        n_token: int,
        mesh: FoldCPProcessMesh,
        synchronize_allocations: bool = False,
        workspace: Optional[_FoldCPAttentionWorkspace] = None,
    ) -> torch.Tensor:
        """Transpose a 1xP column tile into this rank's contiguous query rows."""

        side = int(mesh.layout.shape[1])
        tile_cols = int(bias_tile.shape[-1])
        query_tile_rows = int(math.ceil(n_token / side))

        def _allocate_transpose() -> tuple[torch.Tensor, torch.Tensor]:
            send_bias_tensor = bias_tile.new_zeros(
                side,
                *bias_tile.shape[:-2],
                query_tile_rows,
                tile_cols,
            )
            for destination_col in range(side):
                destination_start, destination_end = _foldcp_diffusion_query_range(
                    n_token=n_token,
                    cp_size=side,
                    cp_rank=destination_col,
                )
                send_bias_tensor[
                    destination_col,
                    ...,
                    : destination_end - destination_start,
                    :,
                ] = bias_tile[..., destination_start:destination_end, :]
            return send_bias_tensor, torch.empty_like(send_bias_tensor)

        row_bias_workspace = None
        preparation_error: Exception | None = None
        if workspace is not None:
            (
                send_bias_tensor,
                received_bias_tensor,
                row_bias_workspace,
            ) = workspace.bias_buffers(bias_tile, n_token=n_token, mesh=mesh)
            try:
                send_bias_tensor[..., query_tile_rows, :].fill_(1)
                _zero_foldcp_drain_buffer(send_bias_tensor[..., :query_tile_rows, :])
                for destination_col in range(side):
                    destination_start, destination_end = _foldcp_diffusion_query_range(
                        n_token=n_token,
                        cp_size=side,
                        cp_rank=destination_col,
                    )
                    send_bias_tensor[
                        destination_col,
                        ...,
                        : destination_end - destination_start,
                        :,
                    ] = bias_tile[..., destination_start:destination_end, :]
            except Exception as exc:
                preparation_error = detach_rank_local_error_traceback(exc)
            if preparation_error is None:
                try:
                    send_bias_tensor[..., query_tile_rows, :].zero_()
                except Exception as exc:
                    # The source-local flag is already armed, so a failed clear
                    # can be reported through the required all-to-all itself.
                    preparation_error = detach_rank_local_error_traceback(exc)
        else:
            transpose_buffers = run_group_rank_action_synchronized(
                _allocate_transpose,
                group=mesh.group_2d,
                description="Fold-CP diffusion bias transpose allocation",
            )
            if transpose_buffers is None:  # pragma: no cover
                raise RuntimeError(
                    "Fold-CP diffusion bias transpose returned no buffers."
                )
            send_bias_tensor, received_bias_tensor = transpose_buffers
        dist.all_to_all_single(
            received_bias_tensor,
            send_bias_tensor.contiguous(),
            group=mesh.group_2d,
        )

        if workspace is not None:
            failure_flags = received_bias_tensor[..., query_tile_rows, :]
            failure_message = "Fold-CP diffusion bias preparation failed on a CP rank."
            if failure_flags.is_cuda:
                # The all-to-all gives every destination one flag row from every
                # source. Enqueue the same assertion on all ranks without a hot-
                # path host synchronization.
                torch._assert_async(
                    torch.all(failure_flags == 0),
                    failure_message,
                )
            elif bool(torch.any(failure_flags != 0).item()):
                if preparation_error is not None:
                    raise RuntimeError(failure_message) from preparation_error
                raise RuntimeError(failure_message)
            del failure_flags, preparation_error

        def _assemble_transpose() -> torch.Tensor:
            if row_bias_workspace is None:
                row_bias = torch.cat(
                    [received_bias_tensor[source_col] for source_col in range(side)],
                    dim=-1,
                )[..., :n_token].contiguous()
            else:
                row_bias = row_bias_workspace
                for source_col in range(side):
                    destination_start = source_col * tile_cols
                    destination_end = min(destination_start + tile_cols, n_token)
                    if destination_start < destination_end:
                        row_bias[..., destination_start:destination_end].copy_(
                            received_bias_tensor[source_col][
                                ...,
                                :query_tile_rows,
                                : destination_end - destination_start,
                            ]
                        )
            query_start, query_end = _foldcp_diffusion_query_range(
                n_token=n_token,
                cp_size=side,
                cp_rank=int(mesh.coord[1]),
            )
            return row_bias[..., : query_end - query_start, :]

        if workspace is not None and not synchronize_allocations:
            return _assemble_transpose()
        row_bias = run_group_rank_action_synchronized(
            _assemble_transpose,
            group=mesh.group_2d,
            description="Fold-CP diffusion bias transpose assembly",
        )
        if row_bias is None:  # pragma: no cover
            raise RuntimeError("Fold-CP diffusion bias transpose returned no result.")
        return row_bias

    def _project_attention_bias(
        self,
        z: torch.Tensor,
        extra_attn_bias: Optional[torch.Tensor] = None,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        if enable_efficient_fusion and z.shape[-3] > 0 and z.shape[-2] > 0:
            layernorm_z_weight = cast(torch.Tensor, self.layernorm_z.weight)
            weight = (self.linear_nobias_z.weight * layernorm_z_weight[None, :])[
                :, :, None, None
            ]
            bias = F.conv2d(permute_final_dims(z, [2, 0, 1]), weight)
        else:
            # conv2d rejects zero spatial dimensions. A pure-padding 1xP rank
            # still needs an empty projected bias so it can join the transpose.
            bias = self.linear_nobias_z(self.layernorm_z(z))
            bias = permute_final_dims(bias, [2, 0, 1])
        if extra_attn_bias is not None:
            while len(extra_attn_bias.shape) < len(bias.shape) - 1:
                extra_attn_bias = extra_attn_bias.unsqueeze(dim=0)
            if len(extra_attn_bias.shape) == len(bias.shape) - 1:
                extra_attn_bias = extra_attn_bias.unsqueeze(dim=-3)
            bias = bias + extra_attn_bias.to(dtype=bias.dtype, device=bias.device)
        return bias.contiguous()

    def project_foldcp_attention_bias_local(
        self,
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        extra_attn_bias: Optional[torch.Tensor] = None,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        """Project the static local pair bias once for diffusion sampling."""

        (
            row_start,
            row_end,
            col_start,
            col_end,
            valid_rows,
            valid_cols,
        ) = self._foldcp_valid_ranges(z_spec)
        z_chunk = z_local[..., :valid_rows, :valid_cols, :]
        extra_local = None
        if extra_attn_bias is not None:
            if extra_attn_bias.shape[-2:] == z_local.shape[-3:-1]:
                extra_local = extra_attn_bias[..., :valid_rows, :valid_cols]
            else:
                extra_local = extra_attn_bias[..., row_start:row_end, col_start:col_end]
        return self._project_attention_bias(
            z=z_chunk,
            extra_attn_bias=extra_local,
            enable_efficient_fusion=enable_efficient_fusion,
        )

    def local_multihead_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
        n_queries: int = 32,
        n_keys: int = 128,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Used by Algorithm 24, with beta_ij being the local mask. Used in AtomTransformer.

        Args:
            q (torch.Tensor): query embedding
                [..., N_atom, c_a]
            kv (torch.Tensor): key/value embedding
                [..., N_atom, c_a]
            z (torch.Tensor): atom-atom pair embedding, in trunked dense shape. Used for computing pair bias.
                [..., n_blocks, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. Defaults to 32.
            n_keys (int, optional): local window size of key tensor. Defaults to 128.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_atom, c_a]
        """

        assert n_queries == z.size(-3)
        assert n_keys == z.size(-2)
        assert len(z.shape) == len(q.shape) + 2

        # Multi-head attention bias
        bias = self.linear_nobias_z(
            self.layernorm_z(z)
        )  # [..., n_blocks, n_queries, n_keys, n_heads]
        bias = permute_final_dims(
            bias, [3, 0, 1, 2]
        )  # [..., n_heads, n_blocks, n_queries, n_keys]
        bias = self._align_bias_to_query(bias, q, n_pair_dims=3)

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)
        q = self.attention(
            q_x=q,
            kv_x=kv,
            trunked_attn_bias=bias,
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        return q

    def local_multihead_attention_foldcp_window(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z_local: torch.Tensor,
        window_spec: FoldCPWindowShardSpec,
        mesh: FoldCPProcessMesh,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        def _prepare_window_attention():
            q_proj, k_proj, v_proj = self.attention._prep_qkv(
                q_x=q,
                kv_x=kv,
                apply_scale=True,
            )
            q_blocks, _, pad_info = rearrange_qk_to_dense_trunk(
                q=q,
                k=kv,
                dim_q=-2,
                dim_k=-2,
                n_queries=window_spec.n_queries,
                n_keys=window_spec.n_keys,
                compute_mask=True,
            )
            q_proj_blocks, kv_proj_blocks, _ = rearrange_qk_to_dense_trunk(
                q=q_proj,
                k=[k_proj, v_proj],
                dim_q=-2,
                dim_k=[-2, -2],
                n_queries=window_spec.n_queries,
                n_keys=window_spec.n_keys,
                compute_mask=False,
            )
            block_start, block_end = window_spec.block_range
            blocks_per_rank = block_end - block_start
            valid_end = min(block_end, window_spec.n_windows)
            local_blocks = q.new_zeros(
                *q_blocks.shape[:-3],
                blocks_per_rank,
                window_spec.n_queries,
                self.attention.num_heads,
                self.attention.c_hidden,
            )
            return (
                q_proj_blocks,
                kv_proj_blocks,
                pad_info,
                block_start,
                valid_end,
                local_blocks,
            )

        prepared = (
            run_group_rank_action_synchronized(
                _prepare_window_attention,
                group=mesh.group_2d,
                description="Fold-CP atom-window attention preparation",
            )
            if int(mesh.layout.shape[0]) == 1 and int(mesh.layout.shape[1]) > 1
            else _prepare_window_attention()
        )
        if prepared is None:  # pragma: no cover
            raise RuntimeError("Fold-CP atom-window attention was not prepared.")
        (
            q_proj_blocks,
            kv_proj_blocks,
            pad_info,
            block_start,
            valid_end,
            local_blocks,
        ) = prepared
        local_compute_error: Exception | None = None
        if block_start < valid_end:
            try:
                valid_blocks = valid_end - block_start
                block_slice = slice(block_start, valid_end)
                q_proj_local = q_proj_blocks[..., block_slice, :, :]
                k_proj_local = kv_proj_blocks[0][..., block_slice, :, :]
                v_proj_local = kv_proj_blocks[1][..., block_slice, :, :]

                z_valid = z_local[..., :valid_blocks, :, :, :]
                bias = self.linear_nobias_z(self.layernorm_z(z_valid))
                bias = permute_final_dims(bias, [3, 0, 1, 2])
                while bias.dim() < q_proj_local.dim():
                    bias = bias.unsqueeze(dim=0)

                mask = pad_info["mask_trunked"][..., block_slice, :, :]
                attn_bias = q_proj_local.new_zeros(
                    q_proj_local.shape[:-1] + (window_spec.n_keys,)
                )
                while mask.dim() < attn_bias.dim():
                    mask = mask.unsqueeze(dim=0)
                attn_bias = attn_bias.masked_fill(~mask, -1e10)
                attn_bias = attn_bias + bias.to(
                    dtype=attn_bias.dtype, device=attn_bias.device
                )

                out = _attention(
                    q=q_proj_local,
                    k=k_proj_local,
                    v=v_proj_local,
                    attn_bias=attn_bias,
                    use_efficient_implementation=(
                        self.attention.use_efficient_implementation
                    ),
                    inplace_safe=inplace_safe,
                )
                out = out.movedim(-4, -2).contiguous()
                local_blocks[..., :valid_blocks, :, :, :] = out
            except Exception as exc:
                # The zero-filled local gather source already exists. Retain a
                # rank-local attention failure while every rank drains the same
                # window ring, then propagate it at the common completion point.
                local_compute_error = detach_rank_local_error_traceback(exc)
        full_blocks = gather_window_blocks(
            local_blocks,
            window_spec,
            mesh.group_2d,
            block_dim=-4,
        )

        def _finish_window_attention() -> torch.Tensor:
            if local_compute_error is not None:
                raise local_compute_error
            full_out = full_blocks.reshape(
                *full_blocks.shape[:-4],
                full_blocks.shape[-4] * full_blocks.shape[-3],
                full_blocks.shape[-2],
                full_blocks.shape[-1],
            )
            if window_spec.q_pad > 0:
                full_out = full_out[..., : -window_spec.q_pad, :, :]
            return self.attention._wrap_up(full_out, q)

        result = (
            run_group_rank_action_synchronized(
                _finish_window_attention,
                group=mesh.group_2d,
                description="Fold-CP atom-window attention completion",
            )
            if int(mesh.layout.shape[0]) == 1 and int(mesh.layout.shape[1]) > 1
            else _finish_window_attention()
        )
        if result is None:  # pragma: no cover
            raise RuntimeError("Fold-CP atom-window attention returned no result.")
        return result

    def standard_multihead_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
        extra_attn_bias: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        """Used by Algorithm 7/20

        Args:
            q (torch.Tensor): the query embedding
                [..., N_token, c_a]
            kv (torch.Tensor): the key/value embedding
                [..., N_token, c_a]
            z (torch.Tensor): pair embedding, used for computing pair bias.
                [..., N_token, N_token, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            enable_efficient_fusion (bool): Whether to enable efficient fusion of bias calculation in attention to speed up. Defaults to False. (Alg 24)

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_token, c_a]
        """

        row_chunk_size = self._foldcp_diffusion_bias_row_chunk_size()
        if (
            row_chunk_size > 0
            and not enable_efficient_fusion
            and z.shape[-3] > row_chunk_size
        ):
            return self._standard_multihead_attention_stream_pair_bias(
                q=q,
                kv=kv,
                z=z,
                extra_attn_bias=extra_attn_bias,
                inplace_safe=inplace_safe,
                row_chunk_size=row_chunk_size,
            )

        # Multi-head attention bias
        if enable_efficient_fusion:
            layernorm_z_weight = cast(torch.Tensor, self.layernorm_z.weight)
            weight = (self.linear_nobias_z.weight * layernorm_z_weight[None, :])[
                :, :, None, None
            ]
            bias = F.conv2d(z, weight)
        else:
            bias = self.linear_nobias_z(self.layernorm_z(z))
            bias = permute_final_dims(
                bias, [2, 0, 1]
            )  # [..., n_heads, N_token, N_token]
        if extra_attn_bias is not None:
            while len(extra_attn_bias.shape) < len(bias.shape) - 1:
                extra_attn_bias = extra_attn_bias.unsqueeze(dim=0)
            if len(extra_attn_bias.shape) == len(bias.shape) - 1:
                extra_attn_bias = extra_attn_bias.unsqueeze(dim=-3)
            bias = bias + extra_attn_bias.to(dtype=bias.dtype, device=bias.device)
        bias = self._align_bias_to_query(bias, q, n_pair_dims=2)

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)
        q = self.attention(q_x=q, kv_x=kv, attn_bias=bias, inplace_safe=inplace_safe)

        return q

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        enable_efficient_fusion: bool = False,
        extra_attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Details are given in local_forward and standard_forward"""
        # Input projections
        if self.has_s:
            a = self.layernorm_a(a=a, s=s)
        else:
            a = self.layernorm_a(a)

        if self.cross_attention_mode:
            if self.has_s:
                kv = self.layernorm_kv(a=a, s=s)
            else:
                kv = self.layernorm_kv(a)
        else:
            kv = a

        # Multihead attention with pair bias
        if n_queries and n_keys:
            a = self.local_multihead_attention(
                a,
                kv,
                z,
                n_queries,
                n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        else:
            a = self.standard_multihead_attention(
                a,
                kv,
                z,
                extra_attn_bias=extra_attn_bias,
                inplace_safe=inplace_safe,
                enable_efficient_fusion=enable_efficient_fusion,
            )

        # Output projection (from adaLN-Zero [27])
        if self.has_s:
            if inplace_safe:
                a *= torch.sigmoid(self.linear_a_last(s))
            else:
                a = torch.sigmoid(self.linear_a_last(s)) * a

        return a

    def forward_foldcp_local_z(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        mesh: FoldCPProcessMesh,
        inplace_safe: bool = False,
        extra_attn_bias: Optional[torch.Tensor] = None,
        enable_efficient_fusion: bool = False,
        projected_bias_local: Optional[
            torch.Tensor | FoldCPQueryOwnedAttentionBias
        ] = None,
        workspace: Optional[_FoldCPAttentionWorkspace] = None,
    ) -> torch.Tensor:
        def _prepare_attention_state():
            if self.has_s:
                prepared_a = self.layernorm_a(a=a, s=s)
            else:
                prepared_a = self.layernorm_a(a)

            if self.cross_attention_mode:
                if self.has_s:
                    kv = self.layernorm_kv(a=prepared_a, s=s)
                else:
                    kv = self.layernorm_kv(prepared_a)
            else:
                kv = prepared_a
            return prepared_a, kv

        prepared = (
            run_group_rank_action_synchronized(
                _prepare_attention_state,
                group=mesh.group_2d,
                description="Fold-CP diffusion attention state preparation",
            )
            if int(mesh.layout.shape[0]) == 1 and int(mesh.layout.shape[1]) > 1
            else _prepare_attention_state()
        )
        if prepared is None:  # pragma: no cover
            raise RuntimeError("Fold-CP diffusion attention state was not prepared.")
        a, kv = prepared

        a = self.standard_multihead_attention_foldcp_local_z(
            q=a,
            kv=kv,
            z_local=z_local,
            z_spec=z_spec,
            mesh=mesh,
            extra_attn_bias=extra_attn_bias,
            inplace_safe=inplace_safe,
            enable_efficient_fusion=enable_efficient_fusion,
            projected_bias_local=projected_bias_local,
            workspace=workspace,
        )

        if self.has_s:
            if inplace_safe:
                a *= torch.sigmoid(self.linear_a_last(s))
            else:
                a = torch.sigmoid(self.linear_a_last(s)) * a
        return a

    def forward_foldcp_window(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z_local: torch.Tensor,
        window_spec: FoldCPWindowShardSpec,
        mesh: FoldCPProcessMesh,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        def _prepare_window_attention_state():
            if self.has_s:
                prepared_a = self.layernorm_a(a=a, s=s)
            else:
                prepared_a = self.layernorm_a(a)

            if self.cross_attention_mode:
                if self.has_s:
                    kv = self.layernorm_kv(a=prepared_a, s=s)
                else:
                    kv = self.layernorm_kv(prepared_a)
            else:
                kv = prepared_a
            return prepared_a, kv

        prepared = (
            run_group_rank_action_synchronized(
                _prepare_window_attention_state,
                group=mesh.group_2d,
                description="Fold-CP atom-window attention state preparation",
            )
            if int(mesh.layout.shape[0]) == 1 and int(mesh.layout.shape[1]) > 1
            else _prepare_window_attention_state()
        )
        if prepared is None:  # pragma: no cover
            raise RuntimeError("Fold-CP atom-window attention state was not prepared.")
        a, kv = prepared

        a = self.local_multihead_attention_foldcp_window(
            q=a,
            kv=kv,
            z_local=z_local,
            window_spec=window_spec,
            mesh=mesh,
            inplace_safe=inplace_safe,
        )

        if self.has_s:
            if inplace_safe:
                a *= torch.sigmoid(self.linear_a_last(s))
            else:
                a = torch.sigmoid(self.linear_a_last(s)) * a
        return a


class DiffusionTransformerBlock(nn.Module):
    """
    Implements Algorithm 23[Line2-Line3] in AF3

    Args:
        c_a (int): single embedding dimension.
        c_s (int): single embedding dimension.
        c_z (int): pair embedding dimension.
        n_heads (int): number of heads for DiffusionTransformerBlock.
        biasinit (float, optional): bias initialization value. Defaults to -2.0.
        cross_attention_mode (bool, optional): whether to use cross attention. Defaults to False.
    """

    def __init__(
        self,
        c_a: int,  # could be 128 or 768 in AF3
        c_s: int,  # could be c_s or c_atom
        c_z: int,  # could be c_z or c_atompair
        n_heads: int,  # could be 16 or 4 or ... in AF3
        biasinit: float = -2.0,
        cross_attention_mode: bool = False,
    ) -> None:
        super(DiffusionTransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.attention_pair_bias = AttentionPairBias(
            has_s=True,
            create_offset_ln_z=False,
            n_heads=n_heads,
            c_a=c_a,
            c_s=c_s,
            c_z=c_z,
            biasinit=biasinit,
            cross_attention_mode=cross_attention_mode,
        )
        self.conditioned_transition_block = ConditionedTransitionBlock(
            n=2, c_a=c_a, c_s=c_s, biasinit=biasinit
        )
        self.residual_path = nn.Identity()

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        enable_efficient_fusion: bool = False,
        extra_attn_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N, c_a]
            s (torch.Tensor): single embedding
                [..., N, c_s]
            z (torch.Tensor): pair embedding
                [..., N, N, c_z] or [..., n_block, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to None.
            n_keys (int, optional): local window size of key tensor. Defaults to None.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            enable_efficient_fusion (bool): Whether to enable efficient fusion of bias calculation in attention to speed up. Defaults to False. (Alg 24)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - out_a: the output of DiffusionTransformerBlock [..., N, c_a]
                - s: the single embedding [..., N, c_s]
                - z: the pair embedding
        """
        attn_out = self.residual_path(
            self.attention_pair_bias(
                a=a,
                s=s,
                z=z,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                enable_efficient_fusion=enable_efficient_fusion,
                extra_attn_bias=extra_attn_bias,
            )
        )
        if inplace_safe:
            attn_out += a
        else:
            attn_out = attn_out + a
        ff_out = self.residual_path(self.conditioned_transition_block(a=attn_out, s=s))
        out_a = ff_out + attn_out
        # Avoid s/z to be deleted by torch.utils.checkpoint
        return out_a, s, z

    def forward_foldcp_local_z(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        mesh: FoldCPProcessMesh,
        inplace_safe: bool = False,
        extra_attn_bias: Optional[torch.Tensor] = None,
        enable_efficient_fusion: bool = False,
        projected_bias_local: Optional[
            torch.Tensor | FoldCPQueryOwnedAttentionBias
        ] = None,
        workspace: Optional[_FoldCPAttentionWorkspace] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out = self.residual_path(
            self.attention_pair_bias.forward_foldcp_local_z(
                a=a,
                s=s,
                z_local=z_local,
                z_spec=z_spec,
                mesh=mesh,
                extra_attn_bias=extra_attn_bias,
                inplace_safe=inplace_safe,
                enable_efficient_fusion=enable_efficient_fusion,
                projected_bias_local=projected_bias_local,
                workspace=workspace,
            )
        )
        if inplace_safe:
            attn_out += a
        else:
            attn_out = attn_out + a
        ff_out = self.residual_path(self.conditioned_transition_block(a=attn_out, s=s))
        out_a = ff_out + attn_out
        return out_a, s, z_local

    def forward_foldcp_window(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z_local: torch.Tensor,
        window_spec: FoldCPWindowShardSpec,
        mesh: FoldCPProcessMesh,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_out = self.residual_path(
            self.attention_pair_bias.forward_foldcp_window(
                a=a,
                s=s,
                z_local=z_local,
                window_spec=window_spec,
                mesh=mesh,
                inplace_safe=inplace_safe,
            )
        )
        if inplace_safe:
            attn_out += a
        else:
            attn_out = attn_out + a
        ff_out = self.residual_path(self.conditioned_transition_block(a=attn_out, s=s))
        out_a = ff_out + attn_out
        return out_a, s, z_local


class DiffusionTransformer(nn.Module):
    """
    Implements Algorithm 23 in AF3

    Args:
        c_a (int): single embedding dimension.
        c_s (int): single embedding dimension.
        c_z (int): pair embedding dimension.
        n_blocks (int): number of blocks in DiffusionTransformer.
        n_heads (int): number of heads in attention.
        cross_attention_mode (bool, optional): whether to use cross attention. Defaults to False.
        blocks_per_ckpt (int, optional): number of DiffusionTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        c_a: int,  # could be 128 or 768 in AF3
        c_s: int,  # could be c_s or c_atom
        c_z: int,  # could be c_z or c_atompair
        n_blocks: int,  # could be 3 or 24 in AF3
        n_heads: int,  # could be 16 or 4 or ... in AF3
        cross_attention_mode: bool = False,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(DiffusionTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.blocks_per_ckpt = blocks_per_ckpt
        self._foldcp_attention_workspace: Optional[_FoldCPAttentionWorkspace] = None

        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            block = DiffusionTransformerBlock(
                n_heads=n_heads,
                c_a=c_a,
                c_s=c_s,
                c_z=c_z,
                cross_attention_mode=cross_attention_mode,
            )
            self.blocks.append(block)

    def clear_foldcp_attention_workspace(self) -> None:
        """Release rollout-scoped collective buffers between inference jobs."""

        self._foldcp_attention_workspace = None

    def _prep_blocks(
        self,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        enable_efficient_fusion: bool = False,
        extra_attn_bias: Optional[torch.Tensor] = None,
    ) -> list[Any]:
        blocks = [
            partial(
                b,
                n_queries=n_queries,
                n_keys=n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                enable_efficient_fusion=enable_efficient_fusion,
                extra_attn_bias=extra_attn_bias,
            )
            for b in self.blocks
        ]
        return blocks

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        enable_efficient_fusion: bool = False,
        extra_attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
                Args:
                    a (torch.Tensor): the single feature aggregate per-atom representation
                        [..., N, c_a]
                    s (torch.Tensor): single embedding
                        [..., N, c_s]
                    z (torch.Tensor): pair embedding
                        [..., N, N, c_z]
                    n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to None.
                    n_keys (int, optional): local window size of key tensor. Defaults to None.
        enable_efficient_fusion (bool): Whether to enable efficient fusion of bias calculation in attention to speed up. Defaults to False. (Alg 24)

                Returns:
                    torch.Tensor: the output of DiffusionTransformer
                        [..., N, c_a]
        """
        blocks = self._prep_blocks(
            n_queries=n_queries,
            n_keys=n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            enable_efficient_fusion=enable_efficient_fusion,
            extra_attn_bias=extra_attn_bias,
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        a, s, z = checkpoint_blocks(
            blocks, args=(a, s, z), blocks_per_ckpt=blocks_per_ckpt
        )
        del s, z
        return a

    def forward_foldcp_local_z(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        mesh: FoldCPProcessMesh,
        inplace_safe: bool = False,
        extra_attn_bias: Optional[torch.Tensor] = None,
        enable_efficient_fusion: bool = False,
        projected_bias_local: Optional[
            list[torch.Tensor | FoldCPQueryOwnedAttentionBias]
        ] = None,
    ) -> torch.Tensor:
        if projected_bias_local is not None and len(projected_bias_local) != len(
            self.blocks
        ):
            raise ValueError(
                "Fold-CP projected bias cache must contain one tensor per block."
            )
        workspace = None
        if (
            not torch.is_grad_enabled()
            and mesh.layout.shape[0] == 1
            and mesh.layout.shape[1] > 1
        ):
            if self._foldcp_attention_workspace is None:
                self._foldcp_attention_workspace = _FoldCPAttentionWorkspace()
            workspace = self._foldcp_attention_workspace
        for block_idx, block in enumerate(self.blocks):
            block_result = None
            block_error: Exception | None = None
            try:
                block_result = block.forward_foldcp_local_z(
                    a=a,
                    s=s,
                    z_local=z_local,
                    z_spec=z_spec,
                    mesh=mesh,
                    inplace_safe=inplace_safe,
                    extra_attn_bias=extra_attn_bias,
                    enable_efficient_fusion=enable_efficient_fusion,
                    projected_bias_local=(
                        None
                        if projected_bias_local is None
                        else projected_bias_local[block_idx]
                    ),
                    workspace=workspace,
                )
            except Exception as exc:
                # Data collectives inside the block already drain their own
                # schedules. Retain a failure from the communication-free tail
                # (attention wrap-up / final gate) so healthy ranks cannot enter
                # the next block's gather alone.
                block_error = detach_rank_local_error_traceback(exc)

            if int(mesh.layout.shape[0]) == 1 and int(mesh.layout.shape[1]) > 1:

                def _finish_block():
                    if block_error is not None:
                        raise block_error
                    return block_result

                block_result = run_group_rank_action_synchronized(
                    _finish_block,
                    group=mesh.group_2d,
                    description=f"Fold-CP diffusion block {block_idx} completion",
                )
            elif block_error is not None:
                raise block_error

            if block_result is None:  # pragma: no cover
                raise RuntimeError(
                    f"Fold-CP diffusion block {block_idx} returned no result."
                )
            a, s, z_local = block_result
        del s, z_local
        return a

    def prepare_foldcp_attention_bias_cache(
        self,
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        mesh: FoldCPProcessMesh,
        extra_attn_bias: Optional[torch.Tensor] = None,
        enable_efficient_fusion: bool = False,
    ) -> Optional[list[torch.Tensor | FoldCPQueryOwnedAttentionBias]]:
        """Build per-block local pair-bias projections shared by denoise steps.

        Returns None when the resident cache would exceed its memory budget; the
        caller then leaves the per-step projection in place.
        """

        if not self.blocks:
            return []
        (
            row_start,
            row_end,
            col_start,
            col_end,
            valid_rows,
            valid_cols,
        ) = self.blocks[0].attention_pair_bias._foldcp_valid_ranges(z_spec)
        n_token = int(z_spec.original_shape[z_spec.pair_dims[1]])
        tile_cols = int(z_spec.local_shape[z_spec.pair_dims[1]])
        is_one_by_p = mesh.layout.shape[0] == 1 and mesh.layout.shape[1] > 1
        # The 1xP transpose turns each column tile into contiguous query rows,
        # so the resident tile is [H, ceil(N / P), N] instead of [H, rows, cols].
        mesh_cols = int(mesh.layout.shape[1])
        bias_rows = (
            (n_token + mesh_cols - 1) // mesh_cols if is_one_by_p else valid_rows
        )
        bias_cols = n_token if is_one_by_p else valid_cols
        n_heads = self.blocks[0].attention_pair_bias.n_heads
        if not foldcp_diffusion_bias_cache_is_safe(
            n_blocks=len(self.blocks),
            n_heads=n_heads,
            bias_rows=bias_rows,
            bias_cols=bias_cols,
            element_size=z_local.element_size(),
        ):
            resident_gib = (
                len(self.blocks)
                * n_heads
                * bias_rows
                * bias_cols
                * z_local.element_size()
                / 1024**3
            )
            # Never degrade silently: without this line the sampling loop simply
            # runs several times slower and still looks healthy.
            logger.warning(
                "Fold-CP diffusion pair-bias resident cache disabled: %.2f GiB "
                "of resident storage needed for %d blocks at n_token=%d, "
                "mesh=%dx%d, but the resident budget is %.2f GiB. Each denoise "
                "step will reproject the bias. Raise "
                "OPENDDE_FOLDCP_DIFFUSION_BIAS_CACHE_MAX_BYTES to re-enable it.",
                resident_gib,
                len(self.blocks),
                n_token,
                mesh.layout.shape[0],
                mesh.layout.shape[1],
                _foldcp_diffusion_bias_cache_max_bytes() / 1024**3,
            )
            return None

        if not is_one_by_p:
            return [
                block.attention_pair_bias.project_foldcp_attention_bias_local(
                    z_local=z_local,
                    z_spec=z_spec,
                    extra_attn_bias=extra_attn_bias,
                    enable_efficient_fusion=enable_efficient_fusion,
                )
                for block in self.blocks
            ]

        attention_pair_bias = self.blocks[0].attention_pair_bias
        query_start, query_end = _foldcp_diffusion_query_range(
            n_token=n_token,
            cp_size=mesh_cols,
            cp_rank=int(mesh.coord[1]),
        )

        # Every block projects the same query-owned pair source. Exchange each
        # source rank's local columns only with the destination that owns those
        # query rows, then build all block caches locally. This avoids both the
        # full N x N source gather and 24 projected-bias transposes. Padding and
        # optional-bias staging stay in the synchronized allocation boundary.
        def _allocate_shared_source():
            return _prepare_foldcp_diffusion_bias_cache_source(
                z_local,
                extra_attn_bias,
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                valid_rows=valid_rows,
                valid_cols=valid_cols,
                tile_cols=tile_cols,
                mesh_cols=mesh_cols,
            )

        shared_source = run_group_rank_action_synchronized(
            _allocate_shared_source,
            group=mesh.group_2d,
            description="Fold-CP diffusion bias-cache source allocation",
        )
        if shared_source is None:  # pragma: no cover - every rank runs the action
            raise RuntimeError("Fold-CP diffusion bias cache returned no source.")
        (
            local_cols_front,
            gathered_cols_front,
            packed_extra_bias,
            extra_local,
        ) = shared_source
        # The synchronized result tuple owns the same source buffers as the
        # unpacked locals.  Drop it before the allocation-heavy projection;
        # otherwise the completed full gather remains live even after its
        # named aliases are released below.
        del shared_source
        del _allocate_shared_source
        dist.all_to_all_single(
            gathered_cols_front,
            local_cols_front,
            group=mesh.group_2d,
        )
        del local_cols_front

        def _extract_projection_source(
            gathered_cols_front: torch.Tensor = gathered_cols_front,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            # ``gathered_cols_front`` stores global columns first and already
            # contains only this rank's query rows. Move the global-column
            # dimension into the historical projection layout
            # [..., N, ceil(N/P), C] without changing launch size or ordering.
            projection_shared_pair = gathered_cols_front.movedim(0, -3)[
                ..., :n_token, : query_end - query_start, :
            ]
            valid_query_rows = query_end - query_start

            def _pad_projection_rows(value: torch.Tensor) -> torch.Tensor:
                if valid_query_rows == tile_cols:
                    return value.contiguous()
                padded = value.new_zeros(*value.shape[:-2], tile_cols, value.shape[-1])
                padded[..., :valid_query_rows, :] = value
                return padded

            if packed_extra_bias:
                return (
                    _pad_projection_rows(projection_shared_pair[..., :-1]),
                    _pad_projection_rows(
                        projection_shared_pair[..., -1:].contiguous()
                    ).squeeze(-1),
                )
            return _pad_projection_rows(projection_shared_pair), None

        projection_source = run_group_rank_action_synchronized(
            _extract_projection_source,
            group=mesh.group_2d,
            description="Fold-CP diffusion bias-cache source assembly",
        )
        if projection_source is None:  # pragma: no cover
            raise RuntimeError(
                "Fold-CP diffusion bias cache returned no projection source."
            )
        projection_z, projection_extra_bias = projection_source
        del projection_source
        del _extract_projection_source, gathered_cols_front
        if extra_local is not None and not packed_extra_bias:
            query_owned_extra_bias = (
                attention_pair_bias._foldcp_transpose_bias_to_query_rows(
                    bias_tile=extra_local,
                    n_token=n_token,
                    mesh=mesh,
                    synchronize_allocations=True,
                )
            )
            projection_extra_bias = query_owned_extra_bias.new_zeros(
                *query_owned_extra_bias.shape[:-2], n_token, tile_cols
            )
            projection_extra_bias[..., : query_end - query_start] = (
                query_owned_extra_bias.transpose(-2, -1)
            )
            del query_owned_extra_bias
        del extra_local

        # The source already has the historical [N, ceil(N/P)] spatial shape.
        # This is the final allocation-heavy phase before ranks re-enter model
        # collectives, so propagate any local failure before returning.
        def _build_query_owned_biases() -> list[FoldCPQueryOwnedAttentionBias]:
            query_owned_biases = []
            for block in self.blocks:
                projected_bias = block.attention_pair_bias._project_attention_bias(
                    z=projection_z,
                    extra_attn_bias=projection_extra_bias,
                    enable_efficient_fusion=enable_efficient_fusion,
                )
                query_owned_biases.append(
                    FoldCPQueryOwnedAttentionBias(
                        projected_bias.transpose(-2, -1)[
                            ..., : query_end - query_start, :
                        ].contiguous()
                    )
                )
                del projected_bias
            return query_owned_biases

        query_owned_biases = run_group_rank_action_synchronized(
            _build_query_owned_biases,
            group=mesh.group_2d,
            description="Fold-CP diffusion bias-cache projection",
        )
        if query_owned_biases is None:  # pragma: no cover
            raise RuntimeError("Fold-CP diffusion bias cache returned no projections.")
        return query_owned_biases

    def forward_foldcp_window(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z_local: torch.Tensor,
        window_spec: FoldCPWindowShardSpec,
        mesh: FoldCPProcessMesh,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        for block_idx, block in enumerate(self.blocks):
            block_result = None
            block_error: Exception | None = None
            try:
                block_result = block.forward_foldcp_window(
                    a=a,
                    s=s,
                    z_local=z_local,
                    window_spec=window_spec,
                    mesh=mesh,
                    inplace_safe=inplace_safe,
                )
            except Exception as exc:
                block_error = detach_rank_local_error_traceback(exc)

            if int(mesh.layout.shape[0]) == 1 and int(mesh.layout.shape[1]) > 1:

                def _finish_window_block():
                    if block_error is not None:
                        raise block_error
                    return block_result

                block_result = run_group_rank_action_synchronized(
                    _finish_window_block,
                    group=mesh.group_2d,
                    description=(
                        f"Fold-CP atom-window transformer block {block_idx} completion"
                    ),
                )
            elif block_error is not None:
                raise block_error

            if block_result is None:  # pragma: no cover
                raise RuntimeError(
                    f"Fold-CP atom-window transformer block {block_idx} "
                    "returned no result."
                )
            a, s, z_local = block_result
        del s, z_local
        return a


class AtomTransformer(nn.Module):
    """
    Implements Algorithm 7 in AF3

    Performs local transformer among atom embeddings, with bias predicted from atom pair embeddings

    Args:
        c_atom (int, optional): embedding dim for atom feature. Defaults to 128.
        c_atompair (int, optional): embedding dim for atompair feature. Defaults to 16.
        n_blocks (int, optional): number of block in AtomTransformer. Defaults to 3.
        n_heads (int, optional): number of heads in attention. Defaults to 4.
        n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to 32.
        n_keys (int, optional): local window size of key tensor. Defaults to 128.
        blocks_per_ckpt (int, optional): number of AtomTransformer/DiffusionTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(AtomTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.diffusion_transformer = DiffusionTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_a=c_atom,
            c_s=c_atom,
            c_z=c_atompair,
            cross_attention_mode=True,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        p: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            c (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            p (torch.Tensor): atompair embedding in dense block shape.
                [..., n_blocks, n_queries, n_keys, c_atompair]

        Returns:
            torch.Tensor: the output of AtomTransformer
                [..., N_atom, c_atom]
        """
        n_blocks, n_queries, n_keys = p.shape[-4:-1]

        assert n_queries == self.n_queries
        assert n_keys == self.n_keys
        return self.diffusion_transformer(
            a=q,
            s=c,
            z=p,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

    def forward_foldcp_window(
        self,
        q: torch.Tensor,
        c: torch.Tensor,
        p_local: torch.Tensor,
        window_spec: FoldCPWindowShardSpec,
        mesh: FoldCPProcessMesh,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        assert window_spec.n_queries == self.n_queries
        assert window_spec.n_keys == self.n_keys
        return self.diffusion_transformer.forward_foldcp_window(
            a=q,
            s=c,
            z_local=p_local,
            window_spec=window_spec,
            mesh=mesh,
            inplace_safe=inplace_safe,
        )


class ConditionedTransitionBlock(nn.Module):
    """
    Implements Algorithm 25 in AF3

    Args:
        c_a (int): single embedding dim (single feature aggregated atom info).
        c_s (int):  single embedding dim.
        n (int, optional): channel scale factor. Defaults to 2.
        biasinit (float, optional): bias initialization value. Defaults to -2.0.
    """

    def __init__(self, c_a: int, c_s: int, n: int = 2, biasinit: float = -2.0) -> None:
        super(ConditionedTransitionBlock, self).__init__()
        self.c_a = c_a
        self.c_s = c_s
        self.n = n
        self.adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        self.linear_nobias_a1 = LinearNoBias(
            in_features=c_a, out_features=n * c_a, initializer="relu"
        )
        self.linear_nobias_a2 = LinearNoBias(
            in_features=c_a, out_features=n * c_a, initializer="relu"
        )
        self.linear_nobias_b = LinearNoBias(in_features=n * c_a, out_features=c_a)
        self.linear_s = BiasInitLinear(
            in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
        )

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N, c_a]
            s (torch.Tensor): single embedding
                [..., N, c_s]

        Returns:
            torch.Tensor: the updated a from ConditionedTransitionBlock
                [..., N, c_a]
        """
        a = self.adaln(a, s)
        b = F.silu((self.linear_nobias_a1(a))) * self.linear_nobias_a2(a)
        # Output projection (from adaLN-Zero [27])
        a = torch.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)
        return a


class AtomAttentionEncoder(nn.Module):
    """
    Implements Algorithm 5 in AF3

    Args:
        has_coords (bool): whether the module input will contains coordinates (r_l).
        c_token (int): token embedding dim.
        c_atom (int, optional): atom embedding dim. Defaults to 128.
        c_atompair (int, optional): atompair embedding dim. Defaults to 16.
        c_s (int, optional):  single embedding dim. Defaults to 384.
        c_z (int, optional): pair embedding dim. Defaults to 128.
        n_blocks (int, optional): number of blocks in AtomTransformer. Defaults to 3.
        n_heads (int, optional): number of heads in AtomTransformer. Defaults to 4.
        n_queries (int, optional): local window size of query tensor. Defaults to 32.
        n_keys (int, optional): local window size of key tensor. Defaults to 128.
        blocks_per_ckpt (int, optional): number of AtomAttentionEncoder/AtomTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        has_coords: bool,
        c_token: int,  # 384 or 768
        c_atom: int = 128,
        c_atompair: int = 16,
        c_s: int = 384,
        c_z: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(AtomAttentionEncoder, self).__init__()
        self.has_coords = has_coords
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.input_feature = {
            # "ref_pos": 3,
            # "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }
        self.linear_no_bias_ref_pos = LinearNoBias(
            in_features=3, out_features=self.c_atom, precision=torch.float32
        )  # use high precision for ref_pos
        self.linear_no_bias_ref_charge = LinearNoBias(
            in_features=1, out_features=self.c_atom
        )
        self.linear_no_bias_f = LinearNoBias(
            in_features=sum(self.input_feature.values()), out_features=self.c_atom
        )
        self.linear_no_bias_d = LinearNoBias(
            in_features=3, out_features=self.c_atompair, precision=torch.float32
        )
        self.linear_no_bias_invd = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )
        self.linear_no_bias_v = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )

        if self.has_coords:
            # Line9
            self.layernorm_s = LayerNorm(self.c_s, create_offset=False)
            self.linear_no_bias_s = LinearNoBias(
                in_features=self.c_s,
                out_features=self.c_atom,
                initializer="zeros",
                precision=torch.float32,
            )
            # Line10
            self.layernorm_z = LayerNorm(
                self.c_z, create_offset=False
            )  # memory bottleneck
            self.linear_no_bias_z = LinearNoBias(
                in_features=self.c_z,
                out_features=self.c_atompair,
                initializer="zeros",
                precision=torch.float32,
            )
            # Line11
            self.linear_no_bias_r = LinearNoBias(
                in_features=3, out_features=self.c_atom, precision=torch.float32
            )
        self.linear_no_bias_cl = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.linear_no_bias_cm = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.small_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair,
                out_features=self.c_atompair,
                initializer="relu",
            ),
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair,
                out_features=self.c_atompair,
                initializer="relu",
            ),
            nn.ReLU(),
            LinearNoBias(
                in_features=self.c_atompair,
                out_features=self.c_atompair,
                initializer="zeros",
            ),
        )
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.linear_no_bias_q = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_token
        )

    def _add_token_pair_context_to_atom_pair(
        self,
        p_lm: torch.Tensor,
        z: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Add token-pair trunk context to atom local-window pair features.

        This is the same computation as
        ``p_lm + Linear(LayerNorm(broadcast_token_to_local_atom_pair(z)))``, but
        the gather/layernorm/projection is streamed over atom-window blocks so a
        full ``[n_blocks, n_queries, n_keys, c_z]`` temporary is not kept alive.
        """

        window_chunk_size = 64

        atom_to_token_idx_q, atom_to_token_idx_k, _ = rearrange_qk_to_dense_trunk(
            atom_to_token_idx,
            atom_to_token_idx,
            dim_q=-1,
            dim_k=-1,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=False,
        )
        p_lm = p_lm.unsqueeze(dim=-5)
        n_windows = atom_to_token_idx_q.shape[0]
        for start in range(0, n_windows, window_chunk_size):
            end = min(start + window_chunk_size, n_windows)
            z_token_pair = gather_pair_embedding_in_dense_trunk(
                z,
                idx_q=atom_to_token_idx_q[start:end],
                idx_k=atom_to_token_idx_k[start:end],
            )
            z_token_pair = self.linear_no_bias_z(self.layernorm_z(z_token_pair))
            if z_token_pair.dim() == p_lm.dim() - 1:
                z_token_pair = z_token_pair.unsqueeze(dim=-5)
            target = [slice(None)] * p_lm.dim()
            target[-4] = slice(start, end)
            p_lm[tuple(target)] = p_lm[tuple(target)] + z_token_pair
            del z_token_pair
        return p_lm

    def _add_atom_single_context_and_mlp(
        self,
        p_lm: torch.Tensor,
        c_l_q: torch.Tensor,
        c_l_k: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Add atom single context and the pair MLP in atom-window chunks."""

        window_chunk_size = 64

        n_windows = p_lm.shape[-4]
        for start in range(0, n_windows, window_chunk_size):
            end = min(start + window_chunk_size, n_windows)
            p_target = [slice(None)] * p_lm.dim()
            p_target[-4] = slice(start, end)
            q_target = [slice(None)] * c_l_q.dim()
            q_target[-3] = slice(start, end)
            k_target = [slice(None)] * c_l_k.dim()
            k_target[-3] = slice(start, end)

            p_chunk = p_lm[tuple(p_target)]
            p_chunk = (
                p_chunk
                + self.linear_no_bias_cl(F.relu(c_l_q[tuple(q_target)][..., None, :]))
                + self.linear_no_bias_cm(
                    F.relu(c_l_k[tuple(k_target)][..., None, :, :])
                )
            )
            p_lm[tuple(p_target)] = p_chunk + self.small_mlp(p_chunk)
        return p_lm

    def _add_atom_single_context_and_mlp_local(
        self,
        p_lm: torch.Tensor,
        c_l_q: torch.Tensor,
        c_l_k: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        if inplace_safe:
            p_lm = p_lm + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
            p_lm += self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
            p_lm += self.small_mlp(p_lm)
            return p_lm
        p_lm = (
            p_lm
            + self.linear_no_bias_cl(F.relu(c_l_q[..., None, :]))
            + self.linear_no_bias_cm(F.relu(c_l_k[..., None, :, :]))
        )
        return p_lm + self.small_mlp(p_lm)

    def _add_atom_single_context_and_mlp_foldcp_local(
        self,
        p_lm: torch.Tensor,
        c_l_q: torch.Tensor,
        c_l_k: torch.Tensor,
        *,
        block_start: int,
        n_windows: int,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Preserve serial 64-window CUDA launch boundaries for a local shard."""

        window_chunk_size = 64
        local_windows = p_lm.shape[-4]
        block_end = min(block_start + local_windows, n_windows)
        output = torch.zeros_like(p_lm)
        canonical_start = (block_start // window_chunk_size) * window_chunk_size

        for chunk_start in range(canonical_start, block_end, window_chunk_size):
            chunk_end = min(chunk_start + window_chunk_size, n_windows)
            overlap_start = max(block_start, chunk_start)
            overlap_end = min(block_end, chunk_end)
            if overlap_start >= overlap_end:
                continue

            launch_windows = chunk_end - chunk_start
            local_start = overlap_start - block_start
            local_end = overlap_end - block_start
            launch_start = overlap_start - chunk_start
            launch_end = overlap_end - chunk_start

            p_shape = list(p_lm.shape)
            p_shape[-4] = launch_windows
            q_shape = list(c_l_q.shape)
            q_shape[-3] = launch_windows
            k_shape = list(c_l_k.shape)
            k_shape[-3] = launch_windows
            p_launch = p_lm.new_zeros(p_shape)
            q_launch = c_l_q.new_zeros(q_shape)
            k_launch = c_l_k.new_zeros(k_shape)

            p_launch[..., launch_start:launch_end, :, :, :] = p_lm[
                ..., local_start:local_end, :, :, :
            ]
            q_launch[..., launch_start:launch_end, :, :] = c_l_q[
                ..., local_start:local_end, :, :
            ]
            k_launch[..., launch_start:launch_end, :, :] = c_l_k[
                ..., local_start:local_end, :, :
            ]
            updated = self._add_atom_single_context_and_mlp(
                p_lm=p_launch,
                c_l_q=q_launch,
                c_l_k=k_launch,
                inplace_safe=inplace_safe,
            )
            output[..., local_start:local_end, :, :, :] = updated[
                ..., launch_start:launch_end, :, :, :
            ]
            del p_launch, q_launch, k_launch, updated
        return output

    def _project_pair_embedding_in_dense_trunk_from_foldcp_local(
        self,
        *,
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        mesh: FoldCPProcessMesh,
        out: torch.Tensor,
        block_start: int = 0,
        n_windows: Optional[int] = None,
        window_chunk_size: int = 64,
        source_rows: Optional[int] = None,
    ) -> torch.Tensor:
        """Project atom-window pair context directly from Fold-CP local tiles."""

        def _prepare_index_gather():
            if z_local.ndim != 3:
                raise ValueError(
                    "Fold-CP atom-window pair lookup expects z_local=[T,T,C]."
                )
            if idx_q.ndim != 2 or idx_k.ndim != 2:
                raise ValueError("idx_q and idx_k must be [N_block, N_query/key].")
            if n_windows is not None and int(n_windows) < 0:
                raise ValueError("n_windows must be non-negative.")
            if int(window_chunk_size) <= 0:
                raise ValueError("window_chunk_size must be positive.")
            prepared_idx_q = idx_q.long()
            prepared_idx_k = idx_k.long()
            tile_rows = z_spec.local_shape[z_spec.pair_dims[0]]
            tile_cols = z_spec.local_shape[z_spec.pair_dims[1]]
            n_token = z_spec.original_shape[z_spec.pair_dims[0]]
            return (
                prepared_idx_q,
                prepared_idx_k,
                tile_rows,
                tile_cols,
                n_token,
                [torch.empty_like(prepared_idx_q) for _ in range(mesh.layout.numel)],
                [torch.empty_like(prepared_idx_k) for _ in range(mesh.layout.numel)],
            )

        index_state = run_group_rank_action_synchronized(
            _prepare_index_gather,
            group=mesh.group_2d,
            description="atom-window index-gather preparation",
        )
        if index_state is None:  # pragma: no cover
            raise RuntimeError("atom-window index gather returned no state.")
        idx_q, idx_k, tile_rows, tile_cols, n_token, idx_q_all, idx_k_all = index_state
        group_rank = dist.get_rank(mesh.group_2d)
        dist.all_gather(idx_q_all, idx_q, group=mesh.group_2d)
        dist.all_gather(idx_k_all, idx_k, group=mesh.group_2d)

        def _build_transfer_schedule(
            source_ranges: tuple[tuple[int, int], ...],
        ) -> frozenset[tuple[int, int, int]]:
            """Build one rank-identical P2P schedule without CUDA temporaries."""

            idx_q_cpu = [tensor.detach().to(device="cpu") for tensor in idx_q_all]
            idx_k_cpu = [tensor.detach().to(device="cpu") for tensor in idx_k_all]
            schedule: set[tuple[int, int, int]] = set()
            for source_index, (source_start, source_end) in enumerate(source_ranges):
                for cp_rank in range(mesh.layout.numel):
                    row_coord, col_coord = mesh.layout.to_coord(cp_rank)
                    row_start = row_coord * tile_rows
                    row_end = min(row_start + tile_rows, n_token)
                    col_start = col_coord * tile_cols
                    col_end = min(col_start + tile_cols, n_token)
                    for dst_rank in range(mesh.layout.numel):
                        dst_block_start = dst_rank * idx_q.shape[-2]
                        dst_overlap_start = max(dst_block_start, source_start)
                        dst_overlap_end = min(
                            dst_block_start + idx_q.shape[-2], source_end
                        )
                        if dst_overlap_start >= dst_overlap_end:
                            continue
                        dst_slice = slice(
                            dst_overlap_start - dst_block_start,
                            dst_overlap_end - dst_block_start,
                        )
                        dst_idx_q = idx_q_cpu[dst_rank][dst_slice]
                        dst_idx_k = idx_k_cpu[dst_rank][dst_slice]
                        q_in_tile = (dst_idx_q >= row_start) & (dst_idx_q < row_end)
                        k_in_tile = (dst_idx_k >= col_start) & (dst_idx_k < col_end)
                        if bool(q_in_tile.any()) and bool(k_in_tile.any()):
                            schedule.add((source_index, cp_rank, dst_rank))
            return frozenset(schedule)

        def _fill_transfer(
            transfer: torch.Tensor,
            *,
            dst_idx_q: torch.Tensor,
            dst_idx_k: torch.Tensor,
            row_start: int,
            row_end: int,
            col_start: int,
            col_end: int,
        ) -> None:
            """Fill a preallocated transfer while keeping failures drainable."""

            q_in_tile = (dst_idx_q >= row_start) & (dst_idx_q < row_end)
            k_in_tile = (dst_idx_k >= col_start) & (dst_idx_k < col_end)
            q_local = (dst_idx_q - row_start).clamp(0, tile_rows - 1)
            k_local = (dst_idx_k - col_start).clamp(0, tile_cols - 1)
            dst_window = z_local[
                q_local[..., :, None],
                k_local[..., None, :],
                :,
            ]
            valid_mask = (q_in_tile[..., :, None] & k_in_tile[..., None, :]).unsqueeze(
                -1
            )
            transfer.copy_(dst_window.masked_fill(~valid_mask, 0.0))

        if n_windows is None:
            transfer_schedule = run_group_rank_action_synchronized(
                lambda: _build_transfer_schedule(
                    ((0, idx_q.shape[-2] * mesh.layout.numel),)
                ),
                group=mesh.group_2d,
                description="legacy atom-window transfer scheduling",
            )
            if transfer_schedule is None:  # pragma: no cover
                raise RuntimeError("legacy atom-window transfer returned no schedule.")
            legacy_buffers = run_group_rank_action_synchronized(
                lambda: (
                    z_local.new_zeros(*out.shape[:-1], z_local.shape[-1]),
                    z_local.new_empty(*out.shape[:-1], z_local.shape[-1]),
                ),
                group=mesh.group_2d,
                description="legacy atom-window transfer allocation",
            )
            if legacy_buffers is None:  # pragma: no cover
                raise RuntimeError("legacy atom-window transfer returned no buffers.")
            z_window, transfer = legacy_buffers
            compute_error: Exception | None = None
            for cp_rank in range(mesh.layout.numel):
                row_coord, col_coord = mesh.layout.to_coord(cp_rank)
                row_start = row_coord * tile_rows
                col_start = col_coord * tile_cols
                col_end = min(col_start + tile_cols, n_token)
                src_global_rank = dist.get_global_rank(mesh.group_2d, cp_rank)
                for dst_rank in range(mesh.layout.numel):
                    dst_idx_q = idx_q_all[dst_rank]
                    dst_idx_k = idx_k_all[dst_rank]
                    dst_needs_tile = (0, cp_rank, dst_rank) in transfer_schedule
                    if not dst_needs_tile:
                        continue

                    if compute_error is None:
                        try:
                            _zero_foldcp_drain_buffer(transfer)
                        except Exception as exc:
                            compute_error = detach_rank_local_error_traceback(exc)
                    if group_rank == cp_rank:
                        if compute_error is None:
                            try:
                                _fill_transfer(
                                    transfer,
                                    dst_idx_q=dst_idx_q,
                                    dst_idx_k=dst_idx_k,
                                    row_start=row_start,
                                    row_end=min(row_start + tile_rows, n_token),
                                    col_start=col_start,
                                    col_end=col_end,
                                )
                            except Exception as exc:
                                compute_error = detach_rank_local_error_traceback(exc)
                        if dst_rank == group_rank:
                            if compute_error is None:
                                try:
                                    z_window.add_(transfer)
                                except Exception as exc:
                                    compute_error = detach_rank_local_error_traceback(
                                        exc
                                    )
                        else:
                            dst_global_rank = dist.get_global_rank(
                                mesh.group_2d, dst_rank
                            )
                            dist.send(
                                transfer,
                                dst=dst_global_rank,
                                group=mesh.group_2d,
                            )
                    elif group_rank == dst_rank:
                        dist.recv(
                            transfer,
                            src=src_global_rank,
                            group=mesh.group_2d,
                        )
                        if compute_error is None:
                            try:
                                z_window.add_(transfer)
                            except Exception as exc:
                                compute_error = detach_rank_local_error_traceback(exc)

            def _finish_legacy_window() -> torch.Tensor:
                if compute_error is not None:
                    raise compute_error
                z_norm = self.layernorm_z(z_window)
                if source_rows is None:
                    return self.linear_no_bias_z(z_norm)
                local_rows = (
                    int(z_norm.numel() // z_norm.shape[-1]) if z_norm.shape[-1] else 0
                )
                if source_rows <= local_rows:
                    return self.linear_no_bias_z(z_norm)
                flat = z_norm.contiguous().reshape(local_rows, z_norm.shape[-1])
                launch = flat.new_zeros(int(source_rows), flat.shape[-1])
                launch[:local_rows].copy_(flat)
                projected = self.linear_no_bias_z(launch)[:local_rows]
                return projected.reshape(*z_norm.shape[:-1], -1)

            result = run_group_rank_action_synchronized(
                _finish_legacy_window,
                group=mesh.group_2d,
                description="legacy atom-window pair computation",
            )
            if result is None:  # pragma: no cover
                raise RuntimeError("legacy atom-window pair returned no result.")
            return result

        prefix_shape = out.shape[:-4]
        blocks_per_rank = idx_q.shape[-2]
        local_block_end = block_start + blocks_per_rank
        max_chunk_blocks = min(int(window_chunk_size), blocks_per_rank)
        source_ranges = tuple(
            (source_start, min(source_start + int(window_chunk_size), int(n_windows)))
            for source_start in range(0, int(n_windows), int(window_chunk_size))
        )
        transfer_schedule = run_group_rank_action_synchronized(
            lambda: _build_transfer_schedule(source_ranges),
            group=mesh.group_2d,
            description="atom-window transfer scheduling",
        )
        if transfer_schedule is None:  # pragma: no cover
            raise RuntimeError("atom-window transfer returned no schedule.")

        def _allocate_window_stream():
            return (
                z_local.new_zeros(*out.shape[:-1], self.c_atompair),
                z_local.new_empty(
                    *prefix_shape,
                    max_chunk_blocks,
                    self.n_queries,
                    self.n_keys,
                    z_local.shape[-1],
                ),
                z_local.new_empty(
                    max_chunk_blocks,
                    self.n_queries,
                    self.n_keys,
                    z_local.shape[-1],
                ),
            )

        stream_buffers = run_group_rank_action_synchronized(
            _allocate_window_stream,
            group=mesh.group_2d,
            description="atom-window pair stream allocation",
        )
        if stream_buffers is None:  # pragma: no cover
            raise RuntimeError("atom-window pair stream returned no buffers.")
        projected, z_window_buffer, transfer_buffer = stream_buffers
        compute_error: Exception | None = None
        for source_index, (source_start, source_end) in enumerate(source_ranges):
            overlap_start = max(block_start, source_start)
            overlap_end = min(local_block_end, source_end)
            local_chunk_blocks = max(0, overlap_end - overlap_start)
            local_slice = slice(
                overlap_start - block_start,
                overlap_end - block_start,
            )
            z_window_chunk = z_window_buffer[..., :local_chunk_blocks, :, :, :]
            if compute_error is None:
                try:
                    _zero_foldcp_drain_buffer(z_window_chunk)
                except Exception as exc:
                    compute_error = detach_rank_local_error_traceback(exc)

            for cp_rank in range(mesh.layout.numel):
                row_coord, col_coord = mesh.layout.to_coord(cp_rank)
                row_start = row_coord * tile_rows
                col_start = col_coord * tile_cols
                col_end = min(col_start + tile_cols, n_token)
                src_global_rank = dist.get_global_rank(mesh.group_2d, cp_rank)
                for dst_rank in range(mesh.layout.numel):
                    dst_block_start = dst_rank * blocks_per_rank
                    dst_block_end = dst_block_start + blocks_per_rank
                    dst_overlap_start = max(dst_block_start, source_start)
                    dst_overlap_end = min(dst_block_end, source_end)
                    if dst_overlap_start >= dst_overlap_end:
                        continue
                    dst_slice = slice(
                        dst_overlap_start - dst_block_start,
                        dst_overlap_end - dst_block_start,
                    )
                    dst_idx_q = idx_q_all[dst_rank][dst_slice]
                    dst_idx_k = idx_k_all[dst_rank][dst_slice]
                    dst_needs_tile = (
                        source_index,
                        cp_rank,
                        dst_rank,
                    ) in transfer_schedule
                    if not dst_needs_tile:
                        continue

                    transfer_blocks = dst_overlap_end - dst_overlap_start
                    transfer = transfer_buffer[:transfer_blocks]
                    if compute_error is None:
                        try:
                            _zero_foldcp_drain_buffer(transfer)
                        except Exception as exc:
                            compute_error = detach_rank_local_error_traceback(exc)
                    if group_rank == cp_rank:
                        if compute_error is None:
                            try:
                                _fill_transfer(
                                    transfer,
                                    dst_idx_q=dst_idx_q,
                                    dst_idx_k=dst_idx_k,
                                    row_start=row_start,
                                    row_end=min(row_start + tile_rows, n_token),
                                    col_start=col_start,
                                    col_end=col_end,
                                )
                            except Exception as exc:
                                compute_error = detach_rank_local_error_traceback(exc)
                        if dst_rank == group_rank:
                            if compute_error is None:
                                try:
                                    z_window_chunk.add_(transfer)
                                except Exception as exc:
                                    compute_error = detach_rank_local_error_traceback(
                                        exc
                                    )
                        else:
                            dst_global_rank = dist.get_global_rank(
                                mesh.group_2d, dst_rank
                            )
                            dist.send(
                                transfer,
                                dst=dst_global_rank,
                                group=mesh.group_2d,
                            )
                    elif group_rank == dst_rank:
                        dist.recv(
                            transfer,
                            src=src_global_rank,
                            group=mesh.group_2d,
                        )
                        if compute_error is None:
                            try:
                                z_window_chunk.add_(transfer)
                            except Exception as exc:
                                compute_error = detach_rank_local_error_traceback(exc)

            z_norm = flat = launch = projected_chunk = None
            if local_chunk_blocks > 0 and compute_error is None:
                try:
                    z_norm = self.layernorm_z(z_window_chunk)
                    local_rows = (
                        int(z_norm.numel() // z_norm.shape[-1])
                        if z_norm.shape[-1]
                        else 0
                    )
                    source_chunk_rows = (
                        (source_end - source_start) * self.n_queries * self.n_keys
                    )
                    if source_chunk_rows <= local_rows:
                        projected_chunk = self.linear_no_bias_z(z_norm)
                    else:
                        flat = z_norm.contiguous().reshape(local_rows, z_norm.shape[-1])
                        launch = flat.new_zeros(int(source_chunk_rows), flat.shape[-1])
                        launch[:local_rows].copy_(flat)
                        projected_chunk = self.linear_no_bias_z(launch)[
                            :local_rows
                        ].reshape(*z_norm.shape[:-1], -1)
                    projected[..., local_slice, :, :, :] = projected_chunk
                except Exception as exc:
                    compute_error = detach_rank_local_error_traceback(exc)
            del z_norm, flat, launch, projected_chunk

        def _finish_window_stream() -> torch.Tensor:
            if compute_error is not None:
                raise compute_error
            return projected

        result = run_group_rank_action_synchronized(
            _finish_window_stream,
            group=mesh.group_2d,
            description="atom-window pair stream computation",
        )
        if result is None:  # pragma: no cover
            raise RuntimeError("atom-window pair stream returned no result.")
        return result

    def _warmup_foldcp_atom_window_p2p(
        self,
        *,
        mesh: FoldCPProcessMesh,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initialize NCCL P2P communicators before high-memory atom windows.

        The flat ring below is exactly `Ring2DComm.comm_row` on the 1xP launch
        layout, where every column and skew shift collapses to a self-comm, and
        circulating a token also keeps the mesh in step. A square mesh shifts
        along the column axis and by -row/-col as well, so those links are
        warmed first instead of connecting lazily inside the atom-window loop.

        Each skip below depends only on the coordinate the partner shares, so
        every rank on a given link makes the same decision and the sends and
        receives stay matched.
        """

        if getattr(self, "_foldcp_atom_window_p2p_warmed", False):
            return
        group_rank = dist.get_rank(mesh.group_2d)
        coord = mesh.layout.to_coord(group_rank)
        skew_links: list[tuple[int, int]] = []
        for axis, shift in ((1, -coord[0]), (0, -1), (0, -coord[1])):
            send_rank = mesh.layout.shifted_rank(coord, axis=axis, shift=shift)
            recv_rank = mesh.layout.shifted_rank(coord, axis=axis, shift=-shift)
            if send_rank != group_rank or recv_rank != group_rank:
                skew_links.append((send_rank, recv_rank))

        def _allocate_warmup_buffers():
            token = torch.zeros(1, device=device, dtype=dtype)
            skew_pairs = [
                (torch.zeros_like(token), torch.empty_like(token)) for _ in skew_links
            ]
            ring_receives = [
                torch.empty_like(token) for _ in range(1, mesh.layout.numel)
            ]
            return token, skew_pairs, ring_receives

        buffers = run_group_rank_action_synchronized(
            _allocate_warmup_buffers,
            group=mesh.group_2d,
            description="atom-window P2P warmup allocation",
        )
        if buffers is None:  # pragma: no cover
            raise RuntimeError("atom-window P2P warmup returned no buffers.")
        token, skew_pairs, ring_receives = buffers

        skew_ops: list[dist.P2POp] = []
        for (send_rank, recv_rank), (send_chunk, recv_chunk) in zip(
            skew_links, skew_pairs, strict=True
        ):
            skew_ops.extend(
                (
                    dist.P2POp(
                        dist.isend,
                        send_chunk,
                        dist.get_global_rank(mesh.group_2d, send_rank),
                        mesh.group_2d,
                    ),
                    dist.P2POp(
                        dist.irecv,
                        recv_chunk,
                        dist.get_global_rank(mesh.group_2d, recv_rank),
                        mesh.group_2d,
                    ),
                )
            )
        if skew_ops:
            dispatch_p2p_batch_and_wait(skew_ops)
        del skew_ops, skew_pairs

        send_chunk = token
        for recv_chunk in ring_receives:
            send_rank = (group_rank - 1) % mesh.layout.numel
            recv_rank = (group_rank + 1) % mesh.layout.numel
            operations = [
                dist.P2POp(
                    dist.isend,
                    send_chunk,
                    dist.get_global_rank(mesh.group_2d, send_rank),
                    mesh.group_2d,
                ),
                dist.P2POp(
                    dist.irecv,
                    recv_chunk,
                    dist.get_global_rank(mesh.group_2d, recv_rank),
                    mesh.group_2d,
                ),
            ]
            dispatch_p2p_batch_and_wait(operations)
            send_chunk = recv_chunk
        del ring_receives
        self._foldcp_atom_window_p2p_warmed = True

    def prepare_cache_foldcp_window(
        self,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_element: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
        pad_info: dict[str, Any],
        mesh: FoldCPProcessMesh,
        r_l: Union[torch.Tensor, bool, None] = None,
        z: Optional[torch.Tensor] = None,
        z_spec: Optional[FoldCPPairShardSpec] = None,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, FoldCPWindowShardSpec]:
        if r_l is not None and z_spec is not None and d_lm.is_cuda:
            self._warmup_foldcp_atom_window_p2p(
                mesh=mesh,
                device=d_lm.device,
                dtype=d_lm.dtype,
            )

        def _prepare_local_atom_window_state():
            batch_shape = ref_pos.shape[:-2]
            n_atom = ref_pos.shape[-2]
            c_l = self.linear_no_bias_ref_pos(ref_pos) + self.linear_no_bias_ref_charge(
                torch.arcsinh(ref_charge).reshape(*batch_shape, n_atom, 1)
            )
            ref_features = torch.cat(
                [
                    ref_mask.reshape(*batch_shape, n_atom, 1),
                    ref_element.reshape(*batch_shape, n_atom, 128),
                    ref_atom_name_chars.reshape(*batch_shape, n_atom, 4 * 64),
                ],
                dim=-1,
            ).to(dtype=c_l.dtype)
            if inplace_safe:
                c_l += self.linear_no_bias_f(ref_features)
                c_l *= ref_mask.reshape(*batch_shape, n_atom, 1)
            else:
                c_l = c_l + self.linear_no_bias_f(ref_features)
                c_l = c_l * ref_mask.reshape(*batch_shape, n_atom, 1)

            mask_trunked = pad_info["mask_trunked"]
            assert mask_trunked is not None
            n_windows = mask_trunked.shape[-3]
            block_range = window_block_range(n_windows, mesh)
            block_start, block_end = block_range
            blocks_per_rank = block_end - block_start
            valid_end = min(block_end, n_windows)
            p_lm = d_lm.new_zeros(
                *d_lm.shape[:-4],
                blocks_per_rank,
                self.n_queries,
                self.n_keys,
                self.c_atompair,
            )

            def _linear_with_atom_window_source_rows(
                linear: nn.Module,
                x: torch.Tensor,
                *,
                source_rows: int,
            ) -> torch.Tensor:
                local_rows = int(x.numel() // x.shape[-1]) if x.shape[-1] else 0
                if source_rows <= local_rows:
                    return linear(x)
                flat = x.contiguous().reshape(local_rows, x.shape[-1])
                launch = flat.new_zeros(int(source_rows), flat.shape[-1])
                launch[:local_rows].copy_(flat)
                out = linear(launch)[:local_rows]
                return out.reshape(*x.shape[:-1], -1)

            source_rows = int(d_lm.numel() // d_lm.shape[-1])
            if block_start < valid_end:
                valid_blocks = valid_end - block_start
                valid_slice = slice(block_start, valid_end)
                d_local = d_lm[..., valid_slice, :, :, :]
                v_local = v_lm[..., valid_slice, :, :, :]
                mask_local = mask_trunked[..., valid_slice, :, :]
                p_valid = (
                    _linear_with_atom_window_source_rows(
                        self.linear_no_bias_d,
                        d_local,
                        source_rows=source_rows,
                    )
                    * v_local
                ) * mask_local.unsqueeze(dim=-1)
                invd_local = 1 / (1 + (d_local**2).sum(dim=-1, keepdim=True))
                invd_projected = _linear_with_atom_window_source_rows(
                    self.linear_no_bias_invd,
                    invd_local,
                    source_rows=source_rows,
                )
                v_projected = _linear_with_atom_window_source_rows(
                    self.linear_no_bias_v,
                    v_local.to(dtype=p_valid.dtype),
                    source_rows=source_rows,
                )
                if inplace_safe:
                    p_valid += invd_projected * v_local
                    p_valid += v_projected
                else:
                    p_valid = p_valid + invd_projected * v_local
                    p_valid = p_valid + v_projected
                p_lm[..., :valid_blocks, :, :, :] = p_valid

            z_token_pair = None
            local_idx_q = None
            local_idx_k = None
            if r_l is not None:
                assert z is not None
                atom_to_token_idx_q, atom_to_token_idx_k, _ = atom_window_token_indices(
                    atom_to_token_idx,
                    n_queries=self.n_queries,
                    n_keys=self.n_keys,
                    compute_mask=False,
                )
                z_token_pair = p_lm.new_zeros(
                    blocks_per_rank,
                    self.n_queries,
                    self.n_keys,
                    self.c_atompair,
                )
                valid_blocks = max(0, valid_end - block_start)
                valid_slice = slice(block_start, valid_end)
                if z_spec is not None:
                    local_idx_q = atom_to_token_idx_q.new_zeros(
                        blocks_per_rank, self.n_queries
                    )
                    local_idx_k = atom_to_token_idx_k.new_zeros(
                        blocks_per_rank, self.n_keys
                    )
                    if valid_blocks > 0:
                        local_idx_q[:valid_blocks] = atom_to_token_idx_q[valid_slice]
                        local_idx_k[:valid_blocks] = atom_to_token_idx_k[valid_slice]
                elif valid_blocks > 0:
                    z_valid = gather_pair_embedding_in_dense_trunk(
                        z,
                        idx_q=atom_to_token_idx_q[valid_slice],
                        idx_k=atom_to_token_idx_k[valid_slice],
                    )
                    z_token_pair[:valid_blocks] = self.linear_no_bias_z(
                        self.layernorm_z(z_valid)
                    )
            return (
                c_l,
                p_lm,
                int(n_atom),
                int(n_windows),
                block_range,
                blocks_per_rank,
                block_start,
                source_rows,
                z_token_pair,
                local_idx_q,
                local_idx_k,
            )

        prepared = run_group_rank_action_synchronized(
            _prepare_local_atom_window_state,
            group=mesh.group_2d,
            description="Fold-CP atom-window local feature preparation",
        )
        if prepared is None:  # pragma: no cover
            raise RuntimeError("Fold-CP atom-window local state was not prepared.")
        (
            c_l,
            p_lm,
            n_atom,
            n_windows,
            block_range,
            blocks_per_rank,
            block_start,
            source_rows,
            z_token_pair,
            local_idx_q,
            local_idx_k,
        ) = prepared

        if r_l is not None and z_spec is not None:
            assert z is not None
            assert z_token_pair is not None
            assert local_idx_q is not None and local_idx_k is not None
            z_token_pair = (
                self._project_pair_embedding_in_dense_trunk_from_foldcp_local(
                    z_local=z,
                    z_spec=z_spec,
                    idx_q=local_idx_q,
                    idx_k=local_idx_k,
                    mesh=mesh,
                    out=z_token_pair,
                    block_start=block_start,
                    n_windows=n_windows,
                    window_chunk_size=64,
                    source_rows=source_rows,
                )
            )

        def _finalize_local_atom_window():
            local_p = p_lm
            if r_l is not None:
                if z_token_pair is None:
                    raise RuntimeError("Fold-CP atom-window pair context is missing.")
                local_p = local_p.unsqueeze(dim=-5)
                local_z_pair = z_token_pair
                if local_z_pair.dim() == local_p.dim() - 1:
                    local_z_pair = local_z_pair.unsqueeze(dim=-5)
                local_p = local_p + local_z_pair

            window_spec = FoldCPWindowShardSpec(
                n_atom=n_atom,
                n_windows=n_windows,
                n_queries=int(self.n_queries),
                n_keys=int(self.n_keys),
                q_pad=int(pad_info["q_pad"]),
                block_range=block_range,
                size_cp=mesh.config.size_cp,
                padded_n_windows=int(blocks_per_rank * mesh.config.size_cp),
            )
            return local_p.contiguous(), c_l, window_spec

        finalized = run_group_rank_action_synchronized(
            _finalize_local_atom_window,
            group=mesh.group_2d,
            description="Fold-CP atom-window local feature finalization",
        )
        if finalized is None:  # pragma: no cover
            raise RuntimeError("Fold-CP atom-window local state was not finalized.")
        return finalized

    def prepare_cache(
        self,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_element: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
        pad_info: dict[str, Any],
        r_l: Union[torch.Tensor, bool, None] = None,
        z: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape = ref_pos.shape[:-2]
        N_atom = ref_pos.shape[-2]
        c_l = self.linear_no_bias_ref_pos(ref_pos) + self.linear_no_bias_ref_charge(
            # use arcsinh for ref_charge
            torch.arcsinh(ref_charge).reshape(*batch_shape, N_atom, 1)
        )
        if inplace_safe:
            c_l += self.linear_no_bias_f(
                torch.cat(
                    [
                        ref_mask.reshape(*batch_shape, N_atom, 1),
                        ref_element.reshape(*batch_shape, N_atom, 128),
                        ref_atom_name_chars.reshape(*batch_shape, N_atom, 4 * 64),
                    ],
                    dim=-1,
                ).to(dtype=c_l.dtype)
            )
            c_l *= ref_mask.reshape(*batch_shape, N_atom, 1)
        else:
            c_l = c_l + self.linear_no_bias_f(
                torch.cat(
                    [
                        ref_mask.reshape(*batch_shape, N_atom, 1),
                        ref_element.reshape(*batch_shape, N_atom, 128),
                        ref_atom_name_chars.reshape(*batch_shape, N_atom, 4 * 64),
                    ],
                    dim=-1,
                ).to(dtype=c_l.dtype)
            )
            c_l = c_l * ref_mask.reshape(*batch_shape, N_atom, 1)

        mask_trunked = pad_info["mask_trunked"]
        assert mask_trunked is not None
        p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * mask_trunked.unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, C_atompair]

        # Line5-Line6: Embed pairwise inverse squared distances, and the valid mask
        if inplace_safe:
            p_lm += (
                self.linear_no_bias_invd(1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True)))
                * v_lm
            )
            p_lm += self.linear_no_bias_v(
                v_lm.to(dtype=p_lm.dtype)
            )  # not multipling v_lm
        else:
            p_lm = (
                p_lm
                + self.linear_no_bias_invd(
                    1 / (1 + (d_lm**2).sum(dim=-1, keepdim=True))
                )
                * v_lm
            )
            p_lm = p_lm + self.linear_no_bias_v(
                v_lm.to(dtype=p_lm.dtype)
            )  # not multipling v_lm

        # Line7: Initialise the atom single representation as the single conditioning
        # q_l = c_l.clone()

        # If provided, add trunk embeddings and noisy positions
        if r_l is not None:
            assert z is not None
            p_lm = self._add_token_pair_context_to_atom_pair(
                p_lm=p_lm,
                z=z,
                atom_to_token_idx=atom_to_token_idx,
            )  # [..., N_sample, n_blocks, n_queries, n_keys, c_atompair]
        return p_lm, c_l

    def forward(
        self,
        atom_to_token_idx: torch.Tensor,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        ref_element: torch.Tensor,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
        pad_info: dict[str, Any],
        r_l: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        p_lm: Optional[torch.Tensor] = None,
        c_l: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            atom_to_token_idx (torch.Tensor): atom_to_token_idx
            ref_pos (torch.Tensor): ref_pos
            ref_charge (torch.Tensor): ref_charge
            ref_mask (torch.Tensor): ref_mask
            ref_atom_name_chars (torch.Tensor): ref_atom_name_chars
            ref_element (torch.Tensor): ref_element
            r_l (torch.Tensor, optional): noisy position.
                [..., N_sample, N_atom, 3] if has_coords else None.
            s (torch.Tensor, optional): single embedding.
                [..., N_sample, N_token, c_s] if has_coords else None.
            z (torch.Tensor, optional): pair embedding
                [..., N_token, N_token, c_z] or
                [..., N_sample, N_token, N_token, c_z] if has_coords else None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: the output of AtomAttentionEncoder
            a:
                [..., (N_sample), N_token, c_token]
            q_l:
                [..., (N_sample), N_atom, c_atom]
            c_l:
                [..., (N_sample), N_atom, c_atom]
            p_lm:
                [..., (N_sample), N_atom, N_atom, c_atompair]

        """

        if self.has_coords:
            assert r_l is not None
            assert s is not None
            assert z is not None

        if p_lm is None or c_l is None:
            p_lm, c_l = self.prepare_cache(
                ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_mask=ref_mask,
                ref_atom_name_chars=ref_atom_name_chars,
                ref_element=ref_element,
                atom_to_token_idx=atom_to_token_idx,
                d_lm=d_lm,
                v_lm=v_lm,
                pad_info=pad_info,
                r_l=r_l,
                z=z,
                inplace_safe=inplace_safe,
            )
        else:
            if inplace_safe:
                p_lm_clone = p_lm.clone()
                c_l_clone = c_l.clone()
                p_lm = p_lm_clone
                c_l = c_l_clone

        # Line7: Initialise the atom single representation as the single conditioning
        # q_l = c_l.clone()

        # If provided, add trunk embeddings and noisy positions
        n_token = None
        if r_l is not None:
            assert s is not None
            # Broadcast the single and pair embedding from the trunk
            n_token = s.size(-2)
            c_l = c_l.unsqueeze(dim=-3) + broadcast_token_to_atom(
                x_token=self.linear_no_bias_s(self.layernorm_s(s)),
                atom_to_token_idx=atom_to_token_idx,
            )  # [..., N_sample, N_atom, c_atom]

            # Add the noisy positions
            # Different from paper!!
            q_l = c_l + self.linear_no_bias_r(r_l)  # [..., N_sample, N_atom, c_atom]
        else:
            q_l = c_l.clone()

        # Add the combined single conditioning to the pair representation
        c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
            q=c_l,
            k=c_l,
            dim_q=-2,
            dim_k=-2,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=False,
        )
        p_lm = self._add_atom_single_context_and_mlp(
            p_lm=p_lm,
            c_l_q=c_l_q,
            c_l_k=c_l_k,
            inplace_safe=inplace_safe,
        )

        # Cross attention transformer
        q_l = self.atom_transformer(
            q_l, c_l, p_lm, chunk_size=chunk_size
        )  # [..., (N_sample), N_atom, c_atom]

        # Aggregate per-atom representation to per-token representation
        a = aggregate_atom_to_token(
            x_atom=F.relu(self.linear_no_bias_q(q_l)),
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="mean",
        )  # [..., (N_sample), N_token, c_token]
        return a, q_l, c_l, p_lm

    def forward_foldcp_window(
        self,
        atom_to_token_idx: torch.Tensor,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_mask: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        ref_element: torch.Tensor,
        d_lm: torch.Tensor,
        v_lm: torch.Tensor,
        pad_info: dict[str, Any],
        mesh: FoldCPProcessMesh,
        r_l: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        p_lm: Optional[torch.Tensor] = None,
        c_l: Optional[torch.Tensor] = None,
        window_spec: Optional[FoldCPWindowShardSpec] = None,
        z_spec: Optional[FoldCPPairShardSpec] = None,
        inplace_safe: bool = False,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, FoldCPWindowShardSpec
    ]:
        if self.has_coords:
            assert r_l is not None
            assert s is not None
            assert z is not None

        reuse_cached_inputs = (
            p_lm is not None and c_l is not None and window_spec is not None
        )
        if not reuse_cached_inputs:
            p_lm, c_l, window_spec = self.prepare_cache_foldcp_window(
                ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_mask=ref_mask,
                ref_atom_name_chars=ref_atom_name_chars,
                ref_element=ref_element,
                atom_to_token_idx=atom_to_token_idx,
                d_lm=d_lm,
                v_lm=v_lm,
                pad_info=pad_info,
                mesh=mesh,
                r_l=r_l,
                z=z,
                z_spec=z_spec,
                inplace_safe=inplace_safe,
            )
        if p_lm is None or c_l is None or window_spec is None:  # pragma: no cover
            raise RuntimeError("Fold-CP atom encoder cache was not prepared.")

        def _prepare_encoder_transformer_inputs():
            prepared_p_lm = (
                p_lm.clone() if reuse_cached_inputs and inplace_safe else p_lm
            )
            prepared_c_l = c_l.clone() if reuse_cached_inputs and inplace_safe else c_l
            n_token = None
            if r_l is not None:
                if s is None:  # pragma: no cover - validated above for has_coords
                    raise RuntimeError("Fold-CP atom encoder is missing single input.")
                n_token = s.size(-2)
                prepared_c_l = prepared_c_l.unsqueeze(dim=-3) + broadcast_token_to_atom(
                    x_token=self.linear_no_bias_s(self.layernorm_s(s)),
                    atom_to_token_idx=atom_to_token_idx,
                )
                q_l = prepared_c_l + self.linear_no_bias_r(r_l)
            else:
                q_l = prepared_c_l.clone()

            c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
                q=prepared_c_l,
                k=prepared_c_l,
                dim_q=-2,
                dim_k=-2,
                n_queries=self.n_queries,
                n_keys=self.n_keys,
                compute_mask=False,
            )
            block_start, block_end = window_spec.block_range
            blocks_per_rank = block_end - block_start
            valid_end = min(block_end, window_spec.n_windows)
            c_l_q_local = c_l_q.new_zeros(
                *c_l_q.shape[:-3],
                blocks_per_rank,
                self.n_queries,
                c_l_q.shape[-1],
            )
            c_l_k_local = c_l_k.new_zeros(
                *c_l_k.shape[:-3],
                blocks_per_rank,
                self.n_keys,
                c_l_k.shape[-1],
            )
            if block_start < valid_end:
                valid_blocks = valid_end - block_start
                valid_slice = slice(block_start, valid_end)
                c_l_q_local[..., :valid_blocks, :, :] = c_l_q[..., valid_slice, :, :]
                c_l_k_local[..., :valid_blocks, :, :] = c_l_k[..., valid_slice, :, :]
            prepared_p_lm = self._add_atom_single_context_and_mlp_foldcp_local(
                p_lm=prepared_p_lm,
                c_l_q=c_l_q_local,
                c_l_k=c_l_k_local,
                block_start=block_start,
                n_windows=window_spec.n_windows,
                inplace_safe=inplace_safe,
            )
            return q_l, prepared_c_l, prepared_p_lm, n_token

        prepared = (
            run_group_rank_action_synchronized(
                _prepare_encoder_transformer_inputs,
                group=mesh.group_2d,
                description="Fold-CP atom encoder transformer-input preparation",
            )
            if int(mesh.layout.shape[0]) == 1 and int(mesh.layout.shape[1]) > 1
            else _prepare_encoder_transformer_inputs()
        )
        if prepared is None:  # pragma: no cover
            raise RuntimeError("Fold-CP atom encoder inputs were not prepared.")
        q_l, c_l, p_lm, n_token = prepared

        q_l = self.atom_transformer.forward_foldcp_window(
            q=q_l,
            c=c_l,
            p_local=p_lm,
            window_spec=window_spec,
            mesh=mesh,
            inplace_safe=inplace_safe,
        )

        def _finish_atom_encoder():
            a = aggregate_atom_to_token(
                x_atom=F.relu(self.linear_no_bias_q(q_l)),
                atom_to_token_idx=atom_to_token_idx,
                n_token=n_token,
                reduce="mean",
            )
            return a, q_l, c_l, p_lm, window_spec

        result = (
            run_group_rank_action_synchronized(
                _finish_atom_encoder,
                group=mesh.group_2d,
                description="Fold-CP atom encoder completion",
            )
            if int(mesh.layout.shape[0]) == 1 and int(mesh.layout.shape[1]) > 1
            else _finish_atom_encoder()
        )
        if result is None:  # pragma: no cover
            raise RuntimeError("Fold-CP atom encoder returned no result.")
        return result


class AtomAttentionDecoder(nn.Module):
    """
    Implements Algorithm 6 in AF3

    Args:
        n_blocks (int, optional): number of blocks for AtomTransformer. Defaults to 3.
        n_heads (int, optional): number of heads for AtomTransformer. Defaults to 4.
        c_token (int, optional): feature channel of token (single a). Defaults to 384.
        c_atom (int, optional): embedding dim for atom embedding. Defaults to 128.
        c_atompair (int, optional): embedding dim for atom pair embedding. Defaults to 16.
        n_queries (int, optional): local window size of query tensor. Defaults to 32.
        n_keys (int, optional): local window size of key tensor. Defaults to 128.
        blocks_per_ckpt (int, optional): number of AtomAttentionDecoder/AtomTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        n_blocks: int = 3,
        n_heads: int = 4,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(AtomAttentionDecoder, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.linear_no_bias_a = LinearNoBias(in_features=c_token, out_features=c_atom)
        self.layernorm_q = LayerNorm(c_atom, create_offset=False)
        self.linear_no_bias_out = LinearNoBias(
            in_features=c_atom, out_features=3, precision=torch.float32
        )
        self.atom_transformer = AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def forward(
        self,
        atom_to_token_idx: torch.Tensor,
        a: torch.Tensor,
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            atom_to_token_idx (torch.Tensor): the atom to token index
                [..., N_atom]
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_token]
            q_skip (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            c_skip (torch.Tensor): atom single embedding
                [..., N_atom, c_atom]
            p_skip (torch.Tensor): atompair single embedding
                [..., n_blocks, n_queries, n_keys, c_atompair]

        Returns:
            torch.Tensor: the updated noisy coordinates
                [..., N_atom, 3]
        """
        # Broadcast per-token activiations to per-atom activations and add the skip connection
        q = (
            broadcast_token_to_atom(
                x_token=self.linear_no_bias_a(a),  # [..., N_token, c_atom]
                atom_to_token_idx=atom_to_token_idx,
            )  # [..., N_atom, c_atom]
            + q_skip
        )

        # Cross attention transformer
        q = self.atom_transformer(
            q, c_skip, p_skip, inplace_safe=inplace_safe, chunk_size=chunk_size
        )

        # Map to positions update
        q = self.layernorm_q(q)
        r = self.linear_no_bias_out(q)

        return r

    def forward_foldcp_window(
        self,
        atom_to_token_idx: torch.Tensor,
        a: torch.Tensor,
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip_local: torch.Tensor,
        window_spec: FoldCPWindowShardSpec,
        mesh: FoldCPProcessMesh,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        def _prepare_atom_decoder_input() -> torch.Tensor:
            return (
                broadcast_token_to_atom(
                    x_token=self.linear_no_bias_a(a),
                    atom_to_token_idx=atom_to_token_idx,
                )
                + q_skip
            )

        q = (
            run_group_rank_action_synchronized(
                _prepare_atom_decoder_input,
                group=mesh.group_2d,
                description="Fold-CP atom decoder transformer-input preparation",
            )
            if int(mesh.layout.shape[0]) == 1 and int(mesh.layout.shape[1]) > 1
            else _prepare_atom_decoder_input()
        )
        if q is None:  # pragma: no cover
            raise RuntimeError("Fold-CP atom decoder input was not prepared.")
        q = self.atom_transformer.forward_foldcp_window(
            q=q,
            c=c_skip,
            p_local=p_skip_local,
            window_spec=window_spec,
            mesh=mesh,
            inplace_safe=inplace_safe,
        )

        def _finish_atom_decoder() -> torch.Tensor:
            return self.linear_no_bias_out(self.layernorm_q(q))

        result = (
            run_group_rank_action_synchronized(
                _finish_atom_decoder,
                group=mesh.group_2d,
                description="Fold-CP atom decoder completion",
            )
            if int(mesh.layout.shape[0]) == 1 and int(mesh.layout.shape[1]) > 1
            else _finish_atom_decoder()
        )
        if result is None:  # pragma: no cover
            raise RuntimeError("Fold-CP atom decoder returned no result.")
        return result
