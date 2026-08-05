# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import os
from functools import partial
from typing import Any, Optional, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
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


def _attention_pair_bias_row_chunk_size(n_token: int) -> int:
    """Return the P-independent row launch size used by distributed FoldCP."""

    n_token = int(n_token)
    if n_token <= 512:
        return n_token
    value = int(os.environ.get("OPENDDE_PAIR_BIAS_ROW_CHUNK", "112"))
    if value <= 0:
        return n_token
    return min(n_token, value)


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
        gathered: list[torch.Tensor | None] = [None for _ in range(side)]
        gathered[mesh.coord[0]] = local_out

        ready = local_out
        for step in range(1, side):
            ready = ring.comm_col.exchange(ready.contiguous())
            source_row = (mesh.coord[0] + step) % side
            gathered[source_row] = ready

        if any(item is None for item in gathered):
            raise RuntimeError("failed to collect Fold-CP diffusion attention rows.")
        full_out = torch.cat(
            [item for item in gathered if item is not None], dim=row_dim
        )
        slices = [slice(None)] * full_out.dim()
        slices[row_dim] = slice(0, n_token)
        return full_out[tuple(slices)].contiguous()

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
    ) -> torch.Tensor:
        q_proj, k_proj, v_proj = self.attention._prep_qkv(
            q_x=q,
            kv_x=kv,
            apply_scale=True,
        )
        (
            row_start,
            row_end,
            col_start,
            col_end,
            valid_rows,
            valid_cols,
        ) = self._foldcp_valid_ranges(z_spec)
        tile_rows = z_spec.local_shape[z_spec.pair_dims[0]]
        tile_cols = z_spec.local_shape[z_spec.pair_dims[1]]
        local_raw = q.new_zeros(
            *q_proj.shape[:-3],
            tile_rows,
            q_proj.shape[-3],
            q_proj.shape[-1],
        )
        if valid_rows > 0 and valid_cols > 0:
            q_chunk = q[..., row_start:row_end, :]
            q_proj_chunk = q_proj[..., row_start:row_end, :]
            z_chunk = z_local[..., :valid_rows, :valid_cols, :]
            if enable_efficient_fusion:
                layernorm_z_weight = cast(torch.Tensor, self.layernorm_z.weight)
                weight = (self.linear_nobias_z.weight * layernorm_z_weight[None, :])[
                    :, :, None, None
                ]
                bias_local = F.conv2d(permute_final_dims(z_chunk, [2, 0, 1]), weight)
            else:
                bias_local = self.linear_nobias_z(self.layernorm_z(z_chunk))
                bias_local = permute_final_dims(bias_local, [2, 0, 1])
            if extra_attn_bias is not None:
                if extra_attn_bias.shape[-2:] == z_local.shape[-3:-1]:
                    extra_local = extra_attn_bias[..., :valid_rows, :valid_cols]
                else:
                    extra_local = extra_attn_bias[
                        ..., row_start:row_end, col_start:col_end
                    ]
                while len(extra_local.shape) < len(bias_local.shape) - 1:
                    extra_local = extra_local.unsqueeze(dim=0)
                if len(extra_local.shape) == len(bias_local.shape) - 1:
                    extra_local = extra_local.unsqueeze(dim=-3)
                bias_local = bias_local + extra_local.to(
                    dtype=bias_local.dtype,
                    device=bias_local.device,
                )
            bias_local = self._align_bias_to_query(
                bias_local,
                q_chunk,
                n_pair_dims=2,
            ).contiguous()

            if bias_local.shape[-1] != tile_cols:
                bias_tile = bias_local.new_zeros(*bias_local.shape[:-1], tile_cols)
                bias_tile[..., : bias_local.shape[-1]] = bias_local
            else:
                bias_tile = bias_local

            side = mesh.layout.shape[1]
            ring = mesh.ring_comm()
            gathered_bias: list[torch.Tensor | None] = [None for _ in range(side)]
            gathered_bias[mesh.coord[1]] = bias_tile
            ready = bias_tile
            for step in range(1, side):
                ready = ring.comm_row.exchange(ready.contiguous())
                source_col = (mesh.coord[1] + step) % side
                gathered_bias[source_col] = ready
            if any(item is None for item in gathered_bias):
                raise RuntimeError(
                    "failed to collect Fold-CP diffusion attention bias columns."
                )
            row_bias = torch.cat(
                [item for item in gathered_bias if item is not None],
                dim=-1,
            )[..., : q.shape[-2]].contiguous()
            raw_chunk = _attention(
                q=q_proj_chunk,
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
        if block_start < valid_end:
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
                use_efficient_implementation=self.attention.use_efficient_implementation,
                inplace_safe=inplace_safe,
            )
            out = out.movedim(-4, -2).contiguous()
            local_blocks[..., :valid_blocks, :, :, :] = out
        full_blocks = gather_window_blocks(
            local_blocks,
            window_spec,
            mesh.group_2d,
            block_dim=-4,
        )
        full_out = full_blocks.reshape(
            *full_blocks.shape[:-4],
            full_blocks.shape[-4] * full_blocks.shape[-3],
            full_blocks.shape[-2],
            full_blocks.shape[-1],
        )
        if window_spec.q_pad > 0:
            full_out = full_out[..., : -window_spec.q_pad, :, :]
        return self.attention._wrap_up(full_out, q)

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
    ) -> torch.Tensor:
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

        a = self.standard_multihead_attention_foldcp_local_z(
            q=a,
            kv=kv,
            z_local=z_local,
            z_spec=z_spec,
            mesh=mesh,
            extra_attn_bias=extra_attn_bias,
            inplace_safe=inplace_safe,
            enable_efficient_fusion=enable_efficient_fusion,
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
    ) -> torch.Tensor:
        for block in self.blocks:
            a, s, z_local = block.forward_foldcp_local_z(
                a=a,
                s=s,
                z_local=z_local,
                z_spec=z_spec,
                mesh=mesh,
                inplace_safe=inplace_safe,
                extra_attn_bias=extra_attn_bias,
                enable_efficient_fusion=enable_efficient_fusion,
            )
        del s, z_local
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
        for block in self.blocks:
            a, s, z_local = block.forward_foldcp_window(
                a=a,
                s=s,
                z_local=z_local,
                window_spec=window_spec,
                mesh=mesh,
                inplace_safe=inplace_safe,
            )
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

        if z_local.ndim != 3:
            raise ValueError("Fold-CP atom-window pair lookup expects z_local=[T,T,C].")
        if idx_q.ndim != 2 or idx_k.ndim != 2:
            raise ValueError("idx_q and idx_k must be [N_block, N_query/key].")

        idx_q = idx_q.long()
        idx_k = idx_k.long()
        tile_rows = z_spec.local_shape[z_spec.pair_dims[0]]
        tile_cols = z_spec.local_shape[z_spec.pair_dims[1]]
        n_token = z_spec.original_shape[z_spec.pair_dims[0]]
        group_rank = dist.get_rank(mesh.group_2d)
        idx_q_all = [torch.empty_like(idx_q) for _ in range(mesh.layout.numel)]
        idx_k_all = [torch.empty_like(idx_k) for _ in range(mesh.layout.numel)]
        dist.all_gather(idx_q_all, idx_q, group=mesh.group_2d)
        dist.all_gather(idx_k_all, idx_k, group=mesh.group_2d)

        if n_windows is None:
            z_window = z_local.new_zeros(*out.shape[:-1], z_local.shape[-1])
            for cp_rank in range(mesh.layout.numel):
                row_coord, col_coord = mesh.layout.to_coord(cp_rank)
                row_start = row_coord * tile_rows
                col_start = col_coord * tile_cols
                col_end = min(col_start + tile_cols, n_token)
                src_global_rank = dist.get_global_rank(mesh.group_2d, cp_rank)
                for dst_rank in range(mesh.layout.numel):
                    dst_idx_q = idx_q_all[dst_rank]
                    dst_idx_k = idx_k_all[dst_rank]
                    q_in_tile = (dst_idx_q >= row_start) & (
                        dst_idx_q < min(row_start + tile_rows, n_token)
                    )
                    k_in_tile = (dst_idx_k >= col_start) & (dst_idx_k < col_end)
                    dst_needs_tile = bool(q_in_tile.any()) and bool(k_in_tile.any())
                    if not dst_needs_tile:
                        continue

                    if group_rank == cp_rank:
                        q_local = (dst_idx_q - row_start).clamp(0, tile_rows - 1)
                        k_local = (dst_idx_k - col_start).clamp(0, tile_cols - 1)
                        dst_window = z_local[
                            q_local[..., :, None],
                            k_local[..., None, :],
                            :,
                        ]
                        valid_mask = (
                            q_in_tile[..., :, None] & k_in_tile[..., None, :]
                        ).unsqueeze(-1)
                        dst_window = dst_window.masked_fill(
                            ~valid_mask, 0.0
                        ).contiguous()
                        if dst_rank == group_rank:
                            z_window += dst_window
                        else:
                            dst_global_rank = dist.get_global_rank(
                                mesh.group_2d, dst_rank
                            )
                            dist.send(
                                dst_window.contiguous(),
                                dst=dst_global_rank,
                                group=mesh.group_2d,
                            )
                        del dst_window
                    elif group_rank == dst_rank:
                        recv_window = z_local.new_empty(
                            *out.shape[:-1], z_local.shape[-1]
                        )
                        dist.recv(
                            recv_window,
                            src=src_global_rank,
                            group=mesh.group_2d,
                        )
                        z_window += recv_window
                        del recv_window
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

        projected = z_local.new_zeros(*out.shape[:-1], self.c_atompair)
        prefix_shape = out.shape[:-4]
        blocks_per_rank = idx_q.shape[-2]
        local_block_end = block_start + blocks_per_rank
        for source_start in range(0, int(n_windows), int(window_chunk_size)):
            source_end = min(source_start + int(window_chunk_size), int(n_windows))
            overlap_start = max(block_start, source_start)
            overlap_end = min(local_block_end, source_end)
            local_chunk_blocks = max(0, overlap_end - overlap_start)
            local_slice = slice(
                overlap_start - block_start,
                overlap_end - block_start,
            )
            z_window_chunk = z_local.new_zeros(
                *prefix_shape,
                local_chunk_blocks,
                self.n_queries,
                self.n_keys,
                z_local.shape[-1],
            )

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
                    q_in_tile = (dst_idx_q >= row_start) & (
                        dst_idx_q < min(row_start + tile_rows, n_token)
                    )
                    k_in_tile = (dst_idx_k >= col_start) & (dst_idx_k < col_end)
                    dst_needs_tile = bool(q_in_tile.any()) and bool(k_in_tile.any())
                    if not dst_needs_tile:
                        continue

                    if group_rank == cp_rank:
                        q_local = (dst_idx_q - row_start).clamp(0, tile_rows - 1)
                        k_local = (dst_idx_k - col_start).clamp(0, tile_cols - 1)
                        dst_window = z_local[
                            q_local[..., :, None],
                            k_local[..., None, :],
                            :,
                        ]
                        valid_mask = (
                            q_in_tile[..., :, None] & k_in_tile[..., None, :]
                        ).unsqueeze(-1)
                        dst_window = dst_window.masked_fill(
                            ~valid_mask, 0.0
                        ).contiguous()
                        if dst_rank == group_rank:
                            z_window_chunk += dst_window
                        else:
                            dst_global_rank = dist.get_global_rank(
                                mesh.group_2d, dst_rank
                            )
                            dist.send(
                                dst_window.contiguous(),
                                dst=dst_global_rank,
                                group=mesh.group_2d,
                            )
                        del dst_window
                    elif group_rank == dst_rank:
                        recv_window = z_local.new_empty(
                            *prefix_shape,
                            dst_overlap_end - dst_overlap_start,
                            self.n_queries,
                            self.n_keys,
                            z_local.shape[-1],
                        )
                        dist.recv(
                            recv_window,
                            src=src_global_rank,
                            group=mesh.group_2d,
                        )
                        z_window_chunk += recv_window
                        del recv_window

            if local_chunk_blocks > 0:
                z_norm = self.layernorm_z(z_window_chunk)
                local_rows = (
                    int(z_norm.numel() // z_norm.shape[-1]) if z_norm.shape[-1] else 0
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
            del z_window_chunk
        return projected

    def _warmup_foldcp_atom_window_p2p(
        self,
        *,
        mesh: FoldCPProcessMesh,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initialize NCCL P2P communicators before high-memory atom windows."""

        if getattr(self, "_foldcp_atom_window_p2p_warmed", False):
            return
        group_rank = dist.get_rank(mesh.group_2d)
        token = torch.zeros(1, device=device, dtype=dtype)
        for src_rank in range(mesh.layout.numel):
            src_global_rank = dist.get_global_rank(mesh.group_2d, src_rank)
            if group_rank == src_rank:
                for dst_rank in range(mesh.layout.numel):
                    if dst_rank == src_rank:
                        continue
                    dst_global_rank = dist.get_global_rank(mesh.group_2d, dst_rank)
                    dist.send(token, dst=dst_global_rank, group=mesh.group_2d)
            else:
                dist.recv(token, src=src_global_rank, group=mesh.group_2d)
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
                z_token_pair = (
                    self._project_pair_embedding_in_dense_trunk_from_foldcp_local(
                        z_local=z,
                        z_spec=z_spec,
                        idx_q=local_idx_q,
                        idx_k=local_idx_k,
                        mesh=mesh,
                        out=z_token_pair,
                        block_start=block_start,
                        n_windows=int(n_windows),
                        window_chunk_size=64,
                    )
                )
            elif valid_blocks > 0:
                z_valid = gather_pair_embedding_in_dense_trunk(
                    z,
                    idx_q=atom_to_token_idx_q[valid_slice],
                    idx_k=atom_to_token_idx_k[valid_slice],
                )
                z_token_pair[:valid_blocks] = self.linear_no_bias_z(
                    self.layernorm_z(z_valid)
                )
            p_lm = p_lm.unsqueeze(dim=-5)
            if z_token_pair.dim() == p_lm.dim() - 1:
                z_token_pair = z_token_pair.unsqueeze(dim=-5)
            p_lm = p_lm + z_token_pair

        window_spec = FoldCPWindowShardSpec(
            n_atom=int(n_atom),
            n_windows=int(n_windows),
            n_queries=int(self.n_queries),
            n_keys=int(self.n_keys),
            q_pad=int(pad_info["q_pad"]),
            block_range=block_range,
            size_cp=mesh.config.size_cp,
            padded_n_windows=int(blocks_per_rank * mesh.config.size_cp),
        )
        return p_lm.contiguous(), c_l, window_spec

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

        if p_lm is None or c_l is None or window_spec is None:
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
        elif inplace_safe:
            p_lm = p_lm.clone()
            c_l = c_l.clone()

        n_token = None
        if r_l is not None:
            assert s is not None
            n_token = s.size(-2)
            c_l = c_l.unsqueeze(dim=-3) + broadcast_token_to_atom(
                x_token=self.linear_no_bias_s(self.layernorm_s(s)),
                atom_to_token_idx=atom_to_token_idx,
            )
            q_l = c_l + self.linear_no_bias_r(r_l)
        else:
            q_l = c_l.clone()

        c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
            q=c_l,
            k=c_l,
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
        p_lm = self._add_atom_single_context_and_mlp_foldcp_local(
            p_lm=p_lm,
            c_l_q=c_l_q_local,
            c_l_k=c_l_k_local,
            block_start=block_start,
            n_windows=window_spec.n_windows,
            inplace_safe=inplace_safe,
        )

        q_l = self.atom_transformer.forward_foldcp_window(
            q=q_l,
            c=c_l,
            p_local=p_lm,
            window_spec=window_spec,
            mesh=mesh,
            inplace_safe=inplace_safe,
        )
        a = aggregate_atom_to_token(
            x_atom=F.relu(self.linear_no_bias_q(q_l)),
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="mean",
        )
        return a, q_l, c_l, p_lm, window_spec


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
        q = (
            broadcast_token_to_atom(
                x_token=self.linear_no_bias_a(a),
                atom_to_token_idx=atom_to_token_idx,
            )
            + q_skip
        )
        q = self.atom_transformer.forward_foldcp_window(
            q=q,
            c=c_skip,
            p_local=p_skip_local,
            window_spec=window_spec,
            mesh=mesh,
            inplace_safe=inplace_safe,
        )
        q = self.layernorm_q(q)
        return self.linear_no_bias_out(q)
