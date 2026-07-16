# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# pylint: disable=C0114
import os
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from opendde.data.constants import STD_RESIDUES_WITH_GAP
from opendde.distributed.foldcp.config import FoldCPConfig
from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.launch import (
    foldcp_linear_with_source_launch_shape,
    foldcp_module_with_source_launch_shape,
    foldcp_pair_row_slab_linear_with_source_launch_policy,
)
from opendde.distributed.foldcp.msa_pair_weighted import (
    distributed_msa_pair_weighted_average_with_full_value,
    gather_msa_rows_from_cp,
    serial_msa_pair_weighted_average,
)
from opendde.distributed.foldcp.opm import (
    shard_msa_tensor_for_opm,
)
from opendde.distributed.foldcp.pair_sharding import (
    FoldCPPairShardSpec,
    gather_pair_tensor,
    make_pair_shard_spec,
    shard_pair_tensor,
)
from opendde.distributed.foldcp.real_pairformer import (
    distributed_pairformer_block_pair_update,
    distributed_pairformer_stack_pair_update,
    distributed_pairformer_stack_single_bridge_update,
)
from opendde.model.modules.primitives import LinearNoBias, Transition
from opendde.model.modules.transformer import AttentionPairBias
from opendde.model.msa_sampling import subsample_msa_feature_dict_valid_first
from opendde.model.triangular.layers import (
    LayerNorm,
    OuterProductMean,
)
from opendde.model.triangular.triangular import (
    TriangleAttention,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from opendde.model.utils import (
    checkpoint_blocks,
    expand_at_dim,
    is_fp16_enabled,
)


class PairformerBlock(nn.Module):
    """Implements Algorithm 17 [Line2-Line8] in AF3

    c_hidden_mul is set as openfold
    Ref to:
    https://github.com/aqlaboratory/openfold/blob/feb45a521e11af1db241a33d58fb175e207f8ce0/openfold/model/evoformer.py#L123

    Args:
        n_heads (int, optional): number of head [for AttentionPairBias]. Defaults to 16.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_hidden_mul (int, optional): hidden dim [for TriangleMultiplicationOutgoing].
            Defaults to 128.
        c_hidden_pair_att (int, optional): hidden dim [for TriangleAttention]. Defaults to 32.
        no_heads_pair (int, optional): number of head [for TriangleAttention]. Defaults to 4.
        num_intermediate_factor (int, optional): number of intermediate factor for pair_transition. Defaults to 4.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        c_hidden_mul: int = 128,
        c_hidden_pair_att: int = 32,
        no_heads_pair: int = 4,
        num_intermediate_factor: int = 4,
        hidden_scale_up: bool = False,
    ) -> None:
        super(PairformerBlock, self).__init__()
        self.n_heads = n_heads
        if hidden_scale_up:
            no_heads_pair = c_z // c_hidden_pair_att
            c_hidden_mul = c_z
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z=c_z, c_hidden=c_hidden_mul
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z=c_z, c_hidden=c_hidden_mul)
        self.tri_att_start = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.tri_att_end = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.pair_transition = Transition(c_in=c_z, n=num_intermediate_factor)
        self.c_s = c_s
        if self.c_s > 0:
            self.attention_pair_bias = AttentionPairBias(
                has_s=False, create_offset_ln_z=True, n_heads=n_heads, c_a=c_s, c_z=c_z
            )
            self.single_transition = Transition(c_in=c_s, n=4)

    def _maybe_forward_foldcp_pair_only(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor],
        chunk_size: Optional[int] = None,
    ) -> Optional[tuple[Optional[torch.Tensor], torch.Tensor]]:
        """Run a single c_s=0 PairformerBlock through the Fold-CP pair path."""

        if os.environ.get("OPENDDE_FOLDCP_MODE", "single") != "distributed":
            return None
        if self.c_s != 0 or s is not None:
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None

        foldcp = FoldCPConfig.from_runtime_args(
            mode="distributed",
            size_dp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_DP", "1")),
            size_cp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_CP", "4")),
            devices=os.environ.get("OPENDDE_FOLDCP_DEVICES", ""),
            metrics_jsonl=os.environ.get("OPENDDE_FOLDCP_METRICS_JSONL", ""),
        )
        mesh = FoldCPProcessMesh.create(foldcp)
        z_local, z_spec = shard_pair_tensor(z, mesh, pair_dims=(-3, -2))
        del z
        if pair_mask is None:
            mask_local = None
        else:
            mask_local, _ = shard_pair_tensor(pair_mask, mesh, pair_dims=(-2, -1))
        z_local = distributed_pairformer_block_pair_update(
            self,
            z_local,
            mesh,
            mask_local,
            z_spec,
            chunk_size,
        )
        z = gather_pair_tensor(z_local, z_spec, mesh.group_2d)
        return s, z

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        extra_attn_bias: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the PairformerBlock.

        Args:
            s (Optional[torch.Tensor]): single feature
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": Cuequivariance implementation
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuEquivariance implementation
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[Optional[torch.Tensor], torch.Tensor]: the update of s[Optional] and z
                [..., N_token, c_s] | None
                [..., N_token, N_token, c_z]
        """
        foldcp_result = self._maybe_forward_foldcp_pair_only(s, z, pair_mask, chunk_size)
        if foldcp_result is not None:
            return foldcp_result

        if inplace_safe:
            z = self.tri_mul_out(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=True,
                triangle_multiplicative=triangle_multiplicative,
            )
            z = self.tri_mul_in(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=True,
                triangle_multiplicative=triangle_multiplicative,
            )
            z += self.tri_att_start(
                z,
                mask=pair_mask,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()
            z += self.tri_att_end(
                z,
                mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()
            z += self.pair_transition(z)
        else:
            tmu_update = self.tri_mul_out(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=False,
                triangle_multiplicative=triangle_multiplicative,
            )
            z = z + tmu_update
            del tmu_update
            tmu_update = self.tri_mul_in(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=False,
                triangle_multiplicative=triangle_multiplicative,
            )
            z = z + tmu_update
            del tmu_update
            z = z + self.tri_att_start(
                z,
                mask=pair_mask,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()
            z = z + self.tri_att_end(
                z,
                mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()

            z = z + self.pair_transition(z)
        if self.c_s > 0:
            s = s + self.attention_pair_bias(
                a=s,
                s=None,
                z=z,
                extra_attn_bias=extra_attn_bias,
            )
            s = s + self.single_transition(s)
        return s, z


class PairformerStack(nn.Module):
    """
    Implements Algorithm 17 [PairformerStack] in AF3

    Args:
        n_blocks (int, optional): number of blocks [for PairformerStack]. Defaults to 48.
        n_heads (int, optional): number of head [for AttentionPairBias]. Defaults to 16.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        num_intermediate_factor (int, optional): number of intermediate factor for transition. Defaults to 4.
        blocks_per_ckpt (int, optional): number of Pairformer blocks in each activation checkpoint. Defaults to None.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        n_blocks: int = 48,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        num_intermediate_factor: int = 4,
        blocks_per_ckpt: Optional[int] = None,
        hidden_scale_up: bool = False,
    ) -> None:
        super(PairformerStack, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.blocks_per_ckpt = blocks_per_ckpt
        self.blocks = nn.ModuleList()

        for _ in range(n_blocks):
            block = PairformerBlock(
                n_heads=n_heads,
                c_z=c_z,
                c_s=c_s,
                num_intermediate_factor=num_intermediate_factor,
                hidden_scale_up=hidden_scale_up,
            )
            self.blocks.append(block)

    def _maybe_forward_foldcp_pair_only(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor],
        extra_attn_bias: Optional[torch.Tensor],
        chunk_size: Optional[int] = None,
    ) -> Optional[tuple[Optional[torch.Tensor], torch.Tensor]]:
        if os.environ.get("OPENDDE_FOLDCP_MODE", "single") != "distributed":
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None

        foldcp = FoldCPConfig.from_runtime_args(
            mode="distributed",
            size_dp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_DP", "1")),
            size_cp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_CP", "4")),
            devices=os.environ.get("OPENDDE_FOLDCP_DEVICES", ""),
            metrics_jsonl=os.environ.get("OPENDDE_FOLDCP_METRICS_JSONL", ""),
        )
        mesh = FoldCPProcessMesh.create(foldcp)
        if self.blocks and getattr(self.blocks[0], "c_s", 0) != 0:
            if s is None:
                return None
            return distributed_pairformer_stack_single_bridge_update(
                self,
                s,
                z,
                mesh,
                pair_mask,
                extra_attn_bias=extra_attn_bias,
                chunk_size=chunk_size,
            )
        if s is not None:
            return None
        z_local, z_spec = shard_pair_tensor(z, mesh, pair_dims=(-3, -2))
        if pair_mask is None:
            mask_local = None
        else:
            mask_local, _ = shard_pair_tensor(pair_mask, mesh, pair_dims=(-2, -1))
        z_local = distributed_pairformer_stack_pair_update(
            self,
            z_local,
            mesh,
            mask_local,
            z_spec,
            chunk_size,
        )
        z = gather_pair_tensor(z_local, z_spec, mesh.group_2d)
        return s, z

    def _prep_blocks(
        self,
        pair_mask: Optional[torch.Tensor],
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        extra_attn_bias: Optional[torch.Tensor] = None,
    ):
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                triangle_multiplicative=triangle_multiplicative,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                extra_attn_bias=extra_attn_bias,
            )
            for b in self.blocks
        ]
        return blocks

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        extra_attn_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s (Optional[torch.Tensor]): single feature
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuequivariance implementation
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuEquivariance implementation
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the update of s and z
                [..., N_token, c_s]
                [..., N_token, N_token, c_z]
        """
        foldcp_result = self._maybe_forward_foldcp_pair_only(
            s,
            z,
            pair_mask,
            extra_attn_bias,
            chunk_size=chunk_size,
        )
        if foldcp_result is not None:
            return foldcp_result


        blocks = self._prep_blocks(
            pair_mask=pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            extra_attn_bias=extra_attn_bias,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return s, z


class MSAPairWeightedAveraging(nn.Module):
    """
    Implements Algorithm 10 [MSAPairWeightedAveraging] in AF3

    Args:
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c (int, optional): hidden dim [for MSAPairWeightedAveraging]. Defaults to 32.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        n_heads (int, optional): number of heads [for MSAPairWeightedAveraging]. Defaults to 8.
    """

    def __init__(
        self, c_m: int = 64, c: int = 32, c_z: int = 128, n_heads: int = 8
    ) -> None:
        super(MSAPairWeightedAveraging, self).__init__()
        self.c = c
        self.n_heads = n_heads
        # Input projections
        self.layernorm_m = LayerNorm(c_m)
        self.linear_no_bias_mv = LinearNoBias(
            in_features=c_m, out_features=self.c * self.n_heads
        )
        self.layernorm_z = LayerNorm(c_z)
        self.linear_no_bias_z = LinearNoBias(in_features=c_z, out_features=self.n_heads)
        self.linear_no_bias_mg = LinearNoBias(
            in_features=c_m,
            out_features=self.c * self.n_heads,
            initializer="zeros",
        )
        # Weighted average with gating
        self.softmax_w = nn.Softmax(dim=-2)
        # Output projection
        self.linear_no_bias_out = LinearNoBias(
            in_features=self.c * self.n_heads,
            out_features=c_m,
            initializer="zeros",
        )

    def _linear_no_bias_z_source_launch(
        self,
        z: torch.Tensor,
        *,
        original_n: int,
        row_start: int,
        col_start: int = 0,
        valid_rows: Optional[int] = None,
        valid_cols: Optional[int] = None,
    ) -> torch.Tensor:
        return foldcp_pair_row_slab_linear_with_source_launch_policy(
            self.linear_no_bias_z,
            z,
            original_n=original_n,
            row_start=row_start,
            col_start=col_start,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
        )

    def _deterministic_pair_weighted_average(
        self,
        pair_logits: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Match the canonical CP4 row-slab softmax/einsum launch shape."""

        if pair_logits.ndim != 3 or value.ndim != 4:
            weights = self.softmax_w(pair_logits)
            return torch.einsum("...ijh,...mjhc->...mihc", weights, value)

        n_token = int(pair_logits.shape[0])
        row_tile = (n_token + 1) // 2
        row_outputs = []
        value_batch = value.unsqueeze(0)
        for row_coord in range(2):
            row_start = row_coord * row_tile
            valid_rows = max(0, min(row_tile, n_token - row_start))
            logits_local = pair_logits.new_zeros(
                (1, row_tile, n_token, pair_logits.shape[-1])
            )
            if valid_rows > 0:
                logits_local[:, :valid_rows].copy_(
                    pair_logits[row_start : row_start + valid_rows].unsqueeze(0)
                )
            local_output = serial_msa_pair_weighted_average(
                logits_local,
                value_batch,
            ).squeeze(0)
            row_outputs.append(local_output[:, :valid_rows])
        return torch.cat(row_outputs, dim=1)

    def _maybe_forward_foldcp(
        self,
        m: torch.Tensor,
        z_local: torch.Tensor,
        z_pair_spec: FoldCPPairShardSpec,
        mesh: FoldCPProcessMesh,
    ) -> Optional[torch.Tensor]:
        """Run MSA pair weighted averaging with CP-sharded pair logits.

        The serial formula is:

        ``softmax_j(linear_z(layernorm_z(z[i, j]))) @ value[j]``.

        Fold-CP keeps output rows local. It gathers the current row slab over
        source-token shards before the z projection, preserving the serial
        LayerNorm/Linear call shape while still producing only local-row MSA
        updates.
        """

        if m.ndim != 3 or z_local.ndim != 3:
            return None

        m_norm = self.layernorm_m(m)
        v = self.linear_no_bias_mv(m_norm)
        v = v.reshape(*v.shape[:-1], self.n_heads, self.c)
        g = torch.sigmoid(self.linear_no_bias_mg(m_norm))
        g = g.reshape(*g.shape[:-1], self.n_heads, self.c)

        row_dim, col_dim = z_pair_spec.pair_dims
        n_token = z_pair_spec.original_shape[row_dim]
        row_start, row_end = z_pair_spec.row_range
        col_start, col_end = z_pair_spec.col_range
        valid_rows = max(0, min(row_end, n_token) - row_start)
        valid_cols = max(0, min(col_end, z_pair_spec.original_shape[col_dim]) - col_start)

        z_norm_local = foldcp_module_with_source_launch_shape(
            self.layernorm_z,
            z_local,
            source_rows=n_token * n_token,
        )
        pair_logits_local = self._linear_no_bias_z_source_launch(
            z_norm_local,
            original_n=n_token,
            row_start=row_start,
            col_start=col_start,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
        )
        local_wv = distributed_msa_pair_weighted_average_with_full_value(
            pair_logits_local.unsqueeze(0),
            v.unsqueeze(0),
            mesh,
            original_tokens=n_token,
        )
        wv = gather_msa_rows_from_cp(
            local_wv,
            mesh,
            token_dim=2,
            original_tokens=n_token,
        ).squeeze(0)
        o = g * wv
        o = o.reshape(*o.shape[:-2], self.n_heads * self.c)
        return self.linear_no_bias_out(o)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        z_pair_spec: Optional[FoldCPPairShardSpec] = None,
        foldcp_mesh: Optional[FoldCPProcessMesh] = None,
    ) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): msa embedding
                [...,n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [...,n_token, n_token, c_z]
        Returns:
            torch.Tensor: updated msa embedding
                [...,n_msa_sampled, n_token, c_m]
        """
        # Input projections
        m = self.layernorm_m(m)  # [...,n_msa_sampled, n_token, c_m]
        v = self.linear_no_bias_mv(m)  # [...,n_msa_sampled, n_token, n_heads * c]
        v = v.reshape(
            *v.shape[:-1], self.n_heads, self.c
        )  # [...,n_msa_sampled, n_token, n_heads, c]
        g = torch.sigmoid(
            self.linear_no_bias_mg(m)
        )  # [...,n_msa_sampled, n_token, n_heads * c]
        g = g.reshape(
            *g.shape[:-1], self.n_heads, self.c
        )  # [...,n_msa_sampled, n_token, n_heads, c]
        b = self.linear_no_bias_z(
            self.layernorm_z(z)
        )  # [...,n_token, n_token, n_heads]
        if (
            not torch.is_grad_enabled()
            and torch.are_deterministic_algorithms_enabled()
        ):
            wv = self._deterministic_pair_weighted_average(b, v)
        else:
            w = self.softmax_w(b)  # [...,n_token, n_token, n_heads]
            wv = torch.einsum(
                "...ijh,...mjhc->...mihc", w, v
            )  # [...,n_msa_sampled,n_token,n_heads,c]
        o = g * wv
        o = o.reshape(
            *o.shape[:-2], self.n_heads * self.c
        )  # [...,n_msa_sampled, n_token, n_heads * c]
        m = self.linear_no_bias_out(o)  # [...,n_msa_sampled, n_token, c_m]
        if m.shape[-3] > 5120:
            del v, b, g, wv, o
        return m


class MSAStack(nn.Module):
    """
    Implements MSAStack Line7-Line8 in Algorithm 8

    Args:
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c (int, optional): hidden [for MSAStack] dim. Defaults to 8.
        msa_chunk_size (int, optional): chunk size for msa. Defaults to 2048.
    """

    def __init__(
        self,
        c_m: int = 64,
        c_z: int = 128,
        c: int = 8,
        msa_chunk_size: Optional[int] = 2048,
    ) -> None:
        super(MSAStack, self).__init__()
        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
            c_m=c_m, c=c, c_z=c_z
        )
        self.transition_m = Transition(c_in=c_m, n=4)
        self.msa_chunk_size = msa_chunk_size

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        z_pair_spec: Optional[FoldCPPairShardSpec] = None,
        foldcp_mesh: Optional[FoldCPProcessMesh] = None,
    ) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): msa embedding
                [...,n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [...,n_token, n_token, c_z]

        Returns:
            torch.Tensor: updated msa embedding
                [...,n_msa_sampled, n_token, c_m]
        """
        return self.inference_forward(
            m,
            z,
            self.msa_chunk_size,
            z_pair_spec=z_pair_spec,
            foldcp_mesh=foldcp_mesh,
        )

    def inference_forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: Optional[int] = 2048,
        z_pair_spec: Optional[FoldCPPairShardSpec] = None,
        foldcp_mesh: Optional[FoldCPProcessMesh] = None,
    ) -> torch.Tensor:
        """Inplace slice forward for saving memory
        Args:
            m (torch.Tensor): msa embedding
                [..., n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [..., n_token, n_token, c_z]
            chunk_num (int): size of each chunk for checkpointed block execution

        Returns:
            torch.Tensor: updated msa embedding
                [..., n_msa_sampled, n_token, c_m]
        """
        num_msa = m.shape[-3]
        if chunk_size is None:
            chunk_size = max(num_msa, 1)
        no_chunks = num_msa // chunk_size + (num_msa % chunk_size != 0)
        for i in range(no_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_msa)
            # Use inplace to save memory
            if foldcp_mesh is None:
                msa_update = self.msa_pair_weighted_averaging(m[start:end, :, :], z)
            else:
                if z_pair_spec is None:
                    raise ValueError("Fold-CP MSAStack requires z_pair_spec.")
                msa_update = self.msa_pair_weighted_averaging._maybe_forward_foldcp(
                    m[start:end, :, :],
                    z,
                    z_pair_spec,
                    foldcp_mesh,
                )
                if msa_update is None:
                    raise ValueError(
                        "Fold-CP MSAStack currently expects m=[S,N,C] and z_local=[T,T,C]."
                    )
            m[start:end, :, :] += msa_update
            m[start:end, :, :] += self.transition_m(m[start:end, :, :])
        return m


class MSABlock(nn.Module):
    """
    Boltz-style MSA block.

    This variant updates the MSA stack before applying OuterProductMean so the
    pair representation receives the latest MSA information in every block,
    including the final block.

    Args:
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_hidden (int, optional): hidden dim [for MSABlock]. Defaults to 32.
        is_last_block (bool, optional): whether this is the final MSAModule block.
            Defaults to False.
        msa_chunk_size (int, optional): chunk size for msa. Defaults to 2048.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        c_m: int = 64,
        c_z: int = 128,
        c_hidden: int = 32,
        is_last_block: bool = False,
        msa_chunk_size: Optional[int] = 2048,
        hidden_scale_up: bool = False,
    ) -> None:
        super(MSABlock, self).__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.is_last_block = is_last_block

        self.msa_stack = MSAStack(
            c_m=self.c_m,
            c_z=self.c_z,
            msa_chunk_size=msa_chunk_size,
        )

        # Communication
        self.outer_product_mean_msa = OuterProductMean(
            c_m=self.c_m,
            c_z=self.c_z,
            c_hidden=self.c_hidden,
        )
        self.pair_stack = PairformerBlock(
            c_z=c_z,
            c_s=0,
            hidden_scale_up=hidden_scale_up,
        )

    def _foldcp_opm_linear_out_chunked(
        self,
        a_local: torch.Tensor,
        b_local: torch.Tensor,
        mask_local: torch.Tensor,
        mesh: FoldCPProcessMesh,
    ) -> torch.Tensor:
        opm = self.outer_product_mean_msa
        ring = mesh.ring_comm()
        mask_local = mask_local.to(dtype=a_local.dtype)
        a_local = a_local * mask_local.unsqueeze(-1)
        b_local = b_local * mask_local.unsqueeze(-1)

        a_ready = ring.comm_2d_trans.exchange(a_local)
        mask_a_ready = ring.comm_2d_trans.exchange(mask_local)
        a_ready = ring.comm_row_init.exchange(a_ready)
        mask_a_ready = ring.comm_row_init.exchange(mask_a_ready)
        b_ready = ring.comm_col_init.exchange(b_local)
        mask_b_ready = ring.comm_col_init.exchange(mask_local)

        batch, _, n_local, c_hidden = a_local.shape
        local_update = a_local.new_zeros((batch, n_local, n_local, opm.c_z))
        norm = a_local.new_zeros((batch, n_local, n_local))
        channel_chunk = int(os.environ.get("OPENDDE_FOLDCP_OPM_CHANNEL_CHUNK", "4"))
        if channel_chunk <= 0:
            channel_chunk = c_hidden

        weight = opm.linear_out.weight
        for step in range(ring.layout.shape[1]):
            norm = norm + torch.einsum("bsi,bsj->bij", mask_a_ready, mask_b_ready)
            for channel_start in range(0, c_hidden, channel_chunk):
                channel_end = min(channel_start + channel_chunk, c_hidden)
                outer_chunk = torch.einsum(
                    "bsic,bsjd->bijcd",
                    a_ready[..., channel_start:channel_end],
                    b_ready,
                )
                outer_chunk = outer_chunk.reshape(outer_chunk.shape[:-2] + (-1,))
                weight_slice = weight[
                    :,
                    channel_start * c_hidden : channel_end * c_hidden,
                ]
                if outer_chunk.dtype is torch.bfloat16:
                    with torch.amp.autocast("cuda", enabled=False):
                        local_update = local_update + F.linear(
                            outer_chunk,
                            weight_slice.to(dtype=outer_chunk.dtype),
                            None,
                        )
                else:
                    local_update = local_update + F.linear(
                        outer_chunk,
                        weight_slice.to(dtype=outer_chunk.dtype),
                        None,
                    )
            if step < ring.layout.shape[1] - 1:
                a_ready = ring.comm_row.exchange(a_ready)
                mask_a_ready = ring.comm_row.exchange(mask_a_ready)
                b_ready = ring.comm_col.exchange(b_ready)
                mask_b_ready = ring.comm_col.exchange(mask_b_ready)

        if opm.linear_out.bias is not None:
            local_update = local_update + opm.linear_out.bias.to(
                dtype=local_update.dtype
            )
        return local_update / (norm[..., None] + opm.eps)

    def _foldcp_opm_norm(
        self,
        mask_local: torch.Tensor,
        mesh: FoldCPProcessMesh,
    ) -> torch.Tensor:
        ring = mesh.ring_comm()
        mask_ready = ring.comm_2d_trans.exchange(mask_local)
        mask_ready = ring.comm_row_init.exchange(mask_ready)
        mask_b_ready = ring.comm_col_init.exchange(mask_local)
        batch, _, n_local = mask_local.shape
        norm = mask_local.new_zeros((batch, n_local, n_local))
        for step in range(ring.layout.shape[1]):
            norm = norm + torch.einsum("bsi,bsj->bij", mask_ready, mask_b_ready)
            if step < ring.layout.shape[1] - 1:
                mask_ready = ring.comm_row.exchange(mask_ready)
                mask_b_ready = ring.comm_col.exchange(mask_b_ready)
        return norm

    def _foldcp_add_opm_to_local_pair_no_grad(
        self,
        a_local: torch.Tensor,
        b_local: torch.Tensor,
        mask_local: torch.Tensor,
        z_local: torch.Tensor,
        mesh: FoldCPProcessMesh,
    ) -> torch.Tensor:
        opm = self.outer_product_mean_msa
        ring = mesh.ring_comm()
        mask_local = mask_local.to(dtype=a_local.dtype)
        a_local = a_local * mask_local.unsqueeze(-1)
        b_local = b_local * mask_local.unsqueeze(-1)

        norm = self._foldcp_opm_norm(mask_local, mesh)
        denom = norm[..., None] + opm.eps
        squeeze_pair_batch = z_local.ndim == 3
        if squeeze_pair_batch:
            z_local = z_local.unsqueeze(0)
        z_local = z_local.to(dtype=a_local.dtype)
        z_local *= denom

        a_ready = ring.comm_2d_trans.exchange(a_local)
        a_ready = ring.comm_row_init.exchange(a_ready)
        b_ready = ring.comm_col_init.exchange(b_local)

        _, _, _, c_hidden = a_local.shape
        channel_chunk = int(os.environ.get("OPENDDE_FOLDCP_OPM_CHANNEL_CHUNK", "4"))
        if channel_chunk <= 0:
            channel_chunk = c_hidden

        weight = opm.linear_out.weight
        for step in range(ring.layout.shape[1]):
            for channel_start in range(0, c_hidden, channel_chunk):
                channel_end = min(channel_start + channel_chunk, c_hidden)
                outer_chunk = torch.einsum(
                    "bsic,bsjd->bijcd",
                    a_ready[..., channel_start:channel_end],
                    b_ready,
                )
                outer_chunk = outer_chunk.reshape(outer_chunk.shape[:-2] + (-1,))
                weight_slice = weight[
                    :,
                    channel_start * c_hidden : channel_end * c_hidden,
                ]
                if outer_chunk.dtype is torch.bfloat16:
                    with torch.amp.autocast("cuda", enabled=False):
                        z_local += F.linear(
                            outer_chunk,
                            weight_slice.to(dtype=outer_chunk.dtype),
                            None,
                        )
                else:
                    z_local += F.linear(
                        outer_chunk,
                        weight_slice.to(dtype=outer_chunk.dtype),
                        None,
                    )
            if step < ring.layout.shape[1] - 1:
                a_ready = ring.comm_row.exchange(a_ready)
                b_ready = ring.comm_col.exchange(b_ready)

        if opm.linear_out.bias is not None:
            z_local += opm.linear_out.bias.to(dtype=z_local.dtype)
        z_local /= denom
        if squeeze_pair_batch:
            z_local = z_local.squeeze(0)
        return z_local

    def _maybe_foldcp_mesh(self) -> Optional[FoldCPProcessMesh]:
        if os.environ.get("OPENDDE_FOLDCP_MODE", "single") != "distributed":
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None

        foldcp = FoldCPConfig.from_runtime_args(
            mode="distributed",
            size_dp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_DP", "1")),
            size_cp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_CP", "4")),
            devices=os.environ.get("OPENDDE_FOLDCP_DEVICES", ""),
            metrics_jsonl=os.environ.get("OPENDDE_FOLDCP_METRICS_JSONL", ""),
        )
        return FoldCPProcessMesh.create(foldcp)

    def _outer_product_mean_tile_update(
        self,
        m: torch.Tensor,
        mesh_shape: tuple[int, int],
        mesh_coord: tuple[int, int],
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute one canonical OPM pair tile.

        Deterministic single-rank inference and Fold-CP call this same helper so
        CUDA sees the same output shape and MSA reduction shape in both modes.
        """

        leading_shape = m.shape[:-3]
        m_work = m.float() if is_fp16_enabled() else m
        m_flat = m_work.reshape((-1,) + m_work.shape[-3:])
        opm = self.outer_product_mean_msa

        n_token = m_flat.shape[-2]
        mesh_rows, mesh_cols = mesh_shape
        row_tile = (n_token + mesh_rows - 1) // mesh_rows
        col_tile = (n_token + mesh_cols - 1) // mesh_cols
        row_start = mesh_coord[0] * row_tile
        col_start = mesh_coord[1] * col_tile
        row_end = min(row_start + row_tile, n_token)
        col_end = min(col_start + col_tile, n_token)

        local_update = m_flat.new_zeros(
            (m_flat.shape[0], row_tile, col_tile, opm.c_z)
        )
        if row_start < row_end and col_start < col_end:
            ln = opm.layer_norm(m_flat)
            a = opm.linear_1(ln).transpose(-2, -3).contiguous()
            b = opm.linear_2(ln).transpose(-2, -3).contiguous()
            local_outer = torch.einsum(
                "brac,bdae->brdce",
                a[:, row_start:row_end],
                b[:, col_start:col_end],
            )
            local_outer = local_outer.reshape(
                local_outer.shape[:-2] + (opm.c_hidden * opm.c_hidden,)
            )
            local_valid = foldcp_linear_with_source_launch_shape(
                opm.linear_out,
                local_outer,
                source_rows=n_token * n_token,
            )
            mask = m_flat.new_ones(m_flat.shape[:-1]).unsqueeze(-1)
            norm = torch.einsum(
                "bsrc,bsdc->brdc",
                mask[:, :, row_start:row_end],
                mask[:, :, col_start:col_end],
            )
            norm = norm + opm.eps
            local_valid = local_valid / norm
            local_update[
                :,
                : row_end - row_start,
                : col_end - col_start,
            ] = local_valid

        if leading_shape:
            local_update = local_update.reshape(
                leading_shape + local_update.shape[-3:]
            )
        else:
            local_update = local_update.squeeze(0)
        return local_update

    def _foldcp_outer_product_mean_local_update(
        self,
        m: torch.Tensor,
        mesh: FoldCPProcessMesh,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self._outer_product_mean_tile_update(
            m,
            mesh_shape=mesh.layout.shape,
            mesh_coord=mesh.coord,
            chunk_size=chunk_size,
        )

    def _deterministic_outer_product_mean_full_update(
        self,
        m: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Assemble the full OPM update using the canonical CP4 2x2 tiling."""

        grid_shape = (2, 2)
        n_token = m.shape[-2]
        leading_shape = m.shape[:-3]
        m_work = m.float() if is_fp16_enabled() else m
        full_update = m_work.new_empty(
            leading_shape + (n_token, n_token, self.outer_product_mean_msa.c_z)
        )
        row_tile = (n_token + grid_shape[0] - 1) // grid_shape[0]
        col_tile = (n_token + grid_shape[1] - 1) // grid_shape[1]
        for row_coord in range(grid_shape[0]):
            row_start = row_coord * row_tile
            row_end = min(row_start + row_tile, n_token)
            for col_coord in range(grid_shape[1]):
                col_start = col_coord * col_tile
                col_end = min(col_start + col_tile, n_token)
                tile = self._outer_product_mean_tile_update(
                    m,
                    mesh_shape=grid_shape,
                    mesh_coord=(row_coord, col_coord),
                    chunk_size=chunk_size,
                )
                full_update[
                    ..., row_start:row_end, col_start:col_end, :
                ] = tile[
                    ...,
                    : row_end - row_start,
                    : col_end - col_start,
                    :,
                ]
        return full_update

    def _maybe_add_foldcp_outer_product_mean(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        mesh = self._maybe_foldcp_mesh()
        if mesh is None:
            return None
        local_update = self._foldcp_outer_product_mean_local_update(m, mesh)
        z_local, z_spec = shard_pair_tensor(z, mesh, pair_dims=(-3, -2))
        z_local = z_local + local_update
        return gather_pair_tensor(z_local, z_spec, mesh.group_2d)

    def _maybe_forward_foldcp_opm_pair_stack(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor],
        mesh: Optional[FoldCPProcessMesh] = None,
        z_local: Optional[torch.Tensor] = None,
        z_spec: Optional[FoldCPPairShardSpec] = None,
    ) -> Optional[torch.Tensor]:
        mesh = mesh or self._maybe_foldcp_mesh()
        if mesh is None:
            return None

        if z_local is None or z_spec is None:
            z_local, z_spec = shard_pair_tensor(z, mesh, pair_dims=(-3, -2))
        if pair_mask is None:
            mask_local = None
        else:
            mask_local, _ = shard_pair_tensor(pair_mask, mesh, pair_dims=(-2, -1))
        z_local = self._forward_foldcp_local_pair_update(
            m=m,
            z_local=z_local,
            mesh=mesh,
            mask_local=mask_local,
        )
        return gather_pair_tensor(z_local, z_spec, mesh.group_2d)

    def _forward_foldcp_local_pair_update(
        self,
        m: torch.Tensor,
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        mesh: FoldCPProcessMesh,
        mask_local: Optional[torch.Tensor],
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply OPM and the pair stack while keeping `z` as a local CP tile."""

        use_inplace_denom = not torch.are_deterministic_algorithms_enabled()
        if torch.is_grad_enabled() or not use_inplace_denom:
            local_update = self._foldcp_outer_product_mean_local_update(
                m,
                mesh,
                chunk_size=chunk_size,
            )
            z_local = z_local + local_update
        else:
            leading_shape = m.shape[:-3]
            m_work = m.float() if is_fp16_enabled() else m
            m_flat = m_work.reshape((-1,) + m_work.shape[-3:])
            mask_flat = m_flat.new_ones(m_flat.shape[:-1])
            m_local, _ = shard_msa_tensor_for_opm(
                m_flat,
                mesh,
                seq_dim=1,
                token_dim=2,
            )
            mask_local_opm, _ = shard_msa_tensor_for_opm(
                mask_flat,
                mesh,
                seq_dim=1,
                token_dim=2,
            )
            opm = self.outer_product_mean_msa
            local_ln = opm.layer_norm(m_local)
            a_local = opm.linear_1(local_ln)
            b_local = opm.linear_2(local_ln)
            z_local = self._foldcp_add_opm_to_local_pair_no_grad(
                a_local,
                b_local,
                mask_local_opm,
                z_local,
                mesh,
            )
            if leading_shape:
                z_local = z_local.reshape(leading_shape + z_local.shape[-3:])
        z_local = distributed_pairformer_block_pair_update(
            self.pair_stack,
            z_local,
            mesh,
            mask_local,
            z_spec,
            chunk_size,
        )
        return z_local

    def forward_foldcp_local(
        self,
        m: torch.Tensor,
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        mesh: FoldCPProcessMesh,
        mask_local: Optional[torch.Tensor],
        chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Run one MSABlock on CP local pair tiles without gathering full z."""

        m = self.msa_stack(
            m,
            z_local,
            z_pair_spec=z_spec,
            foldcp_mesh=mesh,
        )
        z_local = self._forward_foldcp_local_pair_update(
            m=m,
            z_local=z_local,
            z_spec=z_spec,
            mesh=mesh,
            mask_local=mask_local,
            chunk_size=chunk_size,
        )
        if self.is_last_block:
            return None, z_local
        return m, z_local

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        pair_mask,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            m (torch.Tensor): msa embedding
                [...,n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [...,n_token, n_token, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuequivariance implementation
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuEquivariance implementation
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[Optional[torch.Tensor], torch.Tensor]: updated m and z
                [...,n_msa_sampled, n_token, c_m] | None
                [...,n_token, n_token, c_z]
        """
        # Boltz updates MSA first, then writes the refreshed MSA state back to z.
        mesh = self._maybe_foldcp_mesh()
        if mesh is None:
            m = self.msa_stack(m, z)
            foldcp_z = None
        else:
            z_local, z_spec = shard_pair_tensor(z, mesh, pair_dims=(-3, -2))
            mask_local = (
                None
                if pair_mask is None
                else shard_pair_tensor(pair_mask, mesh, pair_dims=(-2, -1))[0]
            )
            m, z_local = self.forward_foldcp_local(
                m=m,
                z_local=z_local,
                z_spec=z_spec,
                mesh=mesh,
                mask_local=mask_local,
                chunk_size=chunk_size,
            )
            foldcp_z = gather_pair_tensor(z_local, z_spec, mesh.group_2d)
        if foldcp_z is None:
            if (
                not torch.is_grad_enabled()
                and torch.are_deterministic_algorithms_enabled()
            ):
                opm_update = self._deterministic_outer_product_mean_full_update(
                    m,
                    chunk_size=chunk_size,
                )
            else:
                opm_update = self.outer_product_mean_msa(
                    m, inplace_safe=inplace_safe, chunk_size=chunk_size
                )
            z = z + opm_update
            _, z = self.pair_stack(
                s=None,
                z=z,
                pair_mask=pair_mask,
                triangle_multiplicative=triangle_multiplicative,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        else:
            z = foldcp_z
        if self.is_last_block:
            return None, z
        return m, z


class MSAModule(nn.Module):
    """
    Boltz-style MSA module.

    This keeps the AF3 block structure but changes the per-block update order
    so the latest MSA state is always written back into the pair representation
    before the pair stack runs.

    Args:
        n_blocks (int, optional): number of blocks [for MSAModule]. Defaults to 4.
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s_inputs (int, optional):
            hidden dim for single embedding from InputFeatureEmbedder. Defaults to 449.
        blocks_per_ckpt: number of MSAModule blocks in each activation checkpoint. Defaults to 1.
        msa_chunk_size (int, optional): chunk size for msa. Defaults to 2048.
        msa_configs (dict, optional): MSA sampling config. Must define explicit
            ``msa_depth``.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        c_m: int = 64,
        c_z: int = 128,
        c_s_inputs: int = 449,
        blocks_per_ckpt: Optional[int] = 1,
        msa_chunk_size: Optional[int] = 2048,
        msa_configs: Optional[dict[str, Any]] = None,
        hidden_scale_up: bool = False,
    ) -> None:
        super(MSAModule, self).__init__()
        self.n_blocks = n_blocks
        self.c_m = c_m
        self.c_s_inputs = c_s_inputs
        self.blocks_per_ckpt = blocks_per_ckpt
        self.msa_chunk_size = msa_chunk_size

        self.input_feature = {
            "msa": 32,
            "has_deletion": 1,
            "deletion_value": 1,
        }

        if msa_configs is None or "msa_depth" not in msa_configs:
            raise ValueError("MSA config must define msa_depth.")
        self.msa_depth = int(msa_configs["msa_depth"])

        if self.msa_depth <= 0:
            raise ValueError("MSA msa_depth must be positive.")
        self.linear_no_bias_m = LinearNoBias(
            in_features=32 + 1 + 1, out_features=self.c_m
        )

        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_m
        )
        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            block = MSABlock(
                c_m=self.c_m,
                c_z=c_z,
                is_last_block=(i + 1 == n_blocks),
                msa_chunk_size=self.msa_chunk_size,
                hidden_scale_up=hidden_scale_up,
            )
            self.blocks.append(block)

    def _prep_blocks(
        self,
        pair_mask: Optional[torch.Tensor],
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ):
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                triangle_multiplicative=triangle_multiplicative,
                triangle_attention=triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            for b in self.blocks
        ]
        return blocks

    def one_hot_fp32(
        self,
        tensor: torch.Tensor,
        num_classes: int,
        dtype=torch.float32,
    ) -> torch.Tensor:
        """like F.one_hot, but output dtype is float32.

        Args:
            tensor (torch.Tensor): the input tensor
            num_classes (int): num_classes
            dtype (torch.float32, optional): the output dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: the one-hot encoded tensor with shape
                [..., n_msa_sampled, N_token, num_classes]
        """
        shape = tensor.shape
        one_hot_tensor = torch.zeros(
            *shape, num_classes, dtype=dtype, device=tensor.device
        )
        one_hot_tensor.scatter_(len(shape), tensor.unsqueeze(-1), 1)
        return one_hot_tensor

    def _maybe_foldcp_mesh(self) -> Optional[FoldCPProcessMesh]:
        if os.environ.get("OPENDDE_FOLDCP_MODE", "single") != "distributed":
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None
        if torch.is_grad_enabled():
            return None

        foldcp = FoldCPConfig.from_runtime_args(
            mode="distributed",
            size_dp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_DP", "1")),
            size_cp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_CP", "4")),
            devices=os.environ.get("OPENDDE_FOLDCP_DEVICES", ""),
            metrics_jsonl=os.environ.get("OPENDDE_FOLDCP_METRICS_JSONL", ""),
        )
        return FoldCPProcessMesh.create(foldcp)

    def _maybe_forward_foldcp_blocks(
        self,
        msa_sample: torch.Tensor,
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor],
        z_spec: Optional[FoldCPPairShardSpec] = None,
        mesh: Optional[FoldCPProcessMesh] = None,
        z_is_local: bool = False,
        return_local_pair: bool = False,
        chunk_size: Optional[int] = None,
    ) -> Optional[torch.Tensor | tuple[torch.Tensor, FoldCPPairShardSpec]]:
        mesh = mesh or self._maybe_foldcp_mesh()
        if mesh is None:
            return None

        if z_is_local:
            if z_spec is None:
                raise ValueError("z_spec is required when z_is_local=True.")
            z_local = z.contiguous()
        else:
            z_local, z_spec = shard_pair_tensor(z, mesh, pair_dims=(-3, -2))
        if pair_mask is None:
            mask_local = None
        else:
            mask_local, _ = shard_pair_tensor(pair_mask, mesh, pair_dims=(-2, -1))

        m: Optional[torch.Tensor] = msa_sample
        for block in self.blocks:
            if m is None:
                break
            m, z_local = block.forward_foldcp_local(
                m=m,
                z_local=z_local,
                z_spec=z_spec,
                mesh=mesh,
                mask_local=mask_local,
                chunk_size=chunk_size,
            )
        if return_local_pair:
            return z_local.contiguous(), z_spec
        return gather_pair_tensor(z_local, z_spec, mesh.group_2d)

    def _prepare_msa_sample(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        z_token_dim: int,
    ) -> Optional[torch.Tensor]:
        # If n_blocks < 1, return z unchanged.
        if self.n_blocks < 1:
            return None
        if "msa" not in input_feature_dict:
            return None
        if input_feature_dict["msa"].dim() < 2:
            return None

        msa_feat = subsample_msa_feature_dict_valid_first(
            feat_dict=input_feature_dict,
            dim_dict={feat_name: -2 for feat_name in self.input_feature},
            num_msa=self.msa_depth,
            msa_mask=input_feature_dict.get("msa_mask"),
            gap_token=self.input_feature["msa"] - 1,
        )
        # pylint: disable=E1102
        if z_token_dim > 2000:
            msa_feat["msa"] = self.one_hot_fp32(
                msa_feat["msa"],
                num_classes=self.input_feature["msa"],
            )
        else:
            msa_feat["msa"] = torch.nn.functional.one_hot(
                msa_feat["msa"],
                num_classes=self.input_feature["msa"],
            )

        target_shape = msa_feat["msa"].shape[:-1]
        msa_sample = torch.cat(
            [
                msa_feat[name].reshape(*target_shape, d)
                for name, d in self.input_feature.items()
            ],
            dim=-1,
        )
        del msa_feat
        msa_sample = self.linear_no_bias_m(msa_sample)
        return msa_sample + self.linear_no_bias_s(s_inputs)

    def forward_foldcp_local_pair(
        self,
        input_feature_dict: dict[str, Any],
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        s_inputs: torch.Tensor,
        pair_mask: Optional[torch.Tensor],
        mesh: FoldCPProcessMesh,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, FoldCPPairShardSpec]:
        msa_sample = self._prepare_msa_sample(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            z_token_dim=z_spec.original_shape[z_spec.pair_dims[0]],
        )
        if msa_sample is None:
            return z_local.contiguous(), z_spec
        result = self._maybe_forward_foldcp_blocks(
            msa_sample=msa_sample,
            z=z_local,
            pair_mask=pair_mask,
            z_spec=z_spec,
            mesh=mesh,
            z_is_local=True,
            return_local_pair=True,
            chunk_size=chunk_size,
        )
        if result is None:
            raise RuntimeError("Fold-CP local MSA path requires an initialized mesh.")
        return result

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        s_inputs: torch.Tensor,
        pair_mask: torch.Tensor,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (dict[str, Any]):
                input meta feature dict
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_token, c_s_inputs]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuequivariance implementation
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuEquivariance implementation
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the updated z
                [..., N_token, N_token, c_z]
        """
        msa_sample = self._prepare_msa_sample(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            z_token_dim=z.shape[-2],
        )
        if msa_sample is None:
            return z
        foldcp_z = self._maybe_forward_foldcp_blocks(
            msa_sample=msa_sample,
            z=z,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
        )
        if foldcp_z is not None:
            return foldcp_z

        blocks = self._prep_blocks(
            pair_mask=pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        msa_sample, z = checkpoint_blocks(
            blocks,
            args=(msa_sample, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return z


class TemplateEmbedder(nn.Module):
    """
    Implements Algorithm 16 in AF3

    Args:
        n_blocks (int, optional): number of blocks for TemplateEmbedder. Defaults to 2.
        c (int, optional): hidden dim of TemplateEmbedder. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        num_intermediate_factor (int, optional): number of intermediate factor for transition. Defaults to 2.
        blocks_per_ckpt (int, optional): number of TemplateEmbedder/Pairformer blocks in each activation
            checkpoint. Defaults to None.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        n_blocks: int = 2,
        c: int = 64,
        c_z: int = 128,
        num_intermediate_factor: int = 2,
        blocks_per_ckpt: Optional[int] = None,
        hidden_scale_up: bool = False,
    ) -> None:
        super(TemplateEmbedder, self).__init__()
        self.n_blocks = n_blocks
        self.c = c
        self.c_z = c_z
        self.input_feature1 = {
            "template_distogram": 39,
            "template_backbone_frame_mask": 1,
            "template_unit_vector": 3,
            "template_pseudo_beta_mask": 1,
        }
        self.input_feature2 = {
            "template_restype_i": 32,
            "template_restype_j": 32,
        }
        self.distogram = {"max_bin": 50.75, "min_bin": 3.25, "no_bins": 39}
        self.inf = 100000.0

        self.linear_no_bias_z = LinearNoBias(in_features=self.c_z, out_features=self.c)
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_a = LinearNoBias(
            in_features=sum(self.input_feature1.values())
            + sum(self.input_feature2.values()),
            out_features=self.c,
        )
        self.pairformer_stack = PairformerStack(
            c_s=0,
            c_z=c,
            n_blocks=self.n_blocks,
            num_intermediate_factor=num_intermediate_factor,
            blocks_per_ckpt=blocks_per_ckpt,
            hidden_scale_up=hidden_scale_up,
        )
        self.layernorm_v = LayerNorm(self.c)
        self.relu = nn.ReLU()
        self.linear_no_bias_u = LinearNoBias(in_features=self.c, out_features=self.c_z)

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
        triangle_attention: str = "torch",
        triangle_multiplicative: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> Union[torch.Tensor, int]:
        """
        Args:
            input_feature_dict (dict[str, Any]): input feature dict
            z (torch.Tensor): pair embedding
                [..., N_token, N_token, c_z]
            pair_mask (torch.Tensor, optional): pair masking. Default to None.
                [..., N_token, N_token]
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuEquivariance implementation

        Returns:
            torch.Tensor: the template feature
                [..., N_token, N_token, c_z]
        """
        # Do not use TemplateEmbedder by setting n_blocks=0
        if "template_aatype" not in input_feature_dict or self.n_blocks < 1:
            # Compatible with the OpenDDE 0.5.0 model series
            return 0
        asym_id = input_feature_dict["asym_id"]
        multichain_mask = (asym_id[:, None] == asym_id[None, :]).to(z.dtype)

        num_residues = z.shape[0]
        # determine whether the number of templates is the configured maximum value, otherwise error out
        num_templates = input_feature_dict["template_aatype"].shape[0]
        query_num_channels = z.shape[-1]

        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])

        z = self.layernorm_z(z)
        u = 0
        for template_id in range(num_templates):
            u = u + self.single_template_forward(
                template_id=template_id,
                input_feature_dict=input_feature_dict,
                z=z,
                pair_mask=pair_mask,
                multichain_mask=multichain_mask,
                triangle_attention=triangle_attention,
                triangle_multiplicative=triangle_multiplicative,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        u = u / (1e-7 + num_templates)
        u = self.linear_no_bias_u(self.relu(u))
        assert u.shape == (num_residues, num_residues, query_num_channels)
        return u

    @staticmethod
    def _local_pair_mask_from_asym_id(
        asym_id: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        row_start, row_end = z_spec.row_range
        col_start, col_end = z_spec.col_range
        n_token = z_spec.original_shape[z_spec.pair_dims[0]]
        valid_row_end = min(row_end, n_token)
        valid_col_end = min(col_end, n_token)
        local = reference.new_zeros(z_spec.local_shape[:-1])
        if row_start >= valid_row_end or col_start >= valid_col_end:
            return local
        row_ids = asym_id[row_start:valid_row_end]
        col_ids = asym_id[col_start:valid_col_end]
        valid_rows = valid_row_end - row_start
        valid_cols = valid_col_end - col_start
        local[:valid_rows, :valid_cols] = (row_ids[:, None] == col_ids[None, :]).to(
            dtype=reference.dtype,
            device=reference.device,
        )
        return local

    @staticmethod
    def _local_valid_pair_mask(
        z_spec: FoldCPPairShardSpec,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        row_start, row_end = z_spec.row_range
        col_start, col_end = z_spec.col_range
        n_token = z_spec.original_shape[z_spec.pair_dims[0]]
        valid_row_end = min(row_end, n_token)
        valid_col_end = min(col_end, n_token)
        local = reference.new_zeros(z_spec.local_shape[:-1])
        if row_start >= valid_row_end or col_start >= valid_col_end:
            return local
        valid_rows = valid_row_end - row_start
        valid_cols = valid_col_end - col_start
        local[:valid_rows, :valid_cols] = 1
        return local

    @staticmethod
    def _local_restype_pair_features(
        aatype: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_start, row_end = z_spec.row_range
        col_start, col_end = z_spec.col_range
        n_token = z_spec.original_shape[z_spec.pair_dims[0]]
        valid_row_end = min(row_end, n_token)
        valid_col_end = min(col_end, n_token)
        n_restype = len(STD_RESIDUES_WITH_GAP)
        row_local = reference.new_zeros(*z_spec.local_shape[:-1], n_restype)
        col_local = reference.new_zeros(*z_spec.local_shape[:-1], n_restype)
        if row_start >= valid_row_end or col_start >= valid_col_end:
            return col_local, row_local

        valid_rows = valid_row_end - row_start
        valid_cols = valid_col_end - col_start
        aatype = F.one_hot(aatype, num_classes=n_restype).to(
            dtype=reference.dtype,
            device=reference.device,
        )
        row_feat = aatype[row_start:valid_row_end]
        col_feat = aatype[col_start:valid_col_end]
        col_local[:valid_rows, :valid_cols, :] = col_feat[None, :, :].expand(
            valid_rows, valid_cols, n_restype
        )
        row_local[:valid_rows, :valid_cols, :] = row_feat[:, None, :].expand(
            valid_rows, valid_cols, n_restype
        )
        return col_local, row_local

    def _shard_template_pair_feature(
        self,
        tensor: torch.Tensor,
        mesh: FoldCPProcessMesh,
    ) -> torch.Tensor:
        pair_dims = (-3, -2) if tensor.ndim >= 3 else (-2, -1)
        local, _ = shard_pair_tensor(tensor, mesh, pair_dims=pair_dims)
        return local

    def _linear_no_bias_source_stride_tile(
        self,
        linear: torch.nn.Module,
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        *,
        source_rows: Optional[int] = None,
    ) -> torch.Tensor:
        row_dim, col_dim = z_spec.pair_dims
        row_start, row_end = z_spec.row_range
        col_start, col_end = z_spec.col_range
        n_row = z_spec.original_shape[row_dim]
        n_col = z_spec.original_shape[col_dim]
        valid_rows = max(0, min(row_end, n_row) - row_start)
        valid_cols = max(0, min(col_end, n_col) - col_start)
        if valid_rows == 0 or valid_cols == 0:
            return linear(z_local)

        local_rows = z_local.shape[row_dim]
        local_cols = z_local.shape[col_dim]
        valid_local_slices = [slice(None)] * z_local.ndim
        valid_local_slices[row_dim] = slice(0, valid_rows)
        valid_local_slices[col_dim] = slice(0, valid_cols)
        z_valid = z_local[tuple(valid_local_slices)]

        z_projected = foldcp_pair_row_slab_linear_with_source_launch_policy(
            linear,
            z_valid,
            original_n=n_row,
            row_start=row_start,
            col_start=col_start,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
        )

        if valid_rows == local_rows and valid_cols == local_cols:
            return z_projected

        local_projected = linear(z_local)
        output_slices = [slice(None)] * local_projected.ndim
        output_slices[row_dim] = slice(0, valid_rows)
        output_slices[col_dim] = slice(0, valid_cols)
        local_projected[tuple(output_slices)] = z_projected
        return local_projected

    def _linear_no_bias_z_source_stride_tile(
        self,
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
    ) -> torch.Tensor:
        return self._linear_no_bias_source_stride_tile(
            self.linear_no_bias_z,
            z_local,
            z_spec,
            source_rows=(
                z_spec.original_shape[z_spec.pair_dims[0]]
                * z_spec.original_shape[z_spec.pair_dims[1]]
            ),
        )

    def _linear_no_bias_a_source_stride_tile(
        self,
        a_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
    ) -> torch.Tensor:
        return self._linear_no_bias_source_stride_tile(
            self.linear_no_bias_a,
            a_local,
            z_spec,
        )

    def forward_foldcp_local_pair(
        self,
        input_feature_dict: dict[str, Any],
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        mesh: FoldCPProcessMesh,
        pair_mask: Optional[torch.Tensor] = None,
        triangle_attention: str = "torch",
        triangle_multiplicative: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], FoldCPPairShardSpec]:
        if "template_aatype" not in input_feature_dict or self.n_blocks < 1:
            return None, z_spec

        num_templates = input_feature_dict["template_aatype"].shape[0]
        pair_mask_local = (
            self._local_valid_pair_mask(z_spec=z_spec, reference=z_local)
            if pair_mask is None
            else shard_pair_tensor(pair_mask, mesh, pair_dims=(-2, -1))[0]
        )
        multichain_mask_local = self._local_pair_mask_from_asym_id(
            asym_id=input_feature_dict["asym_id"],
            z_spec=z_spec,
            reference=z_local,
        )

        z_norm_local = self.layernorm_z(z_local)
        u_local = z_local.new_zeros(*z_spec.local_shape[:-1], self.c)
        for template_id in range(num_templates):
            v_local = self.single_template_forward_foldcp_local(
                template_id=template_id,
                input_feature_dict=input_feature_dict,
                z_local=z_norm_local,
                z_spec=z_spec,
                mesh=mesh,
                pair_mask_local=pair_mask_local,
                multichain_mask_local=multichain_mask_local,
                chunk_size=chunk_size,
            )
            u_local = u_local + v_local
        u_local = u_local / (1e-7 + num_templates)
        u_local = self.linear_no_bias_u(self.relu(u_local))
        return u_local.contiguous(), z_spec

    def single_template_forward_foldcp_local(
        self,
        template_id: int,
        input_feature_dict: dict[str, Any],
        z_local: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        mesh: FoldCPProcessMesh,
        pair_mask_local: torch.Tensor,
        multichain_mask_local: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        dgram = self._shard_template_pair_feature(
            input_feature_dict["template_distogram"][template_id],
            mesh,
        )
        pseudo_beta_mask_2d = self._shard_template_pair_feature(
            input_feature_dict["template_pseudo_beta_mask"][template_id],
            mesh,
        )
        dgram = dgram * multichain_mask_local[..., None] * pair_mask_local[..., None]
        pseudo_beta_mask_2d = (
            pseudo_beta_mask_2d * multichain_mask_local * pair_mask_local
        )

        aatype_col, aatype_row = self._local_restype_pair_features(
            input_feature_dict["template_aatype"][template_id],
            z_spec=z_spec,
            reference=z_local,
        )

        unit_vector = self._shard_template_pair_feature(
            input_feature_dict["template_unit_vector"][template_id],
            mesh,
        )
        unit_vector = (
            unit_vector * multichain_mask_local[..., None] * pair_mask_local[..., None]
        )

        backbone_mask_2d = self._shard_template_pair_feature(
            input_feature_dict["template_backbone_frame_mask"][template_id],
            mesh,
        )
        backbone_mask_2d = backbone_mask_2d * multichain_mask_local * pair_mask_local

        at = torch.concat(
            [
                dgram,
                pseudo_beta_mask_2d.unsqueeze(-1),
                aatype_col,
                aatype_row,
                unit_vector,
                backbone_mask_2d.unsqueeze(-1),
            ],
            dim=-1,
        )
        v_local = self._linear_no_bias_z_source_stride_tile(
            z_local,
            z_spec,
        ) + self._linear_no_bias_a_source_stride_tile(at, z_spec)
        v_local = distributed_pairformer_stack_pair_update(
            self.pairformer_stack,
            v_local,
            mesh,
            pair_mask_local,
            z_spec,
            chunk_size,
        )
        return self.layernorm_v(v_local)

    def single_template_forward(
        self,
        template_id: int,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
        multichain_mask: Optional[torch.Tensor] = None,
        triangle_attention: str = "torch",
        triangle_multiplicative: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        assert pair_mask is not None
        assert multichain_mask is not None
        to_concat = []

        dgram = input_feature_dict["template_distogram"][
            template_id
        ]  # [N_token, N_token, 39]
        pseudo_beta_mask_2d = input_feature_dict["template_pseudo_beta_mask"][
            template_id
        ]
        dgram = dgram * multichain_mask[..., None] * pair_mask[..., None]
        pseudo_beta_mask_2d = (
            pseudo_beta_mask_2d * multichain_mask * pair_mask
        )  # [N_token, N_token]
        to_concat.append(dgram)
        to_concat.append(pseudo_beta_mask_2d.unsqueeze(-1))

        aatype = input_feature_dict["template_aatype"][template_id]  # [N_token]
        aatype = F.one_hot(aatype, num_classes=len(STD_RESIDUES_WITH_GAP))
        to_concat.append(expand_at_dim(aatype, dim=-3, n=z.shape[0]))
        to_concat.append(expand_at_dim(aatype, dim=-2, n=z.shape[0]))

        unit_vector = input_feature_dict["template_unit_vector"][template_id]
        unit_vector = (
            unit_vector * multichain_mask[..., None] * pair_mask[..., None]
        )  # [N_token, N_token, 3]
        to_concat.append(unit_vector)

        backbone_mask_2d = input_feature_dict["template_backbone_frame_mask"][
            template_id
        ]
        backbone_mask_2d = backbone_mask_2d * multichain_mask * pair_mask
        to_concat.append(backbone_mask_2d.unsqueeze(-1))

        at = torch.concat(to_concat, dim=-1)
        v = self.linear_no_bias_z(z) + self.linear_no_bias_a(at)
        _, v = self.pairformer_stack(
            s=None,
            z=v,
            pair_mask=pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        v = self.layernorm_v(v)
        return v
