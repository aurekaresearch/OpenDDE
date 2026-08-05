# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP migration scaffolding for OpenDDE inference."""

from opendde.distributed.foldcp.config import FoldCPConfig
from opendde.distributed.foldcp.comm import One2OneComm, Ring2DComm, TransposeComm
from opendde.distributed.foldcp.layout import FoldCP2DLayout
from opendde.distributed.foldcp.metrics import (
    FoldCPBenchmarkRecorder,
    FoldCPStageMetric,
    measure_foldcp_stage,
)
from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.msa_pair_weighted import (
    distributed_msa_pair_weighted_average,
    serial_msa_pair_weighted_average,
    shard_msa_value_by_token,
)
from opendde.distributed.foldcp.structural_pair import (
    distributed_structural_pair_context,
    serial_structural_pair_context,
    structural_role_pair_type,
)
from opendde.distributed.foldcp.atom_window import (
    FoldCPWindowShardSpec,
    atom_window_token_indices,
    distributed_atom_window_attention,
    distributed_atom_window_pair_context,
    gather_window_attention_output,
    gather_window_blocks,
    serial_atom_window_attention,
    serial_atom_window_pair_context,
    window_block_range,
)
from opendde.distributed.foldcp.confidence import (
    add_confidence_distance_embedding_local,
    distributed_confidence_pair_logits,
)
from opendde.distributed.foldcp.pair_sharding import (
    FoldCPPairShardSpec,
    gather_pair_tensor,
    gather_pair_tensor_like,
    gather_pair_tensor_like_to_rank,
    infer_pair_dims,
    shard_pair_feature_dict,
    shard_pair_tensor,
)
from opendde.distributed.foldcp.real_pairformer import (
    distributed_pairformer_block_pair_update,
    distributed_pairformer_stack_single_bridge_update,
    distributed_pairformer_stack_pair_update,
    distributed_pair_transition_update,
    distributed_triangle_attention_update,
    distributed_triangle_multiplication_update,
)
from opendde.distributed.foldcp.opm import (
    FoldCPMSAShardSpec,
    serial_outer_product_mean,
    shard_msa_tensor_for_opm,
)
from opendde.distributed.foldcp.triangular_mult import (
    TriangleMultiplicationDirection,
    distributed_triangle_multiplication,
    serial_triangle_multiplication,
)
from opendde.distributed.foldcp.online_softmax import online_softmax_update
from opendde.distributed.foldcp.triangle_attention import (
    distributed_ring_attention,
    distributed_triangle_attention_ending,
    distributed_triangle_attention_starting,
    serial_triangle_attention_ending,
    serial_triangle_attention_starting,
)

__all__ = [
    "FoldCPBenchmarkRecorder",
    "FoldCPConfig",
    "FoldCP2DLayout",
    "FoldCPPairShardSpec",
    "FoldCPWindowShardSpec",
    "FoldCPMSAShardSpec",
    "FoldCPProcessMesh",
    "FoldCPStageMetric",
    "One2OneComm",
    "Ring2DComm",
    "TransposeComm",
    "distributed_msa_pair_weighted_average",
    "distributed_atom_window_attention",
    "distributed_atom_window_pair_context",
    "distributed_structural_pair_context",
    "distributed_confidence_pair_logits",
    "add_confidence_distance_embedding_local",
    "TriangleMultiplicationDirection",
    "distributed_ring_attention",
    "distributed_triangle_attention_update",
    "distributed_triangle_attention_ending",
    "distributed_triangle_attention_starting",
    "distributed_triangle_multiplication",
    "distributed_triangle_multiplication_update",
    "gather_pair_tensor",
    "gather_pair_tensor_like",
    "gather_pair_tensor_like_to_rank",
    "gather_window_attention_output",
    "gather_window_blocks",
    "infer_pair_dims",
    "measure_foldcp_stage",
    "online_softmax_update",
    "distributed_pairformer_block_pair_update",
    "distributed_pairformer_stack_single_bridge_update",
    "distributed_pairformer_stack_pair_update",
    "distributed_pair_transition_update",
    "serial_outer_product_mean",
    "serial_triangle_multiplication",
    "serial_triangle_attention_ending",
    "serial_triangle_attention_starting",
    "serial_msa_pair_weighted_average",
    "serial_atom_window_attention",
    "serial_atom_window_pair_context",
    "serial_structural_pair_context",
    "shard_msa_value_by_token",
    "structural_role_pair_type",
    "shard_pair_feature_dict",
    "shard_pair_tensor",
    "shard_msa_tensor_for_opm",
    "atom_window_token_indices",
    "window_block_range",
]
