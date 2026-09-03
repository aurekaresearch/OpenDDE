# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import torch
import torch.nn as nn

from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.pair_sharding import FoldCPPairShardSpec
from opendde.model.modules.primitives import Linear


# Adapted From openfold.model.heads
class DistogramHead(nn.Module):
    """Implements Algorithm 1 [Line17] in AF3

    Computes a distogram probability distribution.
    For use in computation of distogram bin probabilities, subsection 1.9.8 (AF2)

    Args:
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        no_bins (int, optional): Number of distogram bins. Defaults to 64.
    """

    def __init__(self, c_z: int = 128, no_bins: int = 64) -> None:
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(
            in_features=self.c_z, out_features=self.no_bins, initializer="zeros"
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # [*, N, N, C_z]
        """
        Args:
            z (torch.Tensor): pair embedding
                [*, N_token, N_token, C_z]

        Returns:
            torch.Tensor: distogram probability distribution
                [*, N_token, N_token, no_bins]
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits

    def contact_probs_foldcp_local(
        self,
        z_pair_local: torch.Tensor,
        z_pair_spec: FoldCPPairShardSpec,
        mesh: FoldCPProcessMesh,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        thres: float = 8.0,
        gather_to_rank0_only: bool = False,
    ) -> torch.Tensor | None:
        from opendde.distributed.foldcp.distogram import (
            distributed_distogram_contact_probs,
        )

        return distributed_distogram_contact_probs(
            z_pair_local=z_pair_local,
            z_pair_spec=z_pair_spec,
            mesh=mesh,
            linear=self.linear,
            min_bin=min_bin,
            max_bin=max_bin,
            no_bins=no_bins,
            thres=thres,
            gather_to_rank0_only=gather_to_rank0_only,
        )

    def contact_probs_foldcp_from_full_pair(
        self,
        z_pair: torch.Tensor,
        mesh: FoldCPProcessMesh,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        thres: float = 8.0,
        gather_to_rank0_only: bool = False,
    ) -> torch.Tensor | None:
        from opendde.distributed.foldcp.distogram import (
            distributed_distogram_contact_probs_from_full_pair,
        )

        return distributed_distogram_contact_probs_from_full_pair(
            z_pair=z_pair,
            mesh=mesh,
            linear=self.linear,
            min_bin=min_bin,
            max_bin=max_bin,
            no_bins=no_bins,
            thres=thres,
            gather_to_rank0_only=gather_to_rank0_only,
        )
