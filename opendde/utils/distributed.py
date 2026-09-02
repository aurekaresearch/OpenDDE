# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import os

import torch


def distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class DistWrapper:
    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.num_nodes = int(self.world_size // self.local_world_size)
        self.node_rank = int(self.rank // self.local_world_size)

    def refresh(self) -> None:
        """Refresh global coordinates after a process group is initialized.

        ``DIST_WRAPPER`` is constructed at import time.  That is correct for a
        normal ``torchrun`` launch, but libraries can initialize the default
        process group programmatically after this module has already been
        imported.  Keep the environment-derived local coordinates and refresh
        the values that PyTorch can authoritatively report.
        """

        if not distributed_available():
            return
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.num_nodes = max(1, self.world_size // self.local_world_size)
        self.node_rank = self.rank // self.local_world_size

    def all_gather_object(self, obj, group=None):
        """Function to gather objects from several distributed processes.
        It is now only used by sync metrics in logger due to security reason.
        """
        if distributed_available():
            group_world_size = torch.distributed.get_world_size(group=group)
        else:
            group_world_size = 1
        if group_world_size > 1:
            with torch.no_grad():
                obj_list = [None for _ in range(group_world_size)]
                torch.distributed.all_gather_object(obj_list, obj, group=group)
                return obj_list
        return [obj]


DIST_WRAPPER = DistWrapper()


def traverse_and_aggregate(dict_list, aggregation_func=None):
    """Traverse list of dicts and merge into a single dict with leaf values joined to list."""
    merged_dict = {}
    all_keys = set().union(*dict_list)
    for key in all_keys:
        agg_value = [m[key] for m in dict_list if key in m]

        if isinstance(agg_value[0], dict):
            merged_dict[key] = traverse_and_aggregate(
                agg_value, aggregation_func=aggregation_func
            )
        else:
            if aggregation_func is not None:
                agg_value = aggregation_func(agg_value)
            merged_dict[key] = agg_value

    return merged_dict


def gather_and_merge(metrics, aggregation_func=None):
    """Gather metrics from ddp workers and aggregate leaf metrics."""
    gathered_metrics = DIST_WRAPPER.all_gather_object(metrics)  # list of metrics
    merged_metrics = traverse_and_aggregate(gathered_metrics, aggregation_func)
    return merged_metrics
