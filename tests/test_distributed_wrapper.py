# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research


def test_dist_wrapper_refreshes_programmatic_process_group(monkeypatch):
    from opendde.utils import distributed

    wrapper = distributed.DistWrapper()
    wrapper.rank = 0
    wrapper.world_size = 1
    monkeypatch.setattr(distributed, "distributed_available", lambda: True)
    monkeypatch.setattr(distributed.torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(
        distributed.torch.distributed,
        "get_world_size",
        lambda group=None: 8,
    )

    wrapper.refresh()

    assert wrapper.rank == 3
    assert wrapper.world_size == 8


def test_all_gather_object_uses_subgroup_size(monkeypatch):
    from opendde.utils import distributed

    wrapper = distributed.DistWrapper()
    wrapper.world_size = 8
    subgroup = object()

    monkeypatch.setattr(distributed, "distributed_available", lambda: True)
    monkeypatch.setattr(
        distributed.torch.distributed,
        "get_world_size",
        lambda *, group: 2 if group is subgroup else 8,
    )

    def gather(output, obj, *, group):
        assert group is subgroup
        output[:] = [obj, "peer"]

    monkeypatch.setattr(distributed.torch.distributed, "all_gather_object", gather)

    assert wrapper.all_gather_object("local", group=subgroup) == ["local", "peer"]
