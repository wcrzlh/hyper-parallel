# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Redistribute one source step across DP before local packing, without buffering."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.batching.step_packing import PackingStep, balanced_step_partitions
from hyper_parallel.distributed_data.schema import SampleKey
from hyper_parallel.distributed_data.transport import DataGroups, DataPlaneTransport


def balance_packing_step(
    step: PackingStep | None, lengths: list[int], workloads: list[int] | None, *, mesh: Any, device: Any,
) -> tuple[PackingStep | None, list[int], list[int] | None]:
    """Apply equal-count KK within a fixed CP/TP coordinate's DP group.

    All readers validate their samples before entering. CP peers already read
    identical source steps; identical metadata ordering gives identical plans
    in their separate DP groups. TP non-readers retain the existing broadcast
    path. No source cursor or pending next-step data is changed.

    Args:
        step: Validated local source step, or None on TP non-readers.
        lengths: Local encoded sequence lengths.
        workloads: Model-derived costs, or None to use the partitioner's proxy.
        mesh: Trainer MeshContext; its data DP axis is independent of EDP/FSDP.
        device: Device for metadata and payload collectives.

    Returns:
        Redistributed step and its lengths and costs in deterministic order.
    """
    if step is None or mesh is None or mesh.dp_size == 1:
        return step, lengths, workloads
    dp_mesh = mesh.device_mesh["dp"]
    group = dp_mesh.get_group()
    ranks = tuple(dist.get_process_group_ranks(group))
    rank = dist.get_rank()
    groups = DataGroups(ranks, group, group, None, ranks[0], True)
    transport = DataPlaneTransport(groups, rank, communication_device=device)
    metadata = transport.all_gather_object((lengths, workloads))
    all_lengths = [length for rank_lengths, _ in metadata for length in rank_lengths]
    all_costs = None if workloads is None else [cost for _, costs in metadata for cost in costs]
    partitions = balanced_step_partitions(all_lengths, len(ranks), workloads=all_costs, equal_size=True)
    # Rank labels do not change the balance. Source order avoids arbitrary
    # permutations of complete DP batches when KK returns equivalent groups.
    partitions.sort(key=lambda indices: indices[0])
    samples = _exchange_samples(step, partitions, ranks, transport)
    local_indices = partitions[ranks.index(rank)]
    local_lengths = [all_lengths[index] for index in local_indices]
    local_costs = None if all_costs is None else [all_costs[index] for index in local_indices]
    return replace(step, samples=samples), local_lengths, local_costs


def _exchange_samples(
    step: PackingStep, partitions: list[list[int]], ranks: tuple[int, ...], transport: DataPlaneTransport,
) -> list[Any]:
    """Exchange complete encoded samples while preserving the plan order."""
    rank = dist.get_rank()
    sample_count = len(step.samples)
    keys = [SampleKey(source, index) for source in ranks for index in range(sample_count)]
    local_payloads = {SampleKey(rank, index): sample for index, sample in enumerate(step.samples)}
    outgoing, retained = _route_samples(partitions, keys, ranks, rank, local_payloads)
    moved = any(keys[index].reader_rank != target
                for target, partition in zip(ranks, partitions) for index in partition)
    if moved:
        retained.update(transport.exchange_prepared(transport.prepare_exchange(outgoing)))
    return [retained[keys[index]] for index in partitions[ranks.index(rank)]]


def _route_samples(
    partitions: list[list[int]], keys: list[SampleKey], ranks: tuple[int, ...], rank: int,
    payloads: dict[SampleKey, Any],
) -> tuple[dict[int, list[tuple[SampleKey, Any]]], dict[SampleKey, Any]]:
    """Retain local samples and send only samples whose DP owner changes."""
    outgoing = {}
    retained = {}
    for target, partition in zip(ranks, partitions):
        owned = [(keys[index], payloads[keys[index]]) for index in partition if keys[index] in payloads]
        if target == rank:
            retained.update(owned)
        elif owned:
            outgoing[target] = owned
    return outgoing, retained
