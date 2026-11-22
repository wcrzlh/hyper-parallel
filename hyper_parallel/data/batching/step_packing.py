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
"""Deterministic, budgeted partitions within one source optimizer step."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch  # pylint: disable=forbidden-backend-import


PACKING_PLAN_VERSION = 2


def validate_step_packing_length(max_seq_len: int | None, collate_fn: Any) -> None:
    """Require a global token limit compatible with the collator's alignment.

    Args:
        max_seq_len: Maximum packed token count before CP slicing.
        collate_fn: Collator supplying the required sequence alignment.
    """
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("step_packing requires a positive data_transform.max_seq_len")
    alignment = getattr(collate_fn, "alignment_size", 1)
    if max_seq_len % alignment:
        raise ValueError("step_packing max_seq_len must be divisible by the collator's TP * CP alignment")


def balanced_step_partitions(
    lengths: Sequence[int], num_partitions: int, *, workloads: Sequence[int] | None = None,
) -> list[list[int]]:
    """Balance indivisible samples using multiway Karmarkar-Karp differencing.

    Use model-derived workloads when available, otherwise retain the legacy
    verl proxy for model families without an analytical estimate.
    The caller checks the actual token budget separately: balanced workloads
    do not imply bounded token sums. Ties use source indices deterministically.

    Args:
        lengths: Positive encoded lengths in source order.
        num_partitions: Number of nonempty groups to construct.
        workloads: Optional positive per-sample computation estimates.
    """
    if not 1 <= num_partitions <= len(lengths) or any(length <= 0 for length in lengths):
        raise ValueError("Partitions must be nonempty and sample lengths positive")
    if workloads is not None and (len(workloads) != len(lengths) or min(workloads, default=0) <= 0):
        raise ValueError("Workloads must contain one positive cost per sample")
    if num_partitions == 1:
        return [list(range(len(lengths)))]
    if workloads is None:
        workloads = [24576 * length + length * length for length in lengths]
    heap = []
    serial = 0
    for index, cost in enumerate(workloads):
        groups = [(cost, (index,))] + [(0, ())] * (num_partitions - 1)
        heapq.heappush(heap, (-cost, serial, groups))
        serial += 1
    while len(heap) > 1:
        _, _, left = heapq.heappop(heap)
        _, _, right = heapq.heappop(heap)
        groups = sorted(
            [(a[0] + b[0], tuple(sorted(a[1] + b[1]))) for a, b in zip(left, reversed(right))],
            reverse=True,
        )
        heapq.heappush(heap, (-(groups[0][0] - groups[-1][0]), serial, groups))
        serial += 1
    partitions = [list(indices) for _, indices in heap[0][2]]
    partitions.sort(key=lambda group: (sum(workloads[index] for index in group), group[0]), reverse=True)
    return partitions[::2][::-1] + partitions[1::2]


@dataclass
class PackingStep:
    """Stage complete CPU samples; collate only after ranks agree on a plan.

    This object never enters the model. It is yielded by the main-process
    loader iterator, so its callable and tensors need not cross worker queues.
    """

    samples: Sequence[Mapping[str, Any]]
    collate_fn: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]
    max_seq_len: int

    def sample_lengths(self) -> list[int]:
        """Read real encoded lengths without counting alignment padding."""
        lengths = []
        for sample in self.samples:
            ids = sample.get("input_ids")
            if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.numel() == 0:
                raise ValueError("Step packing requires nonempty one-dimensional encoded input_ids")
            lengths.append(ids.numel())
        if not lengths or max(lengths) > self.max_seq_len:
            raise ValueError("Step packing samples must fit data_transform.max_seq_len after transformation")
        return lengths

    def batches(self, partitions: Sequence[Sequence[int]]) -> Iterator[Mapping[str, Any]]:
        """Collate each micro-batch lazily, keeping all samples in this step.

        Args:
            partitions: Groups of source sample indices agreed across ranks.
        """
        for partition in partitions:
            batch = self.collate_fn([self.samples[index] for index in partition])
            ids = batch.get("input_ids")
            if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1
                    or ids.shape[1] > self.max_seq_len or "cu_seq_lens" not in batch):
                raise ValueError("Step packing collator must produce [1, tokens <= max_seq_len] and cu_seq_lens")
            yield batch


def packing_checkpoint_fields(max_seq_len: int) -> dict[str, int]:
    """Record the recipe length and deterministic partition algorithm version.

    Args:
        max_seq_len: Maximum packed token count recorded with the source cursor.
    """
    return {"step_packing_max_seq_len": max_seq_len, "step_packing_plan_version": PACKING_PLAN_VERSION}


def validate_packing_checkpoint(state: Mapping[str, Any], max_seq_len: int) -> None:
    """Allow legacy source cursors, but reject incompatible new packing plans.

    Args:
        state: Saved loader state, possibly predating partition metadata.
        max_seq_len: Current maximum packed token count.
    """
    for key, expected in packing_checkpoint_fields(max_seq_len).items():
        if key in state and state[key] != expected:
            raise ValueError(f"Step packing resume requires unchanged {key}: saved={state[key]}, current={expected}")
