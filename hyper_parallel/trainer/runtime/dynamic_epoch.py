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
"""Synchronize online packing epoch tails before any model or TP collectives."""

from __future__ import annotations

from itertools import islice
from typing import Any, Iterator

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.batching import TokenBatchLoader
from hyper_parallel.data.batching.step_packing import PackingStep, balanced_step_partitions


def uses_token_batch_loader(config: Any) -> bool:
    """Identify the configured loader also on TP ranks that own no dataset."""
    loader = getattr(config, "dataloader", None)
    target = getattr(getattr(loader, "target", None), "callable", None)
    return isinstance(target, type) and issubclass(target, TokenBatchLoader)


def uses_source_epoch(config: Any) -> bool:
    """Identify loaders that agree on source exhaustion before each step."""
    loader_config = getattr(config, "dataloader", None)
    step_packing = bool(getattr(loader_config, "step_packing", False))
    requires_source_epoch_sync = uses_token_batch_loader(config) or step_packing
    return requires_source_epoch_sync


def prepare_token_batch_step(
    data_iterator: Any, *, num_micro_batches: int, device: Any, drop_last: bool, step_packing: bool = False,
) -> tuple[Iterator[Any], int]:
    """Stage one complete CPU step and agree on EOF before broadcasting model inputs.

    Non-reader TP ranks pass None. At optimizer-step checkpoints all staged
    batches have been consumed, so the existing source cursor plus packing
    buffer remains the complete data state. Step packing reads one source step,
    then partitions it within max_seq_len and agrees on the computation count.
    With drop_last, the shortest DP shard bounds the epoch; other shards'
    remaining packs are omitted.

    Args:
        data_iterator: Source reader iterator, or None on non-reader ranks.
        num_micro_batches: Original source batch count when step packing is disabled.
        device: Device used for distributed metadata reductions.
        drop_last: Whether incomplete optimizer steps are discarded.
        step_packing: Whether to partition one complete source step.

    Returns:
        The micro-batch iterator and the forward/backward count for this step.
    """
    source_count = 1 if step_packing else num_micro_batches
    batches = list(islice(data_iterator, source_count)) if data_iterator is not None else []
    available = len(batches) if data_iterator is not None else source_count
    readiness = torch.tensor([available, -int(bool(batches))], dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(readiness, op=dist.ReduceOp.MIN)
    available, negative_has_data = readiness.tolist()
    if available < source_count:
        if not drop_last and negative_has_data:
            raise ValueError(
                "TokenBatchLoader has an incomplete distributed optimizer step; "
                "set drop_last=True or choose a batch configuration with complete epoch tails."
            )
        raise StopIteration
    if step_packing:
        return _partition_packing_step(batches[0] if batches else None, device=device)
    return iter(batches), num_micro_batches


def _global_max(values: list[int], device: Any) -> list[int]:
    """Synchronize over every participating rank, including EP and non-readers."""
    tensor = torch.tensor(values, dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.tolist()


def _partition_packing_step(step: PackingStep | None, *, device: Any) -> tuple[Iterator[Any], int]:
    """Agree on nonempty bounded partitions before any model collectives."""
    lengths = []
    error = None
    if step is not None:
        try:
            if not isinstance(step, PackingStep):
                raise ValueError("step_packing loader must yield a PackingStep")
            lengths = step.sample_lengths()
        except (TypeError, ValueError, AttributeError) as exc:
            error = exc
    initial_count = (sum(lengths) + step.max_seq_len - 1) // step.max_seq_len if lengths else 0
    invalid, negative_min_samples, count = _global_max(
        [int(error is not None), -len(lengths) if step is not None else -(2 ** 60), initial_count], device,
    )
    if invalid:
        detail = error or "see source rank error"
        raise ValueError(f"Invalid step packing samples on a source rank: {detail}") from error
    max_count = -negative_min_samples
    while 1 <= count <= max_count:
        partitions = balanced_step_partitions(lengths, count) if lengths else []
        exceeds = any(sum(lengths[index] for index in group) > step.max_seq_len for group in partitions)
        if not _global_max([int(exceeds)], device)[0]:
            return (step.batches(partitions) if step is not None else iter(())), count
        count += 1
    raise ValueError("Cannot fit this step within max_seq_len with the same nonempty micro-batch count on every rank")
