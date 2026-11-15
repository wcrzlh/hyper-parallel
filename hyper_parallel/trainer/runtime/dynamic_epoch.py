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

from itertools import islice
from typing import Any, Iterator

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.batching import TokenBatchLoader


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
    data_iterator: Any, *, num_micro_batches: int, device: Any, drop_last: bool,
) -> Iterator[Any]:
    """Stage one complete CPU step and agree on EOF before broadcasting model inputs.

    Non-reader TP ranks pass None. At optimizer-step checkpoints all staged
    batches have been consumed, so the existing source cursor plus packing
    buffer remains the complete data state. With drop_last, the shortest DP
    shard bounds the epoch; other shards' remaining packs are omitted.
    """
    batches = list(islice(data_iterator, num_micro_batches)) if data_iterator is not None else []
    available = len(batches) if data_iterator is not None else num_micro_batches
    readiness = torch.tensor([available, -int(bool(batches))], dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.all_reduce(readiness, op=dist.ReduceOp.MIN)
    available, negative_has_data = readiness.tolist()
    if available < num_micro_batches:
        if not drop_last and negative_has_data:
            raise ValueError(
                "TokenBatchLoader has an incomplete distributed optimizer step; "
                "set drop_last=True or choose a batch configuration with complete epoch tails."
            )
        raise StopIteration
    return iter(batches)
