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
"""MTP future targets and loss-count aliases over the generic runtime-input protocol."""

from __future__ import annotations

# Torch-only model integration; there is no backend-dispatch layer.
# pylint: disable=forbidden-backend-import
from collections.abc import Mapping
from typing import Any

from torch.nn import functional

from hyper_parallel.components.losses.mtp_objective import build_mtp_targets
from hyper_parallel.data.batching.runtime_input import (
    RuntimeInputAdapter,
    RuntimeInputContext,
)
from hyper_parallel.models.qwen3_5_moe.adapter.mtp_runtime import _gather_metadata


class QwenMTPRuntime(RuntimeInputAdapter):
    """Shift targets globally before selecting the local CP interval.

    Source movement, padding and CP slicing remain owned by the generic batch.
    The declared loss alias exposes the same future labels to token counting.
    """

    def __init__(self, depth: int = 1, labels_are_shifted: bool = True) -> None:
        """Declare checkpoint MTP depth and the source label alignment."""
        self.depth = depth
        self.labels_are_shifted = labels_are_shifted

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the model-owned future-target field."""
        return ("mtp_targets",)

    def loss_input_aliases(self) -> Mapping[str, str]:
        """Expose an alias for independent future-token objective counting."""
        return {"mtp_labels": "mtp_targets"}

    def build(self, *, batch: Mapping[str, Any], parallel_context: Any,
              options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Resolve the explicit CP group without retaining mutable batch state."""
        context = RuntimeInputContext.from_batch(batch=batch, parallel_context=parallel_context, options=options)
        group = parallel_context.cp_mesh.get_group() if parallel_context.cp_world_size > 1 else None
        return self._build_targets(batch, context, group)

    def build_runtime_inputs(self, *, batch: Mapping[str, Any], context: RuntimeInputContext) -> Mapping[str, Any]:
        """Support standalone CP=1 runtime use without an initialized communicator."""
        if context.parallel_sizes.get("cp", 1) != 1:
            raise ValueError("CP MTP requires build() with its DataLoader parallel context")
        return self._build_targets(batch, context, None)

    def _build_targets(self, batch: Mapping[str, Any], context: RuntimeInputContext,
                       cp_group: Any) -> dict[str, Any]:
        targets = batch["labels"]
        mask = batch.get("loss_mask")
        if mask is not None:
            targets = targets.masked_fill(~mask.bool(), -100)
        if not self.labels_are_shifted:
            targets = targets[..., 1:]
            if targets.shape[-1] == context.local_input_shape[1] - 1:
                targets = functional.pad(targets, (0, 1), value=-100)
        if targets.shape != tuple(context.local_input_shape):
            raise ValueError("MTP targets must align with every local input position")
        if cp_group is not None:
            targets = _gather_metadata(targets, cp_group)
        boundaries = batch.get("cu_seq_lens")
        if boundaries is not None and boundaries.numel() > 2:
            # Generic collate may append a separate, wholly ignored tail pad.
            # Real packed documents remain unsupported; never shift labels across them.
            real_end = int(boundaries[1])
            if not bool(targets[..., real_end:].eq(-100).all()):
                raise NotImplementedError("MTP packed multi-document training is not qualified")
        future = build_mtp_targets(targets, self.depth)
        start = context.parallel_ranks.get("cp", 0) * context.local_input_shape[1]
        return {"mtp_targets": future[..., start:start + context.local_input_shape[1]].contiguous()}
