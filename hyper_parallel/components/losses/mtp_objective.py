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
"""Aligned MTP targets and independently normalized training objectives."""

import math
from collections.abc import Mapping
from typing import Any

import torch


def build_mtp_targets(foundation_targets: torch.Tensor, depth: int) -> torch.Tensor:
    """Shift already next-token-aligned global targets without wrapping tails.

    Args:
        foundation_targets: Masked global targets with shape ``[batch, sequence]``.
        depth: Number of future-token prediction stages.

    Returns:
        Targets with shape ``[depth, batch, sequence]``; unavailable targets are -100.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("MTP depth must be a positive integer")
    if foundation_targets.ndim != 2 or foundation_targets.dtype != torch.long:
        raise ValueError("MTP foundation targets must be a rank-two int64 Tensor")
    sequence = foundation_targets.shape[-1]
    targets = foundation_targets.new_full((depth, *foundation_targets.shape), -100)
    for index in range(depth):
        shift = index + 1
        if shift < sequence:
            targets[index, :, :-shift] = foundation_targets[:, shift:]
    return targets


def validate_mtp_loss_weight(weight: float, *, enabled: bool = False) -> float:
    """Validate the sole loss-layer coefficient, rejecting enabled zero-weight training."""
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError("mtp_loss_weight must be a finite nonnegative number")
    value = float(weight)
    if not math.isfinite(value) or value < 0 or (enabled and value == 0):
        raise ValueError("mtp_loss_weight must be finite and nonnegative, and positive when MTP is enabled")
    return value


def mtp_objectives(output: Any, weight: float) -> dict[str, torch.Tensor]:
    """Return local means; trainer DP/CP aggregation owns global token weighting.

    The MTP sum and count combine all depths before normalization. Only the
    weighted objective is returned, so summing the mapping cannot double count it.
    """
    coefficient = validate_mtp_loss_weight(weight, enabled=True)

    def read(name: str) -> torch.Tensor:
        value = output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError(f"MTP output requires scalar Tensor {name}")
        return value

    foundation_sum = read("loss_sum")
    foundation_count = read("valid_token_count")
    mtp_sum = read("mtp_loss_sum")
    mtp_count = read("mtp_valid_token_count")
    return {
        "foundation_loss": foundation_sum / foundation_count.clamp_min(1).to(foundation_sum.dtype),
        "mtp_loss": coefficient * mtp_sum / mtp_count.clamp_min(1).to(mtp_sum.dtype),
    }
