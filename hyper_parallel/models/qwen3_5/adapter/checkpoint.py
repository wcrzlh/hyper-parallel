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
"""Qwen text checkpoint export policy shared by dense and MoE adapters."""

from typing import Any

from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion


def export_text_checkpoint(
    model: Any, state_dict: dict[str, Any], save_original_format: bool,
) -> tuple[dict[str, Any], bool]:
    """Restore native text layouts without restoring a source VLM namespace.

    Args:
        model: Metadata-isolated HF export view.
        state_dict: Gathered model weights.
        save_original_format: Requested policy; text checkpoints always retain
            their native namespace while undoing replacement layouts.

    Returns:
        Native text weights and False to disable further HF reverse conversion.
    """
    del save_original_format
    replacements = getattr(model, "_hp_used_replacement_weight_conversions", None)
    if replacements:
        original_mapping = getattr(model, "_weight_conversions", None)
        try:
            model._weight_conversions = replacements  # pylint: disable=protected-access
            state_dict = revert_weight_conversion(model, state_dict)
        finally:
            model._weight_conversions = original_mapping  # pylint: disable=protected-access
    # Base PrefixChange mappings describe loading text from VLM, not its export format.
    return state_dict, False
