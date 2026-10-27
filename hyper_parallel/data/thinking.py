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
"""Sample-local thinking policy shared by text and processor transforms."""

from collections.abc import Mapping
from typing import Any


def validate_thinking_mode(mode: str) -> None:
    """Reject unknown policies without changing legacy template defaults."""
    if mode not in ("legacy", "auto"):
        raise ValueError("thinking_mode must be 'legacy' or 'auto'")


def has_reasoning(message: Mapping[str, Any]) -> bool:
    """Detect nonempty structured reasoning or a leading think block."""
    reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("reasoning_content must be a string or None")
    if isinstance(reasoning, str):
        return bool(reasoning.strip())
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(block.get("text", "") for block in content if block.get("type") == "text")
    if not isinstance(content, str):
        return False
    content = content.lstrip()
    if not content.startswith("<think>"):
        return False
    if "</think>" not in content:
        raise ValueError("Assistant reasoning has an unclosed <think> block")
    return bool(content[len("<think>"):content.index("</think>")].strip())


def thinking_kwargs(
        message: Mapping[str, Any], mode: str, explicit: bool | None, defaults: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a turn's mode without mutating shared tokenizer/template state.

    Explicit sample flags override automatic detection. Legacy mode with no
    sample flag returns the original options unchanged.
    """
    validate_thinking_mode(mode)
    if explicit is not None and not isinstance(explicit, bool):
        raise ValueError("sample enable_thinking must be a bool or None")
    result = dict(defaults)
    if mode == "legacy" and explicit is None:
        return result
    reasoning = has_reasoning(message)
    enabled = reasoning if explicit is None else explicit
    if not enabled and reasoning:
        raise ValueError("enable_thinking=False conflicts with nonempty assistant reasoning")
    result["enable_thinking"] = enabled
    # Full-history rendering must retain earlier supervised reasoning spans.
    result.setdefault("preserve_thinking", True)
    return result
