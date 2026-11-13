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
"""Keep Qwen vision FSDP collectives aligned for text-only training batches."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature
from typing import Any, Iterator

# This adapter implements the Torch-only Transformers multimodal contract.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest, _commit_forward_rewrite

_VISION_ENCODED = ContextVar("qwen_vision_encoded", default=False)


@contextmanager
def vision_already_encoded() -> Iterator[None]:
    """Suppress dummy work when CP or encoder parallel already ran the vision branch."""
    token = _VISION_ENCODED.set(True)
    try:
        yield
    finally:
        _VISION_ENCODED.reset(token)


def _attach_dummy_vision(visual: nn.Module, embeddings: torch.Tensor) -> torch.Tensor:
    """Execute one spatial merge group and connect every visual parameter to the loss."""
    config = visual.config
    merge = config.spatial_merge_size
    patch_width = config.in_channels * config.temporal_patch_size * config.patch_size ** 2
    pixels = torch.zeros(merge ** 2, patch_width, device=embeddings.device, dtype=visual.dtype)
    grid = torch.tensor([[1, merge, merge]], device=embeddings.device, dtype=torch.long)
    features = visual(pixels, grid_thw=grid, return_dict=True).pooler_output
    anchor = features.float().mean() * 0.0
    return embeddings + anchor.to(embeddings.dtype)


def install_dummy_vision_forward(model: nn.Module) -> None:
    """Attach zero-valued visual dependencies at the native multimodal base boundary.

    Args:
        model: Qwen conditional-generation model with a native ``model.visual``.

    The embedding hook preserves input IDs and native RoPE bookkeeping. It also
    preserves the embedding -> vision -> decoder FSDP call order on text ranks.
    Precomputed embeddings use the same dependency directly. CP and encoder
    parallel callers suppress this fallback after handling their own vision work.
    """
    base = model.model
    if getattr(base, "_hp_dummy_vision_installed", False):
        return
    original_forward = base.forward
    forward_signature = signature(original_forward)
    pending = ContextVar("qwen_dummy_vision_pending", default=False)

    def embedding_hook(_module: nn.Module, _args: tuple, output: torch.Tensor) -> torch.Tensor:
        """Execute vision after the native text embedding, once per base forward.

        Args:
            _module: Input embedding module invoking this hook.
            _args: Original positional inputs to the embedding module.
            output: Text embeddings to connect to the dummy vision graph.
        """
        if pending.get():
            pending.set(False)
            return _attach_dummy_vision(base.visual, output)
        return output

    @wraps(original_forward)
    def forward(*args: Any, **kwargs: Any) -> Any:
        """Preserve native IDs, positions, outputs, and the real-image/video branches."""
        if not base.training or _VISION_ENCODED.get():
            return original_forward(*args, **kwargs)
        bound = forward_signature.bind(*args, **kwargs) if args else None
        inputs = bound.arguments if bound is not None else kwargs
        if inputs.get("pixel_values") is not None or inputs.get("pixel_values_videos") is not None:
            return original_forward(*args, **kwargs)
        if inputs.get("inputs_embeds") is not None:
            inputs["inputs_embeds"] = _attach_dummy_vision(base.visual, inputs["inputs_embeds"])
            return original_forward(*bound.args, **bound.kwargs) if bound is not None else original_forward(**inputs)
        token = pending.set(True)
        try:
            return original_forward(*args, **kwargs)
        finally:
            pending.reset(token)

    handle = base.get_input_embeddings().register_forward_hook(embedding_hook)
    try:
        _commit_forward_rewrite(_ForwardRewriteRequest(
            base, forward, companion_attrs={"_hp_dummy_vision_installed": True},
        ))
    except Exception:
        handle.remove()
        raise
