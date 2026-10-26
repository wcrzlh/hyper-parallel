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
"""Training bridge from CP-local backbone activations to parallel MTP decoders."""

from collections.abc import Mapping
from dataclasses import replace
from functools import wraps
from typing import Any

import torch
from torch import distributed as dist
from torch.nn import functional as F
from transformers.masking_utils import create_causal_mask

from hyper_parallel.components.losses.chunked_cross_entropy import ChunkedCausalLMOutput, chunked_cross_entropy
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import TP, ModuleShardingSpec
from hyper_parallel.models.qwen3_5_moe.adapter.mtp import Qwen3_5MoeMTP


def mtp_distributed_setup(setup: Any) -> Any:
    """Declare the fusion projection through the existing model override interface."""
    overrides = dict(setup.plan_overrides or {})
    overrides.setdefault("mtp.fc", ModuleShardingSpec(
        params={"weight": {TP: Replicate()}}, in_src={}, in_dst={}, is_boundary=False))
    return replace(setup, plan_overrides=overrides)


class _GatherSequence(torch.autograd.Function):
    """Gather sequences with SUM/scatter adjoint, without averaging remote losses."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: Any) -> torch.Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.size = dist.get_world_size(group)
        parts = [torch.empty_like(tensor) for _ in range(ctx.size)]
        dist.all_gather(parts, tensor.contiguous(), group=group)
        return torch.cat(parts, dim=1)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Clone because the autograd-supplied gradient can alias another consumer.
        summed = grad.contiguous().clone()
        dist.all_reduce(summed, op=dist.ReduceOp.SUM, group=ctx.group)
        return summed.chunk(ctx.size, dim=1)[ctx.rank].contiguous().clone(), None


def _gather_metadata(tensor: torch.Tensor, group: Any) -> torch.Tensor:
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, tensor.contiguous(), group=group)
    return torch.cat(parts, dim=-1)


def _mtp_full_attention_mask(mask: Any, embeds: torch.Tensor) -> torch.Tensor | None:
    """Select a global full-attention mask without mutating backbone metadata."""
    if isinstance(mask, Mapping):
        if "full_attention" not in mask:
            raise ValueError("MTP mask mapping must contain full_attention")
        mask = mask["full_attention"]
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise ValueError("MTP full_attention must be a Tensor or None")
    batch, length = embeds.shape[:2]
    if mask.ndim == 2 and tuple(mask.shape) == (batch, length):
        return mask
    if mask.ndim == 4 and mask.shape[0] == batch and tuple(mask.shape[-2:]) == (length, length):
        return mask
    raise ValueError("MTP attention mask must cover the gathered global sequence, not a CP-local slice")


def mtp_forward(
    model: Any, *, chunk_loss_targets: torch.Tensor, mtp_targets: torch.Tensor,
    chunk_loss_mask: torch.Tensor | None = None, chunk_loss_chunk_size: int = 0,
    chunk_loss_ignore_index: int = -100,
    mtp_attention_mask: torch.Tensor | Mapping[str, Any] | None = None,
    input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None, **kwargs: Any,
) -> ChunkedCausalLMOutput:
    """Run backbone once and score each future depth on this rank's owned interval."""
    if kwargs.get("past_key_values") is not None or kwargs.get("use_cache") or kwargs.get("output_router_logits"):
        raise ValueError("MTP training does not support KV cache or router auxiliary loss")
    if kwargs.pop("logits_to_keep", 0) != 0 or chunk_loss_ignore_index != -100:
        raise ValueError("MTP requires logits_to_keep=0 and ignore_index=-100")
    if kwargs.pop("return_dict", True) is False:
        raise ValueError("MTP requires return_dict=True")
    for field in ("labels", "shift_labels", "loss_mask", "past_key_values", "use_cache"):
        kwargs.pop(field, None)
    if inputs_embeds is None:
        inputs_embeds = model.get_input_embeddings()(input_ids)
    if inputs_embeds.shape[0] != 1:
        raise NotImplementedError("MTP currently requires micro batch size 1")
    if position_ids is None:
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
    outputs = model.model(
        input_ids=None, inputs_embeds=inputs_embeds, position_ids=position_ids,
        use_cache=False, **kwargs,
    )
    local_hidden = outputs.last_hidden_state
    targets = chunk_loss_targets
    if chunk_loss_mask is not None:
        targets = targets.masked_fill(~chunk_loss_mask.bool(), -100)
    if mtp_targets.shape != (model.mtp.depth, *targets.shape):
        raise ValueError("MTP targets must have shape [depth, batch, local_sequence]")

    def ce(hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if chunk_loss_chunk_size:
            return chunked_cross_entropy(hidden, labels, model.lm_head.weight, chunk_size=chunk_loss_chunk_size)
        logits = model.lm_head(hidden).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")

    foundation_sum = ce(local_hidden, targets)
    group = model._hp_mtp_cp_group
    hidden, embeds = local_hidden, inputs_embeds
    start = 0
    global_positions = position_ids
    if group is not None:
        embeds = _GatherSequence.apply(embeds, group)
        global_positions = _gather_metadata(position_ids, group)
        start = dist.get_rank(group) * local_hidden.shape[1]
    rotary_positions = position_ids
    if rotary_positions.ndim == 2:
        rotary_positions = rotary_positions.unsqueeze(0).expand(3, -1, -1)
    elif rotary_positions.shape[0] == 4:
        rotary_positions = rotary_positions[1:]
    text_positions = position_ids if position_ids.ndim == 2 else position_ids[0]
    decoder = model.model.language_model if model.config.model_type == "qwen3_5_moe" else model.model
    rotary = decoder.rotary_emb(hidden, rotary_positions)
    causal_mask = create_causal_mask(
        config=model.mtp.config, inputs_embeds=embeds,
        attention_mask=_mtp_full_attention_mask(mtp_attention_mask, embeds),
        past_key_values=None,
        position_ids=global_positions if global_positions.ndim == 2 else global_positions[0],
    )
    predictions = model.mtp(hidden, embeds, rotary, causal_mask, text_positions, sequence_start=start)
    mtp_sum = sum(ce(prediction, labels)
                  for prediction, labels in zip(predictions, mtp_targets))
    return ChunkedCausalLMOutput(
        loss_sum=foundation_sum, valid_token_count=targets.ne(-100).sum(),
        mtp_loss_sum=mtp_sum, mtp_valid_token_count=mtp_targets.ne(-100).sum(),
    )


def attach_mtp(model: Any, mesh: Any = None) -> None:
    """Create checkpoint parameters before replacement, sharding, and weight loading."""
    if model.config.model_type not in ("qwen3_5_moe_text", "qwen3_5_moe"):
        raise NotImplementedError("MTP currently supports Qwen3.5/3.6 MoE text and image-text models")
    for name in ("tp_size", "pp_size"):
        if getattr(mesh, name, 1) != 1:
            raise NotImplementedError("MTP requires TP=PP=1")
    if getattr(mesh, "sequence_parallel", False) or getattr(mesh, "loss_parallel", False):
        raise NotImplementedError("MTP does not support sequence/loss parallelism")
    config = getattr(model.config, "text_config", model.config)
    reference = model.get_input_embeddings().weight
    # HF's temporary dtype context has ended; new modules must inherit the
    # backbone precision before replacement and checkpoint loading.
    with torch.device(reference.device):
        model.mtp = Qwen3_5MoeMTP(config).to(dtype=reference.dtype)
    decoder = model.model.language_model if model.config.model_type == "qwen3_5_moe" else model.model
    model.mtp.apply(decoder._init_weights)
    model._hp_mtp_enabled = True
    model._hp_mtp_cp_group = mesh.cp_mesh.get_group() if getattr(mesh, "cp_size", 1) > 1 else None
    original_forward = model.forward

    @wraps(original_forward)
    def forward(*args: Any, **kwargs: Any) -> Any:
        if "mtp_targets" not in kwargs:
            return original_forward(*args, **kwargs)
        if args:
            raise TypeError("MTP requires keyword-only training inputs")
        return mtp_forward(model, **kwargs)

    model.forward = forward
