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
"""Canonical checkpoint-shaped Qwen MoE models with optional future-token decoders."""

from __future__ import annotations

# Torch-only model integration; there is no backend-dispatch layer.
# pylint: disable=forbidden-backend-import
import copy
from collections.abc import Iterator
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn import functional as F
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeRMSNorm,
)

from hyper_parallel.models.qwen3_5_moe.adapter.mtp_runtime import mtp_forward


class Qwen3_5MoeMTP(nn.Module):
    """Use checkpoint MTP names without registering a second embedding or head.

    Full attention and experts use the same planner rules as backbone decoders.
    """

    def __init__(self, config: Any) -> None:
        """Construct the configured number of independent full-attention decoder layers."""
        super().__init__()
        depth = getattr(config, "mtp_num_hidden_layers", None)
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("enable_mtp requires positive text config mtp_num_hidden_layers")
        if getattr(config, "mtp_use_dedicated_embeddings", False):
            raise NotImplementedError("MTP dedicated embeddings are not supported")
        if "full_attention" not in config.layer_types:
            raise ValueError("MTP requires a full_attention layer definition")
        if getattr(config, "output_router_logits", False) or getattr(config, "attention_dropout", 0.0):
            raise NotImplementedError("MTP requires router auxiliary loss and attention dropout disabled")
        # CP wrappers are installed by the parallel plan, not HF's kernel registry.
        self.config = copy.deepcopy(config)
        self.config._attn_implementation = "sdpa"
        self.depth = depth
        self.gradient_checkpointing = False
        hidden = config.hidden_size
        self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(hidden, eps=config.rms_norm_eps)
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        layer_index = config.layer_types.index("full_attention")
        self.layers = nn.ModuleList(Qwen3_5MoeDecoderLayer(self.config, layer_index) for _ in range(depth))
        self.norm = Qwen3_5MoeRMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        sequence_start: int = 0,
    ) -> list[torch.Tensor]:
        """Predict each depth over the local sequence, retaining the gradient graph.

        Args:
            hidden_states: Final normalized local backbone states, ``[B, local_S, H]``.
            inputs_embeds: Global input embeddings, already visual-fused for VLM.
            position_embeddings: Local rotary cosine and sine tensors.
            attention_mask: Global causal mask, adapted by the CP attention wrapper.
            position_ids: Local text positions.
            sequence_start: Global offset of this rank's contiguous sequence shard.

        Returns:
            One normalized hidden-state tensor per future-token depth.
        """
        if (hidden_states.ndim != 3 or inputs_embeds.ndim != 3
                or hidden_states.shape[::2] != inputs_embeds.shape[::2]
                or sequence_start < 0 or sequence_start + hidden_states.shape[1] > inputs_embeds.shape[1]):
            raise ValueError("MTP local hidden states must index a valid interval of global embeddings")
        return list(self._predict(
            hidden_states, inputs_embeds, position_embeddings, attention_mask, position_ids, sequence_start))

    def _predict(
        self, hidden_states: torch.Tensor, inputs_embeds: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None, position_ids: torch.Tensor | None,
        sequence_start: int,
    ) -> Iterator[torch.Tensor]:
        sequence = hidden_states.shape[1]
        for index, layer in enumerate(self.layers):
            shift = index + 1
            start = sequence_start + shift
            shifted = F.pad(inputs_embeds, (0, 0, 0, shift))[:, start:start + sequence]
            hidden_states = self.fc(torch.cat((
                self.pre_fc_norm_embedding(shifted), self.pre_fc_norm_hidden(hidden_states),
            ), dim=-1))
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings,
                attention_mask=attention_mask, position_ids=position_ids, past_key_values=None,
            )
            hidden_states = self.norm(hidden_states)
            yield hidden_states


class _QwenMTPModel:
    """Construct executable MTP structure before replacement and checkpoint loading."""

    _hp_mtp_export_class: ClassVar[type]

    def __init__(self, config: Any) -> None:
        """Initialize the HF base through MRO, then its checkpoint-owned MTP layers."""
        # HF does not inherit native conversion rules for custom subclasses.
        # Register this model-owned class without changing native model mappings.
        class_name = type(self).__name__
        if get_checkpoint_conversion_mapping(class_name) is None:
            native_class = self._hp_mtp_export_class
            mapping = get_checkpoint_conversion_mapping(native_class.__name__)
            if mapping is None:
                mapping = get_checkpoint_conversion_mapping(config.model_type)
            register_checkpoint_conversion_mapping(class_name, mapping or [])
        super().__init__(config)
        text_config = getattr(config, "text_config", config)
        reference = getattr(self, "get_input_embeddings")().weight
        with torch.device(reference.device):
            self.mtp = Qwen3_5MoeMTP(text_config).to(dtype=reference.dtype)
        backbone = getattr(self, "model")
        decoder = backbone.language_model if hasattr(backbone, "language_model") else backbone
        self.mtp.apply(decoder._init_weights)
        self._hp_mtp_enabled = True
        self._hp_mtp_cp_group = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Use the native inference path unless explicit MTP training targets exist."""
        if "mtp_targets" not in kwargs:
            return getattr(super(), "forward")(*args, **kwargs)
        if args:
            raise TypeError("MTP training requires keyword-only inputs")
        return mtp_forward(self, **kwargs)


class QwenMTPForCausalLM(_QwenMTPModel, Qwen3_5MoeForCausalLM):
    """Text-only CausalLM, retaining native checkpoint parameter names."""
    _hp_mtp_export_class = Qwen3_5MoeForCausalLM


class QwenMTPForConditionalGeneration(_QwenMTPModel, Qwen3_5MoeForConditionalGeneration):
    """Image-text conditional generation with shared embedding/head MTP."""
    _hp_mtp_export_class = Qwen3_5MoeForConditionalGeneration
