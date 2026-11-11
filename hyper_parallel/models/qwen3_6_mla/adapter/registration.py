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
"""Adapters for the Qwen3.6 MLA model family (Qwen3_6MLAForCausalLM).

Qwen3.6 MLA is a hybrid decoder: interleaved full-attention layers with
KV-latent (MLA) compression and GatedDeltaNet linear-attention layers, plus
fine-grained MoE feed-forward blocks. The family adapter contributes the
context-parallel wrappers that intercept ``F.scaled_dot_product_attention``
(``sdpa_hf_ulysses``) and the GDN primitives (``gdn_ulysses``).

The chunked-CE training objective is resolved through the generic
``qwen3_5_moe_text`` adapter path, so no model-specific loss adapter is
registered here.
"""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _load_context_parallel():
    """Return the Qwen3.6 MLA CP wrapper module through a lazy provider."""
    from hyper_parallel.models.qwen3_6_mla.adapter.distributed import context_parallel  # pylint: disable=C0415
    return context_parallel


def _load_sharding_rules():
    """Classify MLA and GDN weights before generic planner template selection."""
    from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole  # pylint: disable=C0415

    return [
        (["kv_a_proj.", "kv_down_proj."], ParamRole.REPLICATED),
        (["kv_b_proj.", "kv_up_proj."], ParamRole.COLWISE),
        ("in_proj_qkv", ParamRole.FUSED_QKV),
        (["in_proj_z", "in_proj_b", "in_proj_a", "conv1d"], ParamRole.COLWISE),
        (["A_log", "dt_bias"], ParamRole.COLWISE),
        ("out_proj", ParamRole.ROWWISE),
        ("shared_expert_gate", ParamRole.REPLICATED),
    ]


QWEN3_6_MLA_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3_6MLAForCausalLM",
    model_type="qwen36_moe_mla",
    context_parallel=_load_context_parallel,
    sharding_rules=_load_sharding_rules,
)

register_model_adapter(QWEN3_6_MLA_ADAPTER_SPEC)
