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
"""Context parallel wrappers for Qwen3.5-MoE-MLA (Qwen3.6-35B-A3B-kvfirst-mla).

This model has two attention types:
  - full_attention layers: Qwen3_5MoeAttention with MLA KV compression.
    The forward ultimately calls F.scaled_dot_product_attention, so the
    generic ``sdpa_hf_ulysses`` wrapper from the framework's INNER_WRAPPER_REGISTRY
    handles CP by intercepting F.sdpa and running Ulysses seq/head exchange.

  - linear_attention layers: Qwen3_5MoeGatedDeltaNet (GDN).
    The forward calls ``self.causal_conv1d_fn`` and ``self.chunk_gated_delta_rule``,
    so the generic ``gdn_ulysses`` wrapper handles CP by intercepting those primitives.

No model-specific CP wrapper is needed — the generic wrappers in
``hyper_parallel.distributed.context_parallel.wrappers`` cover both attention
types.  This module exists only so that the ModelAdapterSpec's
``context_parallel`` provider resolves to a callable that returns a module
satisfying the planner's interface contract.
"""

from hyper_parallel.distributed.context_parallel.wrappers import (  # noqa: F401
    gdn_ulysses_cp_wrapper,
    sdpa_hf_ulysses_cp_wrapper,
)

__all__ = [
    "gdn_ulysses_cp_wrapper",
    "sdpa_hf_ulysses_cp_wrapper",
]
