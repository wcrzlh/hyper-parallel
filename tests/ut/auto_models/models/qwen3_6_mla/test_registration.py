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
"""Keep MLA language and vision-language adapter identities distinct."""

import unittest

from hyper_parallel.models.registry import get_model_adapter
from tests.common.mark_utils import arg_mark


class TestMLARegistration(unittest.TestCase):
    """Ensure the visual architecture retains its model-owned providers."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_visual_and_text_architectures_resolve_distinct_contracts(self):
        """Feature: MLA adapter registration.

        Description: Resolve text and visual architectures independently.
        Expectation: The VLM retains native visual providers and MLA sharding.
        """
        visual = get_model_adapter("Qwen3_6MLAForConditionalGeneration")
        text = get_model_adapter("Qwen3_6MLAForCausalLM")
        native_visual = get_model_adapter("qwen3_5_moe")
        self.assertEqual(visual.architecture, "Qwen3_6MLAForConditionalGeneration")
        self.assertEqual(text.architecture, "Qwen3_6MLAForCausalLM")
        self.assertIsNot(visual, text)
        for provider in ("materialization", "loss", "fsdp_execution_order", "encoder_parallel"):
            self.assertIs(getattr(visual, provider), getattr(native_visual, provider), provider)
        self.assertIs(visual.sharding_rules, text.sharding_rules)
        self.assertIs(visual.context_parallel, text.context_parallel)
