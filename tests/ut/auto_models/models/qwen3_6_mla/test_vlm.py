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
"""Check shared VLM recipes retain exact remote-class matching."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from hyper_parallel.models.qwen3_5_moe.adapter import vlm
from hyper_parallel.models.replacement import ModuleReplacementSpec, compile_module_replacements, module_replacement
from tests.common.mark_utils import arg_mark


@module_replacement
def _identity_factory(*, module, module_fqn, context):
    """Expose replacement matching independently of accelerator kernels."""
    del module_fqn, context
    return module


class TestSharedVLMSetup(unittest.TestCase):
    """Keep native recipes unchanged and adapt only the declared MLA family."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_native_model_keeps_original_rules(self):
        """Feature: Native VLM setup preservation.

        Description: Prepare a native Qwen config with remote code enabled.
        Expectation: No reference model is built and existing rules are unchanged.
        """
        rules = (object(),)
        setup = SimpleNamespace(module_replacements=rules, plan_overrides={})
        with patch.object(vlm.AutoConfig, "from_pretrained", return_value=SimpleNamespace(
                architectures=["Qwen3_5MoeForConditionalGeneration"])), patch.object(
                vlm.AutoModelForImageTextToText, "from_config") as build:
            vlm._prepare_remote_vlm_setup("local-model", setup, {"trust_remote_code": True})
        self.assertIs(setup.module_replacements, rules)
        build.assert_not_called()
        self.assertEqual(setup.plan_overrides, {})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_remote_rules_match_without_weakening_type_checks(self):
        """Feature: Remote MLA replacement matching.

        Description: Resolve the checkpoint class and retain explicit vision overrides.
        Expectation: Exact-type checks reject unrelated modules.
        """
        native = type("Qwen3_5MoeGatedDeltaNet", (nn.Module,), {
            "__module__": "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"})
        remote = type("Qwen3_5MoeGatedDeltaNet", (nn.Module,), {"__module__": "remote_checkpoint"})
        reference = nn.Module()
        reference.model = nn.Module()
        reference.model.visual = nn.Linear(2, 2)
        reference.model.linear_attn = remote()
        rule = ModuleReplacementSpec(("model.linear_attn",), _identity_factory, native, True)
        custom_visual = object()
        setup = SimpleNamespace(module_replacements=(rule,), plan_overrides={"model.visual": custom_visual})
        with patch.object(vlm.AutoConfig, "from_pretrained", return_value=SimpleNamespace(
                architectures=["Qwen3_6MLAForConditionalGeneration"])), patch.object(
                vlm.AutoModelForImageTextToText, "from_config", return_value=reference):
            vlm._prepare_remote_vlm_setup("local-model", setup, {"trust_remote_code": True})
        resolved = setup.module_replacements[0]
        self.assertIs(resolved.module_type, remote)
        self.assertTrue(resolved.exact_type)
        self.assertIs(setup.plan_overrides["model.visual"], custom_visual)
        self.assertEqual(len(compile_module_replacements(reference, setup.module_replacements).targets), 1)
        reference.model.linear_attn = nn.Linear(2, 2)
        with self.assertRaisesRegex(TypeError, "expected exact Qwen3_5MoeGatedDeltaNet"):
            compile_module_replacements(reference, setup.module_replacements)
