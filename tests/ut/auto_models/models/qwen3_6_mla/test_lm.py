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
"""Check causal-LM construction uses exact checkpoint replacement classes."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from hyper_parallel.models.qwen3_5_moe.adapter import lm, model_setup
from hyper_parallel.models.replacement import ModuleReplacementSpec, compile_module_replacements
from tests.ut.auto_models.models.qwen3_6_mla.test_vlm import _identity_factory


class TestSharedLMSetup(unittest.TestCase):
    """Keep the ordinary builder contract while resolving remote model types."""

    def test_native_model_preserves_rules_and_options(self):
        """Native checkpoints need no reference model or replacement remapping."""
        rules = (object(),)
        setup = SimpleNamespace(module_replacements=rules, plan_overrides={})
        with patch.object(model_setup.AutoConfig, "from_pretrained", return_value=SimpleNamespace(
                architectures=["Qwen3_5MoeForConditionalGeneration"])), patch.object(
                lm.AutoModelForCausalLM, "from_config") as reference, patch.object(
                lm.HyperAutoModelForCausalLM, "from_pretrained") as build:
            output = lm.build_qwen_lm("local-model", distributed_setup=setup,
                                     trust_remote_code=True, load_base_model=True)
        self.assertIs(output, build.return_value)
        self.assertIs(setup.module_replacements, rules)
        self.assertEqual(setup.plan_overrides, {})
        reference.assert_not_called()
        build.assert_called_once_with("local-model", distributed_setup=setup,
                                      trust_remote_code=True, load_base_model=True)

    def test_remote_text_and_visual_checkpoints_keep_exact_checks(self):
        """The causal Auto class maps GDN/experts for both MLA model trees."""
        for architecture, has_vision in (("Qwen3_6MLAForCausalLM", False),
                                         ("Qwen3_6MLAForConditionalGeneration", True)):
            with self.subTest(architecture=architecture):
                config = SimpleNamespace(architectures=[architecture])
                reference = nn.Module()
                reference.model = nn.Module()
                decoder = reference.model
                if has_vision:
                    reference.model.visual = nn.Linear(2, 2)
                    reference.model.language_model = nn.Module()
                    decoder = reference.model.language_model
                layer = nn.Module()
                layer.mlp = nn.Module()
                decoder.layers = nn.ModuleList([layer])
                rules, remote_types = [], []
                for owner, attr, name, pattern in (
                    (layer, "linear_attn", "Qwen3_5MoeGatedDeltaNet", "*.linear_attn"),
                    (layer.mlp, "experts", "Qwen3_5MoeExperts", "*.mlp.experts"),
                ):
                    native = type(name, (nn.Module,), {
                        "__module__": "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"})
                    remote = type(name, (nn.Module,), {"__module__": "remote_checkpoint"})
                    setattr(owner, attr, remote())
                    rules.append(ModuleReplacementSpec((pattern,), _identity_factory, native, True))
                    remote_types.append(remote)
                setup = SimpleNamespace(module_replacements=tuple(rules), plan_overrides={})
                with patch.object(model_setup.AutoConfig, "from_pretrained", return_value=config), patch.object(
                        lm.AutoModelForCausalLM, "from_config", return_value=reference) as construct, patch.object(
                        lm.HyperAutoModelForCausalLM, "from_pretrained", return_value=reference) as build:
                    output = lm.build_qwen_lm("local-model", distributed_setup=setup, trust_remote_code=True)
                self.assertIs(output, reference)
                self.assertIs(construct.call_args.args[0], config)
                build.assert_called_once_with("local-model", distributed_setup=setup, trust_remote_code=True)
                self.assertEqual(len(compile_module_replacements(reference, setup.module_replacements).targets), 2)
                for rule, remote in zip(setup.module_replacements, remote_types):
                    self.assertIs(rule.module_type, remote)
                    self.assertTrue(rule.exact_type)
                self.assertEqual(bool(setup.plan_overrides), has_vision)
                if has_vision:
                    self.assertEqual(set(setup.plan_overrides["model.visual"].params), {"weight", "bias"})
