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
"""Real CPU Qwen forwards for mixed text/image vision dependencies."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from transformers import (
    Qwen3_5Config, Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration,
)

from hyper_parallel.components.losses.mtp_objective import build_mtp_targets
from hyper_parallel.models.qwen3_5_moe.adapter.encoder_parallel import QwenVisionBridge
from hyper_parallel.models.qwen3_5_moe.adapter.vlm import install_qwen_vlm_cp_forward
from hyper_parallel.models.qwen3_5_moe.adapter.vlm_chunk_loss import bind_chunk_loss
from hyper_parallel.models.qwen3_5_moe.modeling_mtp import QwenMTPForConditionalGeneration
from hyper_parallel.models.qwen3_5_moe.vlm_cp import QwenVLMCPInputs
from tests.common.mark_utils import arg_mark


def make_vlm(moe: bool = True) -> torch.nn.Module:
    """Build a tiny native VLM with real vision attention and a full-attention decoder."""
    config_class = Qwen3_5MoeConfig if moe else Qwen3_5Config
    model_class = Qwen3_5MoeForConditionalGeneration if moe else Qwen3_5ForConditionalGeneration
    config = config_class.from_dict({
        "image_token_id": 99, "video_token_id": 98, "vision_start_token_id": 97,
        "text_config": {
            "vocab_size": 128, "hidden_size": 16, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
            "num_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8, "intermediate_size": 32,
            "layer_types": ["full_attention"], "attention_dropout": 0.0,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                                "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
        },
        "vision_config": {"hidden_size": 16, "intermediate_size": 32, "num_heads": 4, "depth": 2,
                          "out_hidden_size": 16, "patch_size": 2, "spatial_merge_size": 2,
                          "temporal_patch_size": 2, "num_position_embeddings": 16},
    })
    config._attn_implementation = "sdpa"
    return model_class(config)


def make_batch(image: bool = False) -> dict:
    """Use equal text lengths so comparisons isolate the visual dependency."""
    ids = torch.arange(1, 13).unsqueeze(0)
    if image:
        ids[:, 2] = 97
        ids[:, 3:7] = 99
    batch = {"input_ids": ids, "labels": ids.masked_fill(ids.eq(99), -100), "use_cache": False}
    if image:
        batch.update(pixel_values=torch.randn(16, 24), image_grid_thw=torch.tensor([[1, 4, 4]]),
                     mm_token_type_ids=ids.eq(99).long())
    return batch


class TestDummyVision(unittest.TestCase):
    """Check loss/gradient parity and visual call counts across adapter entry points."""

    def setUp(self) -> None:
        """Keep an unmodified native reference and record actual visual invocations."""
        torch.manual_seed(42)
        self.model = make_vlm()
        self.reference = copy.deepcopy(self.model)
        self.calls = []
        self.model.model.visual.register_forward_pre_hook(
            lambda _module, args: self.calls.append(args[0].shape))

    def assert_visual_zero_grad(self) -> None:
        """Every visual leaf must participate in backward, with no text gradient contribution."""
        for name, parameter in self.model.model.visual.named_parameters():
            self.assertIsNotNone(parameter.grad, f"missing visual gradient: {name}")
            self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0,
                             f"nonzero dummy gradient: {name}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_native_loss_and_gradients_match_without_changing_state(self):
        """Dummy execution changes neither logits, CE, decoder gradients, nor checkpoint keys.

        Feature: Mixed-modality vision dependencies.
        Description: Dummy execution changes neither logits, CE, decoder gradients, nor checkpoint keys.
        Expectation: All asserted outputs, masks, gradients, and errors match the reference.
        """
        install_qwen_vlm_cp_forward(self.model, cp_enabled=False)
        batch = make_batch()
        output, reference = self.model(**batch), self.reference(**batch)
        torch.testing.assert_close(output.logits, reference.logits, rtol=0, atol=0)
        torch.testing.assert_close(output.loss, reference.loss, rtol=0, atol=0)
        output.loss.backward()
        reference.loss.backward()
        self.assert_visual_zero_grad()
        expected = dict(self.reference.named_parameters())
        for name, parameter in self.model.named_parameters():
            if "visual" not in name:
                torch.testing.assert_close(parameter.grad, expected[name].grad, rtol=0, atol=0)
        self.assertEqual(list(self.model.state_dict()), list(self.reference.state_dict()))
        self.assertEqual(self.calls, [torch.Size([4, 24])])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_real_images_then_text_keep_native_rope_and_gradients(self):
        """Alternating modalities preserve native position bookkeeping and avoid duplicate vision.

        Feature: Mixed-modality vision dependencies.
        Description: Alternating modalities preserve native position bookkeeping and avoid duplicate vision.
        Expectation: All asserted outputs, masks, gradients, and errors match the reference.
        """
        install_qwen_vlm_cp_forward(self.model, cp_enabled=False)
        for image in (True, False, True):
            with self.subTest(image=image):
                self.model.zero_grad(set_to_none=True)
                self.reference.zero_grad(set_to_none=True)
                batch = make_batch(image)
                output, reference = self.model(**batch), self.reference(**batch)
                torch.testing.assert_close(output.logits, reference.logits, rtol=0, atol=0)
                output.loss.backward()
                reference.loss.backward()
                if image:
                    for actual, expected in zip(self.model.parameters(), self.reference.parameters()):
                        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
                else:
                    self.assert_visual_zero_grad()
        self.assertEqual(self.calls, [torch.Size([16, 24]), torch.Size([4, 24]), torch.Size([16, 24])])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_chunk_loss_both_binding_orders(self):
        """Chunk loss cannot bypass dummy work, with or without CP metadata.

        Feature: Mixed-modality vision dependencies.
        Description: Chunk loss cannot bypass dummy work, with or without CP metadata.
        Expectation: All asserted outputs, masks, gradients, and errors match the reference.
        """
        for chunk_first in (False, True):
            for geometry in (False, True):
                with self.subTest(chunk_first=chunk_first, geometry=geometry):
                    model = make_vlm()
                    if chunk_first:
                        bind_chunk_loss(model)
                    install_qwen_vlm_cp_forward(model, cp_enabled=False)
                    if not chunk_first:
                        bind_chunk_loss(model)
                    batch = make_batch()
                    ids = batch["input_ids"]
                    targets = torch.cat((ids[:, 1:], torch.full((1, 1), -100)), dim=1)
                    if geometry:
                        batch["vlm_cp_inputs"] = QwenVLMCPInputs(
                            0, 12, ids, torch.ones_like(ids), None, targets, 1)
                    output = model(**batch, chunk_loss_targets=targets, chunk_loss_mask=None,
                                   chunk_loss_chunk_size=4, chunk_loss_ignore_index=-100)
                    output.loss_sum.backward()
                    for name, parameter in model.model.visual.named_parameters():
                        self.assertIsNotNone(parameter.grad, f"missing chunk gradient: {name}")
                        self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_cp_precomputed_embeddings_and_real_images(self):
        """CP insertion runs vision once; text CP retains zero visual gradients.

        Feature: Mixed-modality vision dependencies.
        Description: CP insertion runs vision once; text CP retains zero visual gradients.
        Expectation: All asserted outputs, masks, gradients, and errors match the reference.
        """
        install_qwen_vlm_cp_forward(self.model)
        for image in (False, True):
            with self.subTest(image=image):
                self.model.zero_grad(set_to_none=True)
                batch = make_batch(image)
                ids = batch["input_ids"]
                targets = ids[:, 7:12]
                targets = torch.cat((targets, torch.full((1, 1), -100)), dim=1)
                batch["vlm_cp_inputs"] = QwenVLMCPInputs(
                    6, 12, ids, torch.ones_like(ids), batch.get("mm_token_type_ids"), targets, 2)
                batch["input_ids"] = ids[:, 6:12]
                self.model(**batch).loss.backward()
                if not image:
                    self.assert_visual_zero_grad()
        self.assertEqual(self.calls, [torch.Size([4, 24]), torch.Size([16, 24])])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_encoder_plan_suppresses_fallback_and_restores_context(self):
        """Encoder parallel owns its own dummy policy, including globally inactive plans.

        Feature: Mixed-modality vision dependencies.
        Description: Encoder parallel owns its own dummy policy, including globally inactive plans.
        Expectation: All asserted outputs, masks, gradients, and errors match the reference.
        """
        install_qwen_vlm_cp_forward(self.model, cp_enabled=False)
        QwenVisionBridge(self.model).install_forward()
        plan = SimpleNamespace(encode_and_route=Mock(return_value=([], None)))
        self.model(**make_batch(), encoder_parallel_plan=plan)
        self.assertEqual(self.calls, [])
        plan.encode_and_route.assert_called_once()
        self.model(**make_batch()).loss.backward()
        self.assert_visual_zero_grad()
        self.assertEqual(len(self.calls), 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_eval_and_positional_base_inputs(self):
        """Evaluation keeps native behavior; positional training calls retain the dependency.

        Feature: Mixed-modality vision dependencies.
        Description: Evaluation keeps native behavior; positional training calls retain the dependency.
        Expectation: All asserted outputs, masks, gradients, and errors match the reference.
        """
        install_qwen_vlm_cp_forward(self.model, cp_enabled=False)
        self.model.eval()
        self.model(**make_batch())
        self.assertEqual(self.calls, [])
        self.model.train()
        output = self.model.model(make_batch()["input_ids"], use_cache=False)
        output.last_hidden_state.sum().backward()
        self.assert_visual_zero_grad()
        self.assertEqual(len(self.calls), 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_dense_model_uses_same_dummy_path(self):
        """Dense Qwen and MoE use the same native visual contract.

        Feature: Mixed-modality vision dependencies.
        Description: Dense Qwen and MoE use the same native visual contract.
        Expectation: All asserted outputs, masks, gradients, and errors match the reference.
        """
        model = make_vlm(moe=False)
        install_qwen_vlm_cp_forward(model, cp_enabled=False)
        model(**make_batch()).loss.backward()
        for name, parameter in model.model.visual.named_parameters():
            self.assertIsNotNone(parameter.grad, f"missing dense gradient: {name}")
            self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_text_mtp_keeps_visual_and_prediction_gradients(self):
        """MTP's precomputed embeddings also retain the zero visual dependency.

        Feature: Mixed-modality vision dependencies.
        Description: MTP's precomputed embeddings also retain the zero visual dependency.
        Expectation: All asserted outputs, masks, gradients, and errors match the reference.
        """
        config = Qwen3_5MoeConfig.from_dict(self.model.config.to_dict())
        config._attn_implementation = "sdpa"
        config.text_config.mtp_num_hidden_layers = 1
        model = QwenMTPForConditionalGeneration(config)
        install_qwen_vlm_cp_forward(model, cp_enabled=False)
        batch = make_batch()
        ids = batch["input_ids"]
        targets = torch.cat((ids[:, 1:], torch.full((1, 1), -100)), dim=1)
        batch["vlm_cp_inputs"] = QwenVLMCPInputs(0, 12, ids, torch.ones_like(ids), None, targets, 1)
        output = model(**batch, chunk_loss_targets=targets, mtp_targets=build_mtp_targets(targets, 1),
                       chunk_loss_chunk_size=4)
        (output.loss_sum + output.mtp_loss_sum.sum()).backward()
        self.assertGreater(torch.count_nonzero(model.mtp.fc.weight.grad).item(), 0)
        for name, parameter in model.model.visual.named_parameters():
            self.assertIsNotNone(parameter.grad, f"missing MTP visual gradient: {name}")
            self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)
