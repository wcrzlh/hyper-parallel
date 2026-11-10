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
"""CPU checks for the model-owned MTP migration, export, and step normalization."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from hyper_parallel.components.losses import ChunkedCausalLMLoss, ModelOutputLoss
from hyper_parallel.components.checkpoint.huggingface_checkpointer import HuggingFaceCheckpointer
from hyper_parallel.components.losses.mtp_objective import build_mtp_targets
from hyper_parallel.models._transformers.checkpoint_loader import CheckpointManager
from hyper_parallel.models.qwen3_5_moe.adapter.mtp_data import QwenMTPRuntime
from hyper_parallel.models.qwen3_5_moe.adapter.mtp_builder import build_qwen_mtp
from hyper_parallel.models.qwen3_5_moe.adapter.vlm import install_qwen_vlm_cp_forward
from hyper_parallel.models.qwen3_5_moe.modeling_mtp import (
    Qwen3_5MoeMTP,
    QwenMTPForCausalLM,
    QwenMTPForConditionalGeneration,
)
from hyper_parallel.models.qwen3_5_moe.vlm_cp import QwenVLMParallelBatch
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.runtime.metrics import mean_global_loss
from tests.common.mark_utils import arg_mark


def make_model(depth: int = 2, kv_heads: int = 1) -> QwenMTPForCausalLM:
    """Build a real tiny decoder with no distributed or fused replacements."""
    config = Qwen3_5MoeTextConfig.from_dict({
        "vocab_size": 32, "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": kv_heads, "head_dim": 8, "num_experts": 2, "num_experts_per_tok": 1,
        "moe_intermediate_size": 8, "shared_expert_intermediate_size": 8,
        "layer_types": ["full_attention"], "mtp_num_hidden_layers": depth,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                            "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
    })
    config._attn_implementation = "sdpa"
    return QwenMTPForCausalLM(config)


def make_structural_reference() -> Qwen3_5MoeForCausalLM:
    """Reproduce beta's native backbone plus MTP tree for a structural inventory.

    This is a state-layout reference, not an independent numerical oracle.
    """
    config = make_model().config
    model = Qwen3_5MoeForCausalLM(config)
    model.mtp = Qwen3_5MoeMTP(config)
    return model


class TestMTPMigration(unittest.TestCase):
    """Validate real eager computation rather than mocked training success."""

    def test_released_vlm_weights_load_into_text_mtp(self) -> None:
        """Both pretrained readers load every backbone and MTP tensor through native renames."""
        reference = make_model(1)
        state = reference.state_dict()
        for released_prefix in (False, True):
            with self.subTest(released_prefix=released_prefix), tempfile.TemporaryDirectory() as directory:
                checkpoint = {
                    (name.replace("model.", "model.language_model.", 1)
                     if released_prefix and name.startswith("model.") else name): tensor.clone().contiguous()
                    for name, tensor in state.items()
                }
                save_file(checkpoint, str(Path(directory) / "model.safetensors"))
                for loader in ("legacy", "dcp"):
                    with self.subTest(loader=loader):
                        model = make_model(1)
                        with torch.no_grad():
                            for parameter in model.parameters():
                                parameter.zero_()
                        HuggingFaceCheckpointer(loader=loader).load(directory, {"model": model})
                        for name, actual in model.state_dict().items():
                            torch.testing.assert_close(actual, state[name], rtol=0, atol=0, msg=name)

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_config_first_factory_consumes_pretrained_only_options(self) -> None:
        """Feature: MTP factory options. Description: Exercise real HF construction.
        Expectation: Load-only kwargs do not reach the model constructor; this
        CPU config-only check explicitly bypasses distributed infrastructure.
        """
        module = "hyper_parallel.models._transformers.auto_model"
        text = make_model(1).config
        sources = [text, Qwen3_5MoeConfig.from_dict({"text_config": text.to_dict()})]
        for config in sources:
            with self.subTest(source_type=config.model_type), tempfile.TemporaryDirectory() as directory:
                config.save_pretrained(directory)
                with patch(f"{module}.instantiate_infrastructure", return_value=(None, None)):
                    with patch(f"{module}.apply_model_infrastructure", side_effect=lambda model, **kwargs: model):
                        model = build_qwen_mtp(
                            directory, load_base_model=False, ignore_mismatched_sizes=True,
                            torch_dtype=torch.float32, local_files_only=True,
                        )
                self.assertIsInstance(model, QwenMTPForCausalLM)
                self.assertTrue(model._hp_mtp_enabled)
                self.assertEqual(model.mtp.depth, 1)
                self.assertFalse(any("visual" in name for name in model.state_dict()))

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_vlm_runtime_preserves_vision_gradients(self) -> None:
        """Feature: VLM MTP. Description: Train one real image through native vision.
        Expectation: Both backbone/vision and future-token parameters receive gradients.
        """
        text = make_model(1).config.to_dict()
        text["vocab_size"] = 128
        config = Qwen3_5MoeConfig.from_dict({
            "text_config": text, "image_token_id": 99, "video_token_id": 98, "vision_start_token_id": 97,
            "vision_config": {"hidden_size": 16, "intermediate_size": 32, "num_heads": 4, "depth": 2,
                              "out_hidden_size": 16, "patch_size": 2, "spatial_merge_size": 2,
                              "temporal_patch_size": 2, "num_position_embeddings": 16},
        })
        config._attn_implementation = "sdpa"
        model = QwenMTPForConditionalGeneration(config)
        install_qwen_vlm_cp_forward(model, cp_enabled=False)
        ids = torch.arange(1, 21).unsqueeze(0)
        ids[:, 3:7] = 99
        labels = ids.masked_fill(ids.eq(99), -100)
        labels[:, :3] = -100
        batch = {"input_ids": ids, "labels": labels, "mm_token_type_ids": ids.eq(99).long(),
                 "pixel_values": torch.randn(16, 24), "image_grid_thw": torch.tensor([[1, 4, 4]])}
        topology = SimpleNamespace(tp_size=1, pp_size=1, cp_size=1, dp_size=1, dp_rank=0, tp_rank=0,
                                   sequence_parallel=False, loss_parallel=False)
        runtime = QwenMTPRuntime(depth=1, labels_are_shifted=False)
        batcher = QwenVLMParallelBatch(mesh_context=topology, device=torch.device("cpu"),
                                       runtime_input_adapter=runtime)
        inputs, loss_inputs = batcher(None, external_batch=batch)
        self.assertIs(loss_inputs["mtp_labels"], inputs["mtp_targets"])
        objective = ChunkedCausalLMLoss(chunk_size=17)
        objective.bind_model(model)
        output = model(**objective.prepare_model_inputs(inputs, loss_inputs), use_cache=False)
        loss = sum(objective(model_output=output, labels=None).values())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.mtp.fc.weight.grad.abs().sum()), 0)
        visual = [parameter for name, parameter in model.named_parameters() if "visual" in name]
        self.assertTrue(visual)
        self.assertTrue(all(parameter.grad is not None for parameter in visual))
        self.assertTrue(any(bool(parameter.grad.ne(0).any()) for parameter in visual))

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_chunk_and_full_logits_have_matching_loss_and_gradients(self) -> None:
        """Feature: Canonical MTP. Description: Compare both CE paths.
        Expectation: Every parameter gradient matches, including MTP parameters.
        """
        torch.manual_seed(41)
        model = make_model()
        reference = copy.deepcopy(model)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        targets = torch.tensor([[2, 3, -100, 5, -100]])
        future = build_mtp_targets(targets, 2)
        losses = []
        for item, loss_fn in ((model, ChunkedCausalLMLoss(chunk_size=2)), (reference, ModelOutputLoss())):
            loss_fn.bind_model(item)
            inputs = loss_fn.prepare_model_inputs(
                {"input_ids": ids, "mtp_targets": future}, {"shift_labels": targets},
            )
            output = item(**inputs, use_cache=False)
            self.assertIsNone(output.logits)
            loss = sum(loss_fn(model_output=output, labels=targets).values())
            loss.backward()
            losses.append(loss.detach())
        torch.testing.assert_close(losses[0], losses[1])
        for name, parameter in model.named_parameters():
            expected = dict(reference.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=2e-5, atol=2e-7, msg=name)

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_text_export_keeps_task_and_mtp_without_mutating_live_config(self) -> None:
        """Feature: HF export. Description: Save a real MTP checkpoint.
        Expectation: CausalLM task, MTP tensors, and inference-compatible text RoPE.
        """
        model = make_model(1)
        original = copy.deepcopy(model.config.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            CheckpointManager(model).save_pretrained(directory)
            with open(f"{directory}/config.json", encoding="utf-8") as source:
                config = json.load(source)
            self.assertEqual(config["architectures"], ["Qwen3_5MoeForCausalLM"])
            self.assertEqual(config["model_type"], "qwen3_5_moe_text")
            self.assertNotIn("mrope_section", config["rope_parameters"])
            self.assertEqual(model.config.to_dict(), original)
            self.assertEqual(type(model), QwenMTPForCausalLM)
            from safetensors import safe_open  # pylint: disable=import-outside-toplevel
            with safe_open(f"{directory}/model.safetensors", framework="pt") as saved:
                expected = {name: tensor for name, tensor in model.state_dict().items() if name.startswith("mtp.")}
                self.assertEqual({name for name in saved.keys() if name.startswith("mtp.")}, set(expected))
                for name, tensor in expected.items():
                    torch.testing.assert_close(saved.get_tensor(name), tensor, rtol=0, atol=0, msg=name)

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_step_plan_normalizes_independent_objectives_across_uneven_batches(self) -> None:
        """Feature: Step-level MTP aggregation. Description: Accumulate unequal counts.
        Expectation: Loss and gradient equal separately normalized global objectives.
        """
        labels = [torch.tensor([[1, 2, 3, -100]]), torch.tensor([[1, -100]])]
        futures = [torch.tensor([[[2, 3, -100, -100]]]), torch.tensor([[[-100, -100]]])]
        batches = [({}, {"shift_labels": left, "mtp_labels": right}) for left, right in zip(labels, futures)]
        mesh = SimpleNamespace(dp_cp_mesh=None, dp_size=1, cp_size=1, sequence_parallel=False)
        trainer = SimpleNamespace(get_batch=next, num_micro_batches=2,
                                  mesh=mesh, device=torch.device("cpu"))
        BaseTrainer.prepare_loss_step(trainer, iter(batches))
        weights = trainer.loss_token_normalization_weights
        torch.testing.assert_close(weights["foundation"], torch.tensor(1.0))
        torch.testing.assert_close(weights["mtp"], torch.tensor(2.0))
        parameter = torch.tensor(2.0, requires_grad=True)
        total = 0
        for foundation, mtp in ((3, 2), (1, 0)):
            loss = {"foundation_loss": parameter.square(), "mtp_loss": parameter * 0.3 if mtp else parameter * 0}
            reduced = mean_global_loss(
                loss, {"foundation_tokens": torch.tensor(foundation), "mtp_tokens": torch.tensor(mtp)},
                mesh, token_normalization_weights=weights,
            )
            total = total + sum(reduced.values())
        total = total / 4
        total.backward()
        torch.testing.assert_close(total, torch.tensor(4.6))
        torch.testing.assert_close(parameter.grad, torch.tensor(4.3))
