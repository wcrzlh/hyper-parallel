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
"""MTP target alignment, chunk gradients, and independent token normalization."""

import copy
import unittest
from types import SimpleNamespace

import torch
from transformers import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM

from hyper_parallel.components.losses import ChunkedCausalLMLoss, ModelOutputLoss
from hyper_parallel.data.batching.build_collate_fn import TextPackingCollator
from hyper_parallel.components.losses.mtp_objective import build_mtp_targets, validate_mtp_loss_weight
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.qwen3_5_moe.adapter.mtp_runtime import (
    attach_mtp, _mtp_full_attention_mask, mtp_distributed_setup,
)
from hyper_parallel.models.qwen3_5_moe.vlm_cp import QwenVLMParallelBatch
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from tests.ut.auto_models.test_qwen_vlm_chunk_loss import make_model as make_vlm, make_batch


def make_model(depth: int = 2) -> Qwen3_5MoeForCausalLM:
    """Construct a tiny real decoder without optional GDN kernels."""
    config = Qwen3_5MoeTextConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, num_experts=2, num_experts_per_tok=1,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        layer_types=["full_attention"], mtp_num_hidden_layers=depth,
        rope_parameters={"rope_type": "default", "rope_theta": 10000,
                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
    )
    config._attn_implementation = "sdpa"
    model = Qwen3_5MoeForCausalLM(config)
    attach_mtp(model)
    return model


class TestMTP(unittest.TestCase):
    """Use a real CPU model as the eager/chunk first-order gradient oracle."""

    def setUp(self):
        torch.manual_seed(41)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def test_mtp_plan_override_is_opt_in_copy(self):
        """MTP's projection declaration must not mutate caller configuration."""
        setup = DistributedSetup()
        configured = mtp_distributed_setup(setup)
        self.assertIsNone(setup.plan_overrides)
        self.assertEqual(set(configured.plan_overrides), {"mtp.fc"})
        repeated = mtp_distributed_setup(configured)
        self.assertIs(repeated.plan_overrides["mtp.fc"], configured.plan_overrides["mtp.fc"])

    def test_padding_does_not_count_as_conversation(self):
        """Explicit sample counts distinguish ignored conversations from padding."""
        sample = {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, -100])}
        collator = TextPackingCollator(sequence_parallel_size=2)
        single = collator([sample])
        self.assertEqual(single["num_sequences"], 1)
        self.assertEqual(single["cu_seq_lens"].tolist(), [0, 3, 4])
        ignored = {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([-100, -100])}
        packed = collator([sample, ignored])
        self.assertEqual(packed["num_sequences"], 2)
        self.assertEqual(packed["cu_seq_lens"].tolist(), [0, 3, 5, 6])

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_attach_inherits_backbone_dtype_and_device(self):
        """Attaching outside HF's dtype context must preserve BF16, including meta."""
        config = make_model().config
        default_dtype = torch.get_default_dtype()
        for device in ("cpu", "meta"):
            with self.subTest(device=device):
                with torch.device(device):
                    model = Qwen3_5MoeForCausalLM._from_config(config, dtype=torch.bfloat16)
                backbone = model.get_input_embeddings().weight
                attach_mtp(model)
                for name, parameter in model.mtp.named_parameters():
                    self.assertEqual(parameter.dtype, backbone.dtype, name)
                    self.assertEqual(parameter.device, backbone.device, name)
                self.assertEqual(torch.get_default_dtype(), default_dtype)

    def test_targets_shift_future_mask_and_tail(self):
        """Masked future targets move with their token, including CP boundaries."""
        targets = torch.tensor([[-100, 2, 3, -100, 5]])
        expected = torch.tensor([[[2, 3, -100, 5, -100]], [[3, -100, 5, -100, -100]]])
        torch.testing.assert_close(build_mtp_targets(targets, 2), expected)
        self.assertTrue(build_mtp_targets(targets, 6)[-1].eq(-100).all())

    def test_mtp_dictionary_mask_matches_none(self):
        """P2P routing metadata must not leak into MTP's HF mask builder."""
        model = make_model()
        labels = torch.tensor([[2, 3, 4, -100]])
        inputs = dict(input_ids=torch.tensor([[1, 2, 3, 4]]), chunk_loss_targets=labels,
                      mtp_targets=build_mtp_targets(labels, 2), chunk_loss_chunk_size=2)
        mask = {"full_attention": None, "linear_attention": None}
        expected = model(**inputs)
        expected_loss = expected.loss_sum + expected.mtp_loss_sum
        expected_loss.backward()
        gradients = {name: param.grad.clone() for name, param in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        actual = model(**inputs, mtp_attention_mask=mask)
        actual_loss = actual.loss_sum + actual.mtp_loss_sum
        actual_loss.backward()
        torch.testing.assert_close(actual_loss, expected_loss)
        for name, param in model.named_parameters():
            torch.testing.assert_close(param.grad, gradients[name], msg=name)
        self.assertEqual(mask, {"full_attention": None, "linear_attention": None})

    def test_mtp_mask_requires_global_extent(self):
        """Keep explicit global masks and reject ambiguous or local masks."""
        embeds = torch.zeros(1, 8, 16)
        mask = torch.ones(1, 1, 8, 8, dtype=torch.bool).tril()
        self.assertIs(_mtp_full_attention_mask({"full_attention": mask}, embeds), mask)
        for invalid in ({}, torch.ones(1, 4), torch.ones(1, 1, 4, 8)):
            with self.subTest(mask=invalid), self.assertRaises(ValueError):
                _mtp_full_attention_mask(invalid, embeds)

    def test_chunk_matches_eager_all_parameter_gradients(self):
        """Both objectives share head/embeddings and backpropagate through all depths."""
        eager = make_model()
        chunked = make_model()
        chunked.load_state_dict(eager.state_dict())
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        targets = torch.tensor([[-100, 3, 4, 5, 6, -100]])
        inputs = {"input_ids": ids, "mtp_targets": build_mtp_targets(targets, 2)}
        losses = {"shift_labels": targets, "labels": targets}
        objectives = (ModelOutputLoss(mtp_loss_weight=0.3), ChunkedCausalLMLoss(chunk_size=2, mtp_loss_weight=0.3))
        results = []
        for model, objective in zip((eager, chunked), objectives):
            objective.bind_model(model)
            output = model(**objective.prepare_model_inputs(inputs, losses))
            values = objective(model_output=output, labels=None)
            self.assertEqual(set(values), {"foundation_loss", "mtp_loss"})
            sum(values.values()).backward()
            results.append(values)
        for name in results[0]:
            torch.testing.assert_close(results[0][name], results[1][name])
        for (name, actual), (_, expected) in zip(chunked.named_parameters(), eager.named_parameters()):
            self.assertIsNotNone(actual.grad, msg=f"Missing gradient for {name}")
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-6, msg=name)

    def test_empty_targets_still_connect_gradients(self):
        """Empty supervision returns a finite graph-connected zero for both branches."""
        model = make_model()
        labels = torch.full((1, 4), -100, dtype=torch.long)
        objective = ChunkedCausalLMLoss(chunk_size=2)
        objective.bind_model(model)
        inputs = objective.prepare_model_inputs(
            {"input_ids": torch.ones_like(labels), "mtp_targets": build_mtp_targets(labels, 2)},
            {"shift_labels": labels},
        )
        values = objective(model_output=model(**inputs), labels=None)
        loss = sum(values.values())
        torch.testing.assert_close(loss, torch.tensor(0.))
        loss.backward()
        self.assertTrue(all(param.grad is not None for param in model.parameters()))

    def test_counts_accumulate_across_micro_batches(self):
        """All depths and micro-batches contribute to the auxiliary denominator."""
        labels = torch.tensor([[1, 2, 3, -100]])
        batch = {"labels": labels, "shift_labels": labels, "mtp_labels": build_mtp_targets(labels, 2)}
        counts = count_loss_token([batch, copy.deepcopy(batch)])
        self.assertEqual(counts["foundation_tokens"].item(), 6)
        self.assertEqual(counts["mtp_tokens"].item(), 6)

    def test_invalid_weights(self):
        """Reject invalid objectives instead of silently loading an untrained branch."""
        for weight in (-1, float("nan"), float("inf"), True, 0):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                validate_mtp_loss_weight(weight, enabled=True)

    def test_checkpoint_names_do_not_duplicate_shared_weights(self):
        """MTP retains its own prefix and never registers the backbone embedding/head."""
        names = set(make_model().state_dict())
        self.assertIn("mtp.fc.weight", names)
        self.assertFalse(any("mtp.embed" in name or "mtp.lm_head" in name for name in names))

    def test_vlm_chunk_gradients_and_visual_dependency(self):
        """Use image-fused embeddings once and preserve visual gradients in both objectives."""
        models = [make_vlm(), make_vlm()]
        for model in models:
            model.config.text_config.mtp_num_hidden_layers = 1
            attach_mtp(model)
        models[1].load_state_dict(models[0].state_dict())
        batch = make_batch()
        context = SimpleNamespace(cp_size=1, tp_size=1, pp_size=1, dp_size=1, dp_rank=0,
                                  cp_rank=0, tp_rank=0, sequence_parallel=False, loss_parallel=False)
        results = []
        for model, objective in zip(models, (ModelOutputLoss(), ChunkedCausalLMLoss(chunk_size=17))):
            objective.bind_model(model)
            batcher = QwenVLMParallelBatch(mesh_context=context, device=torch.device("cpu"))
            batcher.bind_model(model)
            inputs, labels = batcher(None, external_batch=batch)
            output = model(**objective.prepare_model_inputs(inputs, labels))
            loss = objective(model_output=output, labels=None)
            loss["mtp_loss"].backward()
            results.append(loss)
            self.assertTrue(any(param.grad is not None and param.grad.abs().sum() > 0
                                for param in model.model.visual.parameters()))
        for name in results[0]:
            torch.testing.assert_close(results[0][name], results[1][name], atol=2e-6, rtol=2e-5)
        for (name, actual), (_, expected) in zip(models[0].named_parameters(), models[1].named_parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=3e-4, msg=name)
