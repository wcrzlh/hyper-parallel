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
"""Tests for optional model-runtime loss-label aliases."""

import unittest
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch

from hyper_parallel.data.batching.get_batch import TextParallelBatch
from hyper_parallel.data.batching.runtime_input import (
    RuntimeInputAdapter,
    RuntimeInputContext,
)
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models.qwen3_5_moe.adapter.mtp_data import QwenMTPRuntime
from tests.common.mark_utils import arg_mark


class _Runtime(RuntimeInputAdapter):
    """Minimal unrelated model runtime demonstrating the default extension path."""

    def build_runtime_inputs(self, *, batch: Mapping[str, Any], context: RuntimeInputContext) -> Mapping[str, Any]:
        """Return a model-only descriptor, without declaring auxiliary labels."""
        return {"descriptor": batch["input_ids"].shape}


class TestRuntimeLossAliases(unittest.TestCase):
    """Keep ordinary runtime behavior unchanged and reject ownership violations."""

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_absent_provider_adds_no_loss_fields(self) -> None:
        """Feature: Optional loss aliases. Description: Use an ordinary runtime.
        Expectation: The existing forward-only default returns no auxiliary labels.
        """
        self.assertEqual(_Runtime().build_loss_inputs({"descriptor": (1, 3)}), {})

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_present_provider_rejects_invalid_aliases(self) -> None:
        """Feature: Runtime ownership. Description: Exercise invalid declared sources.
        Expectation: Encoded fields cannot be silently replaced or exposed implicitly.
        """
        class InvalidRuntime(_Runtime):
            """Deliberately violate the runtime alias contract."""

            def loss_input_aliases(self) -> Mapping[str, str]:
                """Demonstrate an undeclared source and a forbidden foundation alias."""
                return {"shift_labels": "descriptor"}

        with self.assertRaisesRegex(ValueError, "declared runtime"):
            InvalidRuntime().build_loss_inputs({"descriptor": (1, 3)})

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_mtp_targets_flow_through_generic_text_batch(self) -> None:
        """Feature: Model-owned future targets. Description: Use generic text transport.
        Expectation: Future labels are shifted once, aliased, and have ignored tails.
        """
        runtime = QwenMTPRuntime(depth=2)
        batcher = TextParallelBatch(
            mesh_context=MeshContext(), device=torch.device("cpu"), tokenizer=SimpleNamespace(eod=None),
            data_config={"labels_are_shifted": True, "create_attention_mask_in_dataloader": False},
            pp_shared_data=False, source_type="online", runtime_input_adapter=runtime,
        )
        source = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "labels": torch.tensor([[2, 3, 4, -100]]),
                  "cu_seq_lens": torch.tensor([0, 4], dtype=torch.int32)}
        model_inputs, loss_inputs = batcher(iter([source]))
        self.assertIs(model_inputs["mtp_targets"], loss_inputs["mtp_labels"])
        self.assertEqual(model_inputs["mtp_targets"].tolist(), [[[3, 4, -100, -100]], [[4, -100, -100, -100]]])
