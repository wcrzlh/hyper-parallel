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
"""Unit tests for text Trainer metric inputs."""
# pylint: disable=wrong-import-position

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch

from hyper_parallel.trainer.text_trainer import TextTrainer
from tests.common.mark_utils import arg_mark


class TestTextTrainerMetrics(unittest.TestCase):
    """Verify text training forwards each step's metric inputs."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    @patch("hyper_parallel.trainer.text_trainer.synchronize")
    def test_train_step_passes_loss_inputs_to_step_callbacks(self, _mock_synchronize):
        """Forward every micro-batch label mapping to token accounting callbacks."""
        training_batches = [
            (
                {"input_ids": torch.tensor([[1, 2, 3]])},
                {"labels": torch.tensor([[-100, 2, 3]])},
            ),
            (
                {"input_ids": torch.tensor([[4, 5]])},
                {"labels": torch.tensor([[-100, 5]])},
            ),
        ]
        trainer = TextTrainer.__new__(TextTrainer)
        trainer.base = SimpleNamespace(
            config=SimpleNamespace(training=SimpleNamespace(max_grad_norm=0.0)),
            num_micro_batches=len(training_batches),
            optimizer=[],
            lr_scheduler=None,
            state=SimpleNamespace(global_step=0),
            get_batch=MagicMock(side_effect=training_batches),
            model_reshard=MagicMock(),
            configure_fsdp_gradient_sync=MagicMock(),
        )
        trainer.on_step_begin = MagicMock()
        trainer.on_step_end = MagicMock()
        trainer.forward_backward_step = MagicMock(
            side_effect=[
                (torch.tensor(1.0), {"foundation_loss": torch.tensor(1.0)}),
                (torch.tensor(2.0), {"foundation_loss": torch.tensor(2.0)}),
            ]
        )

        trainer.train_step(data_iterator=iter(()))

        trainer.on_step_begin.assert_called_once_with(
            micro_batches=[loss_inputs for _, loss_inputs in training_batches]
        )
        self.assertEqual(trainer.base.state.global_step, 1)


if __name__ == "__main__":
    unittest.main()
