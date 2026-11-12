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
"""CPU tests of NPU routing-key conversion and dispatch permutation semantics."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from hyper_parallel.distributed.expert_parallel.experts import _argsort_keys
from tests.common.mark_utils import arg_mark


def _device_keys(values: torch.Tensor, device_type: str) -> MagicMock:
    """Emulate device selection while running real conversion and sorting on CPU."""
    keys = MagicMock()
    keys.device = SimpleNamespace(type=device_type)
    keys.to.side_effect = values.to
    keys.argsort.side_effect = values.argsort
    return keys


class TestExpertSortKeys(unittest.TestCase):
    """Check exact integer bounds and preservation of routed outputs and gradients."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_npu_sort_is_exact_up_to_supported_bound(self):
        """Adjacent keys near 2**24 stay distinguishable after conversion."""
        for values in (torch.tensor([], dtype=torch.int64), torch.tensor([2**24 - 1, 0, 2**24 - 2, 1])):
            with self.subTest(size=values.numel()):
                keys = _device_keys(values, "npu")
                order = _argsort_keys(keys, bound=2**24)
                torch.testing.assert_close(values[order], values.sort().values, rtol=0, atol=0)
                keys.to.assert_called_once_with(torch.float32)
                self.assertEqual(order.dtype, torch.int64)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_other_devices_and_large_bounds_keep_integer_sort(self):
        """CPU/CUDA avoid conversion; out-of-range NPU keys cannot lose integer bits."""
        for device_type, bound in (("cpu", 8), ("cuda", 8), ("npu", 2**24 + 2), ("npu", 0)):
            with self.subTest(device=device_type, bound=bound):
                values = torch.tensor([2**24 + 1, 2**24, 0] if bound > 2**24 else [2, 1, 0])
                if bound == 0:
                    values = torch.empty(0, dtype=torch.int64)
                keys = _device_keys(values, device_type)
                order = _argsort_keys(keys, bound=bound)
                torch.testing.assert_close(order, values.argsort())
                keys.to.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_expert_grouping_round_trip_preserves_outputs_and_gradients(self):
        """Repeated expert keys can be permuted and restored without changing the routed computation."""
        indices = torch.tensor([2, 0, 1, 2, 0, 2, 1, 0])
        inputs = torch.arange(24, dtype=torch.float64).reshape(8, 3).requires_grad_()
        weights = torch.arange(18, dtype=torch.float64).reshape(3, 3, 2).requires_grad_()
        reference = torch.bmm(inputs.unsqueeze(1), weights[indices]).squeeze(1)
        order = _argsort_keys(_device_keys(indices, "npu"), bound=3)
        packed = torch.bmm(inputs[order].unsqueeze(1), weights[indices[order]]).squeeze(1)
        restored = packed[order.argsort()]
        torch.testing.assert_close(indices[order], indices.sort().values)
        torch.testing.assert_close(restored, reference, rtol=0, atol=0)
        actual_grads = torch.autograd.grad(restored.square().sum(), (inputs, weights))
        expected_grads = torch.autograd.grad(reference.square().sum(), (inputs, weights))
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
