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
"""Launch equal-count step packing through real DP/CP and DP/TP groups."""

from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(["cpu_linux"], "level0", "allcards", "essential")
@pytest.mark.parametrize("cp_size", [1, 2])
def test_step_balance_gloo(cp_size: int) -> None:
    """Feature: Equal-count cross-DP step packing.
    Description: Exercise routing, CP replicas, TP non-readers and checkpoint replay.
    Expectation: Tokens, labels, normalized gradients and resumed steps match.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_step_balance.py")),
        case_name=f"test_step_balance[gloo-{cp_size}]", num_proc=4,
    )


@arg_mark(["platform_ascend910b"], "level0", "allcards", "essential")
@pytest.mark.parametrize("cp_size", [1, 2])
def test_step_balance_hccl(cp_size: int) -> None:
    """Feature: Equal-count cross-DP step packing on NPU.
    Description: Send encoded samples through HCCL with CP or TP peers.
    Expectation: The production transport preserves source steps and replay exactly.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_step_balance.py")),
        case_name=f"test_step_balance[hccl-{cp_size}]", num_proc=4,
    )
