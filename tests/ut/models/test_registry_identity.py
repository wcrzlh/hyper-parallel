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
"""Task-preserving adapter lookup checks for shared Qwen families."""

import unittest
from types import SimpleNamespace

from hyper_parallel.models.registry import get_model_adapter, resolve_model_adapter
from tests.common.mark_utils import arg_mark


class TestAdapterIdentity(unittest.TestCase):
    """Exact CausalLM identities must not fall back to VLM family aliases."""

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_text_and_vlm_keep_distinct_specs_in_either_lookup_order(self) -> None:
        """Feature: Task identity. Description: Resolve architectures and configs.
        Expectation: Text models select text specs even when VLM was registered first.
        """
        for family in ("qwen3_5", "qwen3_5_moe"):
            with self.subTest(family=family):
                visual = get_model_adapter(family)
                text = get_model_adapter(f"{family}_text")
                self.assertIsNot(visual, text)
                self.assertIs(get_model_adapter(text.architecture), text)
                self.assertIs(get_model_adapter(text.architecture.lower()), text)
                model = SimpleNamespace(config=SimpleNamespace(
                    model_type=text.model_type, architectures=[visual.architecture],
                ))
                self.assertIs(resolve_model_adapter(model), text)
