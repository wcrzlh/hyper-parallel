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
"""Remote-code (auto_map) model loading through the HF-native path."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM

from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models._transformers.model_builder import _init_model


class TestRemoteCodeParentResolution(unittest.TestCase):
    """The parent HF AutoModel class must receive the delegated call."""

    def test_parent_delegate_resolves_hf_auto_base(self) -> None:
        """cls inside the HF classmethod must be the HF AutoModel base.

        With super() delegation, ``cls`` would remain the Hyper-Parallel
        subclass whose ``__name__`` is absent from ``auto_map``.
        """

        def fake_from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls

        with patch.object(
            AutoModelForCausalLM,
            "from_pretrained",
            classmethod(fake_from_pretrained),
        ):
            resolved = HyperAutoModelForCausalLM._from_pretrained_parent_class(
                "some-checkpoint"
            )
        self.assertIs(resolved, AutoModelForCausalLM)


class TestInitModelConfigOmission(unittest.TestCase):
    """An explicit config object must not suppress auto_map resolution."""

    @staticmethod
    def _run(hf_config, **kwargs):
        captured = {}

        def fake_parent(cls, *args, **parent_kwargs):
            del cls, args
            captured.update(parent_kwargs)
            return "model-sentinel"

        with patch.object(
            HyperAutoModelForCausalLM,
            "_from_pretrained_parent_class",
            classmethod(fake_parent),
        ):
            is_custom, model = _init_model(
                HyperAutoModelForCausalLM,
                "some-checkpoint",
                hf_config,
                "sdpa",
                torch.bfloat16,
                True,
                backend="hccl",
                **kwargs,
            )
        return is_custom, model, captured

    def test_trust_remote_code_with_auto_map_omits_config(self) -> None:
        """Trusted remote code with auto_map resolves the class from the path."""
        config = SimpleNamespace(
            architectures=["QwenTestForCausalLM"],
            auto_map={"AutoModelForCausalLM": "modeling_test.QwenTestForCausalLM"},
        )
        is_custom, model, captured = self._run(
            config, trust_remote_code=True, ignore_mismatched_sizes=True
        )
        self.assertFalse(is_custom)
        self.assertEqual(model, "model-sentinel")
        self.assertNotIn("config", captured)
        self.assertTrue(captured["trust_remote_code"])

    def test_auto_map_without_trust_remote_code_keeps_config(self) -> None:
        """Without trust_remote_code the explicit config object is preserved."""
        config = SimpleNamespace(
            architectures=["QwenTestForCausalLM"],
            auto_map={"AutoModelForCausalLM": "modeling_test.QwenTestForCausalLM"},
        )
        _, _, captured = self._run(config, trust_remote_code=False)
        self.assertIs(captured["config"], config)

    def test_no_auto_map_keeps_config(self) -> None:
        """Built-in architectures keep the explicit config object."""
        config = SimpleNamespace(architectures=["LlamaForCausalLM"])
        _, _, captured = self._run(config, trust_remote_code=True)
        self.assertIs(captured["config"], config)


if __name__ == "__main__":
    unittest.main()
