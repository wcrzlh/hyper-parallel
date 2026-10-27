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
"""Regression coverage for tool-aware training length analysis."""

import argparse
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from typing import Any

from tests.common.mark_utils import arg_mark

from examples.qwen3_8.tools.analyze_decif_lengths import analyze_record, resolve_settings
from hyper_parallel.data.text.chat_template import TokenizerTemplate
from tests.ut.data.conftest import FakeTokenizer


class TestAnalyzeDecifLengths(unittest.TestCase):
    """Check tool forwarding and supervision loss after truncation."""

    def setUp(self) -> None:
        """Build a small conversation using the training template."""
        self.tokenizer = FakeTokenizer()
        self.template = TokenizerTemplate(self.tokenizer)
        self.record = {"messages": [{"role": "user", "content": "question"},
                                    {"role": "assistant", "content": "answer"}]}

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_tools_forwarded_to_every_render(self):
        """Feature: Training-aligned dataset length analysis.

        Description: JSON-encoded tools reach both full and truncated training encoders.
        Expectation: Reported settings and supervision match the training path.
        """
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        self.record["tools"] = json.dumps(tools)
        original = self.tokenizer.apply_chat_template

        def render(messages: Any, **kwargs: Any) -> Any:
            """Check every render receives the normalized tool definitions."""
            self.assertEqual(kwargs.pop("tools"), tools)
            return original(messages, **kwargs)

        with patch.object(self.tokenizer, "apply_chat_template", side_effect=render):
            row = analyze_record(self.tokenizer, self.template, self.record, 7, 4096)
        self.assertEqual(row["tool_count"], 1)
        self.assertEqual(row["source_index"], 7)
        self.assertFalse(row["zero_supervision"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_tool_tail_removes_all_supervision(self):
        """Feature: Training-aligned dataset length analysis.

        Description: A long masked tool tail leaves no labels after left truncation.
        Expectation: Reported settings and supervision match the training path.
        """
        self.record["messages"].append({"role": "tool", "content": "x" * 200})
        row = analyze_record(self.tokenizer, self.template, self.record, 3, 32)
        self.assertEqual(row["train_tokens_after_truncation"], 31)
        self.assertEqual(row["supervised_tokens_after_truncation"], 0)
        self.assertTrue(row["supervision_lost_to_truncation"])
        self.assertTrue(row["zero_supervision"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_unsupervised_source_is_not_truncation_loss(self):
        """Feature: Training-aligned dataset length analysis.

        Description: Distinguish an originally unsupervised sample from a truncated target.
        Expectation: Reported settings and supervision match the training path.
        """
        self.record["messages"] = self.record["messages"][:1]
        row = analyze_record(self.tokenizer, self.template, self.record, 9, 4096)
        self.assertTrue(row["zero_supervision"])
        self.assertFalse(row["supervision_lost_to_truncation"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_training_config_and_cli_precedence(self):
        """Feature: Training-aligned dataset length analysis.

        Description: Use training transform options unless a CLI override is explicit.
        Expectation: Reported settings and supervision match the training path.
        """
        config = {"dataset": {"model_assets": {"chat_template": "tokenizer", "chat_template_kwargs": {
            "enable_thinking": True, "preserve_thinking": True}}, "data_transform": {
                "data_type": "conversation", "max_seq_len": 128, "text_keys": "chat",
                "tools_key": "functions", "thinking_mode": "auto"}}}
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "train.yaml"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            args = argparse.Namespace(config=config_path, disable_thinking=False, reasoning_effort=None,
                                      max_seq_len=None, text_key=None, tools_key=None, thinking_mode=None)
            self.assertTrue(resolve_settings(args)["preserve_thinking"])
            self.assertEqual((args.max_seq_len, args.text_key, args.tools_key, args.thinking_mode),
                             (128, "chat", "functions", "auto"))
            args.tools_key = ""
            args.max_seq_len = 64
            args.disable_thinking = True
            self.assertFalse(resolve_settings(args)["enable_thinking"])
            self.assertIsNone(args.tools_key)
            self.assertEqual(args.max_seq_len, 64)
