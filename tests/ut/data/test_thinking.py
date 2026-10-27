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
"""CPU regression tests for sample-local thinking and multi-turn supervision."""

import unittest

import torch

from hyper_parallel.data.omni.omni_transform import AutoProcessorTransform
from hyper_parallel.data.text.chat_template import TokenizerTemplate
from hyper_parallel.data.text.build_data_transform import TextConversationTransform
from hyper_parallel.data.thinking import has_reasoning, thinking_kwargs
from tests.ut.data.conftest import FakeTokenizer


class ThinkingTokenizer(FakeTokenizer):
    """Qwen-style thinking prefixes with a deliberately merged newline token."""

    def __init__(self):
        super().__init__()
        self.vocab.update({"\n\n": 1114112})

    def decode(self, tokens, **kwargs):
        """Decode special tokens losslessly, including merged newlines."""
        del kwargs
        reverse = {value: key for key, value in self.vocab.items()}
        return "".join(reverse[token] if token in reverse else chr(token) for token in tokens)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False,
                            return_dict=True, enable_thinking=True, **kwargs):
        """Render stable history and a mode-specific generation prompt."""
        del kwargs
        parts = []
        for message in messages:
            body = message.get("content", "")
            if isinstance(body, list):
                body = "".join(block["text"] if block["type"] == "text" else "<image_placeholder>"
                               for block in body)
            if message["role"] == "assistant":
                reasoning = message.get("reasoning_content", "")
                if body.startswith("<think>"):
                    reasoning, body = body[len("<think>"):].split("</think>", 1)
                    reasoning, body = reasoning.strip(), body.lstrip()
                body = f"<think>\n{reasoning}\n</think>\n\n{body}"
            parts.append(f"<{message['role']}>{body}</{message['role']}>")
        if add_generation_prompt:
            parts.append("<assistant><think>\n" if enable_thinking else "<assistant><think>\n\n</think>\n\n")
        text = "".join(parts)
        result = self.encode(text, add_special_tokens=False) if tokenize else text
        return {"input_ids": result} if return_dict else result


class ThinkingProcessor:
    """Keep full processor tokens and image ordering observable in tests."""

    def __init__(self):
        self.tokenizer = ThinkingTokenizer()

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        """Return expanded media tokens and deterministic ordered image values."""
        kwargs.pop("return_tensors", None)
        kwargs.pop("return_dict", None)
        result = self.tokenizer.apply_chat_template(messages, tokenize=tokenize, return_dict=False, **kwargs)
        if not tokenize:
            return result
        images = [block["image"] for message in messages if isinstance(message["content"], list)
                  for block in message["content"] if block["type"] == "image"]
        tokens = [token for value in result for token in ([value] * 3 if value == 3 else [value])]
        return {"input_ids": torch.tensor([tokens]),
                "mm_token_type_ids": torch.tensor([[int(token == 3) for token in tokens]]),
                "pixel_values": torch.tensor(images)}


class TestThinking(unittest.TestCase):
    """Exercise both transforms without model weights or distributed setup."""

    @staticmethod
    def messages():
        """Mix a slow answer and a fast answer in one conversation."""
        return [{"role": "user", "content": "question1"},
                {"role": "assistant", "content": "answer1", "reasoning_content": "reason1"},
                {"role": "user", "content": "question2"},
                {"role": "assistant", "content": "answer2"}]

    @staticmethod
    def supervised(tokenizer, labels):
        """Decode only supervised tokens for an exact target golden."""
        return tokenizer.decode([int(token) for token in labels if token != -100])

    def test_policy_and_validation(self):
        """Automatic modes are local; contradictory explicit flags fail closed."""
        defaults = {"preserve_thinking": True}
        self.assertTrue(thinking_kwargs(self.messages()[1], "auto", None, defaults)["enable_thinking"])
        self.assertFalse(thinking_kwargs(self.messages()[3], "auto", None, defaults)["enable_thinking"])
        self.assertEqual(defaults, {"preserve_thinking": True})
        self.assertTrue(has_reasoning({"content": "<think> reason </think>answer"}))
        self.assertFalse(has_reasoning({"content": "mention <think> literally"}))
        for flag in (False, "false", 0):
            with self.assertRaises(ValueError):
                thinking_kwargs(self.messages()[1], "auto", flag, defaults)
        with self.assertRaises(ValueError):
            has_reasoning({"content": "<think>unfinished"})

    def test_text_mixed_turns_and_shift(self):
        """Every assistant is supervised and labels are shifted exactly once."""
        tokenizer = ThinkingTokenizer()
        template = TokenizerTemplate(tokenizer)
        messages = self.messages()
        encoded = template.encode_messages(messages, thinking_mode="auto")
        expected = "reason1\n</think>\n\nanswer1</assistant>answer2</assistant>"
        self.assertEqual(self.supervised(tokenizer, encoded["labels"]), expected)
        transform = TextConversationTransform(template, 4096, "messages", thinking_mode="auto")
        transformed = transform({"messages": messages})[0]
        self.assertEqual(transformed["labels"].tolist(), encoded["labels"][1:])
        self.assertEqual(transformed["input_ids"].tolist(), encoded["input_ids"][:-1])
        self.assertEqual(template.chat_template_kwargs, {})

    def test_text_sample_override_and_legacy(self):
        """Legacy calls remain unchanged and sample flags cannot leak to later calls."""
        tokenizer = ThinkingTokenizer()
        template = TokenizerTemplate(tokenizer)
        messages = self.messages()[2:]
        legacy = template.encode_messages(messages)
        fast = template.encode_messages(messages, enable_thinking=False)
        self.assertEqual(self.supervised(tokenizer, fast["labels"]), "answer2</assistant>")
        self.assertEqual(template.encode_messages(messages), legacy)
        self.assertEqual(template.encode_messages(messages, thinking_mode="legacy"), legacy)

    def test_vlm_all_turns_media_order_and_masks(self):
        """Multi-turn labels do not alter full processor tokens or ordered image payloads."""
        processor = ThinkingProcessor()
        messages = self.messages()
        for index, image in ((0, 21), (2, 42)):
            messages[index]["content"] = [{"type": "image", "image": image},
                                          {"type": "text", "text": "question"}]
        transform = AutoProcessorTransform(processor=processor, max_seq_len=4096, thinking_mode="auto")
        result = transform.encode_sample({"messages": messages})
        expected = "reason1\n</think>\n\nanswer1</assistant>answer2</assistant>"
        self.assertEqual(self.supervised(processor.tokenizer, result["labels"]), expected)
        original = processor.apply_chat_template(messages, add_generation_prompt=False)
        self.assertEqual(result["input_ids"].tolist(), original["input_ids"][0].tolist())
        self.assertEqual(result["pixel_values"].tolist(), [21, 42])
        self.assertTrue(result["labels"][result["mm_token_type_ids"] != 0].eq(-100).all())
        messages[1]["loss_mask"] = 0
        result = transform.encode_sample({"messages": messages})
        self.assertEqual(self.supervised(processor.tokenizer, result["labels"]), "answer2</assistant>")

    def test_vlm_default_all_turns_and_single_turn(self):
        """All-turn supervision needs no switch; single-turn legacy remains the same path."""
        processor = ThinkingProcessor()
        transform = AutoProcessorTransform(processor=processor, max_seq_len=4096)
        result = transform.encode_sample({"messages": self.messages()})
        supervised = self.supervised(processor.tokenizer, result["labels"])
        self.assertIn("answer1</assistant>", supervised)
        self.assertIn("answer2</assistant>", supervised)
        self.assertNotIn("question", supervised)
        single = transform.encode_sample({"messages": self.messages()[2:]})
        explicit = AutoProcessorTransform(processor=processor, max_seq_len=4096, thinking_mode="legacy")
        self.assertEqual(single["labels"].tolist(),
                         explicit.encode_sample({"messages": self.messages()[2:]})["labels"].tolist())

    def test_reverse_order_inline_reasoning_and_tool_mask(self):
        """Fast-then-slow and intervening tool output retain the correct target spans."""
        messages = self.messages()
        messages[1].pop("reasoning_content")
        messages[3]["content"] = "<think>reason2</think>answer2"
        messages.insert(2, {"role": "tool", "content": "TOOL_RESULT"})
        expected = "answer1</assistant>reason2\n</think>\n\nanswer2</assistant>"
        tokenizer = ThinkingTokenizer()
        text = TokenizerTemplate(tokenizer).encode_messages(messages, thinking_mode="auto")
        self.assertEqual(self.supervised(tokenizer, text["labels"]), expected)
        processor = ThinkingProcessor()
        transform = AutoProcessorTransform(processor=processor, max_seq_len=4096, thinking_mode="auto")
        result = transform.encode_sample({"messages": messages})
        self.assertEqual(self.supervised(processor.tokenizer, result["labels"]), expected)

    def test_vlm_crossing_boundary_is_not_supervised(self):
        """A merged token crossing a start or end boundary is excluded in full."""
        transform = AutoProcessorTransform(processor=ThinkingProcessor(), max_seq_len=4096)
        full = torch.tensor([ord("a"), 1114112, ord("b")])
        prefix = torch.tensor([ord("a"), ord("\n")])
        self.assertEqual(transform._prefix_boundary(full, prefix, round_up=True), 2)
        self.assertEqual(transform._prefix_boundary(full, prefix, round_up=False), 1)

    def test_vlm_rejects_rewritten_history(self):
        """Do not silently assign labels when a template drops previous reasoning."""
        processor = ThinkingProcessor()
        original = processor.apply_chat_template

        def rewriting(messages, **kwargs):
            result = original(messages, **kwargs)
            if kwargs.get("tokenize") is False and len(messages) > 2:
                return result.replace("reason1", "")
            return result

        processor.apply_chat_template = rewriting
        transform = AutoProcessorTransform(processor=processor, max_seq_len=4096, thinking_mode="auto")
        with self.assertRaisesRegex(ValueError, "history was rewritten"):
            transform.encode_sample({"messages": self.messages()})
