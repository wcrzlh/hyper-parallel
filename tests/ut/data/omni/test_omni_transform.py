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
"""CPU tests for external image binding before processor encoding."""

import copy
import io
import json
import unittest
from typing import Any

from PIL import Image

from hyper_parallel.data.omni.omni_transform import AutoProcessorTransform
from hyper_parallel.data.online.format import normalize_tools
from hyper_parallel.data.text.chat_template import TokenizerTemplate
from hyper_parallel.data.text.text_transform import TextConversationTransform
from tests.common.mark_utils import arg_mark
from tests.ut.data.test_thinking import ThinkingProcessor, ThinkingTokenizer


def image_bytes(color: str, mode: str = "RGB") -> bytes:
    """Build a tiny real PNG without accessing external files."""
    with io.BytesIO() as buffer:
        Image.new("RGB", (2, 3), color).convert(mode).save(buffer, format="PNG")
        return buffer.getvalue()


class TestOmniImageBinding(unittest.TestCase):
    """Verify image ordering, error handling, and legacy input preservation."""

    def setUp(self) -> None:
        """Build a transform for message preparation independently of tokenization."""
        self.transform = AutoProcessorTransform(max_seq_len=4096, processor=object())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_bytes_bind_across_messages_and_text_blocks(self):
        """Verify image order across a conversation.

        Feature: External image binding.
        Description: Bind three colored images across strings and text blocks.
        Expectation: Pixels follow placeholder order without mutating source records.
        """
        sample = {
            "messages": [
                {"role": "user", "content": "before<image>\nafter"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": [{"type": "text", "text": "<image><image>compare"}]},
            ],
            "images": [{"bytes": image_bytes(color), "path": "missing.png"}
                       for color in ("red", "green", "blue")],
        }
        before = copy.deepcopy(sample)
        messages = self.transform.prepare_messages(sample)
        self.assertEqual(sample, before)
        self.assertEqual(messages[0]["content"][0], {"type": "text", "text": "before"})
        self.assertEqual(messages[0]["content"][2], {"type": "text", "text": "\nafter"})
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "answer"}])
        self.assertEqual(messages[2]["content"][-1], {"type": "text", "text": "compare"})
        images = [block["image"] for message in (messages[0], messages[2])
                  for block in message["content"] if block["type"] == "image"]
        self.assertEqual([image.getpixel((0, 0)) for image in images], [(255, 0, 0), (0, 128, 0), (0, 0, 255)])
        self.assertTrue(all(image.mode == "RGB" and image.size == (2, 3) for image in images))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_path_fallback_and_bare_image_block(self):
        """Verify path-only external images.

        Feature: External image path fallback.
        Description: Mix explicit media and external relative/remote paths.
        Expectation: Only placeholders consume external images; relative paths resolve.
        """
        sample = {
            "__online_source_path__": "/data/shard.parquet",
            "messages": [{"role": "user", "content": [
                {"type": "image", "image": "explicit.png"},
                {"type": "image"}, {"type": "text", "text": "<image>"},
            ]}],
            "images": [{"bytes": None, "path": "relative.png"}, "https://example.com/img.png"],
        }
        content = self.transform.prepare_messages(sample)[0]["content"]
        self.assertEqual([block["image"] for block in content],
                         ["/data/explicit.png", "/data/relative.png", "https://example.com/img.png"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_image_count_mismatch_is_not_silent(self):
        """Verify image-count validation.

        Feature: External image validation.
        Description: Supply missing or unused images.
        Expectation: Every mismatch raises before processor encoding.
        """
        for text, images in (("<image>", []), ("<image><image>", ["a.png"]), ("no placeholder", ["a.png"])):
            with self.subTest(text=text, images=images):
                with self.assertRaisesRegex(ValueError, "image.*(count|placeholder)|placeholder.*image"):
                    self.transform.prepare_messages({"messages": [{"content": text}], "images": images})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_bad_images_fail_with_index(self):
        """Verify malformed-image diagnostics.

        Feature: External image validation.
        Description: Supply corrupt bytes or missing image sources.
        Expectation: Errors identify the image index without path fallback.
        """
        for item in ({"bytes": b"not an image", "path": "a.png"}, {"bytes": b""},
                     {"bytes": "base64"}, {"bytes": None, "path": None}, None):
            with self.subTest(item=item), self.assertRaisesRegex((ValueError, TypeError), r"images\[0\]"):
                self.transform.prepare_messages({"messages": [{"content": "<image>"}], "images": [item]})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_decode_grayscale_to_rgb(self):
        """Verify decoded image ownership.

        Feature: Image bytes decoding.
        Description: Decode a grayscale PNG and inspect pixels after buffer closure.
        Expectation: The returned RGB image owns usable pixel data.
        """
        sample = {"messages": [{"content": "<image>"}], "images": [{"bytes": image_bytes("white", "L")}]}
        image = self.transform.prepare_messages(sample)[0]["content"][0]["image"]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((1, 2)), (255, 255, 255))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_arrow_null_image_fields_are_not_extra_sources(self):
        """Verify nullable Arrow image structs.

        Feature: Parquet image normalization.
        Description: Prepare image blocks containing absent fields represented as nulls.
        Expectation: Null fields do not become extra processor image sources.
        """
        sample = {
            "__online_source_path__": "/data/shard.parquet",
            "messages": [{"content": [
                {"type": "image", "path": "a.png", "url": None, "image": None},
                {"type": "image", "path": None, "url": None, "image": None},
            ]}],
            "images": [{"bytes": image_bytes("red")}],
        }
        content = self.transform.prepare_messages(sample)[0]["content"]
        self.assertEqual(content[0], {"type": "image", "path": "/data/a.png"})
        self.assertEqual(set(content[1]), {"type", "image"})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_no_images_preserves_legacy_messages(self):
        """Verify legacy message compatibility.

        Feature: Omni format compatibility.
        Description: Prepare messages without an external image list.
        Expectation: Ordinary strings and explicit image paths remain unchanged.
        """
        messages = [{"role": "user", "content": "question"},
                    {"role": "user", "content": [{"type": "image", "image": "a.png"}]},
                    {"role": "assistant", "content": "answer"}]
        for images in (None, []):
            with self.subTest(images=images):
                self.assertEqual(self.transform.prepare_messages({"messages": messages, "images": images}), messages)


class ToolsTokenizer(ThinkingTokenizer):
    """Render tool definitions as context and calls as assistant output."""

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Use the same tool prefix for full, prompt, and per-turn rendering."""
        messages = copy.deepcopy(messages)
        tools = kwargs.pop("tools", None)
        if tools:
            messages.insert(0, {"role": "system", "content": json.dumps(tools)})
        for message in messages:
            if message.get("tool_calls"):
                message["content"] = "<tool_call>CALL_ARGUMENT</tool_call>"
        return super().apply_chat_template(messages, **kwargs)


class TestOmniTools(unittest.TestCase):
    """Check shared tools parsing and exact supervision with media and thinking."""

    def setUp(self) -> None:
        """Build isolated templates and a complete tool-use conversation."""
        self.processor = ThinkingProcessor()
        self.processor.tokenizer = ToolsTokenizer()
        self.tools = [{"type": "function", "function": {"name": "TOOL_SCHEMA"}}]
        self.messages = [
            {"role": "system", "content": "SYSTEM_CONTEXT"},
            {"role": "user", "content": "USER_QUERY"},
            {"role": "assistant", "content": "", "reasoning_content": "CALL_REASON",
             "tool_calls": [{"type": "function", "function": {"name": "TOOL_SCHEMA", "arguments": {}}}]},
            {"role": "tool", "content": "TOOL_RESULT"},
            {"role": "assistant", "content": "FINAL_ANSWER"},
        ]

    def supervised(self, labels: Any) -> str:
        """Decode all and only tokens selected for the training loss."""
        return self.processor.tokenizer.decode([int(token) for token in labels if token != -100])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_tools_loss_mask_text_vlm_parity(self) -> None:
        """Schema and tool results are context; reasoning, calls, and answers are targets.

        Feature: Shared tool definitions and supervision.
        Description: Schema and tool results are context; reasoning, calls, and answers are targets.
        Expectation: All asserted outputs, masks, gradients, and errors match the reference.
        """
        transform = AutoProcessorTransform(processor=self.processor, max_seq_len=4096,
                                           thinking_mode="auto", tools_key="functions")
        sample = {"messages": self.messages, "functions": json.dumps(self.tools)}
        original = copy.deepcopy(sample)
        result = transform.encode_sample(sample)
        text = TextConversationTransform(TokenizerTemplate(self.processor.tokenizer), 4096,
                                         text_keys="messages", tools_key="functions", thinking_mode="auto")(sample)[0]
        expected = ("CALL_REASON\n</think>\n\n<tool_call>CALL_ARGUMENT</tool_call></assistant>"
                    "FINAL_ANSWER</assistant>")
        self.assertEqual(self.supervised(result["labels"]), expected)
        self.assertEqual(result["input_ids"][:-1].tolist(), text["input_ids"].tolist())
        self.assertEqual(result["labels"][1:].tolist(), text["labels"].tolist())
        self.assertIn("TOOL_SCHEMA", self.processor.tokenizer.decode(result["input_ids"].tolist()))
        self.assertEqual(sample, original)

        self.messages[1]["content"] = [{"type": "image", "image": 21},
                                       {"type": "text", "text": "USER_QUERY"}]
        media = transform.encode_sample(sample)
        self.assertEqual(self.supervised(media["labels"]), expected)
        self.assertTrue(media["labels"][media["mm_token_type_ids"] != 0].eq(-100).all())
        self.assertEqual(media["pixel_values"].tolist(), [21])
        self.messages[2]["loss_mask"] = 0
        masked = transform.encode_sample(sample)
        self.assertEqual(self.supervised(masked["labels"]), "FINAL_ANSWER</assistant>")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_optional_tools_do_not_leak_between_samples(self) -> None:
        """Missing and empty per-sample tools clear static defaults without mutating them.

        Feature: Shared tool definitions and supervision.
        Description: Missing and empty per-sample tools clear static defaults without mutating them.
        Expectation: All asserted outputs, masks, gradients, and errors match the reference.
        """
        defaults = {"tools": self.tools, "preserve_thinking": True}
        transform = AutoProcessorTransform(processor=self.processor, max_seq_len=4096,
                                           tools_key="tools", chat_template_kwargs=defaults)
        sample = {"messages": [self.messages[1], self.messages[-1]]}
        tool_result = transform.encode_sample(dict(sample, tools=self.tools))
        self.assertIn("TOOL_SCHEMA", self.processor.tokenizer.decode(tool_result["input_ids"].tolist()))
        for value in (None, False, [], "", "[]", "null", "false"):
            with self.subTest(value=value):
                result = transform.encode_sample(dict(sample, tools=value))
                self.assertNotIn("TOOL_SCHEMA", self.processor.tokenizer.decode(result["input_ids"].tolist()))
        self.assertNotIn("TOOL_SCHEMA", self.processor.tokenizer.decode(
            transform.encode_sample(sample)["input_ids"].tolist()))
        self.assertEqual(defaults, {"tools": self.tools, "preserve_thinking": True})

        plain = AutoProcessorTransform(processor=self.processor, max_seq_len=4096)
        self.assertEqual(plain.encode_sample(sample)["input_ids"].tolist(),
                         plain.encode_sample(dict(sample, tools=self.tools))["input_ids"].tolist())
        static = AutoProcessorTransform(processor=self.processor, max_seq_len=4096,
                                        chat_template_kwargs=defaults)
        static_ids = static.encode_sample(sample)["input_ids"].tolist()
        self.assertIn("TOOL_SCHEMA", self.processor.tokenizer.decode(static_ids))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_invalid_tools_fail_in_both_transforms(self) -> None:
        """Malformed JSON, scalar schemas, and invalid tools keys fail explicitly.

        Feature: Shared tool definitions and supervision.
        Description: Malformed JSON, scalar schemas, and invalid tools keys fail explicitly.
        Expectation: All asserted outputs, masks, gradients, and errors match the reference.
        """
        text = TextConversationTransform(TokenizerTemplate(self.processor.tokenizer), 4096,
                                         text_keys="messages", tools_key="tools")
        omni = AutoProcessorTransform(processor=self.processor, max_seq_len=4096, tools_key="tools")
        for value in ("{", "{}", {}, True, 1, ["invalid"]):
            with self.subTest(value=value):
                sample = {"messages": self.messages, "tools": value}
                with self.assertRaises(ValueError):
                    text(sample)
                with self.assertRaises(ValueError):
                    omni.encode_sample(sample)
        for key in ("", False, 1):
            with self.subTest(key=key), self.assertRaises(ValueError):
                AutoProcessorTransform(processor=self.processor, max_seq_len=4096, tools_key=key)
        self.assertEqual(normalize_tools(json.dumps(self.tools), "tools"), self.tools)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_sharegpt_tools_are_context_only(self) -> None:
        """The alternate sample format shares tool parsing and assistant masking.

        Feature: Shared tool definitions and supervision.
        Description: The alternate sample format shares tool parsing and assistant masking.
        Expectation: All asserted outputs, masks, gradients, and errors match the reference.
        """
        transform = AutoProcessorTransform(processor=self.processor, max_seq_len=4096,
                                           sample_format="sharegpt", tools_key="tools", thinking_mode="auto")
        result = transform.encode_sample({"tools": self.tools, "conversations": [
            {"from": "system", "value": "SYSTEM_CONTEXT"},
            {"from": "human", "value": "USER_QUERY"},
            {"from": "gpt", "value": "FINAL_ANSWER"},
        ]})
        self.assertIn("TOOL_SCHEMA", self.processor.tokenizer.decode(result["input_ids"].tolist()))
        self.assertEqual(self.supervised(result["labels"]), "FINAL_ANSWER</assistant>")
