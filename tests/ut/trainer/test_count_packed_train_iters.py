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
"""CPU worker counting must preserve source order and production packing results."""

from contextlib import ExitStack
import json
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from examples.qwen3_6_moe.count_packed_train_iters import _collect_lengths, count_epoch
from hyper_parallel.data.batching import TokenBatchLoader
from hyper_parallel.data.batching.build_collate_fn import build_online_text_collate_fn
from hyper_parallel.data.online.mapping import MappingTransformDataset, _JsonlMappingSource
from hyper_parallel.data.text.text_transform import PlaintextTransform
from hyper_parallel.data.text.build_tokenizer import _HFAutoTokenizer
from hyper_parallel.trainer.config.data import DataLoaderConfig
from hyper_parallel.trainer.config.target import Target
from tests.common.mark_utils import arg_mark


class _WordTokenizer:
    """Provide deterministic local token IDs without a downloaded tokenizer."""

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode one token per word, with no terminal token."""
        del add_special_tokens
        return list(range(len(text.split())))


def _packing_config() -> SimpleNamespace:
    return SimpleNamespace(
        dataloader=DataLoaderConfig(
            target=Target(TokenBatchLoader, target_path="hyper_parallel.data.batching.TokenBatchLoader",
                          token_budget=12, min_buffered_samples=3, drop_last=True, num_workers=0),
            collate_fn=Target(build_online_text_collate_fn, target_path="unused", pad_to_length=12),
            sampler_type="cyclic",
        ),
        training=SimpleNamespace(global_batch_size=4, micro_batch_size=1, seed=17),
        dataset=SimpleNamespace(data_transform=SimpleNamespace(max_seq_len=5)),
    )


class TestCountPackedTrainIters(unittest.TestCase):
    """Exercise actual worker processes and process-local JSONL readers."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_parallel_jsonl_matches_serial_lengths_and_epoch_reports(self):
        """Feature: parallel offline packing count.

        Description: Read varied JSONL records, including empty and multi-output transforms.
        Expectation: Workers preserve every source index and every DP/epoch count.
        """
        word_counts = [1, 4, 15, 8, 22] * 6 + [3]
        expected = [[min(5, words - start - 1) for start in range(0, words - 1, 5)]
                    for words in word_counts]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            path = Path(directory) / "data.jsonl"
            path.write_text("".join(json.dumps({"text": "word " * n}) + "\n" for n in word_counts),
                            encoding="utf-8")
            source = _JsonlMappingSource([str(path)], sample_filter=None)
            cleanup.callback(source._close_file_handles)
            tokenizer = _HFAutoTokenizer.__new__(_HFAutoTokenizer)
            tokenizer.tokenizer = _WordTokenizer()
            dataset = MappingTransformDataset(source, PlaintextTransform(tokenizer, max_seq_len=5))
            # Open the cached source reader before spawning: workers must reopen it.
            dataset.get_item(0)
            serial = _collect_lengths(dataset)
            parallel = _collect_lengths(dataset, num_workers=2, worker_batch_size=3)
            self.assertEqual(serial.lengths, expected)
            self.assertEqual(parallel.lengths, expected)
            config = _packing_config()
            for dp_size in (1, 2):
                for epoch in range(3):
                    with self.subTest(dp_size=dp_size, epoch=epoch):
                        self.assertEqual(count_epoch(config, parallel, dp_size, epoch),
                                         count_epoch(config, serial, dp_size, epoch))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tokenizer_wrapper_pickle_preserves_delegation(self):
        """Feature: tokenizer wrapper worker serialization.

        Description: Round-trip a wrapper before invoking its delegated tokenizer methods.
        Expectation: Unpickling does not recurse and restored options remain delegated.
        """
        tokenizer = _HFAutoTokenizer.__new__(_HFAutoTokenizer)
        with self.assertRaises(AttributeError):
            getattr(tokenizer, "__setstate__")
        tokenizer.tokenizer = _WordTokenizer()
        tokenizer.padding_side = "left"
        restored = pickle.loads(pickle.dumps(tokenizer))
        self.assertEqual(restored.encode("one two three"), [0, 1, 2])
        self.assertEqual(restored.padding_side, "left")
        restored.padding_side = "right"
        self.assertEqual(restored.tokenizer.padding_side, "right")
        with self.assertRaises(AttributeError):
            getattr(restored, "missing_attribute")

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_worker_settings_fail_before_reading_data(self):
        """Feature: counter worker option validation.

        Description: Supply negative worker counts or a nonpositive worker batch size.
        Expectation: Reject invalid options before constructing the data iterator.
        """
        for workers, batch_size in [(-1, 64), (2, 0), (0, -1)]:
            with self.subTest(workers=workers, batch_size=batch_size):
                with self.assertRaisesRegex(ValueError, "num_workers.*worker_batch_size"):
                    _collect_lengths(None, workers, batch_size)
