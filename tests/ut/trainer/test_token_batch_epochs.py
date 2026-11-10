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
"""Online packing epoch iteration and state-compatible mid-epoch resume."""

import copy
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.data.batching import TokenBatchLoader
from hyper_parallel.data.batching.build_collate_fn import build_online_text_collate_fn
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
from hyper_parallel.trainer.config.data import DataLoaderConfig
from hyper_parallel.trainer.config.target import Target
from hyper_parallel.trainer.runtime.dynamic_epoch import prepare_token_batch_step
from hyper_parallel.trainer.text_trainer import TextTrainer
from examples.qwen3_6_moe.count_packed_train_iters import LengthDataset, count_epoch


text_module = importlib.import_module("hyper_parallel.trainer.text_trainer")


def _config():
    loader = DataLoaderConfig(
        target=Target(TokenBatchLoader, target_path="hyper_parallel.data.batching.TokenBatchLoader",
                      token_budget=12, min_buffered_samples=3, drop_last=True, num_workers=0),
        collate_fn=Target(build_online_text_collate_fn, target_path="unused", pad_to_length=12),
        sampler_type="cyclic",
    )
    return SimpleNamespace(
        dataloader=loader, training=SimpleNamespace(global_batch_size=2, micro_batch_size=1, seed=17),
        dataset=SimpleNamespace(data_transform=SimpleNamespace(max_seq_len=12)),
    )


class _Samples:
    def __init__(self, lengths: list[list[int]]) -> None:
        """Store the fixture's per-record output lengths."""
        self.lengths = lengths

    def __len__(self) -> int:
        """Return the source record count."""
        return len(self.lengths)

    def get_item(self, index: int) -> list[dict[str, torch.Tensor]]:
        """Return deterministic records for checking the restored sample order."""
        return [{"input_ids": torch.full((length,), index + 1)} for length in self.lengths[index]]


def _collate(samples):
    return tuple((int(sample["input_ids"][0]), sample["input_ids"].numel()) for sample in samples)


def _trainer(config, lengths, limit, snapshot=None):
    sampler = build_dataset_batch_sampler(
        total_samples=len(lengths), micro_batch_size=1, global_batch_size=2,
        dp_rank=0, dp_world_size=1, drop_last=True, sampler_type="cyclic", seed=17,
    )
    loader = config.dataloader.build(
        dataset=_Samples(lengths), batch_sampler=sampler, collate_fn=_collate,
        batch_size=1, dp_world_size=1, max_seq_len=12, sampler_type="cyclic",
    )
    trainer = TextTrainer.__new__(TextTrainer)
    state = SimpleNamespace(global_step=0, epoch=0)
    weight = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
    optimizer = torch.optim.SGD([weight], lr=0.001, momentum=0.9)
    if snapshot:
        state = copy.deepcopy(snapshot["state"])
        loader.load_state_dict(snapshot["loader"])
        weight.data.copy_(snapshot["weight"])
        optimizer.load_state_dict(snapshot["optimizer"])
    trainer.base = SimpleNamespace(
        config=config, train_dataloader=loader, state=state, train_iters=limit, train_steps=limit,
        train_epochs=1, num_micro_batches=2, device="cpu", local_rank=0,
        optimizer=optimizer, abort_profilers=Mock(), destroy_distributed=Mock(),
    )
    trainer.trace = []
    trainer.losses = []
    trainer.weight = weight
    for name in ("on_train_begin", "on_train_end", "on_epoch_begin", "on_epoch_end"):
        setattr(trainer, name, Mock())

    def _step(iterator):
        batches = [next(iterator), next(iterator)]
        trainer.trace.append((state.epoch, batches))
        target = sum(index * length for batch in batches for index, length in batch) / 100
        loss = (weight - target).square()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        trainer.losses.append(float(loss.detach()))
        state.global_step += 1

    trainer.train_step = _step
    return trainer


def _run(trainer):
    with patch.object(text_module, "memory_profiler"), patch.object(text_module, "synchronize"):
        with patch.object(text_module, "print_device_mem_info"):
            trainer.train()


def _snapshot(trainer):
    return copy.deepcopy({
        "state": trainer.base.state, "loader": trainer.base.train_dataloader.state_dict(),
        "weight": trainer.weight.detach(), "optimizer": trainer.base.optimizer.state_dict(),
    })


class TestTokenBatchEpochs(unittest.TestCase):
    """Use real stateful loader checkpoints with a deterministic CPU optimizer."""

    def test_multiple_epochs_and_resume_match_uninterrupted(self):
        config = _config()
        lengths = [[value] for value in [3, 7, 4, 8, 2, 9, 3, 6, 5, 8, 2, 7, 3, 9, 4, 8, 1, 6, 5, 7]]
        counts = [count_epoch(config, LengthDataset(lengths), 1, epoch)["optimizer_steps"] for epoch in range(3)]
        limit = sum(counts[:2]) + 1
        uninterrupted = _trainer(config, lengths, limit)
        _run(uninterrupted)
        self.assertEqual(uninterrupted.base.state.global_step, limit)
        self.assertEqual([epoch for epoch, _ in uninterrupted.trace],
                         [0] * counts[0] + [1] * counts[1] + [2])
        for stop in (1, counts[0], counts[0] + 1):
            with self.subTest(stop=stop):
                prefix = _trainer(config, lengths, stop)
                _run(prefix)
                saved = _snapshot(prefix)
                self.assertEqual(set(saved["loader"]),
                                 {"dp_world_size", "source_dataloader", "save_by_idx", "buffer", "buffer_token_count"})
                suffix = _trainer(config, lengths, limit, saved)
                _run(suffix)
                self.assertEqual(prefix.trace + suffix.trace, uninterrupted.trace)
                self.assertEqual(prefix.losses + suffix.losses, uninterrupted.losses)
                torch.testing.assert_close(suffix.weight, uninterrupted.weight, rtol=0, atol=0)

    def test_counter_reports_uneven_dp_tail(self):
        config = _config()
        config.training.global_batch_size = 4
        config.dataloader.sampler_type = "single"
        # DP0 gets long samples, DP1 short ones: packing produces different counts.
        lengths = [[length] for _ in range(12) for length in (9, 2)]
        result = count_epoch(config, LengthDataset(lengths), 2, 0)
        self.assertEqual(result["packed_batches_per_dp"], [12, 2])
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["omitted_packed_batches_per_dp"], [10, 0])

    def test_incomplete_epoch_fails_instead_of_looping(self):
        trainer = _trainer(_config(), [[3]], 3)
        with self.assertRaisesRegex(ValueError, "complete distributed optimizer step"):
            _run(trainer)

    def test_partial_step_is_rejected_before_forward(self):
        with self.assertRaises(StopIteration):
            prepare_token_batch_step(iter([1]), num_micro_batches=2, device="cpu", drop_last=True)
        with self.assertRaisesRegex(ValueError, "incomplete distributed"):
            prepare_token_batch_step(iter([1]), num_micro_batches=2, device="cpu", drop_last=False)
        self.assertEqual(list(prepare_token_batch_step(iter([1, 2]), num_micro_batches=2,
                                                       device="cpu", drop_last=True)), [1, 2])
