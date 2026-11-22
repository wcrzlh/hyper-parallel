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
"""Budget, source ownership, metadata, and checkpoint tests for step packing."""

from __future__ import annotations

import random
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from typing import Any

import torch

from hyper_parallel.data.batching.build_collate_fn import OmniCollator, TextPackingCollator
from hyper_parallel.data.batching.build_dataloader import FixedBatchDataLoader, TokenBatchLoader, OmniPackingLoader
from hyper_parallel.data.batching.packing import SamplePacker
from hyper_parallel.data.batching.step_packing import (
    PackingStep, balanced_step_partitions, validate_packing_checkpoint,
)
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
from hyper_parallel.trainer.runtime.dynamic_epoch import prepare_token_batch_step
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer
from tests.common.mark_utils import arg_mark


def make_samples(lengths: list[int]) -> list[dict[str, torch.Tensor]]:
    """Give each source sample recognizable inputs and pre-shifted targets."""
    return [{"input_ids": torch.full((length,), index + 1),
             "labels": torch.tensor([index + 1] * (length - 1) + [-100])}
            for index, length in enumerate(lengths)]


def make_loader(
    samples: Any, max_seq_len: int | None = 10, loader_class: type = FixedBatchDataLoader, num_workers: int = 0,
) -> Any:
    """Build the real source sampler and stateful loader without distributed setup."""
    sampler = build_dataset_batch_sampler(
        total_samples=len(samples), micro_batch_size=1, global_batch_size=4,
        dp_rank=0, dp_world_size=1, sampler_type="single", drop_last=True, seed=42,
    )
    return loader_class(
        dataset=samples, batch_sampler=sampler, batch_size=1, global_batch_size=4,
        dp_world_size=1, step_packing=True, max_seq_len=max_seq_len,
        collate_fn=TextPackingCollator(alignment_size=2), drop_last=True, num_workers=num_workers,
    )


class OmniSamples(list):
    """Minimal Omni encoding contract with visible image/token ownership."""

    sample_packer = SamplePacker()

    @staticmethod
    def encode_selected_sample(sample: dict[str, Any]) -> dict[str, Any]:
        """Return an already encoded fixture without changing media ownership."""
        return sample

    @staticmethod
    def encode_batch(batch: dict[str, Any]) -> dict[str, Any]:
        """Retain the final collated fixture."""
        return batch


class TestStepPacking(unittest.TestCase):
    """A step changes its computation schedule without changing source membership."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_equal_count_kk_balances_cost_without_changing_membership(self):
        """Feature: Equal-count DP assignment.
        Description: Balance skewed costs and cover feasible equal-size partitions.
        Expectation: Counts remain equal, membership is exact and the fixture's peak cost falls.
        """
        groups = balanced_step_partitions([10, 9, 1, 1], 2, workloads=[100, 90, 10, 10], equal_size=True)
        self.assertEqual(sorted(sum([100, 90, 10, 10][i] for i in group) for group in groups), [100, 110])
        rng = random.Random(42)
        for ranks in range(1, 9):
            for per_rank in range(1, 9):
                lengths = [rng.randint(1, 32768) for _ in range(ranks * per_rank)]
                groups = balanced_step_partitions(lengths, ranks, equal_size=True)
                self.assertEqual([len(group) for group in groups], [per_rank] * ranks)
                self.assertEqual(sorted(index for group in groups for index in group), list(range(len(lengths))))
                self.assertEqual(groups, balanced_step_partitions(lengths, ranks, equal_size=True))
        with self.assertRaisesRegex(ValueError, "divisible"):
            balanced_step_partitions([1, 2, 3], 2, equal_size=True)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_unequal_dp_source_counts_fail_on_non_readers_too(self):
        """Feature: Collective-safe DP balancing validation.
        Description: A TP non-reader observes unequal source counts in other ranks.
        Expectation: Reject the input before any payload or model collective.
        """
        with patch("hyper_parallel.trainer.runtime.dynamic_epoch._global_max", return_value=[0, -1, 1, 2]):
            with self.assertRaisesRegex(ValueError, "equal source sample counts"):
                prepare_token_batch_step(
                    None, device="cpu", drop_last=True, num_micro_batches=1, step_packing=True,
                    mesh=SimpleNamespace(dp_size=2),
                )

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_model_workloads_reach_partitioner_and_preserve_samples(self):
        """Use model costs while retaining bounded packs and source membership.

        Feature: Model-aware step packing.
        Description: Pass model workloads through step preparation into KK.
        Expectation: Packs respect the budget and retain every source token.
        """
        samples = make_samples([7, 3, 5, 3])
        config = SimpleNamespace(model_type="fixture")
        with patch("hyper_parallel.trainer.runtime.dynamic_epoch.estimate_packing_workloads",
                   return_value=[90, 30, 50, 30]) as estimate:
            with patch("hyper_parallel.trainer.runtime.dynamic_epoch.balanced_step_partitions",
                       wraps=balanced_step_partitions) as partition:
                iterator, count = prepare_token_batch_step(
                    iter(make_loader(samples)), device="cpu", drop_last=True, num_micro_batches=1,
                    step_packing=True, model_config=config,
                )
                batches = list(iterator)
        estimate.assert_called_once_with(config, [7, 3, 5, 3])
        self.assertEqual(partition.call_args.kwargs["workloads"], [90, 30, 50, 30])
        self.assertEqual(len(batches), count)
        self.assertTrue(all(batch["input_ids"].shape[1] <= 10 for batch in batches))
        tokens = torch.cat([batch["input_ids"].flatten() for batch in batches])
        self.assertEqual([int((tokens == identity).sum()) for identity in range(1, 5)], [7, 3, 5, 3])

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_partition_cost_validation_and_checkpoint_version(self):
        """Reject invalid costs and old plans instead of silently changing replay.

        Feature: Packing plan validation.
        Description: Supply invalid workload arrays and an obsolete plan version.
        Expectation: Invalid costs and incompatible checkpoints raise ValueError.
        """
        for costs in ([1], [1, 0]):
            with self.assertRaisesRegex(ValueError, "Workloads"):
                balanced_step_partitions([2, 3], 1, workloads=costs)
        for version in (1, 2):
            with self.assertRaisesRegex(ValueError, "plan_version"):
                validate_packing_checkpoint({"step_packing_plan_version": version}, 32768)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_budget_can_require_more_than_total_length_lower_bound(self):
        """
        Feature: Source-preserving step packing.
        Description: Three samples cannot share a bounded pack.
        Expectation: Each indivisible sample gets its own micro-batch.
        """
        samples = make_samples([6, 6, 6])
        step = PackingStep(samples, TextPackingCollator(alignment_size=2), 10)
        iterator, count = prepare_token_batch_step(
            iter([step]), device="cpu", drop_last=True, num_micro_batches=1, step_packing=True,
        )
        batches = list(iterator)
        self.assertEqual(count, 3)
        self.assertEqual(sorted(batch["input_ids"].shape[1] for batch in batches), [6, 6, 6])

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_alignment_labels_and_document_boundaries(self):
        """
        Feature: Source-preserving step packing.
        Description: Pack uneven source lengths under alignment constraints.
        Expectation: Every source label and document boundary is retained within budget.
        """
        samples = make_samples([7, 3, 5, 3])
        iterator, count = prepare_token_batch_step(
            iter(make_loader(samples)), device="cpu", drop_last=True, num_micro_batches=1, step_packing=True,
        )
        seen, target_count = [], 0
        for batch in iterator:
            ids, labels = batch["input_ids"][0], batch["labels"][0]
            self.assertLessEqual(ids.numel(), 10)
            self.assertEqual(ids.numel() % 2, 0)
            offsets = batch["cu_seq_lens"].tolist()
            self.assertEqual(offsets[-1], ids.numel())
            for start, end in zip(offsets, offsets[1:]):
                identity = int(ids[start])
                if identity == 0:
                    self.assertTrue((labels[start:end] == -100).all())
                    continue
                seen.append(identity)
                self.assertTrue(torch.equal(ids[start:end], samples[identity - 1]["input_ids"]))
                self.assertTrue(torch.equal(labels[start:end], samples[identity - 1]["labels"]))
            target_count += int((labels != -100).sum())
        self.assertGreater(count, 1)
        self.assertEqual(sorted(seen), [1, 2, 3, 4])
        self.assertEqual(target_count, sum(int((sample["labels"] != -100).sum()) for sample in samples))

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_partitions_are_complete_deterministic_and_nonempty(self):
        """
        Feature: Source-preserving step packing.
        Description: Partition random lengths at every feasible group count.
        Expectation: Groups are deterministic, nonempty, and cover every sample once.
        """
        rng = random.Random(42)
        for size in range(1, 21):
            lengths = [rng.randint(1, 100) for _ in range(size)]
            for count in range(1, size + 1):
                groups = balanced_step_partitions(lengths, count)
                self.assertEqual(groups, balanced_step_partitions(lengths, count))
                self.assertEqual(len(groups), count)
                self.assertTrue(all(groups))
                self.assertEqual(sorted(index for group in groups for index in group), list(range(size)))

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_remote_rank_increases_common_count(self):
        """
        Feature: Source-preserving step packing.
        Description: A remote rank requires more partitions than the local rank.
        Expectation: The local rank emits the globally agreed micro-batch count.
        """
        step = PackingStep(make_samples([2, 2, 2, 2]), TextPackingCollator(), 6)
        with patch("hyper_parallel.trainer.runtime.dynamic_epoch._global_max", side_effect=[[0, -4, 3, 4], [0]]):
            iterator, count = prepare_token_batch_step(
                iter([step]), device="cpu", drop_last=True, num_micro_batches=1, step_packing=True,
            )
            self.assertEqual(count, 3)
            self.assertEqual(len(list(iterator)), 3)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_non_reader_participates_in_plan(self):
        """
        Feature: Source-preserving step packing.
        Description: A TP non-reader joins the partition metadata reductions.
        Expectation: It receives the common count without emitting source batches.
        """
        with patch("hyper_parallel.trainer.runtime.dynamic_epoch._global_max", side_effect=[[0, -4, 2, 4], [0]]):
            iterator, count = prepare_token_batch_step(
                None, device="cpu", drop_last=True, num_micro_batches=1, step_packing=True,
            )
            self.assertEqual(count, 2)
            self.assertEqual(list(iterator), [])

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_source_error_is_agreed_before_forward(self):
        """
        Feature: Source-preserving step packing.
        Description: Local or remote input validation fails.
        Expectation: All participants reject the step before model collectives.
        """
        step = PackingStep(make_samples([11]), TextPackingCollator(), 10)
        with self.assertRaisesRegex(ValueError, "Invalid step packing"):
            prepare_token_batch_step(iter([step]), device="cpu", drop_last=True, num_micro_batches=1, step_packing=True)
        with patch("hyper_parallel.trainer.runtime.dynamic_epoch._global_max", return_value=[1, -1, 1, 1]):
            with self.assertRaisesRegex(ValueError, "source rank"):
                prepare_token_batch_step(None, device="cpu", drop_last=True, num_micro_batches=1, step_packing=True)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_drop_last_applies_to_source_steps(self):
        """
        Feature: Source-preserving step packing.
        Description: The source ends with an incomplete optimizer step.
        Expectation: Only complete source steps contribute packed batches.
        """
        loader = make_loader(make_samples([3] * 7))
        source = iter(loader)
        batches, _ = prepare_token_batch_step(
            source, device="cpu", drop_last=True, num_micro_batches=1, step_packing=True,
        )
        self.assertEqual(sum(batch["input_ids"].count_nonzero().item() for batch in batches), 12)
        with self.assertRaises(StopIteration):
            prepare_token_batch_step(source, device="cpu", drop_last=True, num_micro_batches=1, step_packing=True)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_resume_same_grouping_with_workers_and_legacy_cursor(self):
        """
        Feature: Source-preserving step packing.
        Description: Resume both current and legacy cursors with loader workers.
        Expectation: Tokens, labels, boundaries, and partition counts match uninterrupted iteration.
        """
        for loader_class in (FixedBatchDataLoader, TokenBatchLoader):
            for workers in (0, 2):
                with self.subTest(loader=loader_class.__name__, workers=workers):
                    samples = make_samples([2, 3, 4, 3, 7, 3, 5, 3, 2, 2, 2, 2])
                    loader = make_loader(samples, loader_class=loader_class, num_workers=workers)
                    source = iter(loader)
                    batches, first_count = prepare_token_batch_step(
                        source, device="cpu", drop_last=True, num_micro_batches=1, step_packing=True,
                    )
                    list(batches)
                    state = loader.state_dict()
                    expected, expected_count = prepare_token_batch_step(
                        source, device="cpu", drop_last=True, num_micro_batches=1, step_packing=True,
                    )
                    expected = list(expected)
                    self.assertGreaterEqual(expected_count, first_count)
                    for legacy in (False, True):
                        saved = dict(state)
                        if legacy:
                            saved.pop("step_packing_max_seq_len")
                            saved.pop("step_packing_plan_version")
                        restored = make_loader(samples, loader_class=loader_class, num_workers=workers)
                        restored.load_state_dict(saved)
                        actual, count = prepare_token_batch_step(
                            iter(restored), device="cpu", drop_last=True, num_micro_batches=1, step_packing=True,
                        )
                        self.assertEqual(count, expected_count)
                        for left, right in zip(expected, list(actual)):
                            for key in ("input_ids", "labels", "cu_seq_lens"):
                                self.assertTrue(torch.equal(left[key], right[key]), key)
                    wrong_budget = make_loader(samples, max_seq_len=12, loader_class=loader_class)
                    with self.assertRaisesRegex(ValueError, "max_seq_len"):
                        wrong_budget.load_state_dict(state)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_omni_preserves_media_and_encodes_each_sample_once(self):
        """
        Feature: Source-preserving step packing.
        Description: Partition multimodal samples after individual encoding.
        Expectation: Encoding occurs once and image ownership and loss targets are retained.
        """
        samples = OmniSamples(make_samples([7, 3, 5, 3]))
        for index, sample in enumerate(samples):
            sample["pixel_values"] = torch.full((1, 2), index + 1)
        sampler = build_dataset_batch_sampler(
            total_samples=4, micro_batch_size=1, global_batch_size=4,
            dp_rank=0, dp_world_size=1, sampler_type="single", drop_last=True, seed=42,
        )
        with patch.object(samples, "encode_selected_sample", wraps=samples.encode_selected_sample) as encode:
            loader = OmniPackingLoader(
                dataset=samples, batch_sampler=sampler, batch_size=1, global_batch_size=4,
                dp_world_size=1, step_packing=True, max_seq_len=10,
                collate_fn=OmniCollator(alignment_size=2), drop_last=True,
            )
            iterator, count = prepare_token_batch_step(
                iter(loader), device="cpu", drop_last=True, num_micro_batches=1, step_packing=True,
            )
            batches = list(iterator)
            self.assertEqual(encode.call_count, 4)
            self.assertGreater(count, 1)
            self.assertEqual(sorted(int(row[0]) for batch in batches for row in batch["pixel_values"]), [1, 2, 3, 4])
            self.assertEqual(sum(int((batch["labels"] != -100).sum()) for batch in batches), 14)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_trainers_read_one_step_even_when_previous_micro_count_changes(self):
        """
        Feature: Source-preserving step packing.
        Description: Consecutive source steps require different numbers of packs.
        Expectation: Both trainers consume one source step per optimizer update.
        """
        for trainer_class in (TextTrainer, VLMTrainer):
            with self.subTest(trainer=trainer_class.__name__):
                trainer = trainer_class.__new__(trainer_class)
                trainer.base = SimpleNamespace(
                    train_dataloader=make_loader(make_samples([9, 9, 9, 9, 2, 2, 2, 2])),
                    num_micro_batches=1, device="cpu", train_steps=2, train_iters=2,
                    model_config=SimpleNamespace(model_type="fixture"), mesh=None,
                    state=SimpleNamespace(global_step=0),
                    config=SimpleNamespace(dataloader=SimpleNamespace(step_packing=True, drop_last=True)),
                )
                counts = []

                def consume_step(iterator: Any) -> None:
                    """Consume one complete source step without executing a model."""
                    batches = list(iterator)
                    self.assertEqual(len(batches), trainer.base.num_micro_batches)
                    counts.append(len(batches))
                    trainer.base.state.global_step += 1

                trainer.train_step = consume_step
                self.assertFalse(trainer._train_epoch(epoch=0, balanced_loader=False))
                self.assertEqual(counts, [4, 1])
                self.assertEqual(trainer.base.state.global_step, 2)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_budget_validation_and_nonpacking_compatibility(self):
        """
        Feature: Source-preserving step packing.
        Description: Validate packing budgets and exercise the ordinary loader.
        Expectation: Invalid limits fail and nonpacking iteration remains unchanged.
        """
        with self.assertRaisesRegex(ValueError, "positive"):
            make_loader(make_samples([3] * 4), max_seq_len=None)
        with self.assertRaisesRegex(ValueError, "divisible"):
            make_loader(make_samples([3] * 4), max_seq_len=9)
        loader = FixedBatchDataLoader(dataset=make_samples([3] * 4), batch_size=1,
                                    collate_fn=TextPackingCollator(), step_packing=False)
        self.assertEqual(len(list(loader)), 4)
