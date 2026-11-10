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
"""Count finite text-packing epochs on CPU without loading model weights.

Run: python -m examples.qwen3_6_moe.count_packed_train_iters recipe.yaml --world-size 8 --epochs 3
Dotted YAML overrides are accepted as in scripts.train_lm. The output reports
complete synchronized optimizer steps and packs omitted by drop_last. No
training checkpoint is read or modified. Tokenization must be deterministic;
changing data, topology, ordering or packing settings invalidates the count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hyper_parallel.data.batching import TokenBatchLoader, calculate_num_micro_batches
from hyper_parallel.data.batching.build_collate_fn import build_online_text_collate_fn
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
from hyper_parallel.data.text import build_chat_template
from hyper_parallel.data.text.build_dataset import build_online_text_mapping_dataset
from hyper_parallel.trainer.config.parser import load_training_config


class LengthDataset:
    """Replay token lengths while keeping raw-record and transform-output indices."""

    def __init__(self, lengths: list[list[int]]) -> None:
        """Store transformed lengths grouped by source record."""
        self.lengths = lengths

    def __len__(self) -> int:
        """Return the source-record count used by the sampler."""
        return len(self.lengths)

    def get_item(self, index: int) -> list[dict[str, Any]]:
        """Return shape-only samples understood by the real token packing selector."""
        return [{"input_ids": SimpleNamespace(shape=(length,))} for length in self.lengths[index]]


def _build_lengths(config: Any) -> LengthDataset:
    assets = config.dataset.model_assets
    if assets.tokenizer is None:
        raise ValueError("The offline text counter requires a configured tokenizer.")
    tokenizer = assets.tokenizer.build() if getattr(assets.tokenizer, "pretrained_model_name_or_path", None) else (
        assets.tokenizer.build(pretrained_model_name_or_path=config.model.pretrained_model_name_or_path)
    )
    template = None
    if isinstance(assets.chat_template, str):
        template = build_chat_template(
            assets.chat_template, tokenizer, chat_template_kwargs=assets.chat_template_kwargs,
        )
    elif assets.chat_template is not None:
        template = assets.chat_template.build(tokenizer=tokenizer)
    transform = config.dataset.data_transform.build(tokenizer=tokenizer, chat_template=template)
    dataset = config.dataset.build(
        transform=transform, training_config=config.training, sampler_type=config.dataloader.sampler_type,
    )
    if isinstance(dataset, tuple):
        dataset = dataset[0]
    lengths = []
    for index in range(len(dataset)):
        samples = dataset.get_item(index)
        lengths.append([int(sample["input_ids"].shape[-1]) for sample in samples])
        if (index + 1) % 1000 == 0:
            print(f"Tokenized {index + 1}/{len(dataset)} records", flush=True)
    return LengthDataset(lengths)


def _selected_lengths(samples: list[dict[str, Any]]) -> tuple[int, int]:
    return len(samples), sum(sample["input_ids"].shape[-1] for sample in samples)


def count_epoch(config: Any, dataset: LengthDataset, dp_size: int, epoch: int) -> dict[str, Any]:
    """Use the production sampler and TokenBatchLoader to count every DP shard."""
    training, loader = config.training, config.dataloader
    accumulation = calculate_num_micro_batches(training.global_batch_size, training.micro_batch_size, dp_size)
    seed = training.seed if training.seed is not None else 42
    rank_packs = []
    for rank in range(dp_size):
        sampler = build_dataset_batch_sampler(
            total_samples=len(dataset), micro_batch_size=training.micro_batch_size,
            global_batch_size=training.global_batch_size, dp_rank=rank, dp_world_size=dp_size,
            drop_last=loader.drop_last, data_rearrange_map=loader.data_rearrange_map,
            sampler_type=loader.sampler_type, data_sharding=loader.data_sharding, seed=seed,
        )
        dynamic = loader.build(
            dataset=dataset, batch_sampler=sampler, collate_fn=_selected_lengths,
            batch_size=training.micro_batch_size, dp_world_size=dp_size,
            max_seq_len=config.dataset.data_transform.max_seq_len,
            sampler_type=loader.sampler_type, seed=seed, num_workers=0,
            pin_memory=False, prefetch_factor=None,
        )
        dynamic.set_epoch(epoch)
        packs = list(dynamic)
        pad_length = getattr(loader.collate_fn, "pad_to_length", None)
        if pad_length is not None and any(tokens > pad_length for _, tokens in packs):
            raise ValueError("A selected pack exceeds collate_fn.pad_to_length; training would reject it.")
        rank_packs.append(packs)
    counts = [len(packs) for packs in rank_packs]
    steps = min(counts) // accumulation
    used = steps * accumulation
    if not loader.drop_last and any(size != used for size in counts):
        raise ValueError("Incomplete distributed optimizer step with drop_last=False, as in the training runtime.")
    if steps == 0:
        raise ValueError(f"Epoch {epoch} cannot provide a complete distributed optimizer step: packs={counts}")
    return {
        "epoch": epoch, "optimizer_steps": steps, "micro_batches_per_step_per_dp": accumulation,
        "packed_batches_per_dp": counts, "omitted_packed_batches_per_dp": [size - used for size in counts],
        "trained_model_samples": sum(samples for packs in rank_packs for samples, _ in packs[:used]),
        "trained_tokens_without_padding": sum(tokens for packs in rank_packs for _, tokens in packs[:used]),
        "omitted_model_samples": sum(samples for packs in rank_packs for samples, _ in packs[used:]),
    }


def main() -> None:
    """Read the training YAML and report train_iters for the requested source epochs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args, overrides = parser.parse_known_args()
    if args.world_size <= 0 or args.epochs <= 0 or args.start_epoch < 0:
        parser.error("world-size and epochs must be positive, start-epoch must be nonnegative")
    config = load_training_config(args.config, overrides)
    if (config.dataset.target.callable is not build_online_text_mapping_dataset
            or config.dataloader.target.callable is not TokenBatchLoader
            or config.dataloader.collate_fn.callable is not build_online_text_collate_fn
            or config.dataloader.load_balance is not None):
        raise ValueError("This counter supports finite online text mapping + TokenBatchLoader + text packing collator.")
    model_parallel = config.accelerator.tp_size * config.accelerator.cp_size * config.accelerator.pp_size
    if args.world_size % model_parallel:
        raise ValueError("world-size must be divisible by TP * CP * PP; EP does not divide the data-parallel degree.")
    dp_size = args.world_size // model_parallel
    lengths = _build_lengths(config)
    epochs = [count_epoch(config, lengths, dp_size, epoch)
              for epoch in range(args.start_epoch, args.start_epoch + args.epochs)]
    report = {
        "config": str(Path(args.config).resolve()), "world_size": args.world_size, "dp_size": dp_size,
        "source_records": len(lengths), "source_tokens": sum(sum(row) for row in lengths.lengths),
        "epochs": epochs, "train_iters_for_requested_epochs": sum(row["optimizer_steps"] for row in epochs),
        "note": ("Counts use deterministic text preprocessing and unchanged YAML/data; "
                 "sampler and step tails may drop samples."),
    }
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
