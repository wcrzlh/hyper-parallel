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
"""Measure untruncated Omni samples using the training processor on CPU.

Run from the repository root with ``python -m
examples.qwen3_6_moe.tools.stat_multimodal_data --help``. Only processor assets
are loaded; model weights, dummy vision, packing and CP are not involved.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterator
import glob
from itertools import islice
import json
import logging
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch  # pylint: disable=forbidden-backend-import
import yaml

from hyper_parallel.data.constants import IGNORE_INDEX, ONLINE_SOURCE_PATH_KEY
from hyper_parallel.data.omni.omni_transform import AutoProcessorTransform, build_auto_processor

LOGGER = logging.getLogger(__name__)
_WORKER: dict[str, Any] = {}
_BIN_EDGES = (0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True, type=Path, help="Training YAML with an AutoProcessorTransform")
    parser.add_argument("--data-path", nargs="+", help="Override dataset.data_path: JSON/JSONL/Parquet files or globs")
    parser.add_argument("--model-path", help="Override processor asset directory; no model weights are loaded")
    parser.add_argument("--max-seq-len", type=int, help="Override the reporting threshold; never truncates samples")
    parser.add_argument("--workers", type=int, default=4,
                        help="CPU processes, each with its own processor (default: 4)")
    parser.add_argument("--limit", type=int, help="Scan the first N records; omit to scan all records")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New directory for summary.json and samples.jsonl")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")
    return args


def _configuration(args: argparse.Namespace) -> tuple[dict, dict, int, list[Path]]:
    config = yaml.safe_load(args.yaml.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    assets = dict(dataset["model_assets"])
    target = assets.pop("_target_", "")
    if target != "hyper_parallel.data.omni.omni_transform.build_auto_processor":
        raise ValueError("This tool requires dataset.model_assets._target_=build_auto_processor")
    if args.model_path:
        assets["pretrained_model_name_or_path"] = args.model_path
    transform = dict(dataset["data_transform"])
    target = transform.pop("_target_", "")
    if target != "hyper_parallel.data.omni.omni_transform.AutoProcessorTransform":
        raise ValueError("This tool requires AutoProcessorTransform to match training tokenization")
    threshold = args.max_seq_len if args.max_seq_len is not None else transform.pop("max_seq_len")
    if threshold <= 0:
        raise ValueError("max_seq_len must be positive")
    # Count overlength samples instead of failing at the training length guard.
    transform["max_seq_len"] = sys.maxsize
    transform["training_mode"] = dataset.get("training_mode", "sft")
    paths = args.data_path or dataset.get("data_path")
    if dataset.get("data_config"):
        LOGGER.warning("Scanning input files once; dataset.data_config filtering, splits and sampling are not applied")
    return assets, transform, threshold, _resolve_files(paths)


def _resolve_files(paths: str | list[str] | None) -> list[Path]:
    if not paths or isinstance(paths, dict):
        raise ValueError("Pass --data-path with local files; split dictionaries and weighted sources are unsupported")
    files = []
    for pattern in [paths] if isinstance(paths, str) else paths:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"No files match {pattern!r}")
        for match in matches:
            path = Path(match).resolve()
            if not path.is_file() or path.suffix.lower() not in (".json", ".jsonl", ".parquet"):
                raise ValueError(f"Expected a JSON/JSONL/Parquet file: {path}")
            if path not in files:
                files.append(path)
    return files


def _file_rows(path: Path) -> Iterator[tuple[int, Any, str | None]]:
    if path.suffix.lower() == ".parquet":
        row_number = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                row_number += 1
                yield row_number, row, None
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"JSON input must be a list of records: {path}")
        for number, row in enumerate(rows, 1):
            yield number, row, None
    else:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    yield number, None, str(error)
                else:
                    yield number, row, None


def _records(files: list[Path]) -> Iterator[tuple[dict, Any, str | None]]:
    for path in files:
        for number, row, error in _file_rows(path):
            identity = {"file": str(path), "row": number}
            if isinstance(row, dict):
                identity.update(id=str(row.get("id", number)), source=str(row.get("source", path.stem)))
                row[ONLINE_SOURCE_PATH_KEY] = str(path)
            yield identity, row, error


def _initialize(assets: dict, transform: dict, threshold: int) -> None:
    torch.set_num_threads(1)
    processor = build_auto_processor(**assets)
    _WORKER["transform"] = AutoProcessorTransform(processor=processor, **transform)
    _WORKER["max_seq_len"] = threshold


def _analyze(record: tuple[dict, Any, str | None]) -> dict:
    identity, sample, error = record
    result = dict(identity)
    try:
        if error:
            raise ValueError(error)
        if not isinstance(sample, dict):
            raise ValueError("Sample must be a JSON object")
        with torch.inference_mode():
            encoded = _WORKER["transform"].encode_sample(sample)
        length = encoded["input_ids"].numel()
        images = len(encoded.get("image_grid_thw", []))
        videos = len(encoded.get("video_grid_thw", []))
        if "pixel_values" in encoded and not images:
            raise ValueError("Cannot count images: processor returned pixels without image_grid_thw")
        if "pixel_values_videos" in encoded and not videos:
            raise ValueError("Cannot count videos: processor returned pixels without video_grid_thw")
        modality = "image_video" if images and videos else "image" if images else "video" if videos else "text"
        token_types = encoded.get("mm_token_type_ids")
        result.update(
            status="ok", modality=modality, tokens=length,
            supervised_tokens=int(encoded["labels"][1:].ne(IGNORE_INDEX).sum()),
            media_tokens=None if token_types is None else int(token_types.ne(0).sum()),
            images=images, videos=videos, over_max_seq_len=length > _WORKER["max_seq_len"],
        )
    except Exception as exc:  # A bad sample must remain visible without aborting the distribution scan.
        result.update(status="error", error_type=type(exc).__name__, error=str(exc))
    return result


def _distribution(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.int64)
    percentiles = np.percentile(array, [50, 90, 95, 99])
    return {"count": len(values), "total": int(array.sum()), "min": int(array.min()), "max": int(array.max()),
            "mean": float(array.mean()), **dict(zip(("p50", "p90", "p95", "p99"), percentiles.tolist()))}


def _summarize(rows: list[dict], threshold: int) -> dict:
    lengths = [row["tokens"] for row in rows]
    buckets = {}
    for low, high in zip(_BIN_EDGES, _BIN_EDGES[1:]):
        buckets[f"({low}, {high}]"] = sum(low < value <= high for value in lengths)
    buckets[f">{_BIN_EDGES[-1]}"] = sum(value > _BIN_EDGES[-1] for value in lengths)
    over = sum(value > threshold for value in lengths)
    return {
        "tokens": _distribution(lengths),
        "supervised_tokens": _distribution([row["supervised_tokens"] for row in rows]),
        "media_tokens": _distribution([row["media_tokens"] for row in rows if row["media_tokens"] is not None]),
        "images": _distribution([row["images"] for row in rows]),
        "videos": _distribution([row["videos"] for row in rows]),
        "length_buckets": buckets, "over_max_seq_len": over,
        "over_max_seq_len_fraction": over / len(rows) if rows else None,
    }


def _write_results(results: Iterator[dict], args: argparse.Namespace, metadata: dict) -> int:
    groups = {"by_modality": defaultdict(list), "by_source": defaultdict(list)}
    successful = []
    errors = Counter()
    scanned = 0
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            scanned += 1
            if row["status"] == "error":
                errors[row["error_type"]] += 1
            else:
                successful.append(row)
                groups["by_modality"][row["modality"]].append(row)
                groups["by_source"][row["source"]].append(row)
            if scanned % 50 == 0:
                LOGGER.info("Scanned %d records, %d failed", scanned, sum(errors.values()))
    threshold = metadata["max_seq_len"]
    summary = dict(metadata, scanned=scanned, succeeded=len(successful), failed=sum(errors.values()),
                   errors_by_type=dict(errors), overall=_summarize(successful, threshold))
    for name, grouped in groups.items():
        summary[name] = {key: _summarize(rows, threshold) for key, rows in sorted(grouped.items())}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    LOGGER.info("Finished: scanned=%d succeeded=%d failed=%d over_limit=%d; results: %s",
                scanned, len(successful), sum(errors.values()), summary["overall"]["over_max_seq_len"], args.output_dir)
    return 2 if errors or not scanned else 0


def main() -> int:
    """Scan local records and return a nonzero status if any records failed encoding."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _arguments()
    assets, transform, threshold, files = _configuration(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "yaml": str(args.yaml.resolve()), "files": [str(path) for path in files], "processor": assets,
        "transform": transform, "max_seq_len": threshold, "limit": args.limit, "workers": args.workers,
        "notes": [
            "Each input file is scanned once, without training splits, filtering, sampling or shuffling.",
            "Lengths are full processor input_ids before collator padding, packing and CP; no dummy image.",
            "Only the analysis transform length guard is relaxed; training configuration is unchanged.",
            "Supervised tokens count labels[1:] != -100, matching next-token loss shifting.",
            "Media tokens count mm_token_type_ids != 0 when available; unavailable is null per sample.",
            "Distributions and overlength fractions cover successfully encoded samples only.",
        ],
    }
    records = islice(_records(files), args.limit)
    if args.workers == 1:
        _initialize(assets, transform, threshold)
        return _write_results(map(_analyze, records), args, metadata)
    # Initialize once before spawning so invalid processor settings fail without worker restart loops.
    _initialize(assets, transform, threshold)
    with mp.get_context("spawn").Pool(
        args.workers, initializer=_initialize, initargs=(assets, transform, threshold),
    ) as pool:
        return _write_results(pool.imap(_analyze, records, chunksize=1), args, metadata)


if __name__ == "__main__":
    sys.exit(main())
