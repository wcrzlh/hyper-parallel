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
"""Distributed sample conservation, token-normalized updates and resume coverage."""

import copy
import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.data.batching.build_collate_fn import TextPackingCollator
from hyper_parallel.data.batching.build_dataloader import FixedBatchDataLoader
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
from hyper_parallel.distributed_data.transport import all_gather_control_object, broadcast_control_object
from hyper_parallel.trainer.runtime.dynamic_epoch import prepare_token_batch_step


class _Samples:
    """Ten global steps with deliberately uneven source lengths and masked labels."""

    def __len__(self) -> int:
        """Return ten complete global batches of four samples."""
        return 40

    def __getitem__(self, index: int) -> dict:
        """Encode sample identity while masking a subset of target tokens."""
        length = [31, 2, 23, 3, 8, 4, 17, 5][index % 8]
        labels = torch.full((length,), index + 1, dtype=torch.long)
        labels[::3] = -100
        return {"input_ids": torch.full((length,), index + 1, dtype=torch.long), "labels": labels}


def _loader(dp_rank: int) -> FixedBatchDataLoader:
    sampler = build_dataset_batch_sampler(
        total_samples=40, micro_batch_size=1, global_batch_size=4,
        dp_rank=dp_rank, dp_world_size=2, sampler_type="single", drop_last=True, seed=42,
    )
    return FixedBatchDataLoader(
        dataset=_Samples(), batch_sampler=sampler, batch_size=1, global_batch_size=4,
        dp_world_size=2, step_packing=True, max_seq_len=32,
        collate_fn=TextPackingCollator(alignment_size=2), drop_last=True, num_workers=0,
    )


def _step(iterator, mesh, device, balance: bool):
    """Collect exact inputs and an additive loss/gradient before and after routing."""
    packed, count = prepare_token_batch_step(
        iterator, num_micro_batches=1, device=device, drop_last=True, step_packing=True,
        mesh=mesh if balance else None,
    )
    batches = list(packed)
    tp_mesh = mesh.device_mesh["tp"]
    if mesh.tp_size > 1:
        batches = broadcast_control_object(
            batches, src=dist.get_process_group_ranks(tp_mesh.get_group())[0],
            group=tp_mesh.get_group(), device=device,
        )
    signature = [(b["input_ids"].tolist(), b["labels"].tolist(), b["cu_seq_lens"].tolist()) for b in batches]
    peers = all_gather_control_object(signature, group=mesh.device_mesh["cp"].get_group(), device=device)
    assert all(peer == signature for peer in peers), "CP peers received different packed samples"
    tokens = []
    weight = torch.tensor(0.5, device=device, requires_grad=True)
    totals = torch.zeros(2, device=device)
    for batch in batches:
        ids, labels = batch["input_ids"][0], batch["labels"][0]
        tokens.extend((int(i), int(label)) for i, label in zip(ids, labels) if int(i) != 0)
        ids = ids[mesh.cp_rank::mesh.cp_size].to(device).float()
        labels = labels[mesh.cp_rank::mesh.cp_size].to(device)
        valid = labels != -100
        loss = ((weight * ids - 1).square() * valid).sum()
        loss.backward()
        totals += torch.stack((loss.detach(), valid.sum().float()))
    if mesh.tp_rank == 0:
        # CP replicas cover disjoint token positions; only TP readers contribute.
        contribution = torch.stack((totals[0], totals[1], weight.grad))
    else:
        contribution = torch.zeros(3, device=device)
    dist.all_reduce(contribution)
    expected_count = len(set(i for i, _ in tokens))
    assert expected_count == 2, "DP reassignment changed the rank's source sample count"
    gathered = all_gather_control_object(tokens if mesh.cp_rank == mesh.tp_rank == 0 else [], device=device)
    global_tokens = sorted(token for rank_tokens in gathered for token in rank_tokens)
    return signature, count, global_tokens, contribution.cpu()


@pytest.mark.parametrize("backend,cp_size", [("gloo", 1), ("gloo", 2), ("hccl", 1), ("hccl", 2)])
def test_step_balance(backend: str, cp_size: int) -> None:
    """Compare ten balanced steps to source ownership, then replay the last five."""
    device = torch.device("cpu")
    if backend == "hccl":
        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
        torch.npu.set_device(device)
    dist.init_process_group(backend, timeout=timedelta(seconds=120))
    try:
        tp_size = 2 // cp_size
        root = init_device_mesh(device.type, (2, cp_size, tp_size), mesh_dim_names=("dp", "cp", "tp"))
        rank = dist.get_rank()
        mesh = SimpleNamespace(
            device_mesh=root, dp_size=2, cp_size=cp_size, tp_size=tp_size,
            cp_rank=(rank // tp_size) % cp_size, tp_rank=rank % tp_size,
        )
        balanced_loader = _loader(rank // 2) if mesh.tp_rank == 0 else None
        baseline_loader = _loader(rank // 2) if mesh.tp_rank == 0 else None
        balanced_iter = iter(balanced_loader) if balanced_loader is not None else None
        baseline_iter = iter(baseline_loader) if baseline_loader is not None else None
        traces, snapshot = [], None
        changed = False
        for index in range(10):
            baseline = _step(baseline_iter, mesh, device, False)
            balanced = _step(balanced_iter, mesh, device, True)
            assert baseline[2] == balanced[2], "DP routing changed inputs or loss masks"
            torch.testing.assert_close(baseline[3], balanced[3], rtol=0, atol=0)
            traces.append(balanced)
            changed |= baseline[0] != balanced[0]
            if index == 4 and balanced_loader is not None:
                snapshot = copy.deepcopy(balanced_loader.state_dict())
        assert changed, "Skewed fixture did not exercise sample reassignment"
        if balanced_loader is not None:
            balanced_loader = _loader(rank // 2)
            balanced_loader.load_state_dict(snapshot)
        resumed_iter = iter(balanced_loader) if balanced_loader is not None else None
        for expected in traces[5:]:
            actual = _step(resumed_iter, mesh, device, True)
            assert actual[:3] == expected[:3], "Restored plan differs from uninterrupted execution"
            torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
