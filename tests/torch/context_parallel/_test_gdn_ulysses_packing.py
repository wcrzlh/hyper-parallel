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
"""Native HF GDN Ulysses packing versus independent documents, on Gloo or HCCL."""

import argparse
import copy
import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet

from hyper_parallel import init_device_mesh
from hyper_parallel.components.functional.linear_attention_packed import PackedSequenceInfo
from hyper_parallel.components.modules.gdn_triton import TritonGDN
from hyper_parallel.distributed.context_parallel.gated_delta_net import install_gdn_ulysses_cp


def run_case(mesh: Any, device: torch.device, offsets: tuple[int, ...] | None, recompute: bool) -> None:
    """Check outputs, all gradients and zero cross-document gradient leakage.

    Args:
        mesh: Active one-dimensional CP mesh.
        device: Worker computation device.
        offsets: Global document offsets, or None for the dense baseline.
        recompute: Whether to checkpoint the wrapped forward.
    """
    torch.manual_seed(123)
    accelerated = device.type == "npu"
    dtype = torch.bfloat16 if accelerated else torch.float32
    dim = 128 if accelerated else 8
    config = Qwen3_5MoeTextConfig.from_dict({
        "hidden_size": 32, "linear_num_value_heads": 8, "linear_num_key_heads": 4,
        "linear_key_head_dim": dim, "linear_value_head_dim": dim,
    })
    model = Qwen3_5MoeGatedDeltaNet(config, 0).to(device=device, dtype=dtype)
    if accelerated:
        model = TritonGDN(module=model)
    reference = copy.deepcopy(model)
    install_gdn_ulysses_cp(model, mesh)
    length = 512
    start = mesh.get_local_rank() * length // mesh.size()
    end = start + length // mesh.size()
    full_input = torch.randn(1, length, 32, device=device, dtype=dtype, requires_grad=True)
    local_input = full_input.detach()[:, start:end].clone().requires_grad_()
    boundaries = offsets or (0, length)
    expected = torch.cat([reference(full_input[:, left:right])
                          for left, right in zip(boundaries, boundaries[1:])], dim=1)
    kwargs = {"packed_sequence_info": PackedSequenceInfo(offsets, start, end)} if offsets else {}
    if recompute:
        actual = checkpoint(model, local_input, use_reentrant=False, **kwargs)
    else:
        actual = model(local_input, **kwargs)
    # Exercise a nonzero backward halo across the CP cut when a document spans
    # it; the other documents remain unsupervised to detect gradient leakage.
    cut = length // mesh.size()
    index = next((i for i, (left, right) in enumerate(zip(boundaries, boundaries[1:]))
                  if left < cut < right), 1 if len(boundaries) > 2 else 0)
    left, right = boundaries[index:index + 2]
    objective = torch.zeros_like(expected)
    objective[:, left:right] = torch.randn_like(objective[:, left:right])
    (expected.float() * objective.float()).sum().backward()
    (actual.float() * objective[:, start:end].float()).sum().backward()
    tolerance = {"atol": 3e-2, "rtol": 3e-2} if accelerated else {"atol": 3e-5, "rtol": 4e-4}
    torch.testing.assert_close(actual, expected[:, start:end], **tolerance)
    torch.testing.assert_close(local_input.grad, full_input.grad[:, start:end], **tolerance)
    for name, parameter in model.named_parameters():
        target = dict(reference.named_parameters())[name]
        assert parameter.grad is not None and target.grad is not None, f"Missing gradient for {name}"
        reduced_grad = parameter.grad.float()
        dist.all_reduce(reduced_grad, group=mesh.get_group())
        torch.testing.assert_close(reduced_grad, target.grad.float(), **tolerance,
                                   msg=lambda message, name=name: f"{name}: {message}")
    outside = torch.ones(length, dtype=torch.bool, device=device)
    outside[left:right] = False
    leakage = local_input.grad[:, outside[start:end]]
    torch.testing.assert_close(leakage, torch.zeros_like(leakage), atol=0, rtol=0)
    if dist.get_rank() == 0:
        print(f"PASS CP={mesh.size()} device={device.type} offsets={offsets} recompute={recompute}", flush=True)


def run_distributed(backend: str) -> None:
    """Run dense regression and adversarial packed boundaries with recompute.

    Args:
        backend: Gloo for CPU validation or HCCL for NPU validation.
    """
    torch.set_num_threads(1)
    if backend == "hccl":
        import torch_npu  # pylint: disable=import-outside-toplevel,unused-import
        # Optional hardware dependency; Gloo checks must work without torch_npu.
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend, timeout=timedelta(seconds=300))
    try:
        mesh = init_device_mesh(device.type, (dist.get_world_size(),), mesh_dim_names=("cp",))
        for offsets in (None, (0, 512), (0, 64, 128, 256, 512), (0, 1, 3, 63, 65, 127, 129, 257, 512)):
            for recompute in (False, True):
                run_case(mesh, device, offsets, recompute)
    finally:
        dist.destroy_process_group()


def test_packed_ulysses_triton():
    """Eight-rank NPU packing parity using the native HF forward."""
    run_distributed("hccl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "hccl"), default="gloo")
    run_distributed(parser.parse_args().backend)
