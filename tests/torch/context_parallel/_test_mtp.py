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
"""Two-rank CPU MTP CP losses/gradients against a complete-sequence model."""

import copy

import torch
from torch import distributed as dist

from hyper_parallel.components.losses import ChunkedCausalLMLoss
from hyper_parallel.components.losses.mtp_objective import build_mtp_targets
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed._builder.forward_rewriter import (
    _commit_forward_rewrite,
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.context_parallel.wrappers import (
    sdpa_hf_cp_wrapper,
    sdpa_hf_ulysses_cp_wrapper,
)
from tests.ut.auto_models.test_mtp_migration import make_model


def test_mtp_cp() -> None:
    """Check KV all-gather/Ulysses, future targets crossing CP, and ignored tails."""
    dist.init_process_group("gloo")
    try:
        torch.set_num_threads(1)
        mesh = init_device_mesh("cpu", (1, 2, 1), mesh_dim_names=("dp", "cp", "tp"))
        rank = dist.get_rank()
        for wrapper in (sdpa_hf_cp_wrapper, sdpa_hf_ulysses_cp_wrapper):
            torch.manual_seed(42)
            # The released model has two KV heads; CP=2 must divide Q/K/V.
            reference = make_model(kv_heads=2)
            model = copy.deepcopy(reference)
            model._hp_mtp_cp_group = mesh["cp"].get_group()
            for layer in (*model.model.layers, *model.mtp.layers):
                module = layer.self_attn
                _commit_forward_rewrite(_ForwardRewriteRequest(
                    module, wrapper(module, mesh, mesh["tp"], mesh["cp"], None),
                ))
            for supervised_start in (0, 5, 8):
                ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
                targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, -100]])
                targets[:, :supervised_start] = -100
                future = build_mtp_targets(targets, 2)
                logged = []
                for local, candidate in ((False, reference), (True, model)):
                    candidate.zero_grad(set_to_none=True)
                    part = slice(4 * rank, 4 * (rank + 1)) if local else slice(None)
                    loss_fn = ChunkedCausalLMLoss(chunk_size=3)
                    loss_fn.bind_model(candidate)
                    inputs = loss_fn.prepare_model_inputs({
                        "input_ids": ids[:, part],
                        "position_ids": torch.arange(8).unsqueeze(0)[:, part],
                        "mtp_targets": future[..., part],
                    }, {"shift_labels": targets[:, part]})
                    output = candidate(**inputs, use_cache=False)
                    values = loss_fn(model_output=output, labels=None)
                    if local:
                        for name, current, full in (
                            ("foundation_loss", targets[:, part], targets),
                            ("mtp_loss", future[..., part], future),
                        ):
                            values[name] = values[name] * current.ne(-100).sum() / full.ne(-100).sum().clamp_min(1)
                    loss = sum(values.values())
                    loss.backward()
                    scalar = loss.detach().clone()
                    if local:
                        dist.all_reduce(scalar)
                        for name, parameter in candidate.named_parameters():
                            if parameter.grad is None:
                                raise AssertionError(f"Disconnected CP parameter: {name}")
                            dist.all_reduce(parameter.grad)
                    logged.append(scalar)
                torch.testing.assert_close(logged[1], logged[0], atol=3e-6, rtol=3e-5)
                for (name, actual), expected in zip(model.named_parameters(), reference.parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, atol=4e-6, rtol=5e-4, msg=name)
            if rank == 0:
                print(f"PASS MTP CP losses/gradients: {wrapper.__name__}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_mtp_cp()
