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
"""Two-rank MTP CP gradients against a complete-sequence, non-CP oracle."""

from types import SimpleNamespace

import torch
from torch import distributed as dist

from hyper_parallel.components.losses import ChunkedCausalLMLoss
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest, _commit_forward_rewrite
from hyper_parallel.distributed.context_parallel.wrappers import sdpa_hf_cp_wrapper, sdpa_hf_ulysses_cp_wrapper
from hyper_parallel.models.qwen3_5.adapter.distributed.context_parallel import (
    qwen3_5_gdn_p2p_cp_wrapper, qwen3_5_gdn_ulysses_cp_wrapper,
)
from hyper_parallel.models.qwen3_5_moe.adapter.mtp_runtime import attach_mtp
from hyper_parallel.models.qwen3_5_moe.vlm_cp import QwenVLMParallelBatch
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from tests.ut.auto_models.test_qwen_vlm_chunk_loss import make_model, make_batch


def test_mtp_cp() -> None:
    """Compare both CP algorithms to no CP, including every MTP and vision gradient."""
    dist.init_process_group("gloo")
    try:
        torch.set_num_threads(1)
        mesh = init_device_mesh("cpu", (1, 2, 1), mesh_dim_names=("dp", "cp", "tp"))
        common = dict(tp_size=1, pp_size=1, dp_size=1, dp_rank=0, tp_rank=0,
                      sequence_parallel=False, loss_parallel=False)
        context = SimpleNamespace(device_mesh=mesh, cp_mesh=mesh["cp"], cp_size=2,
                                  cp_rank=dist.get_rank(), **common)
        standalone = SimpleNamespace(cp_size=1, cp_rank=0, **common)
        modes = ((qwen3_5_gdn_ulysses_cp_wrapper, sdpa_hf_ulysses_cp_wrapper),
                 (qwen3_5_gdn_p2p_cp_wrapper, sdpa_hf_cp_wrapper))
        for gdn, attention in modes:
            torch.manual_seed(42)
            models = [make_model(), make_model()]
            for model, topology in zip(models, (standalone, context)):
                model.config.text_config.mtp_num_hidden_layers = 2
                attach_mtp(model, topology)
            models[1].load_state_dict(models[0].state_dict())
            layers = models[1].model.language_model.layers
            _commit_forward_rewrite(gdn(layers[0].linear_attn, mesh, mesh["tp"], mesh["cp"], None))
            module = layers[1].self_attn
            _commit_forward_rewrite(_ForwardRewriteRequest(
                module, attention(module, mesh, mesh["tp"], mesh["cp"], None)))
            for layer in models[1].mtp.layers:
                module = layer.self_attn
                _commit_forward_rewrite(_ForwardRewriteRequest(
                    module, attention(module, mesh, mesh["tp"], mesh["cp"], None)))
            batchers, objectives = [], []
            for model, topology in zip(models, (standalone, context)):
                objective = ChunkedCausalLMLoss(chunk_size=17)
                objective.bind_model(model)
                batcher = QwenVLMParallelBatch(mesh_context=topology, device=torch.device("cpu"))
                batcher.bind_model(model)
                batchers.append(batcher)
                objectives.append(objective)
            for supervised_start in (2, 65, 100):
                torch.manual_seed(24)
                batch = make_batch(62, supervised_start)
                losses = []
                for index, (model, batcher, objective) in enumerate(zip(models, batchers, objectives)):
                    model.zero_grad(set_to_none=True)
                    inputs, labels = batcher(None, external_batch=batch)
                    counts = count_loss_token(labels)
                    values = objective(model_output=model(**objective.prepare_model_inputs(inputs, labels)), labels=None)
                    if index == 1:
                        for key in values:
                            count = counts[key.replace("_loss", "_tokens")]
                            total = count.clone()
                            dist.all_reduce(total)
                            values[key] = values[key] * count / total.clamp_min(1)
                    loss = sum(values.values())
                    loss.backward()
                    logged = torch.stack([value.detach() for value in values.values()])
                    if index == 1:
                        dist.all_reduce(logged)
                        for param in model.parameters():
                            if param.grad is None:
                                raise AssertionError("MTP CP path has a disconnected parameter")
                            dist.all_reduce(param.grad)
                    losses.append(logged)
                torch.testing.assert_close(losses[1], losses[0], atol=3e-6, rtol=3e-5)
                for (name, actual), (_, expected) in zip(models[1].named_parameters(), models[0].named_parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, atol=4e-6, rtol=5e-4, msg=name)
            if dist.get_rank() == 0:
                print(f"PASS MTP versus non-CP: {gdn.__name__}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_mtp_cp()
