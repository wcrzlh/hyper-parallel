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
"""Dense-reference parity for packed KV-prefix attention under arbitrary CP cuts."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeAttention

from hyper_parallel.components.functional.linear_attention_packed import PackedSequenceInfo
from hyper_parallel.distributed.context_parallel import packed_attention as packed


def _reference_varlen(module, query, key, value, attention_mask, **kwargs):
    """Emulate NPU TND sparse_mode=3 with explicit right-aligned causal masks."""
    del module, attention_mask
    output = []
    cu_q, cu_k = kwargs["cu_seq_lens_q"], kwargs["cu_seq_lens_k"]
    for q0, q1, k0, k1 in zip(cu_q, cu_q[1:], cu_k, cu_k[1:]):
        q_len, k_len = q1 - q0, k1 - k0
        mask = torch.arange(k_len)[None, :] <= torch.arange(q_len)[:, None] + k_len - q_len
        output.append(F.scaled_dot_product_attention(  # pylint: disable=not-callable
            query[:, :, q0:q1], key[:, :, k0:k1], value[:, :, k0:k1],
            attn_mask=mask, scale=kwargs.get("scaling"), enable_gqa=True,
        ))
    return torch.cat(output, dim=2).transpose(1, 2), None


class TestPackedAttention(unittest.TestCase):
    """Compare outputs and all Q/K/V gradients to one global document mask."""

    def test_forward_backward_across_cp_cuts(self):
        torch.manual_seed(17)
        length = 32
        # Include exact cuts, tiny documents, and a document spanning all ranks.
        for offsets in ((0, 32), (0, 8, 16, 24, 32), (0, 3, 17, 18, 29, 32)):
            for cp_size in (1, 2, 4, 8):
                with self.subTest(offsets=offsets, cp_size=cp_size):
                    inputs = [torch.randn(1, heads, length, 8, dtype=torch.float64, requires_grad=True)
                              for heads in (6, 2, 2)]
                    copies = [tensor.detach().clone().requires_grad_() for tensor in inputs]
                    docs = torch.bucketize(torch.arange(length), torch.tensor(offsets[1:]), right=True)
                    mask = (docs[:, None] == docs[None, :]) & torch.ones(length, length, dtype=torch.bool).tril()
                    expected = F.scaled_dot_product_attention(  # pylint: disable=not-callable
                        *inputs, attn_mask=mask, enable_gqa=True,
                    )
                    chunks = []
                    local = length // cp_size
                    for rank in range(cp_size):
                        mesh = SimpleNamespace(size=lambda: cp_size, get_local_rank=lambda: rank)
                        module = SimpleNamespace(packed_attention_cp_mesh=mesh, is_causal=True)
                        start, end = rank * local, (rank + 1) * local
                        info = PackedSequenceInfo(offsets, start, end)
                        with patch.object(packed, "flex_cp_allgather", return_value=tuple(copies[1:])):
                            with patch.object(packed, "npu_fusion_attention_forward", side_effect=_reference_varlen):
                                result, _ = packed.packed_allgather_attention_forward(
                                    module, *(tensor[:, :, start:end] for tensor in copies), None,
                                    packed_sequence_info=info, cu_seq_lens_q=offsets, cu_seq_lens_k=offsets,
                                )
                        chunks.append(result.transpose(1, 2))
                    actual = torch.cat(chunks, dim=2)
                    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                    upstream = torch.randn_like(expected)
                    expected.backward(upstream)
                    actual.backward(upstream)
                    for original, copied in zip(inputs, copies):
                        torch.testing.assert_close(copied.grad, original.grad, atol=1e-12, rtol=1e-12)

    def test_reject_dense_mask_and_inconsistent_metadata(self):
        tensor = torch.zeros(1, 2, 8, 4)
        mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 1)
        module = SimpleNamespace(packed_attention_cp_mesh=mesh, is_causal=True)
        with self.assertRaisesRegex(ValueError, "no dense mask"):
            packed.packed_allgather_attention_forward(module, tensor, tensor, tensor, torch.ones(8, 16))
        with self.assertRaisesRegex(ValueError, "matching global"):
            packed.packed_allgather_attention_forward(
                module, tensor, tensor, tensor, None, cu_seq_lens_q=(0, 16), cu_seq_lens_k=(0, 8, 16),
            )
        with self.assertRaisesRegex(ValueError, "current contiguous"):
            packed.packed_allgather_attention_forward(
                module, tensor, tensor, tensor, None, cu_seq_lens_q=(0, 16), cu_seq_lens_k=(0, 16),
                packed_sequence_info=PackedSequenceInfo((0, 16), 0, 8),
            )

    def test_hf_interface_forwards_boundaries_without_building_mask(self):
        """Exercise actual HF Qwen projections/RoPE and the custom-interface mask bypass."""
        config = Qwen3_5MoeTextConfig.from_dict({
            "hidden_size": 24, "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 4,
        })
        config._attn_implementation = "sdpa"
        module = Qwen3_5MoeAttention(config, 3)
        mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 1)
        forward = packed.packed_allgather_hf_cp_wrapper(module, None, None, mesh, None)
        hidden = torch.randn(1, 8, 24)
        mask = create_causal_mask(
            config=config, inputs_embeds=hidden, attention_mask=None, past_key_values=None,
            position_ids=torch.tensor([[5, 6, 7, 8, 0, 1, 2, 3]]),
        )
        self.assertIsNone(mask)

        def _gather(key, value, *_):
            return torch.cat((key, key), dim=2), torch.cat((value, value), dim=2)

        with patch.object(packed, "flex_cp_allgather", side_effect=_gather):
            with patch.object(packed, "npu_fusion_attention_forward", side_effect=_reference_varlen) as backend:
                result, _ = forward(
                    hidden, position_embeddings=(torch.ones(1, 8, 4), torch.zeros(1, 8, 4)),
                    attention_mask=mask, cu_seq_lens_q=(0, 3, 12, 16), cu_seq_lens_k=(0, 3, 12, 16),
                )
        self.assertEqual(result.shape, hidden.shape)
        self.assertEqual(backend.call_args.kwargs["cu_seq_lens_q"], (0, 4, 8))
        self.assertEqual(backend.call_args.kwargs["cu_seq_lens_k"], (0, 9, 13))
