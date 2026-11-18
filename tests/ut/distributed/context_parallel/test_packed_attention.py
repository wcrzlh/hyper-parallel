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

from contextlib import ExitStack
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
from tests.common.mark_utils import arg_mark


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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_forward_backward_across_cp_cuts(self):
        """Feature: Packed attention across CP cuts.

        Description: Compare local fragments against a global document mask.
        Expectation: Outputs and gradients match the reference.
        """
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_ulysses_global_boundaries_and_gradients(self):
        """Feature: Packed Ulysses attention.

        Description: Emulate differentiable head exchange across CP cuts.
        Expectation: Outputs and gradients match independent document attention.
        """
        torch.manual_seed(19)
        for offsets in ((0, 32), (0, 16, 32), (0, 1, 15, 17, 31, 32)):
            with self.subTest(offsets=offsets):
                inputs = [torch.randn(1, heads, 32, 8, dtype=torch.float64, requires_grad=True)
                          for heads in (6, 2, 2)]
                copies = [tensor.detach().clone().requires_grad_() for tensor in inputs]
                expected, _ = _reference_varlen(
                    None, *inputs, None, cu_seq_lens_q=offsets, cu_seq_lens_k=offsets,
                )
                head_outputs = []
                for rank in range(2):
                    mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: rank)
                    module = SimpleNamespace(packed_attention_cp_mesh=mesh, is_causal=True)
                    shards = [tensor.chunk(2, dim=1)[rank] for tensor in copies]

                    def _restore(output, seq_dim, head_dim, cp_mesh):
                        self.assertEqual((seq_dim, head_dim), (1, 2))
                        self.assertIs(cp_mesh, mesh)
                        return output

                    with patch.object(packed, "ulysses_seq_to_head", side_effect=shards) as exchange, \
                            patch.object(packed, "ulysses_head_to_seq", side_effect=_restore), \
                            patch.object(packed, "npu_fusion_attention_forward", side_effect=_reference_varlen) as kernel:
                        result, _ = packed.packed_ulysses_attention_forward(
                            module, *(tensor[:, :, rank * 16:(rank + 1) * 16] for tensor in copies), None,
                            cu_seq_lens_q=offsets, cu_seq_lens_k=offsets,
                        )
                    self.assertEqual(exchange.call_count, 3)
                    for call in exchange.call_args_list:
                        self.assertEqual(call.args[1:3], (2, 1))
                    self.assertEqual(kernel.call_args.kwargs["cu_seq_lens_q"], offsets)
                    head_outputs.append(result)
                actual = torch.cat(head_outputs, dim=2)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                # Supervise one document: every other document must have zero Q/K/V gradient.
                upstream = torch.zeros_like(expected)
                left, right = offsets[:2]
                upstream[:, left:right] = torch.randn_like(upstream[:, left:right])
                expected.backward(upstream)
                actual.backward(upstream)
                for original, copied in zip(inputs, copies):
                    torch.testing.assert_close(copied.grad, original.grad, atol=1e-12, rtol=1e-12)
                    torch.testing.assert_close(copied.grad[:, :, right:],
                                               torch.zeros_like(copied.grad[:, :, right:]), atol=0, rtol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_reject_dense_mask_and_inconsistent_metadata(self):
        """Feature: Packed attention input validation.

        Description: Supply masks, inconsistent boundaries or non-divisible KV heads.
        Expectation: Invalid configurations raise before communication.
        """
        mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 1)
        module = SimpleNamespace(packed_attention_cp_mesh=mesh, is_causal=True)
        tensor = torch.zeros(1, 2, 8, 4)
        for forward in (packed.packed_allgather_attention_forward, packed.packed_ulysses_attention_forward):
            with self.subTest(forward=forward.__name__):
                with self.assertRaisesRegex(ValueError, "no dense mask"):
                    forward(module, tensor, tensor, tensor, torch.ones(8, 16))
                with self.assertRaisesRegex(ValueError, "matching global"):
                    forward(module, tensor, tensor, tensor, None,
                            cu_seq_lens_q=(0, 16), cu_seq_lens_k=(0, 8, 16))
                with self.assertRaisesRegex(ValueError, "current contiguous"):
                    forward(
                        module, tensor, tensor, tensor, None, cu_seq_lens_q=(0, 16), cu_seq_lens_k=(0, 16),
                        packed_sequence_info=PackedSequenceInfo((0, 16), 0, 8),
                    )
        with self.assertRaisesRegex(ValueError, "head count to be divisible"):
            packed.packed_ulysses_attention_forward(
                module, tensor, tensor[:, :1], tensor[:, :1], None,
                cu_seq_lens_q=(0, 16), cu_seq_lens_k=(0, 16),
            )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_hf_interface_forwards_boundaries_without_building_mask(self):
        """Feature: Packed HF attention interface.

        Description: Exercise HF projections and RoPE for both CP algorithms.
        Expectation: Global boundaries reach Ulysses and HF builds no dense mask.
        """
        for use_ulysses in (False, True):
            with self.subTest(ulysses=use_ulysses):
                config = Qwen3_5MoeTextConfig.from_dict({
                    "hidden_size": 24, "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 4,
                })
                config._attn_implementation = "sdpa"
                module = Qwen3_5MoeAttention(config, 3)
                mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 1)
                wrapper = packed.packed_ulysses_hf_cp_wrapper if use_ulysses else packed.packed_allgather_hf_cp_wrapper
                forward = wrapper(module, None, None, mesh, None)
                hidden = torch.randn(1, 8, 24)
                mask = create_causal_mask(
                    config=config, inputs_embeds=hidden, attention_mask=None, past_key_values=None,
                    position_ids=torch.tensor([[5, 6, 7, 8, 0, 1, 2, 3]]),
                )
                self.assertIsNone(mask)

                def _gather(key, value, *_):
                    return torch.cat((key, key), dim=2), torch.cat((value, value), dim=2)

                def _seq_to_head(tensor, *_):
                    return torch.cat((tensor, tensor), dim=2).chunk(2, dim=1)[1]

                def _head_to_seq(tensor, *_):
                    return torch.cat((tensor, tensor), dim=2)[:, 8:]

                with ExitStack() as stack:
                    stack.enter_context(patch.object(packed, "flex_cp_allgather", side_effect=_gather))
                    stack.enter_context(patch.object(packed, "ulysses_seq_to_head", side_effect=_seq_to_head))
                    stack.enter_context(patch.object(packed, "ulysses_head_to_seq", side_effect=_head_to_seq))
                    backend = stack.enter_context(
                        patch.object(packed, "npu_fusion_attention_forward", side_effect=_reference_varlen),
                    )
                    result, _ = forward(
                        hidden, position_embeddings=(torch.ones(1, 8, 4), torch.zeros(1, 8, 4)),
                        attention_mask=mask, cu_seq_lens_q=(0, 3, 12, 16), cu_seq_lens_k=(0, 3, 12, 16),
                    )
                self.assertEqual(result.shape, hidden.shape)
                expected_q = (0, 3, 12, 16) if use_ulysses else (0, 4, 8)
                expected_k = (0, 3, 12, 16) if use_ulysses else (0, 9, 13)
                self.assertEqual(backend.call_args.kwargs["cu_seq_lens_q"], expected_q)
                self.assertEqual(backend.call_args.kwargs["cu_seq_lens_k"], expected_k)
