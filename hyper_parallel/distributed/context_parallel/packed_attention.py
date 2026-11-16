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
"""Packed HF attention with KV all-gather or Ulysses all-to-all."""

from __future__ import annotations

from bisect import bisect_right
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import
from transformers import AttentionInterface

from hyper_parallel.components.functional.linear_attention_packed import PackedSequenceInfo
from hyper_parallel.components.functional.npu_fusion_attention import npu_fusion_attention_forward
from hyper_parallel.distributed.context_parallel.collectives import (
    _validate_ulysses_head_counts,
    flex_cp_allgather,
    ulysses_head_to_seq,
    ulysses_seq_to_head,
)
from hyper_parallel.distributed.recipe_spec import inner_wrapper


_ATTENTION_NAME = "hyper_parallel_packed_allgather"


def packed_kv_prefix(
    info: PackedSequenceInfo,
) -> tuple[slice, tuple[int, ...], tuple[int, ...]]:
    """Pair local Q fragments with document KV prefixes for right-aligned causality.

    The final document is truncated at the local Q end. Consequently every Q
    fragment is a suffix of its corresponding KV prefix, including when both
    CP cuts lie inside one document. No zero-length varlen segments are needed.
    """
    first = bisect_right(info.global_offsets, info.start) - 1
    kv_start = info.global_offsets[first]
    prefix_length = info.start - kv_start
    query_offsets = info.local_offsets
    key_offsets = (0, *(offset + prefix_length for offset in query_offsets[1:]))
    return slice(kv_start, info.end), query_offsets, key_offsets


def _packed_cp_sequence(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    kwargs: dict[str, Any],
) -> PackedSequenceInfo:
    """Validate global boundaries against the current contiguous CP shard."""
    if attention_mask is not None:
        raise ValueError("Packed CP attention requires attention_mode=compressed and no dense mask.")
    if query.shape[0] != 1 or query.shape[2] != key.shape[2] or key.shape != value.shape:
        raise ValueError("Packed CP requires batch size one and matching local Q/K/V sequence lengths.")
    if not kwargs.get("is_causal", getattr(module, "is_causal", True)):
        raise ValueError("Packed CP currently supports causal attention only.")
    mesh = module.packed_attention_cp_mesh
    start = mesh.get_local_rank() * query.shape[2]
    offsets_q = kwargs.get("cu_seq_lens_q")
    offsets_k = kwargs.get("cu_seq_lens_k")
    if offsets_q is None or offsets_k is None:
        raise ValueError("Packed CP requires global cu_seq_lens_q and cu_seq_lens_k.")
    if isinstance(offsets_q, torch.Tensor) or isinstance(offsets_k, torch.Tensor):
        raise ValueError("Pass CPU sequence tuples from the runtime adapter to avoid per-layer device synchronization.")
    if tuple(offsets_q) != tuple(offsets_k):
        raise ValueError("Packed self-attention requires matching global Q/KV document boundaries.")
    info = kwargs.get("packed_sequence_info")
    if info is None:
        info = PackedSequenceInfo(tuple(offsets_q), start, start + query.shape[2])
    if (info.start != start or info.length != query.shape[2]
            or info.global_offsets != tuple(offsets_q)
            or info.global_offsets[-1] != query.shape[2] * mesh.size()):
        raise ValueError("Packed boundaries must describe the current contiguous CP shard and global sequence.")
    return info


def packed_allgather_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run packed causal NPU attention without a sequence-sized dense mask.

    Inputs use BNSD with batch size one and equal contiguous CP shards. The
    existing all-gather autograd function sums remote KV gradients in backward.
    Head counts impose no additional restriction on the CP degree.
    """
    info = _packed_cp_sequence(module, query, key, value, attention_mask, kwargs)
    mesh = module.packed_attention_cp_mesh
    global_key, global_value = flex_cp_allgather(key.contiguous(), value.contiguous(), 2, mesh)
    interval, query_offsets, key_offsets = packed_kv_prefix(info)
    return npu_fusion_attention_forward(
        module, query, global_key[:, :, interval], global_value[:, :, interval], None,
        dropout=dropout, scaling=scaling,
        cu_seq_lens_q=query_offsets, cu_seq_lens_k=key_offsets,
    )


@inner_wrapper
def packed_allgather_hf_cp_wrapper(
    target_module: torch.nn.Module, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
) -> Callable:
    """Select the packed AttentionInterface while retaining HF projections and RoPE.

    Custom HF attention implementations without a mask-registry entry bypass
    causal-mask construction. Document isolation is supplied by cu_seq_lens.
    The recipe must select compressed data attention and supply packed metadata.
    """
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None:
        raise ValueError("Packed all-gather requires an active CP mesh.")
    AttentionInterface.register(_ATTENTION_NAME, packed_allgather_attention_forward)
    target_module.packed_attention_cp_mesh = cp_mesh
    target_module.config._attn_implementation = _ATTENTION_NAME
    return target_module.forward


def packed_ulysses_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Exchange sequence shards for head shards, attending within global documents.

    Q/K/V use BNSD and the HF result uses BSND. All-to-all preserves global
    token order, so the varlen kernel consumes the original global offsets.

    Args:
        module: HF attention module carrying the CP mesh.
        query: Local query sequence shard in BNSD layout.
        key: Local key sequence shard in BNSD layout.
        value: Local value sequence shard in BNSD layout.
        attention_mask: None; document isolation comes from cumulative lengths.
        dropout: Attention dropout probability.
        scaling: Optional attention-logit scaling factor.
        **kwargs: Global document offsets and optional packed sequence metadata.

    Returns:
        Local attention output in BSND layout and no attention weights.
    """
    info = _packed_cp_sequence(module, query, key, value, attention_mask, kwargs)
    mesh = module.packed_attention_cp_mesh
    _validate_ulysses_head_counts(query, key, value, mesh.size(), "Packed Ulysses CP")
    query, key, value = (
        ulysses_seq_to_head(tensor, 2, 1, mesh) for tensor in (query, key, value)
    )
    output, _ = npu_fusion_attention_forward(
        module, query, key, value, None, dropout=dropout, scaling=scaling,
        cu_seq_lens_q=info.global_offsets, cu_seq_lens_k=info.global_offsets,
    )
    return ulysses_head_to_seq(output, 1, 2, mesh), None


@inner_wrapper
def packed_ulysses_hf_cp_wrapper(
    target_module: torch.nn.Module, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
) -> Callable:
    """Keep HF projections/RoPE and select packed Pure Ulysses attention.

    The custom interface bypasses HF dense-mask construction. Q/K/V head
    counts must each be divisible by the CP degree; metadata is global.

    Args:
        target_module: HF attention module to wrap.
        mesh: Root mesh, unused by this wrapper.
        tp_mesh: Tensor-parallel mesh, unused by this wrapper.
        cp_mesh: Active context-parallel mesh.
        ep_mesh: Expert-parallel mesh, unused by this wrapper.

    Returns:
        The original forward using the registered packed attention interface.
    """
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("Packed Ulysses requires an active CP mesh with size greater than one.")
    name = "hyper_parallel_packed_ulysses"
    AttentionInterface.register(name, packed_ulysses_attention_forward)
    target_module.packed_attention_cp_mesh = cp_mesh
    target_module.config._attn_implementation = name
    return target_module.forward
