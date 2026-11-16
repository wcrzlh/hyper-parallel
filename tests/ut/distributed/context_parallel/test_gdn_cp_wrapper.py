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
"""Unit tests for the Gated DeltaNet CP inner wrapper."""

import pytest
import torch
from torch.nn import functional as F

from hyper_parallel.components.functional.linear_attention_packed import PackedSequenceInfo
from hyper_parallel.distributed.context_parallel import wrappers as cp_wrappers
from hyper_parallel.distributed.context_parallel import gated_delta_net as cp_gdn


def causal_conv1d_fn(
        hidden_states=None, weight=None, bias=None, activation=None, *, x=None):
    """Fake Transformers Conv1d primitive used by the fake GDN forward."""
    del weight, bias, activation
    if hidden_states is None:
        hidden_states = x
    return hidden_states + 1


def torch_chunk_gated_delta_rule(
        query, key, value, *, g, beta, initial_state=None,
        output_final_state=False):
    """Fake Transformers GDN primitive used by the fake GDN forward."""
    del key, value, g, beta, initial_state, output_final_state
    return query + 2, None


def _hook_wrapper(forward_func):
    """Mimic the Transformers hook decorator without ``functools.wraps``."""
    def wrapped(self, *args, **kwargs):
        return forward_func(self, *args, **kwargs)

    return wrapped


class _FakeCPMesh:
    """Minimal active CP mesh required by wrapper validation."""

    @staticmethod
    def size():
        """Return an active two-rank CP size."""
        return 2


class FakeGatedDeltaNet:
    """Minimal GDN whose forward calls the two intercepted primitives."""

    def __init__(self):
        """Initialize the fake GDN head metadata."""
        self.num_v_heads = 2

    def forward(self, hidden_states, cache_params=None, **kwargs):
        """Run fake Conv1d and GDN rule primitives."""
        del cache_params, kwargs
        mixed = causal_conv1d_fn(hidden_states, torch.ones(1, 2))
        decay = torch.zeros(mixed.shape[:-1])
        beta = torch.zeros_like(decay)
        output, _ = torch_chunk_gated_delta_rule(
            mixed,
            mixed,
            mixed,
            g=decay,
            beta=beta,
            initial_state=None,
            output_final_state=False,
        )
        return output


class HookedGatedDeltaNet:
    """GDN variant using instance primitives below an opaque hook wrapper."""

    def __init__(self):
        """Initialize instance primitive attributes and head metadata."""
        self.num_v_heads = 2
        self.causal_conv1d_fn = causal_conv1d_fn
        self.chunk_gated_delta_rule = torch_chunk_gated_delta_rule

    @_hook_wrapper
    def forward(self, hidden_states, cache_params=None, **kwargs):
        """Run the same attribute primitive calls as Transformers 5.13."""
        del cache_params, kwargs
        mixed = self.causal_conv1d_fn(
            x=hidden_states,
            weight=torch.ones(1, 2),
        )
        decay = torch.zeros(mixed.shape[:-1])
        beta = torch.zeros_like(decay)
        output, _ = self.chunk_gated_delta_rule(
            mixed,
            mixed,
            mixed,
            g=decay,
            beta=beta,
            initial_state=None,
            output_final_state=False,
        )
        return output


def test_gdn_wrapper_intercepts_primitives_without_copying_forward(monkeypatch):
    """The wrapper delegates model semantics to the original GDN forward."""
    calls = {"conv": 0, "to_hp": 0, "to_cp": 0}

    def fake_cp_conv(original, cp_mesh, *args, **kwargs):
        del cp_mesh
        calls["conv"] += 1
        return original(*args, **kwargs)

    def fake_to_hp(query, key, value, decay, beta, cp_mesh):
        del cp_mesh
        calls["to_hp"] += 1
        return query, key, value, decay, beta

    def fake_to_cp(output, cp_mesh):
        del cp_mesh
        calls["to_cp"] += 1
        return output

    monkeypatch.setattr(cp_gdn, "_gdn_cp_causal_conv1d", fake_cp_conv)
    monkeypatch.setattr(cp_gdn, "_gdn_rule_cp_to_hp", fake_to_hp)
    monkeypatch.setattr(cp_gdn, "_gdn_rule_hp_to_cp", fake_to_cp)

    module = FakeGatedDeltaNet()
    original_conv = causal_conv1d_fn
    original_rule = torch_chunk_gated_delta_rule
    cp_wrappers.gdn_ulysses_cp_wrapper(
        module,
        mesh=None,
        tp_mesh=None,
        cp_mesh=_FakeCPMesh(),
        ep_mesh=None,
    )

    inputs = torch.zeros(1, 2, 2, 1)
    output = module.forward(inputs)

    torch.testing.assert_close(output, inputs + 3)
    assert calls == {"conv": 1, "to_hp": 1, "to_cp": 1}
    assert globals()["causal_conv1d_fn"] is original_conv
    assert globals()["torch_chunk_gated_delta_rule"] is original_rule


def test_gdn_halo_exchange_uses_torch_collective_adapter(monkeypatch):
    """The rebased wrapper routes halo exchange through Torch collectives."""
    tail = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    calls = {}

    class FakeMesh:
        """Provide the two-rank mesh surface used by halo exchange."""

        rank_list = (0, 1)

        @staticmethod
        def size():
            """Return the active CP world size."""
            return 2

        @staticmethod
        def get_local_rank():
            """Place this test rank after rank zero."""
            return 1

        @staticmethod
        def get_group():
            """Return an opaque process-group token."""
            return "cp_group"

    def fake_all_to_all_single(input_tensor, input_splits, output_splits, group):
        """Capture the adapter contract and return the previous-rank halo."""
        calls.update(
            input_tensor=input_tensor,
            input_splits=input_splits,
            output_splits=output_splits,
            group=group,
        )
        return torch.full_like(tail.permute(2, 0, 1), 7)

    monkeypatch.setattr(cp_gdn.dist, "get_process_group_ranks", lambda _group: (0, 1))
    monkeypatch.setattr(
        cp_gdn.communication,
        "differentiable_all_to_all_single",
        fake_all_to_all_single,
    )

    result = cp_gdn._exchange_previous_cp_halo(tail, FakeMesh())

    torch.testing.assert_close(result, torch.full_like(tail, 7))
    torch.testing.assert_close(calls["input_tensor"], tail.permute(2, 0, 1)[:0])
    assert calls["input_splits"] == [0, 0]
    assert calls["output_splits"] == [3, 0]
    assert calls["group"] == "cp_group"


def test_gdn_wrapper_resolves_hooked_instance_primitives(monkeypatch):
    """The wrapper supports the decorated Transformers 5.13 call structure."""
    monkeypatch.setattr(
        cp_gdn,
        "_gdn_cp_causal_conv1d",
        lambda original, cp_mesh, *args, **kwargs: original(*args, **kwargs),
    )
    monkeypatch.setattr(
        cp_gdn,
        "_gdn_rule_cp_to_hp",
        lambda query, key, value, decay, beta, cp_mesh: (
            query, key, value, decay, beta),
    )
    monkeypatch.setattr(
        cp_gdn,
        "_gdn_rule_hp_to_cp",
        lambda output, cp_mesh: output,
    )

    module = HookedGatedDeltaNet()
    original_conv = module.causal_conv1d_fn
    original_rule = module.chunk_gated_delta_rule
    cp_wrappers.gdn_ulysses_cp_wrapper(
        module,
        mesh=None,
        tp_mesh=None,
        cp_mesh=_FakeCPMesh(),
        ep_mesh=None,
    )

    inputs = torch.zeros(1, 2, 2, 1)
    torch.testing.assert_close(module.forward(inputs), inputs + 3)
    assert module.causal_conv1d_fn is original_conv
    assert module.chunk_gated_delta_rule is original_rule


@pytest.mark.parametrize("fused,dilation", [(False, 1), (False, 2), (True, 1)])
def test_packed_conv_preserves_documents_and_halo_gradients(monkeypatch, fused, dilation):
    """Feature: Packed GDN convolution halo.

    Description: Sweep document starts around a CP cut.
    Expectation: Outputs and all halo and parameter gradients match independent documents.
    """
    torch.manual_seed(42)
    halo_width = 3 * dilation
    conv = torch.nn.Conv1d(2, 2, 4, padding=halo_width, dilation=dilation, groups=2).double()
    weight, bias = conv.weight, conv.bias

    def _original(value, conv_weight, conv_bias, activation):
        output = torch.conv1d(value, conv_weight.unsqueeze(1), conv_bias,
                          padding=halo_width, dilation=dilation, groups=2)[..., :value.shape[-1]]
        return F.silu(output) if activation == "silu" else output

    layouts = [(0, 16), *((0, boundary, 16) for boundary in range(1, 16)),
               (0, 1, 7, 8, 9, 15, 16), (0, 5, 7, 9, 10, 16)]
    for offsets in layouts:
        inputs = torch.randn(1, 2, 16, dtype=torch.float64, requires_grad=True)
        expected = torch.cat([
            _original(inputs[..., left:right], weight.squeeze(1), bias, "silu" if fused else None)
            for left, right in zip(offsets, offsets[1:])], dim=-1)
        for rank in range(2):
            start, end = rank * 8, (rank + 1) * 8
            local = inputs[..., start:end].detach().clone().requires_grad_()
            halo = inputs[..., start - halo_width:start] if rank else inputs.new_zeros(1, 2, halo_width)
            halo = halo.detach().clone().requires_grad_()
            monkeypatch.setattr(cp_gdn, "_exchange_previous_cp_halo", lambda tail, mesh: halo)
            info = PackedSequenceInfo(offsets, start, end)
            if fused:
                actual = cp_gdn._gdn_cp_causal_conv1d(
                    _original, _FakeCPMesh(), local, weight.squeeze(1), bias, "silu", sequence_info=info)
            else:
                actual = cp_gdn._gdn_cp_conv1d_module(conv.forward, conv, _FakeCPMesh(), local, sequence_info=info)
            message = f"offsets={offsets}, rank={rank}, fused={fused}, dilation={dilation}"
            torch.testing.assert_close(actual, expected[..., start:end], atol=1e-12, rtol=1e-12, msg=message)
            grad = torch.randn_like(actual)
            expected_grads = torch.autograd.grad(
                expected[..., start:end], (inputs, weight, bias), grad, retain_graph=True)
            # allow_unused=False also checks that a fully masked halo retains an
            # autograd edge: its reverse collective must run even at a document start.
            actual_grads = torch.autograd.grad(actual, (local, halo, weight, bias), grad, allow_unused=False)
            expected_halo = expected_grads[0][..., start - halo_width:start] if rank else torch.zeros_like(halo)
            references = (expected_grads[0][..., start:end], expected_halo, *expected_grads[1:])
            for actual_grad, expected_grad in zip(actual_grads, references):
                torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12, msg=message)
            document_start = max(boundary for boundary in offsets if boundary <= start)
            unused = halo_width - min(halo_width, start - document_start)
            torch.testing.assert_close(
                actual_grads[1][..., :unused], torch.zeros_like(halo[..., :unused]), atol=0, rtol=0, msg=message)


def test_packed_rule_overrides_stale_local_kernel_offsets():
    """Feature: Packed GDN rule metadata.

    Description: Supply stale local offsets before Ulysses exchange.
    Expectation: The kernel receives global document offsets.
    """
    info = PackedSequenceInfo((0, 3, 13, 16), 0, 16)
    query = torch.randn(1, 16, 1, 2)
    decay = torch.zeros(1, 16, 1)

    def _rule(query, key, value, *, g, beta, cu_seqlens):
        del key, value, g, beta
        torch.testing.assert_close(cu_seqlens, torch.tensor([0, 3, 13, 16]))
        return query, None

    output, _ = cp_gdn._gdn_packed_rule(
        _rule, query, query, query, info, g=decay, beta=decay, cu_seqlens=torch.tensor([0, 5, 8]))
    torch.testing.assert_close(output, query)


def test_packed_rule_does_not_trust_kwargs_only_fallback():
    """Feature: Packed GDN reference fallback.

    Description: Invoke a reference primitive that ignores cumulative lengths.
    Expectation: Recurrence resets at every document with zero gradient leakage.
    """
    query = torch.ones(1, 8, 1, 2, requires_grad=True)
    decay = torch.zeros(1, 8, 1)

    def _rule(query, key, value, **kwargs):
        del key, value, kwargs
        return query.cumsum(dim=1), None

    output, _ = cp_gdn._gdn_packed_rule(
        _rule, query, query, query, PackedSequenceInfo((0, 3, 8), 0, 8), g=decay, beta=decay)
    expected = torch.tensor([1, 2, 3, 1, 2, 3, 4, 5]).reshape(1, 8, 1, 1).expand_as(query).float()
    torch.testing.assert_close(output, expected)
    output[:, 3:].sum().backward()
    torch.testing.assert_close(query.grad[:, :3], torch.zeros_like(query.grad[:, :3]))
