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
"""Unit tests for analytical model-FLOPs estimation."""

from types import SimpleNamespace
import unittest

from hyper_parallel.trainer.runtime.flops import estimate_flops_per_token
from hyper_parallel.trainer.runtime.flops import estimate_step_flops
from hyper_parallel.trainer.runtime.flops import estimate_packing_workloads
from tests.common.mark_utils import arg_mark


def qwen3_moe_config(**overrides: object) -> SimpleNamespace:
    """Build a miniature Qwen3-MoE configuration with hand-checkable sizes.

    Defaults describe two all-MoE layers with ``hidden_size=8``, GQA with
    ``num_attention_heads=2``/``num_key_value_heads=1``/``head_dim=4``, a gated
    ``intermediate_size=16`` dense MLP, four experts with top-1 routing and a
    ``vocab_size=32`` head. Every expectation in this module is derived from
    these numbers by hand.
    """
    values = {
        "model_type": "qwen3_moe",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 16,
        "moe_intermediate_size": 4,
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "vocab_size": 32,
        "hidden_act": "silu",
        "mlp_only_layers": [],
        "decoder_sparse_step": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def qwen35_moe_config(layers: int = 40, **overrides: object) -> SimpleNamespace:
    """Describe the GQA 35B-A3B architecture without loading model assets."""
    values = {
        "model_type": "qwen3_5_moe_text", "hidden_size": 2048, "num_hidden_layers": layers,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_conv_kernel_dim": 4,
        "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512, "vocab_size": 248320,
    }
    values["layer_types"] *= layers // 4
    values.update(overrides)
    return SimpleNamespace(**values)


class TestPackingWorkloads(unittest.TestCase):
    """Check architecture-derived weights separately from the partition heuristic."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_40_layer_cost_and_full_recomputation(self):
        """Match the independently derived 40-layer principal-GEMM totals.

        Feature: Model-aware packing workload estimation.
        Description: Estimate four sequence lengths with full recomputation.
        Expectation: Costs match the independently derived 40-layer formula.
        """
        lengths = [65, 4096, 32768, 131072]
        config = qwen35_moe_config()
        actual = estimate_packing_workloads(config, lengths)
        self.assertEqual(actual, [327680 * (70938 * size + size * size) for size in lengths])

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_depth_changes_decoder_but_not_lm_head(self):
        """Doubling decoder depth must not duplicate vocabulary projection work.

        Feature: Architecture-derived packing costs.
        Description: Compare 20-layer, 40-layer, and wrapped text configurations.
        Expectation: Only decoder cost scales with depth; wrappers preserve cost.
        """
        length = 4096
        shallow = estimate_packing_workloads(qwen35_moe_config(20), [length])[0]
        deep = estimate_packing_workloads(qwen35_moe_config(40), [length])[0]
        self.assertEqual(2 * shallow - deep, 3051356160 * length)
        wrapped = SimpleNamespace(text_config=qwen35_moe_config(40))
        self.assertEqual(estimate_packing_workloads(wrapped, [length]), [deep])

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_active_experts_and_unsupported_models(self):
        """Charge active experts, retaining legacy handling for other architectures.

        Feature: Packing estimator applicability.
        Description: Vary active experts, model family, compression, and layer types.
        Expectation: Expert costs scale correctly; unsupported models fall back.
        """
        normal = estimate_packing_workloads(qwen35_moe_config(), [64])[0]
        fewer = estimate_packing_workloads(qwen35_moe_config(num_experts_per_tok=7), [64])[0]
        self.assertEqual(normal - fewer, 1006632960 * 64)
        self.assertIsNone(estimate_packing_workloads(SimpleNamespace(model_type="llama"), [64]))
        mla = qwen35_moe_config(kv_compression={"enabled": True, "mode": "mla"})
        self.assertIsNone(estimate_packing_workloads(mla, [64]))
        with self.assertRaisesRegex(ValueError, "layer_type"):
            estimate_packing_workloads(qwen35_moe_config(layer_types=[]), [64])


class TestEstimateFlopsPerToken(unittest.TestCase):
    """Verify the estimator counts activations, not parameters or experts."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_qwen3_moe_matches_closed_form(self) -> None:
        """Add attention, top-1 MoE and logits terms into one per-token total.

        Feature: Analytical FLOPs estimation.
        Description: Estimate one token of a two-layer Qwen3-MoE at seq_length=3.
        Expectation: 2*1296 attention + 2*576 MoE + 1536 logits = 5280 FLOPs.
        """
        flops = estimate_flops_per_token(qwen3_moe_config(), 3.0)
        self.assertEqual(
            flops,
            5280.0,
            f"expected 2*1296 attention + 2*576 moe + 1536 logits, got={flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_causal_attention_scales_with_sequence_length(self) -> None:
        """Charge only the causal-attention term for longer sequences.

        Feature: Analytical FLOPs estimation.
        Description: Estimate the same model at seq_length=8 instead of 3.
        Expectation: The total grows by 2 layers * 6 * q * (8 - 3) = 480 FLOPs.
        """
        flops = estimate_flops_per_token(qwen3_moe_config(), 8.0)
        self.assertEqual(
            flops,
            5760.0,
            f"expected the 5280 baseline plus 480 attention FLOPs, got={flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_topk_routing_sizes_the_moe_term(self) -> None:
        """Scale the MoE term by activated experts instead of total experts.

        Feature: Analytical FLOPs estimation.
        Description: Estimate the same model with top-2 routing and four experts.
        Expectation: The MoE term doubles to 2*1152, so the total is 6432 FLOPs.
        """
        flops = estimate_flops_per_token(qwen3_moe_config(num_experts_per_tok=2), 3.0)
        self.assertEqual(
            flops,
            6432.0,
            f"expected 2592 attention + 2304 top-2 moe + 1536 logits, got={flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_ffn_projection_count_is_activation_independent(self) -> None:
        """Keep the three gated-FFN GEMMs regardless of ``hidden_act``.

        Feature: Analytical FLOPs estimation.
        Description: Estimate the same model with ``gelu`` and ``relu`` activations.
        Expectation: Qwen3-MoE always runs gate/up/down, so both stay at 5280 FLOPs.
        """
        for hidden_act in ("gelu", "relu"):
            flops = estimate_flops_per_token(
                qwen3_moe_config(hidden_act=hidden_act), 3.0
            )
            self.assertEqual(
                flops,
                5280.0,
                "hidden_act must only select the elementwise activation, "
                f"got={flops} for hidden_act={hidden_act!r}",
            )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_mlp_only_layers_use_the_dense_term(self) -> None:
        """Route ``mlp_only_layers`` entries through the dense MLP term.

        Feature: Analytical FLOPs estimation.
        Description: Estimate a two-layer model whose first layer is listed in ``mlp_only_layers``.
        Expectation: One dense layer (2304) and one MoE layer (576) sum to 7008 FLOPs.
        """
        flops = estimate_flops_per_token(qwen3_moe_config(mlp_only_layers=[0]), 3.0)
        self.assertEqual(
            flops,
            7008.0,
            f"expected 2592 attention + 2304 dense + 576 moe + 1536 logits, got={flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_decoder_sparse_step_selects_every_nth_layer(self) -> None:
        """Apply ``decoder_sparse_step`` to decide which layers are MoE.

        Feature: Analytical FLOPs estimation.
        Description: Estimate four layers with ``decoder_sparse_step=2``.
        Expectation: Two dense and two MoE layers sum to 12480 FLOPs.
        """
        flops = estimate_flops_per_token(
            qwen3_moe_config(num_hidden_layers=4, decoder_sparse_step=2), 3.0
        )
        self.assertEqual(
            flops,
            12480.0,
            f"expected 4*1296 attention + 2*2304 dense + 2*576 moe + 1536 logits, got={flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_zero_experts_falls_back_to_the_dense_term(self) -> None:
        """Treat a zero-expert configuration as a dense decoder.

        Feature: Analytical FLOPs estimation.
        Description: Estimate the same two layers with ``num_experts=0``.
        Expectation: Both layers are dense, so the total is 8736 FLOPs.
        """
        flops = estimate_flops_per_token(qwen3_moe_config(num_experts=0), 3.0)
        self.assertEqual(
            flops,
            8736.0,
            f"expected 2592 attention + 2*2304 dense + 1536 logits, got={flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_unsupported_model_type_returns_none(self) -> None:
        """Return ``None`` instead of guessing for an unsupported model family.

        Feature: Analytical FLOPs estimation.
        Description: Estimate a configuration that only declares ``model_type='llama'``.
        Expectation: No estimate is produced and no missing field is read.
        """
        flops = estimate_flops_per_token(SimpleNamespace(model_type="llama"), 3.0)
        self.assertIsNone(
            flops,
            f"unsupported model families must not report FLOPs, got={flops!r}",
        )


class TestEstimateStepFlops(unittest.TestCase):
    """Verify step FLOPs accumulate the real attention boundaries."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_equal_lengths_match_the_per_token_estimate(self) -> None:
        """Reproduce the per-token estimate when every sequence is equal.

        Feature: Analytical FLOPs step estimation.
        Description: Estimate a step of five length-2 sequences for the same model.
        Expectation: 10 tokens * (4992 base + 96*2 attention) = 51840 FLOPs.
        """
        flops = estimate_step_flops(qwen3_moe_config(), 10.0, 5 * 4.0)
        self.assertEqual(
            flops,
            51840.0,
            f"expected 10*(4992 + 96*2) FLOPs, got={flops}",
        )
        self.assertEqual(
            flops,
            10 * estimate_flops_per_token(qwen3_moe_config(), 2.0),
            "equal-length steps must agree with the per-token estimator",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_variable_lengths_beat_the_mean_length_approximation(self) -> None:
        """Accumulate ``sum(S ** 2)`` instead of one averaged sequence length.

        Feature: Analytical FLOPs step estimation.
        Description: Estimate one length-6 and one length-2 sequence in a step.
        Expectation: 8*4992 + 96*40 = 43776 FLOPs, above the 43008 mean-length value.
        """
        flops = estimate_step_flops(qwen3_moe_config(), 6.0 + 2.0, 36.0 + 4.0)
        self.assertEqual(
            flops,
            43776.0,
            f"expected 8*4992 + 96*(36 + 4) FLOPs, got={flops}",
        )
        mean_length_approximation = 8 * estimate_flops_per_token(
            qwen3_moe_config(), (6.0 + 2.0) / 2.0
        )
        self.assertLess(
            mean_length_approximation,
            flops,
            "averaging sequence lengths must underestimate attention work, "
            f"got={mean_length_approximation} against {flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_unsupported_model_type_returns_none(self) -> None:
        """Return ``None`` instead of guessing for an unsupported model family.

        Feature: Analytical FLOPs step estimation.
        Description: Estimate a step for a configuration declaring ``model_type='llama'``.
        Expectation: No estimate is produced.
        """
        flops = estimate_step_flops(SimpleNamespace(model_type="llama"), 8.0, 40.0)
        self.assertIsNone(
            flops,
            f"unsupported model families must not report FLOPs, got={flops!r}",
        )


if __name__ == "__main__":
    unittest.main()
