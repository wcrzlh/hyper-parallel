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
"""Model-owned MTP construction through the normal HyperAutoModel pipeline."""

from typing import Any

from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models import (
    HyperAutoModelForCausalLM,
    HyperAutoModelForImageTextToText,
)
from hyper_parallel.models.qwen3_5_moe.adapter.mtp_runtime import mtp_distributed_setup
from hyper_parallel.models.qwen3_5_moe.modeling_mtp import (
    QwenMTPForCausalLM,
    QwenMTPForConditionalGeneration,
)


class HyperAutoMTPForCausalLM(HyperAutoModelForCausalLM):
    """Keep generic construction free of model-family switches."""

    @classmethod
    def _from_config_parent_class(cls, config: Any, **kwargs: Any) -> Any:
        """Construct only the native text architecture covered by this adapter."""
        if getattr(config, "auto_map", None):
            raise NotImplementedError("Qwen MTP text construction does not cover remote/custom architectures")
        # Match AutoModelForCausalLM: a released VLM checkpoint may carry the
        # native text decoder configuration under text_config.
        if config.model_type == "qwen3_5_moe":
            config = config.get_text_config()
        if config.model_type != "qwen3_5_moe_text":
            raise NotImplementedError("Qwen MTP requires the native MoE text configuration")
        return QwenMTPForCausalLM._from_config(config, **kwargs)


class HyperAutoMTPForImageTextToText(HyperAutoModelForImageTextToText):
    """Build the complete VLM/MTP tree before distributed infrastructure."""

    @classmethod
    def _from_config_parent_class(cls, config: Any, **kwargs: Any) -> Any:
        """Retain the native visual architecture rather than guessing remote-code structure."""
        if config.model_type != "qwen3_5_moe" or getattr(config, "auto_map", None):
            raise NotImplementedError("Qwen MTP VLM construction does not cover remote/custom architectures")
        return QwenMTPForConditionalGeneration._from_config(config, **kwargs)


def build_qwen_mtp(pretrained_model_name_or_path: str, *, distributed_setup: Any = None,
                   multimodal: bool = False, **kwargs: Any) -> Any:
    """Build an opt-in Qwen MoE MTP model using config-first HF weight loading.

    Args:
        pretrained_model_name_or_path: Native HF checkpoint, including MTP tensors.
        distributed_setup: Normalized sharding and replacement contracts.
        multimodal: Keep the visual backbone and conditional-generation task.
        **kwargs: Options for the normal HyperAutoModel build pipeline.

    Returns:
        Fully built model with native parameter names and MTP training support.
    """
    setup = distributed_setup or DistributedSetup()
    mesh = setup.mesh_context
    if mesh.tp_size != 1 or mesh.pp_size != 1 or mesh.sequence_parallel or mesh.loss_parallel:
        raise NotImplementedError("MTP requires TP=PP=1 and sequence/loss parallelism disabled")
    if kwargs.get("quantization_config") is not None or kwargs.get("codegen"):
        raise NotImplementedError("MTP quantized construction and codegen are not qualified")
    compile_config = kwargs.get("compile_config")
    compile_enabled = (compile_config.get("enabled", False) if isinstance(compile_config, dict)
                       else getattr(compile_config, "enabled", False))
    if compile_enabled:
        raise NotImplementedError("MTP graph compilation is not qualified; use compile.enabled=false")
    kwargs["force_hf"] = True
    auto = HyperAutoMTPForImageTextToText if multimodal else HyperAutoMTPForCausalLM
    model = auto.from_pretrained(
        pretrained_model_name_or_path, distributed_setup=mtp_distributed_setup(setup), **kwargs,
    )
    if not getattr(model, "_hp_mtp_enabled", False):
        raise NotImplementedError("MTP requires unquantized config-first model construction")
    model._hp_mtp_cp_group = mesh.cp_mesh.get_group() if mesh.cp_size > 1 else None
    if multimodal:
        from hyper_parallel.models.qwen3_5_moe.adapter.vlm import (  # pylint: disable=import-outside-toplevel
            install_qwen_vlm_cp_forward,
        )
        install_qwen_vlm_cp_forward(model, cp_enabled=mesh.cp_size > 1)
    return model
