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
"""Shared checkpoint type and layout preparation for Qwen LM and VLM builders."""

from dataclasses import replace
from fnmatch import fnmatchcase
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from transformers import AutoConfig

from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import CP, EP, ModuleShardingSpec


def _resolve_remote_module_replacement(rule: Any, reference: Any) -> Any:
    """Resolve the checkpoint's exact class for one native Qwen replacement."""
    source = rule.module_type
    if source is None or not source.__module__.startswith("transformers.models.qwen3_5_moe."):
        return rule
    types = {type(module) for name, module in reference.named_modules()
             if any(fnmatchcase(name, pattern) for pattern in rule.match)}
    if len(types) != 1 or next(iter(types)).__name__ != source.__name__:
        raise ValueError(f"MLA module replacement does not resolve one {source.__name__}: {rule.match}")
    return replace(rule, module_type=next(iter(types)))


def prepare_remote_model_setup(model_path: str, distributed_setup: Any, options: dict[str, Any], *,
                                auto_model_class: Any) -> None:
    """Resolve MLA checkpoint classes without weakening shared exact-type rules.

    Args:
        model_path: HF checkpoint directory or identifier.
        distributed_setup: Parallel setup whose replacement rules are updated in place.
        options: Model-loading options, including whether remote code is trusted.
        auto_model_class: HF Auto class used by the caller's LM or VLM builder.
    """
    if distributed_setup is None or not options.get("trust_remote_code", False):
        return
    load_options = {key: options[key] for key in ("trust_remote_code", "local_files_only", "revision", "cache_dir")
                    if key in options}
    config = AutoConfig.from_pretrained(model_path, **load_options)
    if not {"Qwen3_6MLAForConditionalGeneration", "Qwen3_6MLAForCausalLM"}.intersection(config.architectures or ()):
        return
    with torch.device("meta"):
        reference = auto_model_class.from_config(config, trust_remote_code=True, dtype=torch.bfloat16)
    distributed_setup.module_replacements = tuple(
        _resolve_remote_module_replacement(rule, reference) for rule in distributed_setup.module_replacements or ()
    )
    # The remote vision classes are not HF planner boundaries. Declare their
    # decoder-CP/EP replication explicitly while retaining FSDP ownership.
    visual = getattr(reference.model, "visual", None)
    if visual is None:
        return
    for name, module in visual.named_modules():
        params = {key: {CP: Replicate(), EP: Replicate()} for key, _ in module.named_parameters(recurse=False)}
        if params:
            fqn = "model.visual" + ("." + name if name else "")
            distributed_setup.plan_overrides.setdefault(fqn, ModuleShardingSpec(params=params))
