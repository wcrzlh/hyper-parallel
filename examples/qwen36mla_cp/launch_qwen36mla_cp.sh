#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Qwen3.6-35B-A3B-kvfirst-mla 长序列 CP 训练启动脚本
# 模型: Qwen3.6-35B-A3B-kvfirst-mla-gkd-thinking (V2 SFT 基座)
# 数据: sft_hybrid_long_16k_plus.jsonl (14963 条, 16K-44K tokens)
# 拓扑: 16 chips, CP=2, FSDP=8, EP=1
# CP: full_attn→sdpa_hf_ulysses, linear_attn→gdn_ulysses
# ============================================================

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_DIR="/data3/w00842045/output/qwen35mla-cp-train"
CONFIG_FILE="${SCRIPT_DIR}/train_qwen35mla_cp.yaml"

export HYPER_PARALLEL_PLATFORM=${HYPER_PARALLEL_PLATFORM:-torch}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}

NPROC=${NPROC:-16}

mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

torchrun \
    --nproc_per_node=${NPROC} \
    --rdzv_id=qwen35mla_cp \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:29500 \
    --module examples.qwen3_8.train_text \
    "${CONFIG_FILE}" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
