#!/bin/bash

# ==========================================
# 1. 读取命令行参数 (例如: ./run.sh 4)
# ==========================================
# 如果没有传入参数，默认使用 4
Q_NUM=${1:-2}

# 验证输入是否为 1, 2, 3, 4, 5 中的一个（防止输错）
if [[ ! "$Q_NUM" =~ ^[1-5]$ ]]; then
    echo "错误: 请输入有效的 Q 编号 (1, 2, 3, 4, 5)"
    echo "用法: $0 [编号]"
    echo "示例: $0 1"
    exit 1
fi

echo "================================================="
echo "🚀 正在运行 Benchmark Q${Q_NUM} 的评测任务..."
echo "================================================="

# Global generation params (injected into all local models that accept them).
# Models already using do_sample=False (InternVL, LLaVA family, Cambrian, Monkey, Gemma3)
# are unaffected; this mainly fixes Qwen3/Qwen3.5 which default to sampling mode.
export VLMEVAL_TEMPERATURE=0
export VLMEVAL_TOP_P=1
export VLMEVAL_TOP_K=1

# ==========================================
# 2. 动态替换 BASE_DATA_ARGS 中的 Q 编号
# ==========================================

# todo:
#  1. remove judge?
#  2. change output path to relative path
#  3. also update run_benchmark_dev.sh
BASE_DATA_ARGS="--data benchmark_q${Q_NUM} --verbose --judge gpt-4o --work-dir Q${Q_NUM}/outputs"
LOG_DIR="logs/q${Q_NUM}_$(date +%Y%m%d_%H%M%S)" # 日志文件夹也加上 Q 编号，方便区分
mkdir -p "$LOG_DIR"

run_in_env() {
    local conda_env="$1"
    local model="$2"
    local logfile="$LOG_DIR/${model}.log"
    echo ""
    echo "====== [$conda_env] $model  →  $logfile ======"
    conda run -n "$conda_env" --no-capture-output bash -c \
        "NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 python run.py $BASE_DATA_ARGS ${EXTRA_ARGS:-} --model '$model'" \
        2>&1 | tee "$logfile"
    echo "[EXIT ${PIPESTATUS[0]}] $model" | tee -a "$logfile"
}

run_in_env_vllm() {
    local conda_env="$1"
    local model="$2"
    local logfile="$LOG_DIR/${model}.log"
    echo ""
    echo "====== [$conda_env] $model (vLLM)  →  $logfile ======"
    conda run -n "$conda_env" --no-capture-output bash -c \
        "NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 VLLM_WORKER_MULTIPROC_METHOD=spawn python run.py $BASE_DATA_ARGS ${EXTRA_ARGS:-} --use-vllm --model '$model'" \
        2>&1 | tee "$logfile"
    echo "[EXIT ${PIPESTATUS[0]}] $model" | tee -a "$logfile"
}

# Run models in parallel on the same GPU.
# Stagger starts by 15s to avoid simultaneous NFS reads and memory-allocation spikes.
# Usage: run_parallel_group "env1:model1" "env2:model2" ...
run_parallel_group() {
    local pids=()
    for spec in "$@"; do
        local env="${spec%%:*}"
        local model="${spec#*:}"
        run_in_env "$env" "$model" &
        pids+=($!)
        sleep 15
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
}

run_parallel_vllm_group() {
    local pids=()
    for spec in "$@"; do
        local env="${spec%%:*}"
        local model="${spec#*:}"
        run_in_env_vllm "$env" "$model" &
        pids+=($!)
        sleep 15
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
}

# ──────────────────────────────────────────────────────────────────────────────
# VRAM budget (80 GB H800, BF16 estimates incl. activation overhead)
#   6B  ≈ 14 GB │  7B ≈ 16 GB │  8B ≈ 18 GB │  9B ≈ 20 GB
#  12B  ≈ 24 GB │ 13B ≈ 30 GB │ 14B ≈ 32 GB │ 27B ≈ 56 GB │ 34B ≈ 78 GB
# vLLM gpu_utils → VRAM:
#   Qwen3-VL-8B  0.30 → 24 GB │ Qwen2.5-VL-3B  0.35 → 28 GB  (pair = 52 GB)
#   Qwen2.5-VL-7B 0.44 → 35 GB │ Gemma3-12B    0.44 → 35 GB  (pair = 70 GB)
#   Qwen3.5-27B  0.90 → 72 GB  (solo, 27B weights alone need ~54 GB)
# ──────────────────────────────────────────────────────────────────────────────

# ---------- Group A: InternVL3-14B + Slime-13B (32+30=62 GB) ----------
run_parallel_group \
    "vlmeval_slime:Slime-13B" \

run_parallel_group \
    "InternVL:InternVL3-14B" 

# # ---------- Group D: cambrian_8b + Slime-8B + idefics2_8b (18+18+18=54 GB) ----------
run_parallel_group \
    "cambrain:cambrian_8b" \

un_parallel_group \
    "vlmeval_slime:Slime-8B" 

# # ---------- Group E: Idefics3 + MGM_7B (18+16=34 GB; Gemma3-12B moved to vLLM solo) ----------
run_parallel_group \
    "mgm:MGM_7B" \

# # ---------- vLLM Pair 1: Qwen3-VL-8B (24 GB) + Qwen2.5-VL-3B (28 GB) = 52 GB ----------
run_parallel_vllm_group \
    "qwen:Qwen3-VL-8B-Instruct" \

run_parallel_vllm_group \
    "qwen:Qwen2.5-VL-3B-Instruct" \

# # ---------- vLLM Pair 2: Qwen2.5-VL-7B (35 GB) + Gemma3-12B (35 GB) = 70 GB ----------
run_parallel_vllm_group \
    "qwen:Qwen2.5-VL-7B-Instruct" \
    # "gemma:Gemma3-12B"

# # # ---------- Group B: sharegpt4v_13b + instructblip_13b (30+30=60 GB) ----------
run_parallel_group \
    "sharegpt:sharegpt4v_13b" \
#     # "instructblip:instructblip_13b"

# # # ---------- Solo: Yi_VL_34B (~78 GB) ----------
run_in_env yi Yi_VL_34B

#     # #---------- Pair: deepseek_vl2 + Yi_VL_6B (56+14=70 GB) ----------
run_parallel_group \
    "yi:Yi_VL_6B" \

# EXTRA_ARGS="--reuse" run_parallel_vllm_group \
#     "qwen:Qwen3-VL-4B-Thinking" \

# # # ---------- Group F: mPLUG-Owl3 + monkey + monkey-chat + llava_onevision (16+16+16+16=64 GB) ----------
run_parallel_group \
    "monkey:monkey" \

run_parallel_group \
    "monkey:monkey-chat" 

# run_parallel_group \
#     "mplug:mPLUG-Owl3" \
#     "llava_one:llava_onevision_qwen2_7b_ov"

# EXTRA_ARGS="--reuse" run_parallel_group \
#     "deepseek:deepseek_vl2" \

# # # ---------- Group D: cambrian_8b + Slime-8B + idefics2_8b (18+18+18=54 GB) ----------
run_parallel_group \
    "IDEFICS:idefics2_8b" \

run_parallel_group \
    "IDEFICS:Idefics3-8B-Llama3" \

# # ---------- Group C: GLM 9B + MiniCPM + InternVL2-8B (20+18+18=56 GB) ----------
run_parallel_group \
    "InternVL:InternVL2-8B" \
#     "MiniCPM-V-4_5:MiniCPM-V-4_5" \
#     "glm4:GLM4_1VThinking-9b"

run_parallel_group \
    "InternVL:InternVL2-2B" \

# # ---------- Group G: instructblip_7b + sharegpt4v_7b + CoVT-7B-seg (16+16+16=48 GB) ----------
run_parallel_group \
    "sharegpt:sharegpt4v_7b" \
    # "covt:CoVT-7B-seg" \
    # "instructblip:instructblip_7b" \

# # # ---------- Solo: Qwen3.5-27B (~72 GB, 27B weights alone need ~54 GB) ----------
# EXTRA_ARGS="--reuse" run_in_env_vllm qwen Qwen3.5-27B

# ---------- 赶 DDL 极速 Thinking 组 (2B + 3B + 4B) ----------
# 预计显存占用极小，且均带有极强的推理能力
EXTRA_ARGS="--reuse" run_parallel_vllm_group \
    "qwen:Qwen3-VL-2B-Thinking" \