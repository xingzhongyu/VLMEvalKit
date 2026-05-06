#!/bin/bash

Q_NUM=${1:-2}

if [[ ! "$Q_NUM" =~ ^[1-5]$ ]]; then
    echo "错误: 请输入有效的 Q 编号 (1, 2, 3, 4, 5)"
    echo "用法: $0 [编号]"
    echo "示例: $0 1"
    exit 1
fi

echo "================================================="
echo "🚀 正在运行 Benchmark Q${Q_NUM} DEV 的评测任务..."
echo "================================================="

export VLMEVAL_TEMPERATURE=0
export VLMEVAL_TOP_P=1
export VLMEVAL_TOP_K=1

BASE_DATA_ARGS="--data benchmark_q${Q_NUM}_dev --verbose --judge gpt-4o --work-dir Q${Q_NUM}/outputs_dev"
LOG_DIR="logs_dev/q${Q_NUM}_$(date +%Y%m%d_%H%M%S)"
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

run_parallel_group \
    "vlmeval_slime:Slime-13B" \

run_parallel_group \
    "InternVL:InternVL3-14B"

run_parallel_group \
    "cambrain:cambrian_8b" \

un_parallel_group \
    "vlmeval_slime:Slime-8B"

run_parallel_group \
    "mgm:MGM_7B" \

run_parallel_vllm_group \
    "qwen:Qwen3-VL-8B-Instruct" \

run_parallel_vllm_group \
    "qwen:Qwen2.5-VL-3B-Instruct" \

run_parallel_vllm_group \
    "qwen:Qwen2.5-VL-7B-Instruct" \

run_parallel_group \
    "sharegpt:sharegpt4v_13b" \

run_in_env yi Yi_VL_34B

run_parallel_group \
    "yi:Yi_VL_6B" \

run_parallel_group \
    "monkey:monkey" \

run_parallel_group \
    "monkey:monkey-chat"

run_parallel_group \
    "IDEFICS:idefics2_8b" \

run_parallel_group \
    "IDEFICS:Idefics3-8B-Llama3" \

run_parallel_group \
    "InternVL:InternVL2-8B" \

run_parallel_group \
    "InternVL:InternVL2-2B" \

run_parallel_group \
    "sharegpt:sharegpt4v_7b" \

EXTRA_ARGS="--reuse" run_parallel_vllm_group \
    "qwen:Qwen3-VL-2B-Thinking" \
