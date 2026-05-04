#!/bin/bash

export VLMEVAL_TEMPERATURE=0
export VLMEVAL_TOP_P=1
export VLMEVAL_TOP_K=1

LOG_DIR="logs/exp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

run_in_env() {
    local conda_env="$1"
    local model="$2"
    local data="$3"
    local logfile="$LOG_DIR/${model}_${data}.log"
    echo ""
    echo "====== [$conda_env] $model  data=$data  →  $logfile ======"
    conda run -n "$conda_env" --no-capture-output bash -c \
        "NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 python run.py \
         --data '$data' --verbose --judge gpt-4o --reuse \
         --work-dir /mnt/nfs/zyxing/VLMEvalKit/origin_Q2/outputs_exp \
         --model '$model'" \
        2>&1 | tee "$logfile"
    echo "[EXIT ${PIPESTATUS[0]}] $model $data" | tee -a "$logfile"
}

# Wait until GPU memory usage drops below threshold (default 10 GB used).
wait_gpu_free() {
    local threshold_mb="${1:-10000}"
    echo "Waiting for GPU to be free (used < ${threshold_mb} MB) ..."
    while true; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | head -1)
        if [ "$used" -lt "$threshold_mb" ]; then
            echo "GPU free (${used} MB used), proceeding."
            break
        fi
        echo "  GPU busy (${used} MB used), sleeping 30s ..."
        sleep 30
    done
}

run_parallel_group() {
    local data="$1"; shift
    local pids=()
    for spec in "$@"; do
        local env="${spec%%:*}"
        local model="${spec#*:}"
        run_in_env "$env" "$model" "$data" &
        pids+=($!)
        sleep 15
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
}

run_dataset() {
    local data="$1"
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Running dataset: $data"
    echo "══════════════════════════════════════════════════════"

    # ---------- Solo: Yi_VL_34B (~78 GB) ----------
    wait_gpu_free

    # ---------- Pair: InternVL3-14B + Slime-13B (32+30=62 GB) ----------
    run_parallel_group "$data" \
        "InternVL:InternVL3-14B" \
        "vlmeval_slime:Slime-13B"

     # ---------- Pair: MiniCPM-V-4_5 + InternVL2-8B (18+18=36 GB) ----------
    run_parallel_group "$data" \
        "MiniCPM-V-4_5:MiniCPM-V-4_5" \
        "InternVL:InternVL2-8B"

    # ---------- Pair: sharegpt4v_13b  (30+14=44 GB) ----------
    run_parallel_group "$data" \
        "sharegpt:sharegpt4v_13b" \

   

    run_in_env yi Yi_VL_34B "$data"
    
}

run_dataset exp1_add_manufacturing
# run_dataset exp2_add_fabricated
