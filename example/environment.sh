export HF_HOME=/mnt/nfs/zyxing/.cache/huggingface 
export http_proxy=http://121.250.209.147:7890
export https_proxy=http://121.250.209.147:7890
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen2.5-7B-Instruct  --port 8001
conda activate vlmeval

NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 python run.py --data benchmark_all_choice --model Qwen3-VL-8B-Instruct --verbose --judge gpt-4o

rm -f outputs/Qwen3-VL-8B-Instruct/Qwen3-VL-8B-Instruct_benchmark_all_choice*     

  │ SmolVLM-256M
  │ SmolVLM-500M
  │ Moondream2
  │ MiniCPM-V-2

# ============================================================
# VLM Model Evaluation Commands
# ============================================================

# ============================================================
# Auto GPU Selection
# Finds GPUs with < 2000 MiB used memory (idle), prints them,
# and sets CUDA_VISIBLE_DEVICES automatically.
# Usage: source environment.sh  (to export into current shell)
#        bash environment.sh    (sets CUDA_VISIBLE_DEVICES for child commands below)
# ============================================================
get_free_gpus() {
    local threshold=${1:-1000}  # MiB threshold, default 1000
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v t="$threshold" '$2 < t {printf "%s,", $1}' \
        | sed 's/,$//'
}

FREE_GPUS=$(get_free_gpus 1000)
if [ -z "$FREE_GPUS" ]; then
    echo "[GPU] No idle GPUs found (all have >= 1000 MiB used). Check nvidia-smi."
    echo "[GPU] Falling back to all GPUs: 0,1,2,3,4,5,6,7"
    FREE_GPUS="0,1,2,3,4,5,6,7"
else
    echo "[GPU] Free GPUs (< 1000 MiB used): $FREE_GPUS"
fi
export CUDA_VISIBLE_DEVICES="$FREE_GPUS"

BASE_CMD="NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 python run.py --data benchmark_all_choice --verbose --judge gpt-4o"
# ---------- Qwen3-VL (replaces Qwen2-VL) ----------
# $BASE_CMD --model Qwen3-VL-2B-Instruct
# $BASE_CMD --model Qwen3-VL-4B-Instruct
eval $BASE_CMD --model Qwen3-VL-8B-Instruct
# $BASE_CMD --model Qwen3-VL-32B-Instruct
# $BASE_CMD --model Qwen3-VL-30B-A3B-Instruct
# $BASE_CMD --model Qwen3-VL-235B-A22B-Instruct

# ---------- Qwen2.5-VL (replaces Qwen-VL / Qwen-VL-Chat) ----------
eval $BASE_CMD --model Qwen2.5-VL-3B-Instruct
$BASE_CMD --model Qwen2.5-VL-7B-Instruct
# $BASE_CMD --model Qwen2.5-VL-32B-Instruct
# $BASE_CMD --model Qwen2.5-VL-72B-Instruct

# ---------- InternVL2 ----------
eval $BASE_CMD --model InternVL2-8B
# $BASE_CMD --model InternVL2-26B
# $BASE_CMD --model InternVL2-76B

# ---------- InternVL3 (replaces InternVL-Chat-V1-5) ----------
# $BASE_CMD --model InternVL3-8B
eval $BASE_CMD --model InternVL3-14B
# $BASE_CMD --model InternVL3-38B
# $BASE_CMD --model InternVL3-78B

# ---------- Claude 3.7 Sonnet (replaces Claude 3.5 Sonnet, API) ----------
$BASE_CMD --model Claude3-7V_Sonnet

# ---------- GPT-4o / GPT-4o-mini (API) ----------
$BASE_CMD --model GPT4o_20241120
$BASE_CMD --model GPT4o_MINI

# ---------- Gemini-2.5-pro (replaces Gemini-1.5-pro, API) ----------
$BASE_CMD --model GeminiPro2-5

# ---------- Mini-Gemini-34B-HD / 7B-HD (no newer version) ----------
$BASE_CMD --model MGM_7B

# ---------- MiniCPM-V-4_5 (replaces MiniCPM-V 2.5) ----------
$BASE_CMD --model MiniCPM-V-4_5

# ---------- CogVLM2 (no newer version) ----------
$BASE_CMD --model cogvlm2-llama3-chat-19B

# ---------- Cambrian-34B / 8B (no newer version) ----------
$BASE_CMD --model cambrian_8b
$BASE_CMD --model cambrian_34b

# ---------- SliME-13B / 8B (no newer version) ----------
$BASE_CMD --model Slime-8B
$BASE_CMD --model Slime-13B

# ---------- Monkey / MiniMonkey (no newer version) ----------
$BASE_CMD --model monkey
$BASE_CMD --model monkey-chat
$BASE_CMD --model minimonkey

# ---------- mPLUG-Owl3 (replaces mPLUG-Owl2) ----------
$BASE_CMD --model mPLUG-Owl3

# ---------- DeepSeek-VL2 (replaces DeepSeek-VL) ----------
$BASE_CMD --model deepseek_vl2

# ---------- Yi-VL-34B (no newer version) ----------
$BASE_CMD --model Yi_VL_34B
$BASE_CMD --model Yi_VL_6B

# ---------- LLaVA-Next ----------
$BASE_CMD --model llava_next_72b
$BASE_CMD --model llava_next_llama3

# ---------- ShareGPT4V ----------
$BASE_CMD --model sharegpt4v_7b
$BASE_CMD --model sharegpt4v_13b

# ---------- LLaVA-OneVision (newer) ----------
$BASE_CMD --model llava_onevision_qwen2_7b_ov
$BASE_CMD --model llava_onevision_qwen2_72b_ov

# ---------- InstructBLIP (no newer version) ----------
$BASE_CMD --model instructblip_7b
$BASE_CMD --model instructblip_13b

# ---------- IDEFICS-9B ----------
$BASE_CMD --model idefics_9b_instruct

# ---------- Idefics3 (newer) ----------
$BASE_CMD --model Idefics3-8B-Llama3

# ---------- VisCoT (CoVT, no newer version) ----------
$BASE_CMD --model CoVT-7B-seg



# VLM 配置：所有 VLMs 都在 vlmeval/config.py 中配置。对于某些 VLMs（如 MiniGPT-4、LLaVA-v1-7B），需要额外的配置（在配置文件中配置代码 / 模型权重根目录）。在评估时，你应该使用 vlmeval/config.py 中 supported_VLM 指定的模型名称来选择 VLM。确保在开始评估之前，你可以成功使用 VLM 进行推理，使用以下命令 vlmutil check {MODEL_NAME}。

# 注：对于Qwen-VL系列模型（Qwen-VL, Qwen2-VL, Qwen2.5-VL），vlmeval/config.py 中所指定的像素数量上下界如下：

# min_pixels=1280 * 28 * 28,
# max_pixels=16384 * 28 * 28,
# 其中，1280为Qwen官方为平衡性能、计算资源与内存的推荐最大值，而16384为模型输入的理论最大值。这种设定对于部分需要高分辨率的视觉任务（如文档理解）有着积极的作用。但考虑这一设定并没有实际的依据，如果需要与官方的设定对齐，可以去掉这两个数值，或是设置为以下来自Qwen官方demo的数值：

# min_pixels=256 * 28 * 28,
# max_pixels=1280 * 28 * 28,
# 第2步 评测