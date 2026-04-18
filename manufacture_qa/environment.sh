
export http_proxy=http://121.250.209.147:7890
export https_proxy=http://121.250.209.147:7890

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 lm_eval  --model vllm  --model_args pretrained=Qwen/Qwen3-1.7B,trust_remote_code=True,dtype=auto,max_model_len=4096,gpu_memory_utilization=0.7 --tasks manufacture_qa  --num_fewshot 0  --apply_chat_template  --output_path ./eval_logs/manufacture_qa  --log_samples

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=5 lm_eval  --model vllm  --model_args pretrained=Qwen/Qwen3-1.7B,trust_remote_code=True,dtype=auto,max_model_len=4096,gpu_memory_utilization=0.7 --tasks manufacture_reasoning_qa  --num_fewshot 0  --apply_chat_template  --output_path ./eval_logs/manufacture_reasoning_qa  --log_samples
Qwen2.5-3B-Instruct
Qwen3-4B-Instruct

CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
manufacture_reasoning_qa