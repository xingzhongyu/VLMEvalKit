## 🌟 ManuVision-Bench (Paper Implementation)

This repository is built upon VLMEvalKit and serves as the official evaluation framework for the paper **ManuVision-Bench: A Benchmark for Image-Grounded Manufacturing Reasoning and Supplier Discovery**.

### 🚀 Quick Guide for ManuVision-Bench
- **Environment Requirement**: Please ensure you are using **Python 3.10**.
- **Task Structure**: The folders `Q1`, `Q2`, `Q3`, `Q4`, and `Q5` in this repository correspond to the subtasks defined in our paper.
- **Reproduction**: You can easily reproduce the experiments reported in the paper by running the provided shell script:
  - For local LLM:(you must have them setup, can refer to [vlmevalkit](https://github.com/open-compass/vlmevalkit) for instruction)
    ```bash
    bash all_q.sh
    ```
  - For API:
    ```bash
    python run.py \
      --data benchmark_q1 benchmark_q2 benchmark_q3 benchmark_q4 benchmark_q5 \
      --model gpt-4o-2024-11-20 \
      --work-dir outputs/openai_all \
      --api-nproc 4 \
      --verbose
    ```
    ```bash
    python run.py \
      --data benchmark_q1 benchmark_q2 benchmark_q3 benchmark_q4 benchmark_q5 \
      --model claude-sonnet-4-5-20250929 \
      --work-dir outputs/claude_all \
      --api-nproc 4 \
      --verbose
    ```
- **Dataset Implementation**: The evaluation script and dataset loader specific to our benchmark can be found at `vlmeval/dataset/manufacture_mcq.py`.
- **Evaluation Settings**: To guarantee deterministic and reproducible results, the parameter `do_sample` is strictly set to `False` across all evaluation methods.
- **Data Acquisition**: For detailed instructions on how to acquire the data for tasks Q1~Q5, please refer to the data section in our original paper.
- **Packaging Note**: Each full benchmark folder includes an `images.zip` archive for distribution and upload. After cloning the repository, extract each `images.zip` into the corresponding `images/` folder before running local evaluation.

### 📁 Folder Structure

Full benchmarks and dev benchmarks are separated.

```text
Q1/
  benchmark_q1.tsv
  images/
  images.zip
Q2/
  benchmark_q2.tsv
  images/
  images.zip
Q3/
  benchmark_q3.tsv
  images/
  images.zip
Q4/
  benchmark_q4.tsv
  images/
  images.zip
Q5/
  benchmark_q5.tsv
  images/
  images.zip

dev/
  Q1_dev/
    benchmark_q1_dev.tsv
    images/
  Q2_dev/
    benchmark_q2_dev.tsv
    images/
  Q3_dev/
    benchmark_q3_dev.tsv
    images/
  Q4_dev/
    benchmark_q4_dev.tsv
    images/
  Q5_dev/
    benchmark_q5_dev.tsv
    images/
```

The benchmark loader is implemented in `vlmeval/dataset/manufacture_mcq.py`.
It automatically maps:

- `benchmark_q1` to `benchmark_q5` to the `Q1` to `Q5` folders
- `benchmark_q1_dev` to `benchmark_q5_dev` to the `dev/Q1_dev` to `dev/Q5_dev` folders

For local runs, the loader reads from the extracted `images/` folders. The `images.zip` files are included only as convenient packaging artifacts for dataset release and transfer.

Before running any local benchmark, unzip the packaged image archives:

```bash
cd Q1 && unzip images.zip
cd ../Q2 && unzip images.zip
cd ../Q3 && unzip images.zip
cd ../Q4 && unzip images.zip
cd ../Q5 && unzip images.zip
```

### 🧪 Dev Testing

Before launching a full run, we recommend running the 10-sample dev split for all five tasks first.

- OpenAI API:
  ```bash
  python run.py \
    --data benchmark_q1_dev benchmark_q2_dev benchmark_q3_dev benchmark_q4_dev benchmark_q5_dev \
    --model gpt-4o-2024-11-20 \
    --work-dir outputs_dev/openai_all \
    --api-nproc 4 \
    --verbose
  ```
- Claude API:
  ```bash
  python run.py \
    --data benchmark_q1_dev benchmark_q2_dev benchmark_q3_dev benchmark_q4_dev benchmark_q5_dev \
    --model claude-sonnet-4-5-20250929 \
    --work-dir outputs_dev/claude_all \
    --api-nproc 4 \
    --verbose
  ```

Each dataset is evaluated separately, so the output metrics depend on the task type:

- `Q1` and `Q3`: `ExactMatch` and `OptionF1`
- `Q2`, `Q4`, and `Q5`: `ACC` and `F1_macro`

Results will be written under the corresponding `outputs/` or `outputs_dev/` directory, with one result file per benchmark.

### ⚙️ CLI Arguments

The main entry point is:

```bash
python run.py --data ... --model ... --work-dir ... --api-nproc ...
```

Common arguments:

- `--data`: One or more benchmark names to run. You can pass a single task such as `benchmark_q2`, or multiple tasks such as `benchmark_q1 benchmark_q2 benchmark_q3 benchmark_q4 benchmark_q5`.
- `--model`: The model name to evaluate. This repository supports direct use of API model names such as `gpt-4o-2024-11-20` and `claude-sonnet-4-5-20250929`.
- `--work-dir`: The output directory for predictions, parsed details, and metric files.
- `--api-nproc`: The number of parallel API requests to use during inference. Larger values can make evaluation faster, but may also hit rate limits depending on the provider and account.
- `--verbose`: Print progress and evaluation summaries to the terminal.

You can try any supported model name you like, as long as it is registered in `vlmeval/config.py` and your API key has access to it.

Examples:

- Run a single dev benchmark with OpenAI:
  ```bash
  python run.py \
    --data benchmark_q2_dev \
    --model gpt-4o-2024-11-20 \
    --work-dir outputs_dev/q2_openai \
    --api-nproc 4 \
    --verbose
  ```
- Run a single dev benchmark with Claude:
  ```bash
  python run.py \
    --data benchmark_q2_dev \
    --model claude-sonnet-4-5-20250929 \
    --work-dir outputs_dev/q2_claude \
    --api-nproc 4 \
    --verbose
  ```
- Run all full benchmarks with a different supported model:
  ```bash
  python run.py \
    --data benchmark_q1 benchmark_q2 benchmark_q3 benchmark_q4 benchmark_q5 \
    --model gpt-4.1 \
    --work-dir outputs/gpt41_all \
    --api-nproc 4 \
    --verbose
  ```
