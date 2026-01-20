set -euo pipefail

REPO_ROOT="/home/lym/LLM-Research/Attention/Graph_Attention/GraphLens"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=1,2,3

MODELS=(
  /home/lym/data1/LLM-model/Qwen/Qwen3-8B
  /home/lym/data1/LLM-model/meta-llama/Meta-Llama-3.1-8B-Instruct
  /home/lym/data1/LLM-model/Qwen/Qwen3-4B
  /home/lym/data1/LLM-model/meta-llama/Llama-3.2-3B-Instruct
  /home/lym/data1/LLM-model/Qwen/Qwen3-14B
  /home/lym/data1/LLM-model/meta-llama/Llama-2-7b-chat-hf
)

MODEL_PATH="/home/lym/data1/LLM-model/meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_DIR="/home/lym/data1/Datasets/GraphWiz/GraphInstruct-Test"

OUT_DIR="${REPO_ROOT}/outputs/attn_viz"

python -m graphlens.viz.plot \
  --model_path "${MODEL_PATH}" \
  --task_name "GraphWiz" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUT_DIR}" \
  --input_column input_prompt \
  --k_median 10 \
  --plot_topk 1 \
  --preferred_min_edges 60 \
  --hard_max_edges -1 \
  --max_seq_len 1600 \
  --sim_metric concentration \
  --binarize_method topk \
  --plot_mode layer