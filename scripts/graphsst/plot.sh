set -euo pipefail

REPO_ROOT="/home/lym/LLM-Research/Attention/Graph_Attention/src/GraphLens"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=1,2,3

MODEL_PATH="/home/lym/data1/LLM-model/meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_DIR="/home/lym/data1/Datasets/Graph-SST"

OUT_DIR="${REPO_ROOT}/outputs/attn_viz"

TASKS=("Graph-SST2" "Graph-SST5" "Graph-Twitter")

for TASK in "${TASKS[@]}"; do
  python -m graphlens.viz.plot \
    --model_path "${MODEL_PATH}" \
    --task_name "${TASK}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_DIR}" \
    --input_column input_prompt \
    --split test \
    --max_seq_len 1600 \
    --sim_metric concentration \
    --binarize_method topk \
    --plot_mode layer \
    --k_median 10 \
    --plot_topk 1
done