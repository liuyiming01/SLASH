set -euo pipefail

REPO_ROOT="/home/lym/LLM-Research/Attention/Graph_Attention/src/GraphLens"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=1,2,3

DATA_DIR="/home/lym/data1/Datasets/GraphWiz/GraphInstruct-Test"

OUT_DIR="${REPO_ROOT}/outputs/scoring"

python -m graphlens.scoring \
  --model_path /home/lym/data1/LLM-model/meta-llama/Meta-Llama-3.1-8B-Instruct \
  --task_name GraphWiz \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUT_DIR}" \
  --input_column input_prompt \
  --sample_num 50 \
  --min_edges 70 \
  --max_edges 130 \
  --score_mode per_layer \
  --sim_metric concentration \
  --binarize_method topk