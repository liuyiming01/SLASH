set -euo pipefail

REPO_ROOT="/home/lym/LLM-Research/Attention/Graph_Attention/src/GraphLens"

# Ensure python can import the package from src/
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Ensure CUDA_VISIBLE_DEVICES is visible to child processes
export CUDA_VISIBLE_DEVICES=1,2,3

GRAPHSST_ROOT="/home/lym/data1/Datasets/Graph-SST"
SST_TASKS=("Graph-SST2" "Graph-SST5" "Graph-Twitter")

OUT_DIR="${REPO_ROOT}/outputs/scoring"

for ds in "${SST_TASKS[@]}"; do
  python -m graphlens.scoring \
    --model_path /home/lym/data1/LLM-model/meta-llama/Meta-Llama-3.1-8B-Instruct \
    --task_name "${ds}" \
    --data_dir "${GRAPHSST_ROOT}" \
    --output_dir "${OUT_DIR}" \
    --input_column input_prompt \
    --sample_num 50 \
    --min_edges 70 \
    --max_edges 130 \
    --score_mode per_layer \
    --sim_metric concentration \
    --binarize_method topk
done