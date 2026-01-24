REPO_ROOT="/home/lym/LLM-Research/SLASH"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="1,2"

DATA_DIR="/home/lym/data1/Datasets/ChemLLMBench/data/property_prediction"

PROMPT_PATH="${REPO_ROOT}/baselines/molecularNet/prompt/property_prediction_graph_prompt3.1.txt"

OUT_DIR="${REPO_ROOT}/outputs/attn_viz"

PLOT_MODE="layer"          # layer | head
SIM_METRIC="concentration" # concentration | gradient
BINARIZE="topk"            # topk | threshold
MAX_SEQ_LEN=1600
SPLIT="sample"

TASK_NAME="Mol_BACE"

K_MEDIAN=10
PLOT_TOPK=1
PREFERRED_MIN_EDGES=40
HARD_MAX_EDGES=100

MODELS=(
  /home/lym/data1/LLM-model/meta-llama/Meta-Llama-3.1-8B-Instruct
  # /home/lym/data1/LLM-model/Qwen/Qwen3-8B
  # /home/lym/data1/LLM-model/Qwen/Qwen3-4B
  # /home/lym/data1/LLM-model/meta-llama/Llama-3.2-3B-Instruct
  # /home/lym/data1/LLM-model/Qwen/Qwen3-14B
  # /home/lym/data1/LLM-model/meta-llama/Llama-2-7b-chat-hf
)

for MODEL_PATH in "${MODELS[@]}"; do
  echo "=== MolecularNet plot | model=${MODEL_PATH} | task=${TASK_NAME} ==="
  python -m graphlens.viz.plot \
    --model_path "${MODEL_PATH}" \
    --task_name "${TASK_NAME}" \
    --data_dir "${DATA_DIR}" \
    --prompt_path "${PROMPT_PATH}" \
    --split "${SPLIT}" \
    --output_dir "${OUT_DIR}" \
    --input_column input_prompt \
    --plot_mode "${PLOT_MODE}" \
    --sim_metric "${SIM_METRIC}" \
    --binarize_method "${BINARIZE}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --k_median "${K_MEDIAN}" \
    --plot_topk "${PLOT_TOPK}" \
    --preferred_min_edges "${PREFERRED_MIN_EDGES}" \
    --hard_max_edges "${HARD_MAX_EDGES}"
done
