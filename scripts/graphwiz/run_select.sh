REPO_ROOT="/home/lym/LLM-Research/SLASH"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="1,3"

TASK_NAME="GraphWiz_cycle"
DATA_DIR="/home/lym/data1/Datasets/GraphWiz/GraphInstruct-Test"
OUT_DIR="${REPO_ROOT}/outputs/final_select/select1.0/${TASK_NAME}"

MODELS=(
  /home/lym/data1/LLM-model/meta-llama/Meta-Llama-3.1-8B-Instruct
  # /home/lym/data1/LLM-model/Qwen/Qwen3-8B
  # /home/lym/data1/LLM-model/meta-llama/Llama-3.2-3B-Instruct
  # /home/lym/data1/LLM-model/Qwen/Qwen3-4B
  # /home/lym/data1/LLM-model/Qwen/Qwen3-14B
  # /home/lym/data1/LLM-model/meta-llama/Llama-2-7b-chat-hf
)
MODELS_GraphWiz=(
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-7B
  /home/lym/data1/LLM-model/GraphWiz/Mistral-7B
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-13B
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-7B-DPO
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-13B-DPO
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-7B-RFT
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-13B-RFT
  /home/lym/data1/LLM-model/GraphWiz/Mistral-7B-RFT
)

MODE="per_head"
SAMPLE_NUM=10
MAX_SEQ_LEN=1600
HARD_MAX_EDGES=150

BINARIZE_METHOD="topk"
SIM_METRIC="concentration"

GW_TASKS=(cycle connectivity hamilton substructure bipartite flow shortest topology triangle)

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH")"
  OUT="${OUT_DIR}/${MODEL_NAME}"
  SCORING_OUT="${OUT}/scoring_results"
  ENTROPY_OUT="${OUT}/entropy_results"
  SELECT_OUT="${OUT}/select_results"

  echo "[run_select] (${TASK_NAME}) model=${MODEL_NAME} -> ${OUT}"

  python -m slash.entropy \
    --model_path "${MODEL_PATH}" \
    --task_name "${TASK_NAME}" --data_dir "${DATA_DIR}" --input_column "input_prompt" \
    --output_dir "${ENTROPY_OUT}" \
    --sample_num "${SAMPLE_NUM}" --max_seq_len "${MAX_SEQ_LEN}" --hard_max_edges "${HARD_MAX_EDGES}" \
    --score_mode "${MODE}"

  python -m slash.scoring \
    --model_path "${MODEL_PATH}" \
    --task_name "${TASK_NAME}" --data_dir "${DATA_DIR}" --input_column "input_prompt" \
    --output_dir "${SCORING_OUT}" \
    --sample_num "${SAMPLE_NUM}" --max_seq_len "${MAX_SEQ_LEN}" --hard_max_edges "${HARD_MAX_EDGES}" \
    --score_mode "${MODE}" --sim_metric "${SIM_METRIC}" \
    --binarize_method "${BINARIZE_METHOD}"

  SCORE_DIR="${SCORING_OUT}/${MODE}_${BINARIZE_METHOD}_${SIM_METRIC}_std"
  for t in "${GW_TASKS[@]}"; do
    python -m slash.final_select \
      --mode "${MODE}" \
      --json_entropy "${ENTROPY_OUT}/${t}_entropy_${MODE}/${t}_selected_layers_middle_peak_entropy.json" \
      --json_scoring "${SCORE_DIR}/${t}_selected_layers_auto_scoring.json" \
      --output_json "${SELECT_OUT}/${t}_${MODE}_intersection.json"
  done
done