set -euo pipefail

REPO_ROOT="/home/lym/LLM-Research/Attention/Graph_Attention/GraphLens"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES=2,3,6

GRAPHWIZ_DIR="/home/lym/data1/Datasets/GraphWiz/GraphInstruct-Test"
GRAPHSST_ROOT="/home/lym/data1/Datasets/Graph-SST"   # DIG root
OUT_DIR="../outputs/select4.0"

MODELS_GraphWiz=(
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-13B-DPO
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-13B-RFT
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-7B-DPO
  /home/lym/data1/LLM-model/GraphWiz/Mistral-7B-RFT
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-7B-RFT
)

MODE="per_layer"
SAMPLE_NUM=50
MAX_SEQ_LEN=1600

BINARIZE_METHOD="topk"
SIM_METRIC="concentration"
PRE_THRESHOLD_FRAC=0.1

ALPHA=1.0
SELECT_TOP_FRACTION=0.4   # 交集用 top_40

GW_TASKS=(cycle connectivity hamilton substructure bipartite flow shortest topology triangle)
SST_TASKS=("Graph-SST2" "Graph-SST5" "Graph-Twitter")

contains_elem () {
  local needle="$1"
  shift
  local e
  for e in "$@"; do
    if [[ "$e" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

run_graphwiz_for_model () {
  local MODEL_PATH="$1"
  local MODEL_NAME
  MODEL_NAME="$(basename "$MODEL_PATH")"

  local OUT="${OUT_DIR}/${MODEL_NAME}"
  local SCORING_OUT="${OUT}/scoring_results"
  local ENTROPY_OUT="${OUT}/entropy_results"
  local SELECT_OUT="${OUT}/select_results"
  mkdir -p "${SCORING_OUT}" "${ENTROPY_OUT}" "${SELECT_OUT}"

  echo "[run_select] (GraphWiz) model=${MODEL_NAME} -> ${OUT}"

  python -m graphlens.entropy3 \
    --model_path "${MODEL_PATH}" --task_name "GraphWiz" --data_dir "${GRAPHWIZ_DIR}" \
    --output_dir "${ENTROPY_OUT}" --input_column "input_prompt" \
    --sample_num "${SAMPLE_NUM}" --max_seq_len "${MAX_SEQ_LEN}" --hard_max_edges 150 \
    --score_mode "${MODE}" --alpha "${ALPHA}" --select_top_fraction "${SELECT_TOP_FRACTION}"

  python -m graphlens.scoring \
    --model_path "${MODEL_PATH}" --task_name "GraphWiz" --data_dir "${GRAPHWIZ_DIR}" \
    --output_dir "${SCORING_OUT}" --input_column "input_prompt" \
    --sample_num "${SAMPLE_NUM}" --max_seq_len "${MAX_SEQ_LEN}" --hard_max_edges 150 \
    --score_mode "${MODE}" --sim_metric "${SIM_METRIC}" \
    --binarize_method "${BINARIZE_METHOD}" --pre_threshold_frac "${PRE_THRESHOLD_FRAC}" \
    --select_top_fraction "${SELECT_TOP_FRACTION}"

  local SCORE_DIR="${SCORING_OUT}/${MODE}_${BINARIZE_METHOD}_${SIM_METRIC}_std"
  for t in "${GW_TASKS[@]}"; do
    python -m graphlens.final_select \
      --mode "${MODE}" \
      --json_entropy "${ENTROPY_OUT}/${t}_entropy_${MODE}_alpha${ALPHA}/${t}_selected_layers_middle_peak_entropy.json" \
      --json_scoring "${SCORE_DIR}/${t}_selected_layers_auto_scoring.json" \
      --output_json "${SELECT_OUT}/GraphWiz_${t}_${MODE}_intersection.json"
  done
}

# 2) MODELS_GraphWiz: only run GraphWiz, skip those already in MODELS
for MODEL_PATH in "${MODELS_GraphWiz[@]}"; do
  if contains_elem "${MODEL_PATH}" "${MODELS[@]}"; then
    continue
  fi
  run_graphwiz_for_model "${MODEL_PATH}"
done