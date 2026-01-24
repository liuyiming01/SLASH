REPO_ROOT="/home/lym/LLM-Research/SLASH"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="0,3"

# MolecularNet CSV dir (should contain BACE.csv, BBBP.csv, etc.)
DATA_DIR="/home/lym/data1/Datasets/ChemLLMBench/data/property_prediction"
PROMPT_PATH="${REPO_ROOT}/baselines/molecularNet/prompt/property_prediction_graph_prompt3.1.txt"
OUT_DIR="${REPO_ROOT}/outputs/final_select/select1.0/MolecularNet"

MODE="per_layer"
SAMPLE_NUM=30
MAX_SEQ_LEN=1600
HARD_MAX_EDGES=150

BINARIZE_METHOD="topk"
SIM_METRIC="concentration"

MODELS=(
  /home/lym/data1/LLM-model/meta-llama/Meta-Llama-3.1-8B-Instruct
  # /home/lym/data1/LLM-model/Qwen/Qwen3-8B
  # /home/lym/data1/LLM-model/meta-llama/Llama-3.2-3B-Instruct
  # /home/lym/data1/LLM-model/Qwen/Qwen3-4B
  # /home/lym/data1/LLM-model/Qwen/Qwen3-14B
  # /home/lym/data1/LLM-model/meta-llama/Llama-2-7b-chat-hf
)

# MolecularNet 任务名
MN_TASKS=(
  "BACE"
  "BBBP"
  "ClinTox"
  "HIV"
  "Tox21"
)

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH")"
  OUT="${OUT_DIR}/${MODEL_NAME}"

  for t in "${MN_TASKS[@]}"; do
    SCORING_OUT="${OUT}/scoring_results/${t}"
    ENTROPY_OUT="${OUT}/entropy_results/${t}"
    SELECT_OUT="${OUT}/select_results"
    mkdir -p "${SCORING_OUT}" "${ENTROPY_OUT}" "${SELECT_OUT}"

    echo "[run_select] (MolecularNet) model=${MODEL_NAME} task=${t} -> ${OUT}"

    python -m slash.entropy \
      --model_path "${MODEL_PATH}" --task_name "Mol_${t}" --data_dir "${DATA_DIR}" \
      --output_dir "${ENTROPY_OUT}" --input_column "input_prompt" \
      --prompt_path "${PROMPT_PATH}" \
      --sample_num "${SAMPLE_NUM}" --max_seq_len "${MAX_SEQ_LEN}" \
      --preferred_min_edges 40 --hard_max_edges 100 \
      --score_mode "${MODE}" --split "sample" \

    python -m slash.scoring \
      --model_path "${MODEL_PATH}" --task_name "Mol_${t}" --data_dir "${DATA_DIR}" \
      --output_dir "${SCORING_OUT}" --input_column "input_prompt" \
      --prompt_path "${PROMPT_PATH}" \
      --sample_num "${SAMPLE_NUM}" --max_seq_len "${MAX_SEQ_LEN}" \
      --preferred_min_edges 40 --hard_max_edges 100 \
      --score_mode "${MODE}" --sim_metric "${SIM_METRIC}"  --split "sample" \
      --binarize_method "${BINARIZE_METHOD}"

    SCORE_DIR="${SCORING_OUT}/${MODE}_${BINARIZE_METHOD}_${SIM_METRIC}_std"
    python -m slash.final_select \
      --mode "${MODE}" \
      --json_entropy "${ENTROPY_OUT}/${t}_entropy_${MODE}/${t}_selected_layers_middle_peak_entropy.json" \
      --json_scoring "${SCORE_DIR}/${t}_selected_layers_auto_scoring.json" \
      --output_json "${SELECT_OUT}/${t}_${MODE}_intersection.json"
  done
done