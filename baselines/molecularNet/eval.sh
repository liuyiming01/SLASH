CUDA=0,3
BATCH_SIZE=2
MAX_NEW_TOKENS=8
LIMIT=400
OUTPUT_DIR=./outputs/Results1.1

SCRIPT_PATH="$(realpath "$0")"
mkdir -p "$OUTPUT_DIR"
cp "$SCRIPT_PATH" "$OUTPUT_DIR/$(basename "$SCRIPT_PATH")"

DATA_DIR="/home/lym/data1/Datasets/ChemLLMBench/data/property_prediction"
PROMPT_PATH="./prompt/property_prediction_graph_prompt3.1.txt"

TASKS=(BACE BBBP ClinTox HIV Tox21)

MODELS=(
  /home/lym/data1/LLM-model/meta-llama/Meta-Llama-3.1-8B-Instruct
  # /home/lym/data1/LLM-model/meta-llama/Llama-3.2-3B-Instruct
  # /home/lym/data1/LLM-model/Qwen/Qwen3-8B
  # /home/lym/data1/LLM-model/Qwen/Qwen3-4B
  # /home/lym/data1/LLM-model/meta-llama/Llama-3.2-3B-Instruct
  # /home/lym/data1/LLM-model/Qwen/Qwen3-14B
  # /home/lym/data1/LLM-model/meta-llama/Llama-2-7b-chat-hf
)

SELECT_ROOT="/home/lym/LLM-Research/SLASH/outputs/final_select/select1.0/MolecularNet"

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH")"
  SELECT_DIR="${SELECT_ROOT}/${MODEL_NAME}/select_results"

  for TASK in "${TASKS[@]}"; do
    # 1) Vanilla
    CUDA_VISIBLE_DEVICES=$CUDA python eval_mol.py \
      --task "$TASK" \
      --model_path "$MODEL_PATH" \
      --data_dir "$DATA_DIR" \
      --prompt_path "$PROMPT_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --split sample \
      --limit 1000 \
      --sample_num "$LIMIT" \
      --preferred_min_edges 40 \
      --hard_max_edges 100 \
      --seed 42 \
      --batch_size "$BATCH_SIZE" \
      --max_new_tokens "$MAX_NEW_TOKENS"

    # 2) SLASH
    # CFG="${SELECT_DIR}/${TASK}_per_layer_intersection.json"
    # if [[ -f "$CFG" ]]; then
    #   for GAMMA in 0.4; do
    #     CUDA_VISIBLE_DEVICES=$CUDA python eval_mol.py \
    #       --task "$TASK" \
    #       --model_path "$MODEL_PATH" \
    #       --data_dir "$DATA_DIR" \
    #       --prompt_path "$PROMPT_PATH" \
    #       --output_dir "$OUTPUT_DIR" \
    #       --split sample \
    #       --limit 1000 \
    #       --sample_num "$LIMIT" \
    #       --preferred_min_edges 60 \
    #       --hard_max_edges 150 \
    #       --seed 42 \
    #       --batch_size "$BATCH_SIZE" \
    #       --max_new_tokens "$MAX_NEW_TOKENS" \
    #       --layer_head_config_path "$CFG" \
    #       --gamma "$GAMMA"
    #   done
    # else
    #   echo "[skip] missing config: $CFG"
    # fi
  done
done