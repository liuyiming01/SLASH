CUDA=0,1
DEFAULT_BATCH_SIZE=1
DEFAULT_MAX_TOKENS=1024
OUTPUT_DIR=./Results1.0

TASKS=(cycle connectivity hamilton substructure bipartite flow shortest topology triangle)
MODELS=(
  GraphWiz/LLaMA2-7B-DPO
  GraphWiz/Mistral-7B-RFT
  GraphWiz/LLaMA2-13B-DPO
)
GAMMA_VALUES=(0.9)


SELECT_ROOT="SLASH/outputs/final_select/select_layer/GraphWiz"
for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH")"

  for T in "${TASKS[@]}"; do
    MAX_TOKENS="$DEFAULT_MAX_TOKENS"

    CUDA_VISIBLE_DEVICES=$CUDA python evaluate_nlg.py \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --batch_size "$DEFAULT_BATCH_SIZE" \
      --max_tokens "$MAX_TOKENS" \
      --tasks "$T"

    CFG="${SELECT_ROOT}/"${MODEL_NAME}"/select_results/${T}_per_layer_intersection.json"
    if [[ -f "$CFG" ]]; then
      for GAMMA in "${GAMMA_VALUES[@]}"; do
        CUDA_VISIBLE_DEVICES=$CUDA python evaluate_nlg.py \
          --model_path "$MODEL_PATH" \
          --output_dir "$OUTPUT_DIR" \
          --batch_size "$DEFAULT_BATCH_SIZE" \
          --max_tokens "$MAX_TOKENS" \
          --tasks "$T" \
          --layer_head_config_path "$CFG" \
          --gamma "$GAMMA"
      done
    else
      echo "[skip] missing config: $CFG"
    fi

  done
done