CUDA=3,7
BASE_BATCH_SIZE=4
DEFAULT_MAX_TOKENS=10
OUTPUT_DIR=./Results

SCRIPT_PATH="$(realpath "$0")"
mkdir -p "$OUTPUT_DIR"
cp "$SCRIPT_PATH" "$OUTPUT_DIR/$(basename "$SCRIPT_PATH")"

TASKS=(cycle connectivity hamilton substructure bipartite flow shortest triangle topology)
TASKS1=(cycle connectivity hamilton substructure bipartite)
TASKS2=(flow shortest triangle)
TASKS3=(topology)

DEFAULT_MAX_TOKENS1=10
DEFAULT_MAX_TOKENS2=16
DEFAULT_MAX_TOKENS3=40

MODELS=(
  meta-llama/Meta-Llama-3.1-8B-Instruct
  Qwen/Qwen3-8B
  meta-llama/Llama-3.2-3B-Instruct
  Qwen/Qwen3-14B
  Qwen/Qwen3-4B
)

GAMMA_VALUES=(0.6)
EDGE_AGG_OPTIONS=(False)

SELECT_ROOT="SLASH/outputs/final_select/select0/GraphWiz"
for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH")"

  for T in "${TASKS[@]}"; do
    if [[ " ${TASKS1[@]} " =~ " $T " ]]; then
      MAX_TOKENS="$DEFAULT_MAX_TOKENS1"
    elif [[ " ${TASKS2[@]} " =~ " $T " ]]; then
      MAX_TOKENS="$DEFAULT_MAX_TOKENS2"
    elif [[ " ${TASKS3[@]} " =~ " $T " ]]; then
      MAX_TOKENS="$DEFAULT_MAX_TOKENS3"
    else
      MAX_TOKENS="$DEFAULT_MAX_TOKENS"
    fi

    BATCH_SIZE="$BASE_BATCH_SIZE"
    if [[ "$T" == "hamilton" ]]; then
      BATCH_SIZE=1
    fi

    for EDGE_AGG in "${EDGE_AGG_OPTIONS[@]}"; do
      CUDA_VISIBLE_DEVICES=$CUDA python evaluate_new.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --max_tokens "$MAX_TOKENS" \
        --tasks "$T" \
        $EDGE_AGG_FLAG

      CFG="${SELECT_ROOT}/"${MODEL_NAME}"/select_results/${T}_per_layer_intersection.json"
      if [[ -f "$CFG" ]]; then
        for GAMMA in "${GAMMA_VALUES[@]}"; do
          CUDA_VISIBLE_DEVICES=$CUDA python evaluate_new.py \
            --model_path "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --max_tokens "$MAX_TOKENS" \
            --tasks "$T" \
            --layer_head_config_path "$CFG" \
            --gamma "$GAMMA" \
            $EDGE_AGG_FLAG
        done
      else
        echo "[skip] missing config: $CFG"
      fi
    done
  done
done