CUDA=0,1
BASE_BATCH_SIZE=1
DEFAULT_MAX_TOKENS=10
OUTPUT_DIR=./Results_other/exp_ablation

SCRIPT_PATH="$(realpath "$0")"
mkdir -p "$OUTPUT_DIR"
cp "$SCRIPT_PATH" "$OUTPUT_DIR/$(basename "$SCRIPT_PATH")"

MODELS=(
  meta-llama/Meta-Llama-3.1-8B-Instruct
  Qwen/Qwen3-8B
  Qwen/Qwen3-14B
)
# TASKS=(cycle connectivity hamilton substructure bipartite flow shortest triangle topology)
TASKS=(hamilton)

TASKS1=(cycle connectivity hamilton substructure bipartite)
TASKS2=(flow shortest triangle)
TASKS3=(bipartite)

DEFAULT_MAX_TOKENS1=10
DEFAULT_MAX_TOKENS2=16
DEFAULT_MAX_TOKENS3=40

GAMMA_VALUES=(0.6)
EDGE_AGG_OPTIONS=(False)

SELECT_ROOT="SLASH/outputs/final_select/select_layer/GraphWiz"
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

    layers_to_modify=({0..39})
    for GAMMA in "${GAMMA_VALUES[@]}"; do
      CUDA_VISIBLE_DEVICES=$CUDA python evaluate_new.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --max_tokens "$MAX_TOKENS" \
        --tasks "$T" \
        --layers_to_modify "${layers_to_modify[@]}" \
        --gamma "$GAMMA"
    done
  
    CFGS=(
      "${SELECT_ROOT}/"${MODEL_NAME}"/scoring_results/per_layer_topk_concentration_std/${T}_selected_layers_auto_scoring.json"
      "${SELECT_ROOT}/"${MODEL_NAME}"/entropy_results/${T}_entropy_per_layer/${T}_selected_layers_auto_entropy.json"
    )
    for CFG in "${CFGS[@]}"; do
      if [[ -f "$CFG" ]]; then
        for GAMMA in "${GAMMA_VALUES[@]}"; do
          CUDA_VISIBLE_DEVICES=$CUDA python evaluate_new.py \
            --model_path "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size "$BATCH_SIZE" \
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
done