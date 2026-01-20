CUDA=1,3
BASE_BATCH_SIZE=2
DEFAULT_MAX_TOKENS=6
OUTPUT_DIR=./Results4.2_test

# GraphWiz tasks
# TASKS=(cycle connectivity hamilton substructure bipartite flow shortest topology triangle)
TASKS=(substructure bipartite)
TASKS1=(cycle connectivity hamilton substructure bipartite)
TASKS2=(flow shortest triangle)
TASKS3=(topology)

DEFAULT_MAX_TOKENS1=5
DEFAULT_MAX_TOKENS2=20
DEFAULT_MAX_TOKENS3=40


MODELS=(
  /home/lym/data1/LLM-model/Qwen/Qwen3-8B
  /home/lym/data1/LLM-model/meta-llama/Meta-Llama-3.1-8B-Instruct
  /home/lym/data1/LLM-model/Qwen/Qwen3-4B
  /home/lym/data1/LLM-model/meta-llama/Llama-3.2-3B-Instruct
  /home/lym/data1/LLM-model/Qwen/Qwen3-14B
  /home/lym/data1/LLM-model/meta-llama/Llama-2-7b-chat-hf
)

# run_select.sh 结果根目录
SELECT_ROOT="/home/lym/LLM-Research/Attention/Graph_Attention/GraphLens/outputs/final_select/select5.1/GraphWiz"

# 两套实验：无 sys_prompt / 有 sys_prompt
# SYS_PROMPT_MODES=("" "--sys_prompt")
SYS_PROMPT_MODES=("")

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH")"

  for T in "${TASKS[@]}"; do
    # per-task overrides
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

    for SYS_FLAG in "${SYS_PROMPT_MODES[@]}"; do
      # 1) None
      # CUDA_VISIBLE_DEVICES=$CUDA python evaluate_new.py \
      #   --model_path "$MODEL_PATH" \
      #   --output_dir "$OUTPUT_DIR" \
      #   --batch_size "$BATCH_SIZE" \
      #   --max_tokens "$MAX_TOKENS" \
      #   --tasks "$T" \
      #   $SYS_FLAG

      # 2) delta=0.4 / 0.6：读取该任务专属 config
      CFG="${SELECT_ROOT}/"${MODEL_NAME}"/select_results/${T}_per_layer_intersection.json"
      if [[ -f "$CFG" ]]; then
        for DR in 0.4; do
          CUDA_VISIBLE_DEVICES=$CUDA python evaluate_new.py \
            --model_path "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --max_tokens "$MAX_TOKENS" \
            --tasks "$T" \
            --layer_head_config_path "$CFG" \
            --delta_ratio "$DR" \
            $SYS_FLAG
        done
      else
        echo "[skip] missing config: $CFG"
      fi
    done

  done
done