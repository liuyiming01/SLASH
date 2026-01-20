CUDA=2,4,5
BATCH_SIZE=4
DEFAULT_MAX_TOKENS=6
LIMIT=400
OUTPUT_DIR=./Results5.4

TASKS=(Graph-SST2 Graph-SST5 Graph-Twitter)

MODELS=(
  /home/lym/data1/LLM-model/meta-llama/Meta-Llama-3.1-8B-Instruct
  /home/lym/data1/LLM-model/Qwen/Qwen3-8B
  /home/lym/data1/LLM-model/meta-llama/Llama-3.2-3B-Instruct
  /home/lym/data1/LLM-model/Qwen/Qwen3-4B
  /home/lym/data1/LLM-model/Qwen/Qwen3-14B
  /home/lym/data1/LLM-model/meta-llama/Llama-2-7b-chat-hf
)

SELECT_ROOT="/home/lym/LLM-Research/Attention/Graph_Attention/GraphLens/outputs/final_select/select5.2/Graph-SST"
PROMPT_PATH="./prompt/graph-sst-prompt2.txt"
# SYS_PROMPT_MODES=("" "--sys_prompt")
SYS_PROMPT_MODES=("")

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH")"
  SELECT_DIR="${SELECT_ROOT}/${MODEL_NAME}/select_results"

  for T in "${TASKS[@]}"; do
    for SYS_FLAG in "${SYS_PROMPT_MODES[@]}"; do
      # 1) None
      CUDA_VISIBLE_DEVICES=$CUDA python eval_sst.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --max_tokens "$DEFAULT_MAX_TOKENS" \
        --data_name "$T" \
        --split test \
        --limit "$LIMIT" \
        --prompt_path "$PROMPT_PATH" \
        --seed 42 \
        $SYS_FLAG

      # 2) delta=0.4 / 0.6：读取该任务专属 config
      CFG="${SELECT_DIR}/${T}_per_layer_intersection.json"
      if [[ -f "$CFG" ]]; then
        for DR in 0.4; do
          CUDA_VISIBLE_DEVICES=$CUDA python eval_sst.py \
            --model_path "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --max_tokens "$DEFAULT_MAX_TOKENS" \
            --data_name "$T" \
            --split test \
            --limit "$LIMIT" \
            --prompt_path "$PROMPT_PATH" \
            --layer_head_config_path "$CFG" \
            --delta_ratio "$DR" \
            --seed 42 \
            $SYS_FLAG
        done
      else
        echo "[skip] missing config: $CFG"
      fi
    done
  done
done