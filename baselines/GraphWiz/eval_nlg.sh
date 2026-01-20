CUDA=0,1
DEFAULT_BATCH_SIZE=1
DEFAULT_MAX_TOKENS=1024
OUTPUT_DIR=./Results5.1

# GraphWiz tasks
TASKS=(cycle connectivity hamilton substructure bipartite flow shortest topology triangle)

MODELS=(
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-7B
  /home/lym/data1/LLM-model/GraphWiz/Mistral-7B
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-7B-RFT
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-7B-DPO
  /home/lym/data1/LLM-model/GraphWiz/Mistral-7B-RFT
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-13B-RFT
  /home/lym/data1/LLM-model/GraphWiz/LLaMA2-13B-DPO
)

# run_select.sh 结果根目录
SELECT_ROOT="/home/lym/LLM-Research/Attention/Graph_Attention/GraphLens/outputs/final_select/select5.1/GraphWiz"

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

    # 2) delta=0.4 / 0.6：读取该任务专属 config
    CFG="${SELECT_ROOT}/"${MODEL_NAME}"/select_results/${T}_per_layer_intersection.json"
    if [[ -f "$CFG" ]]; then
      for DR in 0.3; do
        CUDA_VISIBLE_DEVICES=$CUDA python evaluate_nlg.py \
          --model_path "$MODEL_PATH" \
          --output_dir "$OUTPUT_DIR" \
          --batch_size "$DEFAULT_BATCH_SIZE" \
          --max_tokens "$MAX_TOKENS" \
          --tasks "$T" \
          --layer_head_config_path "$CFG" \
          --delta_ratio "$DR"
      done
    else
      echo "[skip] missing config: $CFG"
    fi

  done
done