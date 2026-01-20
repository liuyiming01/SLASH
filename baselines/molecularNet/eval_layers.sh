CUDA=6,7
BATCH_SIZE=4
MAX_NEW_TOKENS=8
OUTPUT_DIR=./Results3.0

DATA_DIR="/home/lym/LLM-Research/Attention/Graph_Attention/GraphLens/baselines/repos/ChemLLMBench/data/property_prediction"
PROMPT_PATH="./prompt/property_prediction_graph_prompt.txt"

TASKS=(BACE BBBP ClinTox HIV Tox21)

MODELS=(
  /home/lym/data1/LLM-model/meta-llama/Meta-Llama-3.1-8B-Instruct
  # /home/lym/data1/LLM-model/Qwen/Qwen3-8B
  # /home/lym/data1/LLM-model/Qwen/Qwen3-4B
  # /home/lym/data1/LLM-model/meta-llama/Llama-3.2-3B-Instruct
#   /home/lym/data1/LLM-model/Qwen/Qwen3-14B
#   /home/lym/data1/LLM-model/meta-llama/Llama-2-7b-chat-hf
)

for MODEL_PATH in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    # None
    CUDA_VISIBLE_DEVICES=$CUDA python eval_mol.py \
        --task "$TASK" \
        --model_path "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --prompt_path "$PROMPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --split sample \
        --limit 1000 \
        --sample_num 400 \
        --preferred_min_edges 60 \
        --hard_max_edges 150 \
        --seed 42 \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_NEW_TOKENS"

    # layers 8-17
    CUDA_VISIBLE_DEVICES=$CUDA python eval_mol.py \
        --task "$TASK" \
        --model_path "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --prompt_path "$PROMPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --split sample \
        --limit 1000 \
        --sample_num 400 \
        --preferred_min_edges 60 \
        --hard_max_edges 150 \
        --seed 42 \
        --batch_size "$BATCH_SIZE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --layers_to_modify $(seq 8 17)
  done
done