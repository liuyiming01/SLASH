set -euo pipefail

CUDA=0
BATCH_SIZE=4
LIMIT=400
MAX_NEW_TOKENS=8
OUTPUT_DIR=./Results_Property

DATA_DIR="/home/lym/LLM-Research/Attention/Graph_Attention/src/GraphLens/baselines/repos/ChemLLMBench/data/property_prediction"
PROMPT_PATH="/home/lym/LLM-Research/Attention/Graph_Attention/src/GraphLens/baselines/repos/ChemLLMBench/data/property_prediction/property_prediction_prompt.txt"

TASKS=(BACE BBBP ClinTox HIV Tox21)
SHOTS=(0 4)

MODELS=(
  /home/lym/data1/LLM-model/meta-llama/Meta-Llama-3.1-8B-Instruct
  # /home/lym/data1/LLM-model/meta-llama/Llama-2-7b-chat-hf
  # /home/lym/data1/LLM-model/meta-llama/Llama-3.2-3B-Instruct
)

for MODEL_PATH in "${MODELS[@]}"; do
  for TASK in "${TASKS[@]}"; do
    for SHOT in "${SHOTS[@]}"; do
      # None
      CUDA_VISIBLE_DEVICES=$CUDA python evaluate_property_prediction.py \
        --task "$TASK" \
        --model_path "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --prompt_path "$PROMPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --split test \
        --batch_size "$BATCH_SIZE" \
        --limit "$LIMIT" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --shot "$SHOT"

      # layers 7-17
      CUDA_VISIBLE_DEVICES=$CUDA python evaluate_property_prediction.py \
        --task "$TASK" \
        --model_path "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --prompt_path "$PROMPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --split test \
        --batch_size "$BATCH_SIZE" \
        --limit "$LIMIT" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --shot "$SHOT" \
        --layers_to_modify $(seq 7 17)

      # layers 8-17
      CUDA_VISIBLE_DEVICES=$CUDA python evaluate_property_prediction.py \
        --task "$TASK" \
        --model_path "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --prompt_path "$PROMPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --split test \
        --batch_size "$BATCH_SIZE" \
        --limit "$LIMIT" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --shot "$SHOT" \
        --layers_to_modify $(seq 8 17)
    done
  done
done