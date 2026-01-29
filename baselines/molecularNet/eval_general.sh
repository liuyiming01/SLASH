CUDA=6
BATCH_SIZE=4
MAX_NEW_TOKENS=10
LIMIT=400
OUTPUT_DIR=./outputs/Results

SCRIPT_PATH="$(realpath "$0")"
mkdir -p "$OUTPUT_DIR"
cp "$SCRIPT_PATH" "$OUTPUT_DIR/$(basename "$SCRIPT_PATH")"

DATA_DIR="ChemLLMBench/data/property_prediction"
PROMPT_PATH="./prompt/property_prediction_graph_prompt.txt"

TASKS=(BACE BBBP ClinTox HIV Tox21)
MODELS=(
  meta-llama/Meta-Llama-3.1-8B-Instruct
  Qwen/Qwen3-8B
  meta-llama/Llama-3.2-3B-Instruct
  Qwen/Qwen3-14B
  YuyanLiu/merged_MolecularGPT
)

SELECT_ROOT="SLASH/outputs/final_select/select0/GraphWiz"

EDGE_AGG_OPTIONS=(False)
SPLIT_OPTIONS=(sample)

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH")"
  SELECT_DIR="${SELECT_ROOT}/${MODEL_NAME}/select_results"

  for TASK in "${TASKS[@]}"; do
    for SPLIT in "${SPLIT_OPTIONS[@]}"; do
      for EDGE_AGG in "${EDGE_AGG_OPTIONS[@]}"; do
        EDGE_AGG_FLAG=""
        if [[ "$EDGE_AGG" == "True" ]]; then
          EDGE_AGG_FLAG="--edge_agg"
        fi

        # 1) Vanilla
        CUDA_VISIBLE_DEVICES=$CUDA python eval_mol.py \
          --task "$TASK" \
          --model_path "$MODEL_PATH" \
          --data_dir "$DATA_DIR" \
          --prompt_path "$PROMPT_PATH" \
          --output_dir "$OUTPUT_DIR" \
          --split "$SPLIT" \
          --limit 1000 \
          --sample_num "$LIMIT" \
          --preferred_min_edges 40 \
          --hard_max_edges 100 \
          --seed 42 \
          --batch_size "$BATCH_SIZE" \
          --max_new_tokens "$MAX_NEW_TOKENS" \
          $EDGE_AGG_FLAG

        # 2) SLASH
        CFG="${SELECT_DIR}/cycle_per_layer_intersection.json"
        if [[ -f "$CFG" ]]; then
          for GAMMA in 0.4 0.5 0.6 0.7 0.8; do
            CUDA_VISIBLE_DEVICES=$CUDA python eval_mol.py \
              --task "$TASK" \
              --model_path "$MODEL_PATH" \
              --data_dir "$DATA_DIR" \
              --prompt_path "$PROMPT_PATH" \
              --output_dir "$OUTPUT_DIR" \
              --split "$SPLIT" \
              --limit 1000 \
              --sample_num "$LIMIT" \
              --preferred_min_edges 40 \
              --hard_max_edges 100 \
              --seed 42 \
              --batch_size "$BATCH_SIZE" \
              --max_new_tokens "$MAX_NEW_TOKENS" \
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
done