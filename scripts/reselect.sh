REPO_ROOT="/home/lym/LLM-Research/Attention/Graph_Attention/GraphLens"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"


MODELS=(
  Llama-2-7b-chat-hf
  Llama-3.2-3B-Instruct
  LLaMA2-7B-DPO
  LLaMA2-13B-DPO
  LLaMA2-13B-RFT
  Meta-Llama-3.1-8B-Instruct
  Qwen3-4B
  Qwen3-8B
  Qwen3-14B
  Mistral-7B-RFT
  LLaMA2-7B-RFT
)

for MODEL in "${MODELS[@]}"; do
  python -m graphlens.refresh_intersections \
    --entropy_dir "../outputs/select4.0/${MODEL}/entropy_results" \
    --scoring_dir "../outputs/select4.0/${MODEL}/scoring_results/per_layer_topk_concentration_std" \
    --out_dir "../outputs/select4.0/${MODEL}/select_results2.1" \
    --mode per_layer \
    --alpha 1.0
done