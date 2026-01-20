
CUDA_VISIBLE_DEVICES=5 python evaluate_nlg.py \
    --model_path /home/lym/data1/LLM-model/GraphWiz/LLaMA2-7B \
    --streategy Parallel \
    --batch_size 2 \
# &> ./test_graph.log &
&> ./test_graph.log
