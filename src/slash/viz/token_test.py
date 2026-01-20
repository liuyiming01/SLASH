import os
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ..utils import load_model_and_tokenizer
from ..viz_utils import choose_sample_for_task


def compute_token_attention_from_layer(layer_attn: np.ndarray) -> np.ndarray:
    """
    layer_attn: [S, S], 已经是对所有 head 平均后的单层 attention。
    返回: [S]，每个 token 的注意力值（下三角 + 非 0 平均）。
    """
    tril = np.tril(layer_attn)          # 只保留下三角
    S = tril.shape[0]
    token_scores = np.zeros(S, dtype=np.float32)

    for j in range(S):
        col = tril[:, j]
        nonzero = col[col != 0]
        if nonzero.size > 0:
            token_scores[j] = nonzero.mean()
        else:
            token_scores[j] = 0.0

    return token_scores


def parse_args():
    p = argparse.ArgumentParser(
        description="Minimal script: per-task sample, per-layer token attention bar plot"
    )

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--task_name", type=str, required=True,
                   help="e.g. GraphWiz or GraphWiz_cycle")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--input_column", type=str, default="input_prompt")

    p.add_argument("--min_edges", type=int, default=80)
    p.add_argument("--max_edges", type=int, default=120)
    p.add_argument("--max_seq_len", type=int, default=1000)
    p.add_argument("--plot_dpi", type=int, default=250)

    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path} ...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # 与 plot.py / scoring.py 一致的 task 解析逻辑
    tasks = []
    if args.task_name.startswith("GraphWiz"):
        parts = args.task_name.split("_", 1)
        if len(parts) > 1 and parts[1]:
            tasks.append(parts[1])
        else:
            tasks = [
                "cycle", "connectivity", "hamilton", "substructure",
                "bipartite", "flow", "shortest", "topology", "triangle",
            ]
    else:
        raise ValueError(f"Unknown task name: {args.task_name}")

    for task in tasks:
        print(f"\n=== Task: {task} ===")
        data_path = os.path.join(args.data_dir, f"{task}_test.json")
        if not os.path.exists(data_path):
            print(f"  Data file not found: {data_path}, skip.")
            continue

        df = pd.read_json(data_path, lines=True)

        idxs = choose_sample_for_task(
            df,
            min_edges=args.min_edges,
            max_edges=args.max_edges,
            input_column=args.input_column,
            max_samples=5,
        )
        if not idxs:
            print(f"  No sample in [{args.min_edges},{args.max_edges}] edges for task {task}, skip.")
            continue

        chosen_idx = idxs[0]
        item = df.iloc[chosen_idx]
        prompt = item[args.input_column]
        print(f"  Chosen index: {chosen_idx}, prompt_len={len(prompt)}")

        sys_prompt = (
        "Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request step by step.\n\n"
            "### Instruction:\n{query}\n\n### Response:"
        )
        prompt = sys_prompt.format(query=prompt)

        # 编码并截断
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        if enc.input_ids.shape[1] > args.max_seq_len:
            enc.input_ids = enc.input_ids[:, :args.max_seq_len]
            if "attention_mask" in enc:
                enc.attention_mask = enc.attention_mask[:, :args.max_seq_len]

        with torch.no_grad():
            out = model(enc.input_ids, output_attentions=True, use_cache=False)

        # [L, H, S, S]
        attn_np = np.stack(
            [a[0].to(torch.float32).cpu().numpy() for a in out.attentions],
            axis=0,
        )
        num_layers, num_heads, S, _ = attn_np.shape

        token_ids = enc.input_ids[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        token_count = len(tokens)

        # 输出目录: /output_dir/model_name/task_x/sample_idx_y/
        task_save_root = os.path.join(
            args.output_dir,
            os.path.basename(args.model_path),
            f"task_{task}",
            f"sample_{chosen_idx}",
        )
        os.makedirs(task_save_root, exist_ok=True)

        for l in range(num_layers):
            # 该层对所有 head 取平均 -> [S, S]
            layer_attn = attn_np[l].mean(axis=0)
            token_scores = compute_token_attention_from_layer(layer_attn)

            # 只显示实际的 token 数目
            token_scores = token_scores[:token_count]

            # 画柱状图
            plt.figure(figsize=(max(8, token_count * 0.3), 4))
            x = np.arange(token_count)
            plt.bar(x, token_scores)
            plt.xticks(x, tokens, rotation=90, fontsize=6)
            plt.xlabel("Token")
            plt.ylabel("Mean attention (lower triangle, non-zero)")
            plt.title(f"Task: {task} | Sample: {chosen_idx} | Layer: {l}")
            plt.tight_layout()

            fig_path = os.path.join(task_save_root, f"layer_{l}_token_attention.png")
            plt.savefig(fig_path, dpi=args.plot_dpi)
            plt.close()

            print(f"  Saved layer {l} token attention bar plot to: {fig_path}")


if __name__ == "__main__":
    main()