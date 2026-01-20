import os
from typing import Literal, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from tqdm import tqdm

from .utils import (
    load_model_and_tokenizer,
    count_edges_in_prompt,
    select_layers_by_top_fraction,
    select_layers_auto_otsu,
    auto_edge_range,
)
from .datasets import load_and_filter_samples  # NEW


def matrix_based_entropy_from_svals(
    svals: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    根据奇异值计算矩阵基熵（支持 batch）。
    """
    power = svals ** 2
    power_sum = power.sum(dim=-1, keepdim=True) + eps
    p = power / power_sum  # (..., r)

    if abs(alpha - 1.0) < 1e-6:
        log_p = torch.log(p + eps)
        H = -(p * log_p).sum(dim=-1)  # (...)
    else:
        H_alpha = (p ** alpha).sum(dim=-1) + eps
        H = torch.log(H_alpha) / (1.0 - alpha)

    return H


def attention_entropy_per_head(
    attn: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    模式 1: 对每个 (layer, head) 的 attention 矩阵计算矩阵基熵。
    """
    if attn.ndim != 4:
        raise ValueError(f"attn must have shape [L, H, T, T], got {attn.shape}")
    L, H, T, T2 = attn.shape
    if T != T2:
        raise ValueError("Attention matrices must be square [T, T].")

    attn = attn.to(torch.float32)

    ent_lh = torch.empty(L, H, device=attn.device, dtype=torch.float32)
    for l in range(L):
        for h in range(H):
            m = attn[l, h]
            svals = torch.linalg.svdvals(m)
            H_val = matrix_based_entropy_from_svals(svals, alpha=alpha)
            ent_lh[l, h] = H_val.item()

    return ent_lh


def attention_entropy_per_layer_mean_head(
    attn: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    模式 2: 每层先对所有 head 的 attention 做平均，再计算矩阵基熵。
    """
    if attn.ndim != 4:
        raise ValueError(f"attn must have shape [L, H, T, T], got {attn.shape}")
    L, H, T, T2 = attn.shape
    if T != T2:
        raise ValueError("Attention matrices must be square [T, T].")

    attn = attn.to(torch.float32)
    attn_layer = attn.mean(dim=1)  # [L, T, T]

    ent_layer = torch.empty(L, device=attn.device, dtype=torch.float32)
    for l in range(L):
        m = attn_layer[l]
        svals = torch.linalg.svdvals(m)
        H_val = matrix_based_entropy_from_svals(svals, alpha=alpha)
        ent_layer[l] = H_val.item()

    return ent_layer


# ===========================
# 可视化工具
# ===========================
def plot_entropy_heatmap_and_layer_mean(
    ent_lh: torch.Tensor,
    save_dir: str,
    prefix: str = "attn_entropy_per_head",
    show: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    L, H = ent_lh.shape

    ent_np = ent_lh.detach().cpu().numpy()
    layer_idx = np.arange(L)

    plt.figure(figsize=(max(6, H * 0.4), max(4, L * 0.4)))
    im = plt.imshow(ent_np, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label="Matrix-Based Entropy")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Attention Entropy per Layer-Head")
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, f"{prefix}_heatmap.png")
    plt.savefig(heatmap_path, dpi=200)
    if show:
        plt.show()
    plt.close()

    mean_per_layer = ent_np.mean(axis=1)  # [L]
    plt.figure(figsize=(6, 4))
    plt.plot(layer_idx, mean_per_layer, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Mean Entropy over Heads")
    plt.title("Mean Attention Entropy per Layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    line_path = os.path.join(save_dir, f"{prefix}_layer_mean.png")
    plt.savefig(line_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_layer_entropy(
    ent_layer: torch.Tensor,
    save_dir: str,
    prefix: str = "attn_entropy_layer_mean_head",
    show: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    ent_np = ent_layer.detach().cpu().numpy()
    L = ent_np.shape[0]
    layer_idx = np.arange(L)

    plt.figure(figsize=(6, 4))
    plt.plot(layer_idx, ent_np, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Layer-Level Entropy")
    plt.title("Entropy of Mean-Head Attention per Layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{prefix}.png")
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()


# ===========================
# GraphWiz: 找数据文件
# ===========================
def _find_data_file(data_dir: str, task_name: str) -> str:
    cand1 = os.path.join(data_dir, f"{task_name}_test.jsonl")
    cand2 = os.path.join(data_dir, f"{task_name}_test.json")

    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2

    raise FileNotFoundError(
        f"Cannot find data file for task '{task_name}' in {data_dir}. "
        f"Tried: {cand1}, {cand2}"
    )

# ===========================
# 主流程：在 sample_num 个样本上取熵的平均（含自动选区间 + Graph-SST 适配）
# ===========================
@torch.no_grad()
def process_task(
    task_name: str,
    cfg: dict,
    model,
    tokenizer,
):
    data_dir = cfg["data_dir"]
    input_column = cfg["input_column"]
    requested_sample_num = int(cfg["sample_num"])
    max_seq_len = cfg["max_seq_len"]
    score_mode = cfg["score_mode"]
    alpha = cfg["alpha"]
    output_dir = cfg["output_dir"]
    top_fraction = cfg.get("select_top_fraction", 0.4)

    # 1) 先拿“全量样本”（GraphWiz 从文件；Graph-SST 从 DIG 生成 prompt）
    graphsst = cfg.get("graphsst", None)
    data_path = None
    if graphsst is None:
        data_path = cfg.get("data_path", None) or _find_data_file(data_dir, task_name)

    all_df = load_and_filter_samples(
        data_path=data_path,
        input_column=input_column,
        min_edges=0,
        max_edges=10**9,
        sample_num=10**9,          # 会返回全量（n=min(1e9, len(df))）
        graphsst=graphsst,
    )
    if all_df is None or len(all_df) == 0:
        print(f"[{task_name}] No samples. Skip.")
        return

    # 2) 自动选择边数区间，确保覆盖 sample_num
    if "__num_edges" not in all_df.columns:
        all_df["__num_edges"] = all_df[input_column].map(count_edges_in_prompt)

    edge_counts = all_df["__num_edges"].to_numpy(dtype=np.int32)
    true_min_edges = int(edge_counts.min())
    true_max_edges = int(edge_counts.max())
    true_median = float(np.median(edge_counts))
    print(f"[{task_name}] TRUE edge stats: min={true_min_edges}, max={true_max_edges}, median={true_median:.1f}, n={len(edge_counts)}")

    chosen_min, chosen_max, stats = auto_edge_range(edge_counts=edge_counts, sample_num=requested_sample_num, min_chosen_min=60)
    if chosen_min is None:
        print(f"[{task_name}] Cannot auto-select edge range.")
        return

    # 覆写 cfg（便于日志一致；不强依赖）
    cfg["min_edges"] = chosen_min
    cfg["max_edges"] = chosen_max

    subset = all_df[(all_df["__num_edges"] >= chosen_min) & (all_df["__num_edges"] <= chosen_max)]
    if len(subset) < requested_sample_num:
        print(f"[{task_name}] Warning: subset size {len(subset)} < sample_num {requested_sample_num}, fallback to full set.")
        subset = all_df

    samples = subset.sample(n=min(requested_sample_num, len(subset)), random_state=42).reset_index(drop=True)
    print(
        f"[{task_name}] AUTO edge range: chosen_min={chosen_min}, chosen_max={chosen_max}, "
        f"center={stats.get('chosen_center'):.1f}, width={stats.get('chosen_width')}, "
        f"covered={len(subset)}/{len(all_df)}, using_samples={len(samples)}"
    )

    # 3) 熵计算（原逻辑保留）
    score_sum = None
    score_cnt = None
    num_layers = None
    num_heads = None

    for _, row in tqdm(samples.iterrows(), total=len(samples), desc=f"[{task_name}] Entropy"):
        prompt = row[input_column]

        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        if enc.input_ids.shape[1] > max_seq_len:
            enc.input_ids = enc.input_ids[:, :max_seq_len]
            if "attention_mask" in enc:
                enc.attention_mask = enc.attention_mask[:, :max_seq_len]

        out = model(enc.input_ids, output_attentions=True, use_cache=False)
        attn_list = out.attentions  # list of length L, each: [batch, H, S, S]

        L_this = len(attn_list)
        H_this = attn_list[0].shape[1]

        if score_sum is None:
            num_layers = L_this
            num_heads = H_this
            if score_mode == "per_head":
                score_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
                score_cnt = np.zeros((num_layers, num_heads), dtype=np.int32)
            elif score_mode == "per_layer":
                score_sum = np.zeros((num_layers,), dtype=np.float64)
                score_cnt = np.zeros((num_layers,), dtype=np.int32)
            else:
                raise ValueError(f"Unknown score_mode: {score_mode}")

        if score_mode == "per_head":
            for l, attn_l in enumerate(attn_list):
                attn_l_cpu = attn_l[0].detach().to("cpu", torch.float32)  # [H, S, S]
                for h in range(H_this):
                    m = attn_l_cpu[h]
                    svals = torch.linalg.svdvals(m)
                    H_val = matrix_based_entropy_from_svals(svals, alpha=alpha)
                    score_sum[l, h] += float(H_val)
                    score_cnt[l, h] += 1
        else:
            for l, attn_l in enumerate(attn_list):
                attn_l_cpu = attn_l[0].detach().to("cpu", torch.float32)  # [H, S, S]
                m = attn_l_cpu.mean(dim=0)  # [S, S]
                svals = torch.linalg.svdvals(m)
                H_val = matrix_based_entropy_from_svals(svals, alpha=alpha)
                score_sum[l] += float(H_val)
                score_cnt[l] += 1

        del out, attn_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_entropy = score_sum / np.maximum(score_cnt, 1)

    # 保存 & 绘图
    out_dir_mode = os.path.join(output_dir, f"{task_name}_entropy_{score_mode}_alpha{alpha}")
    os.makedirs(out_dir_mode, exist_ok=True)

    cache_file = os.path.join(out_dir_mode, f"{task_name}_entropy_{score_mode}.npz")
    np.savez_compressed(cache_file, avg_entropy=avg_entropy, count=score_cnt)
    print(f"[{task_name}] Cached entropy saved to {cache_file}")

    ent_tensor = torch.from_numpy(avg_entropy.astype(np.float32))
    if score_mode == "per_head":
        plot_entropy_heatmap_and_layer_mean(
            ent_tensor,
            save_dir=out_dir_mode,
            prefix=f"{task_name}_entropy_alpha{alpha}_per_head",
            show=False,
        )
    else:
        plot_layer_entropy(
            ent_tensor,
            save_dir=out_dir_mode,
            prefix=f"{task_name}_entropy_alpha{alpha}_per_layer_mean_head",
            show=False,
        )

    valid_mask = score_cnt > 0

    json_top = os.path.join(out_dir_mode, f"{task_name}_selected_layers_top_{int(top_fraction * 100)}_entropy.json")
    json_auto = os.path.join(out_dir_mode, f"{task_name}_selected_layers_auto_entropy.json")

    select_layers_by_top_fraction(
        scores=avg_entropy,
        valid_mask=valid_mask,
        score_mode=score_mode,
        num_heads=num_heads,
        top_fraction=top_fraction,
        json_path=json_top,
    )

    select_layers_auto_otsu(
        scores=avg_entropy,
        valid_mask=valid_mask,
        score_mode=score_mode,
        num_heads=num_heads,
        json_path=json_auto,
        fallback_top_fraction=top_fraction,
    )

    print(f"[{task_name}] Done.")


def parse_args():
    p = argparse.ArgumentParser(description="GraphLens: matrix-based entropy on attention")

    p.add_argument("--model_path", type=str, required=True,
                   help="Path to HF model (or backend-specific id)")
    p.add_argument("--task_name", type=str, required=True,
                   help="Task name. Use 'GraphWiz' or 'GraphWiz_<subtask>' or Graph-SST2/5/Twitter")
    p.add_argument("--data_dir", type=str, required=True,
                   help="GraphWiz: directory containing task files; Graph-SST: DIG dataset root")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save entropy scores and plots")
    p.add_argument("--input_column", type=str, default="input_prompt",
                   help="Column name that contains the graph prompt text")

    p.add_argument("--sample_num", type=int, default=100,
                   help="Number of samples to use for entropy computation")

    p.add_argument("--max_seq_len", type=int, default=1000,
                   help="Max sequence length for model input")

    p.add_argument("--score_mode", type=str, default="per_head",
                   choices=["per_head", "per_layer"],
                   help="Entropy mode: per head or per layer (mean over heads)")

    p.add_argument("--alpha", type=float, default=1.0,
                   help="Order alpha for matrix-based entropy (alpha=1 is Shannon/von Neumann)")

    p.add_argument("--plot_dpi", type=int, default=200,
                   help="DPI for saved figures")
    p.add_argument("--plot_line_color", type=str, default="C0",
                   help="Line color for per-layer plots")

    p.add_argument("--select_top_fraction", type=float, default=0.4,
                   help="Top fraction used when entropy-based selection falls back from Otsu")

    # Graph-SST specific (NEW)
    p.add_argument("--split", type=str, default="test",
                   help="Graph-SST split: train/val/valid/test")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "input_column": args.input_column,
        "sample_num": args.sample_num,
        "max_seq_len": args.max_seq_len,
        "score_mode": args.score_mode,
        "alpha": args.alpha,
        "plot": {
            "dpi": args.plot_dpi,
            "line_color": args.plot_line_color,
        },
        "select_top_fraction": args.select_top_fraction,
        "data_path": None,
        "graphsst": None,
    }

    print(f"Loading model from {args.model_path} ...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # GraphWiz
    if args.task_name.startswith("GraphWiz"):
        parts = args.task_name.split("_", 1)
        if len(parts) > 1 and parts[1]:
            tasks = [parts[1]]
        else:
            tasks = ["cycle", "connectivity", "hamilton", "substructure", "bipartite", "flow", "shortest", "topology", "triangle"]

        for task in tasks:
            cfg["graphsst"] = None
            cfg["data_path"] = None  # 让 process_task 自己在 data_dir 下找文件
            process_task(
                task_name=task,
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
            )
        return

    # Graph-SST family
    if args.task_name in ("Graph-SST2", "Graph-SST5", "Graph-Twitter"):
        cfg["data_path"] = None
        cfg["graphsst"] = {
            "root": args.data_dir,
            "name": args.task_name,
            "split": args.split,
        }
        process_task(
            task_name=args.task_name,
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
        )
        return

    raise ValueError(f"Unknown task name: {args.task_name}")


if __name__ == "__main__":
    main()