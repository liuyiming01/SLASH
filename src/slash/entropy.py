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
)


def _ensure_tensor(x, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    将输入转换成 torch.Tensor，并放到指定 device。
    """
    if isinstance(x, torch.Tensor):
        t = x
        if device is not None:
            t = t.to(device)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t
    t = torch.as_tensor(x)
    if device is not None:
        t = t.to(device)
    if dtype is not None:
        t = t.to(dtype)
    return t


def matrix_based_entropy_from_svals(
    svals: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    根据奇异值计算矩阵基熵（支持 batch）。

    参数
    ----
    svals : torch.Tensor
        形状 (..., r)，最后一维是奇异值 σ_j >= 0。
    alpha : float
        Renyi 阶数；alpha=1 时退化为 Shannon/von Neumann 型熵：
          H = -sum_j p_j log p_j, p_j = σ_j^2 / sum_k σ_k^2
    eps : float
        数值稳定项，避免 log(0) / 除 0。

    返回
    ----
    H : torch.Tensor
        形状与 svals[..., 0] 相同，即去掉最后一维后的 batch 形状。
    """
    power = svals ** 2
    power_sum = power.sum(dim=-1, keepdim=True) + eps
    p = power / power_sum  # (..., r)

    if abs(alpha - 1.0) < 1e-6:
        # Shannon 熵
        log_p = torch.log(p + eps)
        H = -(p * log_p).sum(dim=-1)  # (...)
    else:
        # Renyi 熵
        H_alpha = (p ** alpha).sum(dim=-1) + eps
        H = torch.log(H_alpha) / (1.0 - alpha)

    return H


# def attention_entropy_per_head(
#     attn: torch.Tensor,
#     alpha: float = 1.0,
# ) -> torch.Tensor:
#     """
#     模式 1: 对每个 (layer, head) 的 attention 矩阵计算矩阵基熵。

#     参数
#     ----
#     attn : torch.Tensor
#         形状 [L, H, T, T] 的注意力矩阵（已经是 softmax 后的权重）。
#     alpha : float
#         Renyi 阶数，默认 1.0（推荐）。

#     返回
#     ----
#     ent_lh : torch.Tensor
#         形状 [L, H]，每个 layer-head 的熵。
#     """
#     if attn.ndim != 4:
#         raise ValueError(f"attn must have shape [L, H, T, T], got {attn.shape}")
#     L, H, T, T2 = attn.shape
#     if T != T2:
#         raise ValueError("Attention matrices must be square [T, T].")

#     device = attn.device
#     dtype = torch.float32
#     attn = attn.to(dtype)

#     # 将 (L, H, T, T) 展平为 (L*H, T, T)，一次性做 SVD
#     attn_flat = attn.reshape(L * H, T, T)  # [L*H, T, T]
#     svals = torch.linalg.svdvals(attn_flat)  # [L*H, T]

#     H_flat = matrix_based_entropy_from_svals(svals, alpha=alpha)  # [L*H]
#     ent_lh = H_flat.reshape(L, H)  # [L, H]
#     return ent_lh.to(device)

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

    # 统一 dtype，避免 half/bfloat 影响 SVD
    attn = attn.to(torch.float32)

    # 不再一次性对 [L*H, T, T] 做 SVD，而是逐个 (L, H) 计算
    ent_lh = torch.empty(L, H, device=attn.device, dtype=torch.float32)

    for l in range(L):
        for h in range(H):
            # [T, T]
            m = attn[l, h]
            svals = torch.linalg.svdvals(m)              # [T]
            H_val = matrix_based_entropy_from_svals(svals, alpha=alpha)  # 标量 tensor
            ent_lh[l, h] = H_val.item()

    return ent_lh


# def attention_entropy_per_layer_mean_head(
#     attn: torch.Tensor,
#     alpha: float = 1.0,
# ) -> torch.Tensor:
#     """
#     模式 2: 每层先对所有 head 的 attention 做平均，再计算矩阵基熵。

#     参数
#     ----
#     attn : torch.Tensor
#         形状 [L, H, T, T] 的注意力矩阵。
#     alpha : float
#         Renyi 阶数，默认 1.0。

#     返回
#     ----
#     ent_layer : torch.Tensor
#         形状 [L]，每层一个熵值。
#     """
#     if attn.ndim != 4:
#         raise ValueError(f"attn must have shape [L, H, T, T], got {attn.shape}")
#     L, H, T, T2 = attn.shape
#     if T != T2:
#         raise ValueError("Attention matrices must be square [T, T].")

#     device = attn.device
#     dtype = torch.float32
#     attn = attn.to(dtype)

#     # 对 head 取平均: [L, T, T]
#     attn_layer = attn.mean(dim=1)  # [L, T, T]
#     svals = torch.linalg.svdvals(attn_layer)  # [L, T]
#     ent_layer = matrix_based_entropy_from_svals(svals, alpha=alpha)  # [L]
#     return ent_layer.to(device)

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
        m = attn_layer[l]                      # [T, T]
        svals = torch.linalg.svdvals(m)        # [T]
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
    """
    模式 1 可视化：
      - (L, H) 的热力图；
      - 每层 head 平均熵的折线图。
    """
    os.makedirs(save_dir, exist_ok=True)
    L, H = ent_lh.shape

    ent_np = ent_lh.detach().cpu().numpy()
    layer_idx = np.arange(L)

    # 1) 热力图
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

    # 2) 每层 head 平均熵折线图
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
    """
    模式 2 可视化：每层平均 head 后的 attention 矩阵熵的折线图。
    """
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
# 数据加载 & 过滤（参考 scoring.py）
# ===========================

def _load_and_filter_samples(
    data_path: str,
    input_column: str,
    min_edges: int,
    max_edges: int,
    sample_num: int,
):
    """
    加载并过滤数据：
      - 支持 json/jsonl
      - 根据边数过滤
    返回：
      samples_df
    """
    if not os.path.exists(data_path):
        print(f"Warning: data file not found: {data_path}")
        return None

    ext = os.path.splitext(data_path)[1].lower()
    if ext in [".json", ".jsonl"]:
        df = pd.read_json(data_path, lines=True)
    else:
        raise ValueError(f"Unsupported data file extension: {ext}, path={data_path}")

    if input_column not in df.columns:
        raise ValueError(f"Input column '{input_column}' not found in data file: {data_path}")

    df["__num_edges"] = df[input_column].map(count_edges_in_prompt)
    df_filt = df[(df["__num_edges"] >= min_edges) &
                 (df["__num_edges"] <= max_edges)]
    print("Filtered samples count:", len(df_filt))
    if len(df_filt) == 0:
        print(f"Warning: no samples with {min_edges}-{max_edges} edges in {data_path}.")
        return None

    n_samples = min(sample_num, len(df_filt))
    samples = df_filt.sample(n=n_samples, random_state=42)

    return samples


def _find_data_file(data_dir: str, task_name: str) -> str:
    """
    在 data_dir 中根据 task_name 查找数据文件，优先使用:
      - {task_name}.jsonl
      - {task_name}.json
    """
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
# 主流程：在 sample_num 个样本上取熵的平均
# ===========================

@torch.no_grad()
def process_task(
    task_name: str,
    cfg: dict,
    model,
    tokenizer,
):
    """
    对单个 task：
      1) 加载并过滤样本
      2) 前向推理获取 attention
      3) 计算矩阵基熵
      4) 在 sample_num 个样本上求平均
      5) 保存 npz + 绘图 + 选择高分 layer/head
    """
    data_dir = cfg["data_dir"]
    input_column = cfg["input_column"]
    min_edges = cfg["min_edges"]
    max_edges = cfg["max_edges"]
    sample_num = cfg["sample_num"]
    max_seq_len = cfg["max_seq_len"]
    score_mode = cfg["score_mode"]
    alpha = cfg["alpha"]
    output_dir = cfg["output_dir"]
    top_fraction = cfg.get("select_top_fraction", 0.4)

    data_path = _find_data_file(data_dir, task_name)
    print(f"[{task_name}] Loading data from {data_path}")
    samples = _load_and_filter_samples(
        data_path=data_path,
        input_column=input_column,
        min_edges=min_edges,
        max_edges=max_edges,
        sample_num=sample_num,
    )
    if samples is None or len(samples) == 0:
        print(f"[{task_name}] No valid samples. Skip.")
        return

    # 累加器（懒初始化，第一次 forward 后根据 L/H 大小创建）
    score_sum = None
    score_cnt = None
    num_layers = None
    num_heads = None

    # for _, row in tqdm(samples.iterrows(), total=len(samples), desc=f"[{task_name}] Entropy"):
    #     prompt = row[input_column]

    #     enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    #     if enc.input_ids.shape[1] > max_seq_len:
    #         enc.input_ids = enc.input_ids[:, :max_seq_len]
    #         if "attention_mask" in enc:
    #             enc.attention_mask = enc.attention_mask[:, :max_seq_len]

    #     out = model(enc.input_ids, output_attentions=True, use_cache=False)
    #     attn_list = out.attentions  # list of length L, each: [batch, H, S, S]

    #     # 只取 batch=0
    #     L_this = len(attn_list)
    #     H_this = attn_list[0].shape[1]

    #     # 懒初始化累加器
    #     if score_sum is None:
    #         num_layers = L_this
    #         num_heads = H_this
    #         if score_mode == "per_head":
    #             score_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
    #             score_cnt = np.zeros((num_layers, num_heads), dtype=np.int32)
    #         elif score_mode == "per_layer":
    #             score_sum = np.zeros((num_layers,), dtype=np.float64)
    #             score_cnt = np.zeros((num_layers,), dtype=np.int32)
    #         else:
    #             raise ValueError(f"Unknown score_mode: {score_mode}")

    #     # 组装为 [L, H, T, T]
    #     attn_tensor = torch.stack([a[0] for a in attn_list], dim=0)  # [L, H, S, S]

    #     if score_mode == "per_head":
    #         ent_lh = attention_entropy_per_head(attn_tensor, alpha=alpha)  # [L, H]
    #         ent_np = ent_lh.detach().cpu().numpy()
    #         score_sum += ent_np
    #         score_cnt += 1  # 对所有位置统一 +1

    #     else:  # "per_layer"
    #         ent_layer = attention_entropy_per_layer_mean_head(attn_tensor, alpha=alpha)  # [L]
    #         ent_np = ent_layer.detach().cpu().numpy()
    #         score_sum += ent_np
    #         score_cnt += 1

    # # 聚合平均
    # avg_entropy = score_sum / np.maximum(score_cnt, 1)

    for _, row in tqdm(samples.iterrows(), total=len(samples), desc=f"[{task_name}] Entropy"):
        prompt = row[input_column]

        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        if enc.input_ids.shape[1] > max_seq_len:
            enc.input_ids = enc.input_ids[:, :max_seq_len]
            if "attention_mask" in enc:
                enc.attention_mask = enc.attention_mask[:, :max_seq_len]

        out = model(enc.input_ids, output_attentions=True, use_cache=False)
        attn_list = out.attentions  # list of length L, each: [batch, H, S, S]

        # 只取 batch=0
        L_this = len(attn_list)
        H_this = attn_list[0].shape[1]

        # 懒初始化累加器
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

        # ===== 关键改动：逐层 / 逐 head 在 CPU + float32 上计算熵 =====
        if score_mode == "per_head":
            for l, attn_l in enumerate(attn_list):
                # attn_l: [1, H, S, S]，先搬到 CPU 并转成 float32
                attn_l_cpu = attn_l[0].detach().to("cpu", torch.float32)  # [H, S, S]
                for h in range(H_this):
                    m = attn_l_cpu[h]                    # [S, S]
                    svals = torch.linalg.svdvals(m)      # [S]
                    H_val = matrix_based_entropy_from_svals(svals, alpha=alpha)  # 标量
                    score_sum[l, h] += float(H_val)
                    score_cnt[l, h] += 1
        else:  # "per_layer"
            for l, attn_l in enumerate(attn_list):
                attn_l_cpu = attn_l[0].detach().to("cpu", torch.float32)  # [H, S, S]
                m = attn_l_cpu.mean(dim=0)             # [S, S]
                svals = torch.linalg.svdvals(m)        # [S]
                H_val = matrix_based_entropy_from_svals(svals, alpha=alpha)
                score_sum[l] += float(H_val)
                score_cnt[l] += 1

        # 及时释放 GPU 上的中间变量
        del out, attn_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 聚合平均
    avg_entropy = score_sum / np.maximum(score_cnt, 1)

    # 保存 & 绘图
    out_dir_mode = os.path.join(output_dir, f"{task_name}_entropy_{score_mode}_alpha{alpha}")
    os.makedirs(out_dir_mode, exist_ok=True)

    cache_file = os.path.join(out_dir_mode, f"{task_name}_entropy_{score_mode}.npz")
    np.savez_compressed(cache_file,
                        avg_entropy=avg_entropy,
                        count=score_cnt)
    print(f"[{task_name}] Cached entropy saved to {cache_file}")

    if score_mode == "per_head":
        ent_tensor = torch.from_numpy(avg_entropy.astype(np.float32))
        plot_entropy_heatmap_and_layer_mean(
            ent_tensor,
            save_dir=out_dir_mode,
            prefix=f"{task_name}_entropy_alpha{alpha}_per_head",
            show=False,
        )
    else:
        ent_tensor = torch.from_numpy(avg_entropy.astype(np.float32))
        plot_layer_entropy(
            ent_tensor,
            save_dir=out_dir_mode,
            prefix=f"{task_name}_entropy_alpha{alpha}_per_layer_mean_head",
            show=False,
        )

    # ========= 基于熵的 layer / head 自动选择 =========
    valid_mask = score_cnt > 0

    json_top = os.path.join(
        out_dir_mode,
        f"{task_name}_entropy_selected_layers_top_{int(top_fraction * 100)}.json",
    )
    json_auto = os.path.join(
        out_dir_mode,
        f"{task_name}_entropy_selected_layers_auto.json",
    )

    # 固定比例 top-k
    select_layers_by_top_fraction(
        scores=avg_entropy,
        valid_mask=valid_mask,
        score_mode=score_mode,
        num_heads=num_heads,
        top_fraction=top_fraction,
        json_path=json_top,
    )

    # Otsu 自动阈值（方差太小则回退到 top_fraction）
    select_layers_auto_otsu(
        scores=avg_entropy,
        valid_mask=valid_mask,
        score_mode=score_mode,
        num_heads=num_heads,
        json_path=json_auto,
        fallback_top_fraction=top_fraction,
    )

    print(f"[{task_name}] Done.")

# ===========================
# 命令行入口（风格参考 scoring.py）
# ===========================

def parse_args():
    p = argparse.ArgumentParser(description="GraphLens: matrix-based entropy on attention")

    # 基本参数
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to HF model (or backend-specific id)")
    p.add_argument("--task_name", type=str, required=True,
                   help="Task name, used for locating data file and output naming")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to dataset directory containing task files")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save entropy scores and plots")
    p.add_argument("--input_column", type=str, default="input_prompt",
                   help="Column name that contains the graph prompt text")

    # 样本筛选
    p.add_argument("--sample_num", type=int, default=100,
                   help="Number of samples to use for entropy computation")
    p.add_argument("--min_edges", type=int, default=80,
                   help="Min number of edges in graph description")
    p.add_argument("--max_edges", type=int, default=120,
                   help="Max number of edges in graph description")

    # 模型推理相关
    p.add_argument("--max_seq_len", type=int, default=1000,
                   help="Max sequence length for model input")

    # 打分模式
    p.add_argument("--score_mode", type=str, default="per_head",
                   choices=["per_head", "per_layer"],
                   help="Entropy mode: per head or per layer (mean over heads)")

    # 熵的阶数
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Order alpha for matrix-based entropy (alpha=1 is Shannon/von Neumann)")

    # 绘图配置（目前只简单使用 dpi，可扩展）
    p.add_argument("--plot_dpi", type=int, default=200,
                   help="DPI for saved figures")
    p.add_argument("--plot_line_color", type=str, default="C0",
                   help="Line color for per-layer plots")

    # 基于熵选择高分 layer / head 的比例（与 scoring.py 的默认值保持一致）
    p.add_argument("--select_top_fraction", type=float, default=0.4,
                   help="Top fraction used when entropy-based selection falls back from Otsu")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "input_column": args.input_column,
        "sample_num": args.sample_num,
        "min_edges": args.min_edges,
        "max_edges": args.max_edges,
        "max_seq_len": args.max_seq_len,
        "score_mode": args.score_mode,
        "alpha": args.alpha,
        "plot": {
            "dpi": args.plot_dpi,
            "line_color": args.plot_line_color,
        },
        "select_top_fraction": args.select_top_fraction,
    }


    print(f"Loading model from {args.model_path} ...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    tasks = []
    if args.task_name.startswith('GraphWiz'):
        parts = args.task_name.split('_', 1)
        if len(parts) > 1 and parts[1]:
            tasks.append(parts[1])
        else:
            tasks = ["cycle", "connectivity", "hamilton", "substructure", "bipartite", "flow", "shortest", "topology", "triangle"]
    else:
        raise ValueError(f"Unknown task name: {args.task_name}")
    for task in tasks:
        process_task(
            task_name=task,
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()