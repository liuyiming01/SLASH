import os
import argparse
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import (
    load_model_and_tokenizer,
    count_edges_in_prompt,
    standardize_prompt_edges,
    select_layers_by_top_fraction,
    select_layers_otsu,
    select_layers_peak_auto_entropy,
)
from .datasets import (
    load_and_filter_samples,
    choose_edge_range,
    molecularnet_iter_task_label_ids,
    molecularnet_sample_prompts_by_edge_range,
)


def matrix_based_entropy_from_svals(
    svals: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    power = svals ** 2
    power_sum = power.sum(dim=-1, keepdim=True) + eps
    p = power / power_sum

    if abs(alpha - 1.0) < 1e-6:
        log_p = torch.log(p + eps)
        return -(p * log_p).sum(dim=-1)

    H_alpha = (p ** alpha).sum(dim=-1) + eps
    return torch.log(H_alpha) / (1.0 - alpha)


def plot_entropy_heatmap_and_layer_mean(
    ent_lh: torch.Tensor,
    save_dir: str,
    prefix: str = "attn_entropy_per_head",
    show: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    L, H = ent_lh.shape
    ent_np = ent_lh.detach().cpu().numpy()

    plt.figure(figsize=(max(6, H * 0.4), max(4, L * 0.4)))
    im = plt.imshow(ent_np, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label="Matrix-Based Entropy")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Attention Entropy per Layer-Head")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_heatmap.png"), dpi=200)
    if show:
        plt.show()
    plt.close()

    mean_per_layer = ent_np.mean(axis=1)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(L), mean_per_layer, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Mean Entropy over Heads")
    plt.title("Mean Attention Entropy per Layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_layer_mean.png"), dpi=200)
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

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(L), ent_np, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Layer-Level Entropy")
    plt.title("Entropy of Mean-Head Attention per Layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}.png"), dpi=200)
    if show:
        plt.show()
    plt.close()


def _find_data_file(data_dir: str, task_name: str) -> str:
    cand1 = os.path.join(data_dir, f"{task_name}_test.jsonl")
    cand2 = os.path.join(data_dir, f"{task_name}_test.json")
    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2
    raise FileNotFoundError(
        f"Cannot find data file for task '{task_name}' in {data_dir}. Tried: {cand1}, {cand2}"
    )


@torch.no_grad()
def process_task(task_name: str, cfg: dict, model, tokenizer):
    data_dir = cfg["data_dir"]
    input_column = cfg["input_column"]
    requested_sample_num = int(cfg["sample_num"])
    max_seq_len = int(cfg["max_seq_len"])
    score_mode = cfg["score_mode"]  # per_head | per_layer
    alpha = float(cfg["alpha"])
    output_dir = cfg["output_dir"]
    top_fraction = float(cfg.get("select_top_fraction", 0.4))

    graphsst = cfg.get("graphsst", None)
    molecularnet = cfg.get("molecularnet", None)

    # ========== MolecularNet ==========
    if molecularnet is not None:
        preferred_min_edges = int(cfg.get("preferred_min_edges", 60))
        hard_max_edges = cfg.get("hard_max_edges", None)
        samples, (chosen_min, chosen_max), stats = molecularnet_sample_prompts_by_edge_range(
            molecularnet=molecularnet,
            input_column=input_column,
            sample_num=requested_sample_num,
            preferred_min_edges=preferred_min_edges,
            hard_max_edges=hard_max_edges,
            seed=42,
            require_label=True,
        )
        if samples is None or len(samples) == 0:
            print(f"[{task_name}] No MolecularNet samples after auto edge_range sampling. Skip.")
            return

        cfg["min_edges"] = int(chosen_min)
        cfg["max_edges"] = int(chosen_max)

        print(
            f"[{task_name}] TRUE edge stats (valid pool): "
            f"min={int(stats.get('true_min_edges', -1))}, "
            f"max={int(stats.get('true_max_edges', -1))}, "
            f"median={float(stats.get('true_median_edges', float('nan'))):.1f}, "
            f"valid_n={int(stats.get('true_valid_pool_n', stats.get('valid_n', -1)))}"
        )
        print(
            f"[{task_name}] CHOSEN edge range: [{chosen_min}, {chosen_max}], "
            f"used_preferred_min={stats.get('used_preferred_min')}, "
            f"in_range_n={stats.get('in_range_n')}, "
            f"chosen_n={stats.get('chosen_n')}, "
            f"built_prompts_n={stats.get('built_prompts_n')}, "
            f"mode={stats.get('mode')}"
        )

        subset = samples
        all_df = samples
    else:
        data_path = None if (graphsst is not None or molecularnet is not None) else (cfg.get("data_path") or _find_data_file(data_dir, task_name))

        all_df = load_and_filter_samples(
            data_path=data_path,
            input_column=input_column,
            min_edges=0,
            max_edges=10**9,
            sample_num=10**9,
            graphsst=graphsst,
            molecularnet=molecularnet,
        )
        if all_df is None or len(all_df) == 0:
            print(f"[{task_name}] No samples. Skip.")
            return

        if "__num_edges" in all_df.columns:
            edge_counts = all_df["__num_edges"].to_numpy(dtype=np.int32)
        else:
            edge_counts = all_df[input_column].map(lambda s: count_edges_in_prompt(standardize_prompt_edges(s))).to_numpy(dtype=np.int32)

        print(
            f"[{task_name}] TRUE edge stats: min={int(edge_counts.min())}, "
            f"max={int(edge_counts.max())}, median={float(np.median(edge_counts)):.1f}, n={len(edge_counts)}"
        )

        preferred_min_edges = int(cfg.get("preferred_min_edges", 60))
        hard_max_edges = cfg.get("hard_max_edges", None)

        chosen_min, chosen_max, stats = choose_edge_range(
            edge_counts=edge_counts,
            sample_num=requested_sample_num,
            preferred_min_edges=preferred_min_edges,
            hard_max_edges=hard_max_edges,
        )
        if chosen_min is None:
            print(f"[{task_name}] Cannot choose edge range.")
            return

        cfg["min_edges"] = int(chosen_min)
        cfg["max_edges"] = int(chosen_max)

        mask = (edge_counts >= chosen_min) & (edge_counts <= chosen_max)
        subset = all_df.loc[mask]
        if len(subset) < requested_sample_num:
            print(f"[{task_name}] Warning: subset size {len(subset)} < sample_num {requested_sample_num}, fallback to full set.")
            subset = all_df

        samples = subset.sample(n=min(requested_sample_num, len(subset)), random_state=42).reset_index(drop=True)

        print(
            f"[{task_name}] CHOSEN edge range: [{chosen_min}, {chosen_max}], "
            f"used_preferred_min={stats.get('used_preferred_min')}, "
            f"mode={stats.get('mode')}"
        )

    # Cache reuse
    out_dir_mode = os.path.join(output_dir, f"{task_name}_entropy_{score_mode}")
    os.makedirs(out_dir_mode, exist_ok=True)

    cache_file = os.path.join(out_dir_mode, f"{task_name}_entropy_{score_mode}.npz")
    meta_file = os.path.join(out_dir_mode, f"{task_name}_entropy_{score_mode}.meta.json")
    print(f"[{task_name}] Cache file: {cache_file}")
    meta = {
        "task_name": task_name,
        "score_mode": score_mode,
        "alpha": float(alpha),
        "sample_num": int(requested_sample_num),
        "max_seq_len": int(max_seq_len),
        "preferred_min_edges": int(preferred_min_edges),
        "hard_max_edges": (None if hard_max_edges is None else int(hard_max_edges)),
        "chosen_min_edges": int(chosen_min),
        "chosen_max_edges": int(chosen_max),
        "used_preferred_min": bool(stats.get("used_preferred_min")),
        "selection_mode": str(stats.get("mode")),
        "n_total": int(len(all_df)),
        "n_subset": int(len(subset)),
        "n_samples": int(len(samples)),
    }

    avg_entropy = None
    score_cnt = None

    if os.path.exists(cache_file) and os.path.exists(meta_file):
        try:
            with open(meta_file, "r") as f:
                old_meta = json.load(f)
            if old_meta == meta:
                data = np.load(cache_file)
                avg_entropy = data["avg_entropy"]
                score_cnt = data["count"]
                print(f"[{task_name}] Cache hit: {cache_file} (skip entropy compute)")
        except Exception as e:
            print(f"[{task_name}] Cache load failed, will recompute. Reason: {e}")

    if avg_entropy is None or score_cnt is None:
        score_sum = None
        score_cnt = None
        num_layers = None
        num_heads = None

        empty_cache_every = int(cfg.get("empty_cache_every", 20))

        for i, row in enumerate(tqdm(samples.itertuples(index=False), total=len(samples), desc=f"[{task_name}] Entropy")):
            prompt = getattr(row, input_column)
            prompt = standardize_prompt_edges(prompt)
            print(i, prompt)
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            if enc.input_ids.shape[1] > max_seq_len:
                enc.input_ids = enc.input_ids[:, :max_seq_len]
                if "attention_mask" in enc:
                    enc.attention_mask = enc.attention_mask[:, :max_seq_len]

            out = model(enc.input_ids, output_attentions=True, use_cache=False)
            attn_list = out.attentions  # list[L], each: [B, H, S, S]

            L_this = len(attn_list)
            H_this = attn_list[0].shape[1]

            if score_sum is None:
                num_layers, num_heads = L_this, H_this
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
                    a = attn_l[0].detach().to("cpu", torch.float32)
                    for h in range(H_this):
                        svals = torch.linalg.svdvals(a[h])
                        score_sum[l, h] += float(matrix_based_entropy_from_svals(svals, alpha=alpha))
                        score_cnt[l, h] += 1
            else:
                for l, attn_l in enumerate(attn_list):
                    a = attn_l[0].detach().to("cpu", torch.float32)
                    m = a.mean(dim=0)
                    svals = torch.linalg.svdvals(m)
                    score_sum[l] += float(matrix_based_entropy_from_svals(svals, alpha=alpha))
                    score_cnt[l] += 1

            if torch.cuda.is_available() and empty_cache_every > 0 and ((i + 1) % empty_cache_every == 0):
                torch.cuda.empty_cache()

        avg_entropy = score_sum / np.maximum(score_cnt, 1)

        np.savez_compressed(cache_file, avg_entropy=avg_entropy, count=score_cnt)
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        print(f"[{task_name}] Cached entropy saved to {cache_file}")
        print(f"[{task_name}] Cache meta saved to {meta_file}")

    ent_tensor = torch.from_numpy(avg_entropy.astype(np.float32))
    if score_mode == "per_head":
        plot_entropy_heatmap_and_layer_mean(
            ent_tensor,
            save_dir=out_dir_mode,
            prefix=f"{task_name}_entropy_alpha{alpha}_per_head",
            show=False,
        )
        num_heads = int(avg_entropy.shape[1])
    else:
        plot_layer_entropy(
            ent_tensor,
            save_dir=out_dir_mode,
            prefix=f"{task_name}_entropy_alpha{alpha}_per_layer_mean_head",
            show=False,
        )
        num_heads = int(getattr(model.config, "num_attention_heads", 0))

    valid_mask = score_cnt > 0

    json_top = os.path.join(out_dir_mode, f"{task_name}_selected_layers_top_{int(top_fraction * 100)}_entropy.json")
    json_otsu = os.path.join(out_dir_mode, f"{task_name}_selected_layers_otsu_entropy.json")
    json_auto = os.path.join(out_dir_mode, f"{task_name}_selected_layers_auto_entropy.json")

    select_layers_by_top_fraction(
        scores=avg_entropy,
        valid_mask=valid_mask,
        score_mode=score_mode,
        num_heads=num_heads,
        top_fraction=top_fraction,
        json_path=json_top,
    )
    select_layers_otsu(
        scores=avg_entropy,
        valid_mask=valid_mask,
        score_mode=score_mode,
        num_heads=num_heads,
        json_path=json_otsu,
        fallback_top_fraction=top_fraction,
    )
    mid_sel, mid_info = select_layers_peak_auto_entropy(
        scores=avg_entropy,
        valid_mask=valid_mask,
        score_mode=score_mode,
        num_heads=num_heads,
        json_path=json_auto,
        fallback_top_fraction=top_fraction,
    )

    print(f"[{task_name}] Done.")


def parse_args():
    p = argparse.ArgumentParser(description="SLASH: matrix-based entropy on attention")

    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--task_name", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--input_column", type=str, default="input_prompt")
    p.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Path to the graph prompt template file.",
    )
    p.add_argument("--sample_num", type=int, default=100)
    p.add_argument("--max_seq_len", type=int, default=1000)

    p.add_argument("--score_mode", type=str, default="per_head", choices=["per_head", "per_layer"])
    p.add_argument("--alpha", type=float, default=1.0)

    p.add_argument("--plot_dpi", type=int, default=200)
    p.add_argument("--plot_line_color", type=str, default="C0")
    p.add_argument("--select_top_fraction", type=float, default=0.4)

    # Graph-SST
    p.add_argument("--split", type=str, default="test")

    # perf
    p.add_argument("--empty_cache_every", type=int, default=20, help="Call torch.cuda.empty_cache every N samples (0 to disable).")

    p.add_argument("--preferred_min_edges", type=int, default=60)
    p.add_argument("--hard_max_edges", type=int, default=-1)

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
        "plot": {"dpi": args.plot_dpi, "line_color": args.plot_line_color},
        "select_top_fraction": args.select_top_fraction,
        "data_path": None,
        "graphsst": None,
        "molecularnet": None,
        "empty_cache_every": args.empty_cache_every,
        "preferred_min_edges": int(args.preferred_min_edges),
        "hard_max_edges": (None if int(args.hard_max_edges) < 0 else int(args.hard_max_edges)),
    }

    print(f"Loading model from {args.model_path} ...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    if args.task_name.startswith("GraphWiz"):
        parts = args.task_name.split("_", 1)
        tasks = [parts[1]] if (len(parts) > 1 and parts[1]) else [
            "cycle", "connectivity", "hamilton", "substructure",
            "bipartite", "flow", "shortest", "topology", "triangle"
        ]
        for task in tasks:
            cfg["graphsst"] = None
            cfg["data_path"] = None
            process_task(task_name=task, cfg=cfg, model=model, tokenizer=tokenizer)
        return

    elif args.task_name in ("Graph-SST", "Graph-SST2", "Graph-SST5", "Graph-Twitter"):
        if args.task_name == "Graph-SST":
            tasks = ["Graph-SST2", "Graph-SST5", "Graph-Twitter"]
        else:
            tasks = [args.task_name]
        
        for task in tasks:
            cfg["data_path"] = None
            cfg["graphsst"] = {
                "root": args.data_dir,
                "name": task,
                "split": args.split,
                "prompt_path": args.prompt_path,
            }
            process_task(task, cfg, model, tokenizer)

    elif args.task_name.startswith("Mol"):
        suffix = ""
        parts = args.task_name.split("_", 1)
        if len(parts) > 1 and parts[1]:
            suffix = parts[1]

        for task_id, task, label_col in molecularnet_iter_task_label_ids(suffix or None):
            cfg["data_path"] = None
            cfg["graphsst"] = None
            cfg["molecularnet"] = {
                "root": args.data_dir,
                "task": task,
                "label_col": label_col,  # Tox21 uses __TOX21_TOXIC_ANY__
                "split": args.split,      # "test" or "sample"
                "prompt_path": args.prompt_path,
                "shot": 0,
                "seed": 42,
                "weighted_edges": False,
            }
            process_task(task_name=task_id, cfg=cfg, model=model, tokenizer=tokenizer)
        return

    else:
        raise ValueError(f"Unknown task name: {args.task_name}")


if __name__ == "__main__":
    main()