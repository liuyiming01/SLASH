import os
import argparse
import json

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ..utils import (
    load_model_and_tokenizer,
    get_token_spans,
    compute_global_span,
    build_local_spans,
    build_sawtooth_mask,
    extract_roi_from_attn,
    preprocess_for_scoring,
    score_attention_map,
    standardize_prompt_edges,
    count_edges_in_prompt,
    auto_edge_range,
)
from ..viz_utils import (
    choose_sample_for_task,
    create_layer_figure,
    create_head_figure,
)
from ..datasets import load_and_filter_samples, choose_edge_range, molecularnet_iter_task_label_ids

def _attach_edge_counts(df: pd.DataFrame, input_column: str) -> pd.DataFrame:
    if "__num_edges" not in df.columns:
        df = df.copy()
        df["__num_edges"] = df[input_column].map(
            lambda x: count_edges_in_prompt(standardize_prompt_edges(x)) if isinstance(x, str) else 0
        ).astype(np.int32)
    return df

def _select_k_near_median(
    df: pd.DataFrame,
    input_column: str,
    k: int,
    task_name: str,
    preferred_min_edges: int = 50,
    hard_max_edges: int = -1,
):
    """
    Select k samples closest to the (eligible) median.
    Prefer edges >= preferred_min_edges; fallback to all if not enough (Graph-SST friendly).
    Optionally apply hard_max_edges cap to control compute.
    """
    df = _attach_edge_counts(df, input_column)

    df_use = df
    cap = (None if int(hard_max_edges) < 0 else int(hard_max_edges))
    if cap is not None:
        df_use = df_use[df_use["__num_edges"] <= cap].copy()
        if len(df_use) == 0:
            print(f"[{task_name}] Empty after hard_max_edges={cap}.")
            return None, None, None

    edge_counts = df_use["__num_edges"].to_numpy(dtype=np.int32)
    print(
        f"[{task_name}] TRUE(edge<=cap) stats: min={int(edge_counts.min())}, max={int(edge_counts.max())}, "
        f"median={float(np.median(edge_counts)):.1f}, n={len(edge_counts)}"
    )

    k_req = int(k)
    k = int(max(1, min(k_req, len(df_use))))

    chosen_min, chosen_max, stats = choose_edge_range(
        edge_counts=edge_counts,
        sample_num=k,
        preferred_min_edges=int(preferred_min_edges),
        hard_max_edges=cap,
    )
    if chosen_min is None or chosen_max is None:
        print(f"[{task_name}] choose_edge_range failed: {stats}")
        return None, None, None

    pool = df_use[(df_use["__num_edges"] >= int(chosen_min)) & (df_use["__num_edges"] <= int(chosen_max))].copy()
    if len(pool) == 0:
        print(f"[{task_name}] Empty after applying chosen range [{chosen_min},{chosen_max}].")
        return None, None, None

    # Pick k samples closest to the median of the eligible pool.
    median_val = float(np.median(pool["__num_edges"].to_numpy(dtype=np.int32)))
    pool["__dist_to_median"] = np.abs(pool["__num_edges"].astype(np.float32) - median_val)
    df_k = pool.sort_values(
        by=["__dist_to_median", "__num_edges"],
        ascending=[True, True],
        kind="mergesort",
    ).head(k).reset_index(drop=True)

    print(
        f"[{task_name}] MEDIAN-topk: k={k}, range=[{int(chosen_min)},{int(chosen_max)}], "
        f"preferred_min_edges={int(preferred_min_edges)}, used_pref={bool(stats.get('used_preferred_min', False))}, "
        f"hard_max_edges={cap}"
    )
    return df_k, int(chosen_min), int(chosen_max)

def _save_prompt_json(save_dir: str, task: str, chosen_idx: int, num_edges: int, prompt: str, standardized: bool):
    """
    Save the input prompt (and minimal metadata) as JSON in the plotting directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "task": task,
        "chosen_idx": int(chosen_idx),
        "num_edges": int(num_edges),
        "standardized": bool(standardized),
        "prompt": prompt,
    }
    out_path = os.path.join(save_dir, "input_prompt.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path

def plot_layer_mode(task: str,
                    df: pd.DataFrame,
                    model,
                    tokenizer,
                    sim_metric: str,
                    binarize_method: str,
                    pre_threshold_frac: float,
                    save_root: str,
                    dpi: int,
                    input_column: str = "input_prompt",
                    sample_min_edges: int = 100,
                    sample_max_edges: int = 120,
                    max_seq_len: int = 1000,
                    chosen_indices: list = None,
                    standardize_prompt: bool = True):
    """模式一：layer 平均模式。"""
    model.eval()
    # 如果 main() 已经选好了（median-topk），这里直接用 chosen_indices
    if chosen_indices is None:
        idxs, num_edges_list = choose_sample_for_task(
            df,
            min_edges=sample_min_edges,
            max_edges=sample_max_edges,
            input_column=input_column,
            max_samples=5,
        )
        if not idxs:
            print(f"  No sample in [{sample_min_edges},{sample_max_edges}] edges for task {task}, skip.")
            return
        chosen_indices = [idxs[0]]
    else:
        # chosen_indices 是 df_k 的索引；num_edges 直接从 df 取
        num_edges_list = [int(df.iloc[i].get("__num_edges", count_edges_in_prompt(df.iloc[i][input_column])))
                          for i in chosen_indices]

    for t_i, chosen_idx in enumerate(chosen_indices):
        item = df.iloc[chosen_idx]
        prompt = item[input_column]

        if standardize_prompt:
            prompt = standardize_prompt_edges(prompt)

        ne = count_edges_in_prompt(prompt)
        print(f"  [layer mode] Chosen index: {chosen_idx}, num_edges={ne}")

        sample_save_root = os.path.join(save_root, f"sample_{chosen_idx}")
        os.makedirs(sample_save_root, exist_ok=True)

        prompt_json_path = _save_prompt_json(
            save_dir=sample_save_root,
            task=task,
            chosen_idx=chosen_idx,
            num_edges=ne,
            prompt=prompt,
            standardized=standardize_prompt,
        )
        print(f"  [layer mode] Saved input prompt to: {prompt_json_path}")

        spans = get_token_spans(prompt, tokenizer)
        if not spans:
            print("  No spans found, skip.")
            continue

        g_start, g_end, span_len, spans_sorted = compute_global_span(spans)
        local_spans = build_local_spans(g_start, spans_sorted)
        ideal_mask = build_sawtooth_mask(span_len, local_spans)

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        ).to(model.device)

        with torch.no_grad():
            out = model(enc.input_ids, output_attentions=True, use_cache=False)

        attn_np = np.stack(
            [a[0].to(torch.float32).detach().cpu().numpy() for a in out.attentions],
            axis=0
        )  # [L, H, S, S]

        if attn_np.shape[2] <= g_end:
            print("  ROI out of range, skip.")
            continue

        num_layers, num_heads = attn_np.shape[:2]

        layer_scores = []
        for l in range(num_layers):
            layer_rois = []
            full_attn_heads = []
            for h in range(num_heads):
                roi = extract_roi_from_attn(attn_np, l, h, g_start, g_end)
                if roi.shape != ideal_mask.shape:
                    continue
                layer_rois.append(roi)
                full_attn_heads.append(attn_np[l, h])

            if not layer_rois:
                layer_scores.append(np.nan)
                continue

            avg_roi = np.mean(np.stack(layer_rois, axis=0), axis=0)

            bin_mask, denoised_mask = preprocess_for_scoring(
                avg_roi,
                binarize_method=binarize_method,
                ideal_mask=ideal_mask,
                pre_threshold_frac=pre_threshold_frac,
            )
            score_avg = score_attention_map(
                avg_roi,
                local_spans,
                ideal_mask,
                sim_metric=sim_metric,
                binarize_method=binarize_method,
                pre_threshold_frac=pre_threshold_frac,
            )
            layer_scores.append(score_avg)

            full_attn_layer = np.mean(np.stack(full_attn_heads, axis=0), axis=0)
            title = f"L{l}_Avg (avg_map_score={score_avg:.3f})"

            fig = create_layer_figure(
                full_attn_layer,
                avg_roi,
                bin_mask,
                denoised_mask,
                title,
                local_spans,
                g_start,
                g_end,
            )
            fig.savefig(
                os.path.join(sample_save_root, f"L{l}_Avg_s{score_avg:.3f}.png"),
                dpi=dpi
            )
            plt.close(fig)

        plt.figure(figsize=(8, 4))
        x = np.arange(num_layers)
        plt.plot(x, layer_scores, marker="o", linewidth=1.5)
        plt.title(f"Layer Score Trend - {task}")
        plt.xlabel("Layer")
        plt.ylabel(f"Score ({sim_metric})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        lineplot_path = os.path.join(sample_save_root, f"layer_scores_{task}.png")
        plt.savefig(lineplot_path, dpi=dpi)
        plt.close()

        print(f"  [layer mode] Layer average images saved to: {sample_save_root}")
        print(f"  [layer mode] Layer score line plot saved to: {lineplot_path}")


def plot_head_mode(task: str,
                   df: pd.DataFrame,
                   model,
                   tokenizer,
                   sim_metric: str,
                   binarize_method: str,
                   pre_threshold_frac: float,
                   save_root: str,
                   dpi: int,
                   input_column: str = "input_prompt",
                   sample_min_edges: int = 20,
                   sample_max_edges: int = 30,
                   max_seq_len: int = 1000,
                   chosen_indices: list = None,
                   standardize_prompt: bool = True):
    """模式二：head 单独评分模式 + layer-head 热力图 + 每层 head 平均分折线图。"""
    model.eval()
    if chosen_indices is None:
        idxs, num_edges_list = choose_sample_for_task(
            df,
            min_edges=sample_min_edges,
            max_edges=sample_max_edges,
            input_column=input_column,
            max_samples=5,
        )
        if not idxs:
            print(f"  No sample in [{sample_min_edges},{sample_max_edges}] edges for task {task}, skip.")
            return
        chosen_indices = [idxs[0]]
    else:
        num_edges_list = [int(df.iloc[i].get("__num_edges", count_edges_in_prompt(df.iloc[i][input_column])))
                          for i in chosen_indices]

    for t_i, chosen_idx in enumerate(chosen_indices):
        item = df.iloc[chosen_idx]
        prompt = item[input_column]

        if standardize_prompt:
            prompt = standardize_prompt_edges(prompt)

        ne = count_edges_in_prompt(prompt)
        print(f"  [head mode] Chosen index: {chosen_idx}, num_edges={ne}")

        sample_save_root = os.path.join(save_root, f"sample_{chosen_idx}")
        os.makedirs(sample_save_root, exist_ok=True)

        prompt_json_path = _save_prompt_json(
            save_dir=sample_save_root,
            task=task,
            chosen_idx=chosen_idx,
            num_edges=ne,
            prompt=prompt,
            standardized=standardize_prompt,
        )
        print(f"  [head mode] Saved input prompt to: {prompt_json_path}")

        spans = get_token_spans(prompt, tokenizer)
        if not spans:
            print("  No spans found, skip.")
            continue

        g_start, g_end, span_len, spans_sorted = compute_global_span(spans)
        local_spans = build_local_spans(g_start, spans_sorted)
        ideal_mask = build_sawtooth_mask(span_len, local_spans)

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        ).to(model.device)

        with torch.no_grad():
            out = model(enc.input_ids, output_attentions=True, use_cache=False)
        attn_np = np.stack(
            [a[0].to(torch.float32).detach().cpu().numpy() for a in out.attentions],
            axis=0
        )  # [L, H, S, S]

        if attn_np.shape[2] <= g_end:
            print("  ROI out of range, skip.")
            continue

        num_layers, num_heads = attn_np.shape[:2]

        scores_lh = np.full((num_layers, num_heads), np.nan, dtype=np.float32)

        for l in range(num_layers):
            for h in range(num_heads):
                roi = extract_roi_from_attn(attn_np, l, h, g_start, g_end)
                if roi.shape != ideal_mask.shape:
                    continue

                bin_mask, denoised_mask = preprocess_for_scoring(
                    roi,
                    binarize_method=binarize_method,
                    ideal_mask=ideal_mask,
                    pre_threshold_frac=pre_threshold_frac,
                )
                score = score_attention_map(
                    roi,
                    local_spans,
                    ideal_mask,
                    sim_metric=sim_metric,
                    binarize_method=binarize_method,
                    pre_threshold_frac=pre_threshold_frac,
                )
                scores_lh[l, h] = score

                full_attn_head = attn_np[l, h]
                title = f"L{l}_H{h} (score={score:.3f})"
                fig = create_head_figure(
                    full_attn_head,
                    roi,
                    bin_mask,
                    denoised_mask,
                    title,
                    local_spans,
                    g_start,
                    g_end,
                )
                fig_path = os.path.join(sample_save_root, f"L{l}_H{h}_s{score:.3f}.png")
                fig.savefig(fig_path, dpi=dpi)
                plt.close(fig)

        plt.figure(figsize=(10, 6))
        im = plt.imshow(scores_lh, aspect="auto", cmap="coolwarm", origin="lower")
        plt.xlabel("Head")
        plt.ylabel("Layer")
        plt.xticks(np.arange(num_heads))
        plt.yticks(np.arange(num_layers))
        plt.title(f"{task} - per-head scores ({sim_metric}, {binarize_method})")
        cbar = plt.colorbar(im)
        cbar.set_label("Score")
        plt.tight_layout()
        heatmap_path = os.path.join(sample_save_root, f"{task}_perhead_heatmap.png")
        plt.savefig(heatmap_path, dpi=dpi)
        plt.close()

        layer_mean_scores = np.nanmean(scores_lh, axis=1)
        plt.figure(figsize=(8, 4))
        x = np.arange(num_layers)
        plt.plot(x, layer_mean_scores, marker="o", linewidth=1.5)
        plt.title(f"Layer-wise Mean Head Score - {task}")
        plt.xlabel("Layer")
        plt.ylabel(f"Mean Score ({sim_metric})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        lineplot_path = os.path.join(sample_save_root, f"{task}_layer_mean_head_scores.png")
        plt.savefig(lineplot_path, dpi=dpi)
        plt.close()

        print(f"  [head mode] Per-head figures saved to: {sample_save_root}")
        print(f"  [head mode] Heatmap saved to: {heatmap_path}")
        print(f"  [head mode] Layer mean head score line plot saved to: {lineplot_path}")

def parse_args():
    p = argparse.ArgumentParser(description="GraphLens attention visualization (per-layer / per-head)")

    p.add_argument("--model_path", type=str, required=True,
                   help="Path to HF model (or backend-specific id)")
    p.add_argument("--task_name", type=str, required=True,
                   help="Task name, used only for output naming (e.g., GraphWiz_cycle or GraphWiz)")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to dataset directory containing task files")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save visualization images")
    p.add_argument("--input_column", type=str, default="input_prompt",
                   help="Column name that contains the graph prompt text")
    p.add_argument(
        "--no_standardize_prompt",
        action="store_true",
        help="Disable prompt edge standardization. Default is ON (standardize).",
    )
    p.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Path to prompt template file (required for Graph-SST; optional for Mol).",
    )
    # Graph-SST specific
    p.add_argument("--split", type=str, default="test",
                   help="Graph-SST split: train/val/valid/test")

    # 关键修改：以中位数为中心选 k 个样本（默认 10，且最多 10）
    p.add_argument("--k_median", type=int, default=10,
                   help="Select top-k samples closest to median edge count (cap at 10).")
    p.add_argument("--plot_topk", type=int, default=1,
                   help="How many of the selected k samples to plot (starting from idx=0). Max=k.")

    # 兼容保留，但 median-topk 策略不再依赖它们
    p.add_argument("--min_edges", type=int, default=80,
                   help="(Deprecated) Min edges. Median-topk selection ignores this.")
    p.add_argument("--max_edges", type=int, default=120,
                   help="(Deprecated) Max edges. Median-topk selection ignores this.")

    p.add_argument("--max_seq_len", type=int, default=1000,
                   help="Max sequence length for model input")

    p.add_argument("--sim_metric", type=str, default="concentration")
    p.add_argument("--binarize_method", type=str, default="topk")
    p.add_argument("--pre_threshold_frac", type=float, default=0.1)

    p.add_argument("--plot_mode", type=str, default="layer",
                   choices=["layer", "head"])
    p.add_argument("--plot_dpi", type=int, default=250)

    p.add_argument("--preferred_min_edges", type=int, default=60)
    p.add_argument("--hard_max_edges", type=int, default=-1)

    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path} ...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    standardize_prompt = (not args.no_standardize_prompt)
    tasks = []
    is_graphsst = False

    if args.task_name.startswith('GraphWiz'):
        parts = args.task_name.split('_', 1)
        if len(parts) > 1 and parts[1]:
            tasks.append(parts[1])
        else:
            tasks = [
                "cycle", "connectivity", "hamilton", "substructure",
                "bipartite", "flow", "shortest", "topology", "triangle"
            ]
    elif args.task_name in ("Graph-SST", "Graph-SST2", "Graph-SST5", "Graph-Twitter"):
        if not args.prompt_path:
            raise ValueError("Graph-SST plotting requires --prompt_path")
        if args.task_name == "Graph-SST":
            tasks = ["Graph-SST2", "Graph-SST5", "Graph-Twitter"]
        else:
            tasks = [args.task_name]
        is_graphsst = True
    elif args.task_name.startswith("Mol"):
        # MolecularNet family: iterate tasks (optionally filtered by suffix)
        suffix = ""
        parts = args.task_name.split("_", 1)
        if len(parts) > 1 and parts[1]:
            suffix = parts[1]

        mol_specs = molecularnet_iter_task_label_ids(suffix or None)
        # Encode each MolecularNet task as a unique "task" in this plotting script.
        tasks = [task_id for task_id, _, _ in mol_specs]
        is_graphsst = False
    else:
        raise ValueError(f"Unknown task name: {args.task_name}")

    for task in tasks:
        print(f"\n=== Plot Task: {task} (mode={args.plot_mode}) ===")

        # Load FULL df
        if args.task_name.startswith("Mol"):
            # task here is task_id
            suffix = ""
            parts = args.task_name.split("_", 1)
            if len(parts) > 1 and parts[1]:
                suffix = parts[1]
            mol_specs = {task_id: (t, label) for task_id, t, label in molecularnet_iter_task_label_ids(suffix or None)}
            if task not in mol_specs:
                print(f"  Unknown MolecularNet task_id={task}, skip.")
                continue
            m_task, m_label = mol_specs[task]
            df_full = load_and_filter_samples(
                data_path=None,
                input_column=args.input_column,
                min_edges=0,
                max_edges=10**9,
                sample_num=10**9,
                molecularnet={
                    "root": args.data_dir,
                    "task": m_task,
                    "label_col": m_label,
                    "split": args.split,
                    "prompt_path": args.prompt_path,
                    "shot": 0,
                    "seed": 42,
                    "weighted_edges": False,
                },
            )
            if df_full is None or len(df_full) == 0:
                print("  No samples loaded for MolecularNet, skip.")
                continue
        elif not is_graphsst:
            data_path = os.path.join(args.data_dir, f"{task}_test.json")
            if not os.path.exists(data_path):
                print(f"  Data file not found: {data_path}, skip.")
                continue
            df_full = pd.read_json(data_path, lines=True)
            if args.input_column not in df_full.columns:
                print(f"  Missing column '{args.input_column}' in {data_path}, skip.")
                continue
        else:
            df_full = load_and_filter_samples(
                data_path=None,
                input_column=args.input_column,
                min_edges=0,
                max_edges=10**9,
                sample_num=10**9,  # load all for median stats / selection
                graphsst={
                    "root": args.data_dir,
                    "name": task,
                    "split": args.split,
                    "prompt_path": args.prompt_path,
                },
            )
            if df_full is None or len(df_full) == 0:
                print("  No samples loaded for Graph-SST, skip.")
                continue

        # Median-centered top-k selection (cap at 10)
        k = int(min(max(args.k_median, 1), 10))
        df_k, chosen_min, chosen_max = _select_k_near_median(
            df_full,
            input_column=args.input_column,
            k=k,
            task_name=task,
            preferred_min_edges=int(args.preferred_min_edges),
            hard_max_edges=int(args.hard_max_edges),
        )
        if df_k is None:
            print("  Median-topk selection produced empty set, skip.")
            continue

        # Plot first plot_topk samples among df_k (idx=0 is the closest-to-median)
        plot_n = int(min(max(args.plot_topk, 1), len(df_k), k))
        chosen_indices = list(range(plot_n))  # [0,1,...]

        task_save_root = os.path.join(
            args.output_dir,
            os.path.basename(args.model_path),
            f"task_{task}",
            f"mode_{args.plot_mode}_{args.binarize_method}_{args.sim_metric}_{'std' if standardize_prompt else 'raw'}",
            f"median_topk_{k}_edges_{chosen_min}_{chosen_max}",
        )

        if args.plot_mode == "layer":
            plot_layer_mode(
                task=task,
                df=df_k,
                model=model,
                tokenizer=tokenizer,
                sim_metric=args.sim_metric,
                binarize_method=args.binarize_method,
                pre_threshold_frac=args.pre_threshold_frac,
                save_root=task_save_root,
                dpi=args.plot_dpi,
                input_column=args.input_column,
                sample_min_edges=chosen_min,
                sample_max_edges=chosen_max,
                max_seq_len=args.max_seq_len,
                chosen_indices=chosen_indices,
                standardize_prompt=standardize_prompt,
            )
        elif args.plot_mode == "head":
            plot_head_mode(
                task=task,
                df=df_k,
                model=model,
                tokenizer=tokenizer,
                sim_metric=args.sim_metric,
                binarize_method=args.binarize_method,
                pre_threshold_frac=args.pre_threshold_frac,
                save_root=task_save_root,
                dpi=args.plot_dpi,
                input_column=args.input_column,
                sample_min_edges=chosen_min,
                sample_max_edges=chosen_max,
                max_seq_len=args.max_seq_len,
                chosen_indices=chosen_indices,
                standardize_prompt=standardize_prompt,
            )
        else:
            print(f"  Unknown plot mode: {args.plot_mode}, skip task {task}.")


if __name__ == "__main__":
    main()