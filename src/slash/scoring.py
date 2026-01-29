import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .utils import *
from .viz_utils import *
from .datasets import (
    load_and_filter_samples,
    choose_edge_range,
    molecularnet_iter_task_label_ids,
    molecularnet_sample_prompts_by_edge_range,
)


def _init_score_accumulators(score_mode: str, num_layers: int, num_heads: int):
    """
    per_head / per_layer
    """
    if score_mode == "per_head":
        score_sum = np.zeros((num_layers, num_heads), dtype=np.float32)
        score_cnt = np.zeros((num_layers, num_heads), dtype=np.int32)
        return {"mode": "per_head", "score_sum": score_sum, "score_cnt": score_cnt}
    elif score_mode == "per_layer":
        score_sum_layer = np.zeros((num_layers,), dtype=np.float32)
        score_cnt_layer = np.zeros((num_layers,), dtype=np.int32)
        return {"mode": "per_layer", "score_sum": score_sum_layer, "score_cnt": score_cnt_layer}
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

def _score_single_example(
    prompt: str,
    cfg: dict,
    model,
    tokenizer,
    accumulators: dict,
    num_layers: int,
    num_heads: int,
):
    """
      1) span & ideal_mask
      2) attention
    """
    if cfg.get("standardize_prompt", True):
        prompt = standardize_prompt_edges(prompt)

    max_seq_len = cfg["max_seq_len"]
    score_mode = accumulators["mode"]

    sim_metric = cfg["sim_metric"]               # "concentration" | "gradient"
    binarize_method = cfg["binarize_method"]     # "threshold" | "topk"
    pre_threshold_frac = cfg["pre_threshold_frac"]

    spans = get_token_spans(prompt, tokenizer)
    if not spans:
        return

    g_start, g_end, span_len, spans_sorted = compute_global_span(spans)
    local_spans = build_local_spans(g_start, spans_sorted)
    ideal_mask = build_sawtooth_mask(span_len, local_spans)

    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    if enc.input_ids.shape[1] > max_seq_len:
        enc.input_ids = enc.input_ids[:, :max_seq_len]
        if "attention_mask" in enc:
            enc.attention_mask = enc.attention_mask[:, :max_seq_len]

    out = model(enc.input_ids, output_attentions=True, use_cache=False)
    attn_list = out.attentions  # [batch, H, S, S]

    seq_len = attn_list[0].shape[-1]
    # print(f"    Sample seq_len: {seq_len}, graph span: [{g_start}, {g_end}], span_len: {span_len}")
    if seq_len <= g_end:
        del out
        return

    if score_mode == "per_head":
        score_sum = accumulators["score_sum"]
        score_cnt = accumulators["score_cnt"]

        for l in range(num_layers):
            attn_layer = attn_list[l][0]  # [H, S, S]
            attn_layer_roi = attn_layer[:, g_start:g_end + 1, g_start:g_end + 1] \
                .to(torch.float32).cpu().numpy()  # [H, N, N]

            if attn_layer_roi.shape[1] != span_len or attn_layer_roi.shape[2] != span_len:
                continue

            for h in range(num_heads):
                attn_roi = attn_layer_roi[h]  # [N, N]
                score = score_attention_map(
                    attn_roi,
                    local_spans,
                    ideal_mask,
                    sim_metric=sim_metric,
                    binarize_method=binarize_method,
                    pre_threshold_frac=pre_threshold_frac,
                )
                score_sum[l, h] += score
                score_cnt[l, h] += 1

    else:  # "per_layer"
        score_sum_layer = accumulators["score_sum"]
        score_cnt_layer = accumulators["score_cnt"]

        layer_avg_maps = []
        for l in range(num_layers):
            attn_layer = attn_list[l][0]  # [H, S, S]
            attn_layer_roi = attn_layer[:, g_start:g_end + 1, g_start:g_end + 1]  # [H, N, N]
            if attn_layer_roi.shape[1] != span_len or attn_layer_roi.shape[2] != span_len:
                layer_avg_maps.append(None)
                continue
            layer_avg = attn_layer_roi.mean(dim=0)  # [N, N]
            layer_avg_maps.append(layer_avg.to(torch.float32).cpu().numpy())

        for l in range(num_layers):
            attn_roi = layer_avg_maps[l]
            if attn_roi is None:
                continue
            score = score_attention_map(
                attn_roi,
                local_spans,
                ideal_mask,
                sim_metric=sim_metric,
                binarize_method=binarize_method,
                pre_threshold_frac=pre_threshold_frac,
            )
            score_sum_layer[l] += score
            score_cnt_layer[l] += 1
        
    del out


def _aggregate_and_save(
    task_name: str,
    cfg: dict,
    accumulators: dict,
    num_layers: int,
    num_heads: int,
):
    output_dir = cfg["output_dir"]
    score_mode = accumulators["mode"]
    sim_metric = cfg["sim_metric"]
    binarize_method = cfg["binarize_method"]
    plot_cfg = cfg["plot"]
    top_fraction = cfg.get("select_top_fraction", 0.4)
    standardize_prompt = cfg.get("standardize_prompt", True)

    out_dir_mode = os.path.join(output_dir, f"{score_mode}_{binarize_method}_{sim_metric}_{'std' if standardize_prompt else 'raw'}")
    os.makedirs(out_dir_mode, exist_ok=True)

    if score_mode == "per_head":
        score_sum = accumulators["score_sum"]
        score_cnt = accumulators["score_cnt"]

        valid_mask = score_cnt > 0
        avg_scores = np.zeros_like(score_sum)
        avg_scores[valid_mask] = score_sum[valid_mask] / np.maximum(score_cnt[valid_mask], 1)

        cache_file = os.path.join(
            out_dir_mode,
            f"{task_name}_perhead_scores.npz"
        )
        np.savez_compressed(cache_file,
                            avg_scores=avg_scores,
                            valid_mask=valid_mask)
        print(f"  -> Cached per-head scores saved to {cache_file}")

        plot_perhead_scores_and_layer_mean(
            task_name=task_name,
            avg_scores=avg_scores,
            valid_mask=valid_mask,
            out_dir=out_dir_mode,
            sim_metric=sim_metric,
            binarize_method=binarize_method,
            plot_cfg=plot_cfg,
        )

        json_top = os.path.join(
            out_dir_mode,
            f"{task_name}_selected_layers_top_{int(top_fraction * 100)}_scoring.json",
        )
        json_auto = os.path.join(
            out_dir_mode,
            f"{task_name}_selected_layers_auto_scoring.json",
        )

        select_layers_by_top_fraction(
            scores=avg_scores,
            valid_mask=valid_mask,
            score_mode="per_head",
            num_heads=num_heads,
            top_fraction=top_fraction,
            json_path=json_top,
        )

        select_layers_auto_otsu(
            scores=avg_scores,
            valid_mask=valid_mask,
            score_mode="per_head",
            num_heads=num_heads,
            json_path=json_auto,
            fallback_top_fraction=top_fraction,
        )
        return avg_scores, valid_mask

    else:
        score_sum_layer = accumulators["score_sum"]
        score_cnt_layer = accumulators["score_cnt"]

        valid_mask = score_cnt_layer > 0
        avg_scores_layer = np.zeros_like(score_sum_layer)
        idx = valid_mask
        avg_scores_layer[idx] = score_sum_layer[idx] / np.maximum(score_cnt_layer[idx], 1)

        plot_perlayer_scores(
            task_name=task_name,
            avg_scores_layer=avg_scores_layer,
            out_dir=out_dir_mode,
            sim_metric=sim_metric,
            binarize_method=binarize_method,
            plot_cfg=plot_cfg,
        )

        json_top = os.path.join(
            out_dir_mode,
            f"{task_name}_selected_layers_top_{int(top_fraction * 100)}_scoring.json",
        )
        json_auto = os.path.join(
            out_dir_mode,
            f"{task_name}_selected_layers_auto_scoring.json",
        )

        select_layers_by_top_fraction(
            scores=avg_scores_layer,
            valid_mask=valid_mask,
            score_mode="per_layer",
            num_heads=num_heads,
            top_fraction=top_fraction,
            json_path=json_top,
        )

        select_layers_auto_otsu(
            scores=avg_scores_layer,
            valid_mask=valid_mask,
            score_mode="per_layer",
            num_heads=num_heads,
            json_path=json_auto,
            fallback_top_fraction=top_fraction,
        )
        return avg_scores_layer, valid_mask


@torch.no_grad()
def process_task(task_name: str, cfg: dict, model, tokenizer):

    print(f"\n>>> Processing Task: {task_name}")

    input_column = cfg["input_column"]
    requested_sample_num = int(cfg["sample_num"])

    preferred_min_edges = int(cfg.get("preferred_min_edges", 60))
    hard_max_edges = cfg.get("hard_max_edges", None)

    # MolecularNet
    if cfg.get("molecularnet", None) is not None:
        mn = cfg["molecularnet"]

        samples, (chosen_min, chosen_max), stats = molecularnet_sample_prompts_by_edge_range(
            molecularnet=mn,
            input_column=input_column,
            sample_num=requested_sample_num,
            preferred_min_edges=preferred_min_edges,
            hard_max_edges=hard_max_edges,
            seed=42,
            require_label=True,
        )

        if samples is None or len(samples) == 0:
            print(f"[{task_name}] Warning: no MolecularNet samples after auto edge_range sampling.")
            return None

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
            f"after_cap_n={stats.get('available_after_cap_n')}, "
            f"eligible_pref_n={stats.get('eligible_after_pref_n')}, "
            f"in_range_n={stats.get('in_range_n')}, "
            f"chosen_n={stats.get('chosen_n')}, "
            f"built_prompts_n={stats.get('built_prompts_n')}, "
            f"mode={stats.get('mode')}"
        )

        # ---- scoring on sampled prompts ----
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        score_mode = cfg["score_mode"]
        accumulators = _init_score_accumulators(score_mode, num_layers, num_heads)

        for _, row in tqdm(samples.iterrows(), total=len(samples), desc=f"[{task_name}]"):
            prompt = row[input_column]
            _score_single_example(
                prompt=prompt,
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                accumulators=accumulators,
                num_layers=num_layers,
                num_heads=num_heads,
            )

        return _aggregate_and_save(
            task_name=task_name,
            cfg=cfg,
            accumulators=accumulators,
            num_layers=num_layers,
            num_heads=num_heads,
        )

    # GraphWiz
    all_df = load_and_filter_samples(
        data_path=cfg.get("data_path", None),
        input_column=input_column,
        min_edges=0,
        max_edges=10**9,
        sample_num=10**9,
        graphsst=cfg.get("graphsst", None),
        molecularnet=cfg.get("molecularnet", None),
    )
    if all_df is None or len(all_df) == 0:
        return None

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
        return None

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
        f"after_cap_n={stats.get('available_after_cap_n')}, "
        f"eligible_pref_n={stats.get('eligible_after_pref_n')}, "
        f"mode={stats.get('mode')}"
    )

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    score_mode = cfg["score_mode"]

    accumulators = _init_score_accumulators(score_mode, num_layers, num_heads)

    for _, row in tqdm(samples.iterrows(), total=len(samples), desc=f"[{task_name}]"):
        prompt = row[input_column]
        _score_single_example(
            prompt=prompt,
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            accumulators=accumulators,
            num_layers=num_layers,
            num_heads=num_heads,
        )

    return _aggregate_and_save(
        task_name=task_name,
        cfg=cfg,
        accumulators=accumulators,
        num_layers=num_layers,
        num_heads=num_heads,
    )

def parse_args():
    p = argparse.ArgumentParser(description="GraphLens scoring script")

    p.add_argument("--model_path", type=str, required=True,
                   help="Path to HF model (or backend-specific id)")
    p.add_argument("--task_name", type=str, required=True,
                   help="Task name, used only for output naming")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to dataset directory containing task files")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save scores and plots")
    p.add_argument("--input_column", type=str, default="input_prompt",
                   help="Column name that contains the graph prompt text")
    p.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Path to the graph prompt template file.",
    )
    p.add_argument(
        "--no_standardize_prompt",
        action="store_true",
        help="Disable prompt edge standardization. Default is ON (standardize).",
    )
    # Graph-SST specific
    p.add_argument("--split", type=str, default="test",
                   help="Graph-SST split: train/val/valid/test")
    p.add_argument("--sample_num", type=int, default=100,
                   help="Number of samples to use for scoring")
    p.add_argument("--min_edges", type=int, default=80,
                   help="Min number of edges in graph description")
    p.add_argument("--max_edges", type=int, default=120,
                   help="Max number of edges in graph description")
    p.add_argument("--min_max_degree", type=int, default=4,
                   help="Max degree of nodes in graph description")

    p.add_argument("--max_seq_len", type=int, default=1000,
                   help="Max sequence length for model input")
    p.add_argument("--score_mode", type=str, default="per_head",
                   choices=["per_head", "per_layer"],
                   help="Score per head or per layer")
    p.add_argument("--sim_metric", type=str, default="concentration",
                   help="Similarity metric between attention and adjacency template")
    p.add_argument("--binarize_method", type=str, default="threshold",
                   help="Binarization method for attention map")
    p.add_argument("--pre_threshold_frac", type=float, default=0.1,
                   help="Fraction for threshold-based binarization")

    p.add_argument("--plot_dpi", type=int, default=250,
                   help="DPI for saved figures")
    p.add_argument("--plot_line_color", type=str, default="C0",
                   help="Line color for per-layer plots")

    p.add_argument("--select_top_fraction", type=float, default=0.4,
                   help="Top fraction used when layer/head selection falls back from Otsu")

    p.add_argument(
        "--preferred_min_edges",
        type=int,
        default=60,
        help="Prefer selecting samples with edges >= this value; will gracefully fallback if not enough (e.g., Graph-SST).",
    )
    p.add_argument(
        "--hard_max_edges",
        type=int,
        default=-1,
        help="Hard cap on edges to control compute cost (-1 disables).",
    )

    return p.parse_args()


def main():
    args = parse_args()
    standardize_prompt = (not args.no_standardize_prompt)
    cfg = {
        "output_dir": args.output_dir,
        "input_column": args.input_column,
        "sample_num": args.sample_num,
        "min_edges": args.min_edges,
        "max_edges": args.max_edges,
        "min_max_degree": args.min_max_degree,
        "max_seq_len": args.max_seq_len,
        "score_mode": args.score_mode,
        "sim_metric": args.sim_metric,
        "binarize_method": args.binarize_method,
        "pre_threshold_frac": args.pre_threshold_frac,
        "standardize_prompt": standardize_prompt,
        "plot": {
            "dpi": args.plot_dpi,
            "line_color": args.plot_line_color,
        },
        "select_top_fraction": args.select_top_fraction,
        "preferred_min_edges": int(args.preferred_min_edges),
        "hard_max_edges": (None if int(args.hard_max_edges) < 0 else int(args.hard_max_edges)),
        "data_path": None,
        "graphsst": None,
        "molecularnet": None,
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

        for task in tasks:
            cfg["graphsst"] = None
            cfg["data_path"] = os.path.join(args.data_dir, f"{task}_test.json")
            process_task(task, cfg, model, tokenizer)

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
                "label_col": label_col,  # now single label; Tox21 uses __TOX21_TOXIC_ANY__
                "split": args.split,      # "test" or "sample"
                "prompt_path": args.prompt_path,
                "shot": 0,
                "seed": 42,
                "weighted_edges": False,
            }
            process_task(task_id, cfg, model, tokenizer)
        return

    else:
        raise ValueError(f"Unknown task name: {args.task_name}")


if __name__ == "__main__":
    main()