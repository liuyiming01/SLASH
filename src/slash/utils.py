import os
import re
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
import cv2

# ============================================================
# Section 0. Configuration & Model Loading
# ============================================================

def load_model_and_tokenizer(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


# ============================================================
# Section 1.  Data Parsing & Span Extraction
# ============================================================
_EDGE_PATTERNS = [
    (re.compile(r'\(\s*(\d+)\s*->\s*([\d\s,]+)\)'), "arrow_multi"), # (0->1,2)
    (re.compile(r'\(\s*(\d+)\s*,\s*([\d\s,]+)\)'), "multi_comma"),  # (0,1,2)
    (re.compile(r'\(\s*(\d+)\s*->\s*(\d+)\s*\)'), "arrow"),     # (0->1)
    (re.compile(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'), "pair"),       # (0,1)
]

def _find_active_edge_pattern(text: str):
    for pat, tag in _EDGE_PATTERNS:
        if pat.search(text):
            return pat, tag
    return None, None

def _parse_edge_matches(prompt: str):
    pat, tag = _find_active_edge_pattern(prompt)
    if pat is None:
        return []

    parsed = []
    for m in pat.finditer(prompt):
        src = int(m.group(1))
        if tag in ("arrow_multi", "multi_comma"):
            tail = m.group(2)
            nums = [int(x) for x in re.findall(r'\d+', tail)]
            if not nums:
                continue
            dst = nums[0]
        else:
            dst = int(m.group(2))

        parsed.append(
            {
                "src": src,
                "dst": dst,
                "start": m.start(),
                "end": m.end(),
                "text": m.group(0),
            }
        )
    return parsed

def find_edge_segments(text: str):
    pat, _ = _find_active_edge_pattern(text)
    if pat is None:
        return []

    matches = list(pat.finditer(text))
    if not matches:
        return []

    segments = []
    cur_node = None
    seg_start = None

    for i, m in enumerate(matches):
        node_str = m.group(1)
        if cur_node is None:
            cur_node = node_str
            seg_start = m.start()
        elif node_str != cur_node:
            seg_end = matches[i - 1].end()
            segments.append(
                {"char_start": seg_start, "char_end": seg_end, "node": int(cur_node)}
            )
            cur_node = node_str
            seg_start = m.start()

    seg_end = matches[-1].end()
    segments.append(
        {"char_start": seg_start, "char_end": seg_end, "node": int(cur_node)}
    )
    return segments

def get_token_spans(prompt: str, tokenizer):
    segments = find_edge_segments(prompt)
    if not segments:
        return []

    enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True,
                    add_special_tokens=True)
    offsets = enc["offset_mapping"][0].tolist()

    spans = []
    for seg in segments:
        cs, ce = seg["char_start"], seg["char_end"]
        t_start, t_end = None, None

        for i, (s, e) in enumerate(offsets):
            if e == 0 and s == 0:
                continue
            if e > cs:
                t_start = i
                break

        for i in range(len(offsets) - 1, -1, -1):
            s, e = offsets[i]
            if e == 0 and s == 0:
                continue
            if s < ce:
                t_end = i
                break

        if t_start is not None and t_end is not None and t_start <= t_end:
            spans.append({"tok_start": t_start, "tok_end": t_end})

    return spans

def parse_edges_in_prompt(prompt: str):
    parsed = _parse_edge_matches(prompt)
    return [(e["src"], e["dst"]) for e in parsed]

def count_nodes_in_prompt(prompt: str) -> int:
    edges = parse_edges_in_prompt(prompt)
    if not edges:
        return 0
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    return len(nodes)

def count_edges_in_prompt(prompt: str) -> int:
    edges = parse_edges_in_prompt(prompt)
    return len(edges)

def standardize_prompt_edges(prompt: str) -> str:
    parsed = _parse_edge_matches(prompt)
    if not parsed:
        return prompt

    parsed_sorted = sorted(parsed, key=lambda x: x["src"])

    prefix = prompt[: parsed[0]["start"]]
    suffix = prompt[parsed[-1]["end"] :]

    edge_strs = [item["text"] for item in parsed_sorted]
    middle = " ".join(edge_strs)
    return prefix + middle + suffix

def standardize_prompt_edges_shuffle_span(prompt: str, seed: int = 42) -> str:
    import random
    from collections import defaultdict

    parsed = _parse_edge_matches(prompt)
    if not parsed:
        return prompt

    groups_by_src = defaultdict(list)
    for item in parsed:
        groups_by_src[item["src"]].append(item)

    rng = random.Random(seed) if seed is not None else random
    new_order = []
    for src in sorted(groups_by_src.keys()):
        group = groups_by_src[src]
        if len(group) > 1:
            rng.shuffle(group)
        new_order.extend(group)

    prefix = prompt[: parsed[0]["start"]]
    suffix = prompt[parsed[-1]["end"] :]

    edge_strs = [item["text"] for item in new_order]
    middle = " ".join(edge_strs)

    return prefix + middle + suffix

def auto_edge_range(edge_counts: np.ndarray, sample_num: int, min_chosen_min: int = 60):
    edge_counts = np.asarray(edge_counts, dtype=np.float32)
    edge_counts = edge_counts[~np.isnan(edge_counts)]
    if edge_counts.size == 0:
        return None, None, {"n": 0}

    edge_sorted = np.sort(edge_counts.astype(np.int32))
    n = int(edge_sorted.size)
    k = int(min(max(sample_num, 1), n))

    med = float(np.median(edge_sorted))
    true_min = int(edge_sorted[0])
    true_max = int(edge_sorted[-1])

    if k == n:
        stats = {
            "n": n,
            "k": k,
            "true_min": true_min,
            "true_max": true_max,
            "median": med,
            "chosen_min": true_min,
            "chosen_max": true_max,
            "chosen_center": float((true_min + true_max) / 2.0),
            "chosen_width": int(true_max - true_min),
            "min_constraint": int(min_chosen_min),
            "used_min_constraint": bool(true_min >= min_chosen_min),
        }
        return true_min, true_max, stats

    best_pref = None  # candidates with min_e >= min_chosen_min
    best_all = None   # all candidates

    # sliding window: [i, i+k-1]
    for i in range(0, n - k + 1):
        j = i + k - 1
        min_e = int(edge_sorted[i])
        max_e = int(edge_sorted[j])
        center = (min_e + max_e) / 2.0
        obj = abs(center - med)
        width = int(max_e - min_e)
        cand = (obj, width, i, j, min_e, max_e, center)

        if best_all is None or cand < best_all:
            best_all = cand
        if min_e >= int(min_chosen_min):
            if best_pref is None or cand < best_pref:
                best_pref = cand

    # Prefer constraint-satisfying window; otherwise relax ("expand left")
    chosen = best_pref if best_pref is not None else best_all
    _, width, i, j, min_e, max_e, center = chosen

    stats = {
        "n": n,
        "k": k,
        "true_min": true_min,
        "true_max": true_max,
        "median": med,
        "chosen_min": int(min_e),
        "chosen_max": int(max_e),
        "chosen_center": float(center),
        "chosen_width": int(width),
        "window_i": int(i),
        "window_j": int(j),
        "min_constraint": int(min_chosen_min),
        "used_min_constraint": bool(best_pref is not None),
    }
    return int(min_e), int(max_e), stats

# ============================================================
# Section 2.  ROI & Template mask
# ============================================================

def compute_global_span(spans):
    if not spans:
        raise ValueError("Empty spans list.")
    spans_sorted = sorted(spans, key=lambda x: x["tok_start"])
    g_start = spans_sorted[0]["tok_start"]
    g_end = spans_sorted[-1]["tok_end"]
    length = g_end - g_start + 1
    return g_start, g_end, length, spans_sorted

def build_local_spans(global_start: int, spans_sorted):
    return [
        (sp["tok_start"] - global_start, sp["tok_end"] - global_start)
        for sp in spans_sorted
    ]

def extract_roi(attn_tensor, g_start: int, g_end: int):
    return attn_tensor[:, :, g_start:g_end + 1, g_start:g_end + 1]

def extract_roi_from_attn(attn_np, layer: int, head: int, g_start: int, g_end: int):
    return attn_np[layer, head, g_start:g_end + 1, g_start:g_end + 1]

def build_sawtooth_mask(N: int, local_spans):
    mask = np.zeros((N, N), dtype=np.float32)
    rows, cols = np.indices((N, N))
    for s, e in local_spans:
        s = max(0, s)
        e = min(N - 1, e)
        if s > e:
            continue
        region = (rows >= s) & (rows <= e) & (cols >= s) & (cols <= rows)
        mask[region] = 1.0
    return mask

# ============================================================
# Section 3.  Preprocessing & Denoising
# ============================================================

def normalize_minmax(x: np.ndarray, eps: float = 1e-9):
    v_min = x.min()
    v_max = x.max()
    if v_max - v_min < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - v_min) / (v_max - v_min)).astype(np.float32)

def threshold_binarize(x: np.ndarray, frac: float = 0.1):
    v_min, v_max = float(x.min()), float(x.max())
    if v_max <= v_min:
        return np.zeros_like(x, dtype=np.float32)
    thr = v_min + frac * (v_max - v_min)
    out = (x > thr).astype(np.float32)
    return out

def topk_binarize(x: np.ndarray, ideal_mask: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    ideal_mask = np.asarray(ideal_mask)

    if x.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    if ideal_mask.shape != x.shape:
        raise ValueError("ideal_mask must have the same shape as x for topk_binarize.")

    k = int(np.round(ideal_mask.sum()))
    if k <= 0:
        return np.zeros_like(x, dtype=np.float32)
    if k >= x.size:
        return np.ones_like(x, dtype=np.float32)

    flat = x.reshape(-1)
    # top-k index
    idx = np.argpartition(-flat, k - 1)[:k]
    out = np.zeros_like(flat, dtype=np.float32)
    out[idx] = 1.0
    return out.reshape(x.shape)

def binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if mask.size == 0:
        return mask.astype(np.float32)

    size = 2 * radius + 1
    selem = np.ones((size, size), dtype=bool)

    mask_bool = mask.astype(bool)
    dilated = ndimage.binary_dilation(mask_bool, structure=selem)
    return dilated.astype(np.float32)

def binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if mask.size == 0:
        return mask.astype(np.float32)

    size = 2 * radius + 1
    selem = np.ones((size, size), dtype=bool)

    mask_bool = mask.astype(bool)
    eroded = ndimage.binary_erosion(mask_bool, structure=selem)
    return eroded.astype(np.float32)

# ============================================================
# Section 4.  Topology Alignment Score
# ============================================================

def score_gradient_correlation(feature_map, target_mask):
    feature_map = np.asarray(feature_map, dtype=np.float32)
    target_mask = np.asarray(target_mask, dtype=np.float32)

    sobel_x_img = cv2.Sobel(feature_map, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_img = cv2.Sobel(feature_map, cv2.CV_64F, 0, 1, ksize=3)

    sobel_x_mask = cv2.Sobel(target_mask, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_mask = cv2.Sobel(target_mask, cv2.CV_64F, 0, 1, ksize=3)

    flat_x_img = sobel_x_img.flatten()
    flat_y_img = sobel_y_img.flatten()
    flat_x_mask = sobel_x_mask.flatten()
    flat_y_mask = sobel_y_mask.flatten()

    if np.all(flat_x_img == 0) or np.all(flat_x_mask == 0):
        corr_x = 0.0
    else:
        corr_x = np.corrcoef(flat_x_img, flat_x_mask)[0, 1]

    if np.all(flat_y_img == 0) or np.all(flat_y_mask == 0):
        corr_y = 0.0
    else:
        corr_y = np.corrcoef(flat_y_img, flat_y_mask)[0, 1]

    if np.isnan(corr_x):
        corr_x = 0.0
    if np.isnan(corr_y):
        corr_y = 0.0

    return float((corr_x + corr_y) / 2.0)

def score_attn_concentration(local_spans,
                             feature_map,
                             max_shift: int = 2) -> float:
    x = np.asarray(feature_map)

    if x.ndim == 2:
        x = x[None, ...]
        squeeze_out = True
    elif x.ndim == 3:
        squeeze_out = False
    else:
        raise ValueError("feature_map must be 2D (H, W) or 3D (B, H, W).")

    B, H, W = x.shape
    if H != W:
        raise ValueError("feature_map must be square (H == W).")

    weighted_sum = np.zeros(B, dtype=np.float64)
    total_weight = 0.0

    tri_mask_cache: dict[tuple[int, int], np.ndarray] = {}

    out_wrong_buf = np.zeros(B, dtype=np.float64)
    in_wrong_buf = np.zeros(B, dtype=np.float64)

    shift_candidates = list(range(-max_shift, max_shift + 1))

    for span in local_spans:
        if span is None or len(span) != 2:
            continue
        s_orig, e_orig = int(span[0]), int(span[1])

        s_orig = max(0, min(H - 1, s_orig))
        e_orig = max(0, min(H - 1, e_orig))
        if e_orig < s_orig:
            continue

        span_len = e_orig - s_orig + 1

        best_span_score = None  # np.ndarray, shape (B,)

        for delta in shift_candidates:
            s = s_orig + delta
            e = e_orig + delta

            if e < 0 or s > H - 1:
                continue

            s = max(0, min(H - 1, s))
            e = max(0, min(H - 1, e))
            if e < s:
                continue

            cur_len = e - s + 1
            rows_len = cur_len

            out_wrong_buf.fill(0.0)
            in_wrong_buf.fill(0.0)
            out_count = 0.0
            in_count = 0.0

            left_start = max(0, e - span_len - 1)
            left_end = s
            if left_start < left_end:
                block_out = x[:, s:e + 1, left_start:left_end]  # (B, rows_len, W_out)
                area = rows_len * (left_end - left_start)
                if area > 0:
                    out_count += float(area)
                    out_wrong_buf += block_out.sum(axis=(1, 2))

            diag_margin = max(round(0.25*span_len), 4)
            block_in = x[:, s:e + 1, s:e + 1]  # (B, L, L)
            L = block_in.shape[1]
            if L > 0:
                key = (int(L), int(diag_margin))
                if key not in tri_mask_cache:
                    rows_idx, cols_idx = np.indices((L, L))
                    tri_mask_cache[key] = (cols_idx <= (rows_idx - diag_margin))
                tri_mask = tri_mask_cache[key]

                tri_vals = block_in[:, tri_mask]  # (B, N_valid)
                n_tri = tri_vals.shape[1]
                if n_tri > 0:
                    in_count += float(n_tri)
                    in_wrong_buf += (tri_vals == 0).sum(axis=1)

            if out_count > 0.0:
                out_wrong_ratio = out_wrong_buf / out_count  # (B,)
            else:
                out_wrong_ratio = np.zeros(B, dtype=np.float64)

            if in_count > 0.0:
                in_wrong_ratio = in_wrong_buf / in_count     # (B,)
            else:
                in_wrong_ratio = np.zeros(B, dtype=np.float64)

            span_score = (1.0 - out_wrong_ratio) * (1.0 - in_wrong_ratio)
            span_score = np.clip(span_score, 0.0, 1.0)       # (B,)

            if best_span_score is None:
                best_span_score = span_score
            else:
                best_span_score = np.maximum(best_span_score, span_score)

        if best_span_score is None:
            continue

        w = float(span_len)
        weighted_sum += w * best_span_score
        total_weight += w

    if total_weight == 0.0:
        score = np.ones(B, dtype=np.float64)
    else:
        score = weighted_sum / total_weight
        score = np.clip(score, 0.0, 1.0)

    if squeeze_out:
        return float(score[0])
    return score

def preprocess_for_scoring(attn_roi: np.ndarray,
                           binarize_method: str = "topk",
                           ideal_mask: np.ndarray = None,
                           pre_threshold_frac: float = 0.1,
                           topp_p: float = 0.9):
    x = normalize_minmax(attn_roi)

    if binarize_method == "topk":
        bin_mask = topk_binarize(x, ideal_mask)
    elif binarize_method == "threshold":
        bin_mask = threshold_binarize(x, frac=pre_threshold_frac)
    else:
        raise ValueError(f"Unknown binarize_method: {binarize_method}")

    denoised_mask = binary_dilate(bin_mask, radius=1)
    denoised_mask = binary_erode(denoised_mask, radius=1)

    return bin_mask, denoised_mask


def score_attention_map(attn_roi: np.ndarray,
                        local_spans: list,
                        ideal_mask: np.ndarray,
                        sim_metric: str = "concentration",
                        binarize_method: str = "topk",
                        pre_threshold_frac: float = 0.1) -> float:
    if attn_roi.shape != ideal_mask.shape:
        raise ValueError("attn_roi and ideal_mask must have the same shape.")

    bin_mask, denoised_mask = preprocess_for_scoring(
        attn_roi,
        binarize_method=binarize_method,
        ideal_mask=ideal_mask,
        pre_threshold_frac=pre_threshold_frac,
    )

    if sim_metric == "gradient":
        return float(score_gradient_correlation(denoised_mask, ideal_mask))
    elif sim_metric == "concentration":
        return float(score_attn_concentration(local_spans, denoised_mask))
    else:
        raise ValueError(f"Unknown sim_metric: {sim_metric}")


# ============================================================
# Section 5.  Layer / Head Selection Helpers
# ============================================================

def _otsu_threshold_1d(values: np.ndarray) -> tuple:
    vals = np.asarray(values, dtype=np.float64)
    n = vals.size
    if n == 0:
        return float("nan"), 0.0
    if n == 1:
        return float(vals[0]), 0.0

    vals_sorted = np.sort(vals)
    total_sum = float(vals_sorted.sum())
    total_mean = total_sum / n

    best_thr = float(vals_sorted[0])
    best_var = -1.0

    sum0 = 0.0
    count0 = 0

    for i in range(0, n - 1):
        v = vals_sorted[i]
        sum0 += v
        count0 += 1

        w0 = count0 / n
        w1 = 1.0 - w0
        if w0 <= 0.0 or w1 <= 0.0:
            continue

        mu0 = sum0 / count0
        sum1 = total_sum - sum0
        mu1 = sum1 / (n - count0)

        var_between = w0 * w1 * (mu0 - mu1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thr = 0.5 * (vals_sorted[i] + vals_sorted[i + 1])

    return float(best_thr), float(best_var)


def select_layers_by_top_fraction(scores: np.ndarray,
                                  valid_mask: np.ndarray,
                                  score_mode: str,
                                  num_heads: int,
                                  top_fraction: float,
                                  json_path: str):
    scores = np.asarray(scores, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if not (0.0 < top_fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1].")

    valid_scores = scores[valid_mask]
    if valid_scores.size == 0:
        selected = {}
        with open(json_path, "w") as f:
            json.dump(selected, f, indent=2)
        return selected, float("nan")

    n_valid = valid_scores.size
    k = max(1, int(np.ceil(top_fraction * n_valid)))
    k = min(k, n_valid)

    kth_value = np.partition(valid_scores, -k)[-k]
    threshold = float(kth_value)

    high_mask = (scores >= threshold) & valid_mask

    selected = {}
    if score_mode == "per_layer":
        L = scores.shape[0]
        for l in range(L):
            if not valid_mask[l]:
                continue
            if high_mask[l]:
                selected[int(l)] = list(range(num_heads))
    elif score_mode == "per_head":
        L, H = scores.shape
        for l in range(L):
            heads = [int(h) for h in range(H) if high_mask[l, h]]
            if heads:
                selected[int(l)] = heads
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    with open(json_path, "w") as f:
        json.dump(selected, f, indent=2)

    return selected, threshold


def select_layers_otsu(scores: np.ndarray,
                        valid_mask: np.ndarray,
                        score_mode: str,
                        num_heads: int,
                        json_path: str,
                        fallback_top_fraction: float = 0.4):

    scores = np.asarray(scores, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    valid_scores = scores[valid_mask]
    if valid_scores.size == 0:
        selected = {}
        with open(json_path, "w") as f:
            json.dump(selected, f, indent=2)
        return selected, float("nan")

    thr_otsu, var_between = _otsu_threshold_1d(valid_scores)

    if not np.isfinite(var_between) or var_between <= 1e-6:

        n_valid = valid_scores.size
        k = max(1, int(np.ceil(fallback_top_fraction * n_valid)))
        k = min(k, n_valid)
        kth_value = np.partition(valid_scores, -k)[-k]
        threshold = float(kth_value)
    else:
        threshold = float(thr_otsu)

    high_mask = (scores >= threshold) & valid_mask

    selected = {}
    if score_mode == "per_layer":
        L = scores.shape[0]
        for l in range(L):
            if not valid_mask[l]:
                continue
            if high_mask[l]:
                selected[int(l)] = list(range(num_heads))
    elif score_mode == "per_head":
        L, H = scores.shape
        for l in range(L):
            heads = [int(h) for h in range(H) if high_mask[l, h]]
            if heads:
                selected[int(l)] = heads
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    with open(json_path, "w") as f:
        json.dump(selected, f, indent=2)

    return selected, threshold

def select_layers_peak_auto_entropy(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    score_mode: str,
    num_heads: int,
    json_path: str,
    smooth_sigma: float = None,
    min_prominence_frac: float = 0.08,
    relative_height: float = 0.8,
    fallback_top_fraction: float = 0.4,
):
    scores = np.asarray(scores, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if score_mode not in ("per_layer", "per_head"):
        raise ValueError(f"Unknown score_mode: {score_mode}")

    if score_mode == "per_layer":
        if scores.ndim != 1:
            raise ValueError("per_layer scores must be 1D")
        layer_vals = scores.copy()
        layer_valid = valid_mask.copy()
    else:
        if scores.ndim != 2:
            raise ValueError("per_head scores must be 2D (L,H)")
        L, H = scores.shape
        layer_vals = np.full((L,), np.nan, dtype=np.float32)
        layer_valid = np.any(valid_mask, axis=1)
        for l in range(L):
            m = valid_mask[l]
            if np.any(m):
                layer_vals[l] = float(np.mean(scores[l][m]))

    if not np.any(layer_valid):
        selected = {}
        with open(json_path, "w") as f:
            json.dump(selected, f, indent=2)
        return selected, {"mode": "empty"}

    L = int(layer_vals.shape[0])
    center = (L - 1) / 2.0

    # ---- fill invalid with mean (so smoothing won't crash) ----
    mean_val = float(np.nanmean(layer_vals[layer_valid]))
    filled = layer_vals.copy()
    filled[~layer_valid] = mean_val

    # ---- winsorize to reduce domination by extreme early peaks ----
    v_valid = filled[layer_valid].astype(np.float64)
    lo = float(np.quantile(v_valid, 0.05))
    hi = float(np.quantile(v_valid, 0.95))
    wins = np.clip(filled.astype(np.float64), lo, hi)

    # ---- smooth ----
    if smooth_sigma is None:
        smooth_sigma = max(1.0, L / 70.0)
    smooth = gaussian_filter(wins, sigma=float(smooth_sigma))

    dyn = float(np.max(smooth) - np.min(smooth))
    prom_thr = float(min_prominence_frac * dyn) if dyn > 0 else 0.0

    # ---- peak detection ----
    peaks = None
    prominences = None
    try:
        from scipy.signal import find_peaks, peak_prominences

        peaks, _ = find_peaks(smooth, prominence=prom_thr)
        if peaks.size > 0:
            prominences = peak_prominences(smooth, peaks)[0].astype(np.float64)
    except Exception:
        cand = []
        for i in range(1, L - 1):
            if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
                cand.append(i)
        peaks = np.asarray(cand, dtype=np.int32)

    def _fallback_top_fraction():
        v = smooth.copy()
        valid_vals = v[layer_valid]
        n_valid = int(valid_vals.size)
        k = max(1, int(np.ceil(float(fallback_top_fraction) * n_valid)))
        k = min(k, n_valid)
        thr = float(np.partition(valid_vals, -k)[-k])
        sel_layers = [int(i) for i in range(L) if layer_valid[i] and v[i] >= thr]
        selected = {int(l): list(range(num_heads)) for l in sel_layers}
        with open(json_path, "w") as f:
            json.dump(selected, f, indent=2)
        return selected, {"mode": "fallback_top_fraction", "threshold": thr, "k": k}

    if peaks is None or peaks.size == 0:
        return _fallback_top_fraction()

    # ---- percentile rank ----
    valid_series = smooth[layer_valid].astype(np.float64)
    valid_sorted = np.sort(valid_series)

    def _pct_rank(x: float) -> float:
        # in [0,1]
        return float(np.searchsorted(valid_sorted, x, side="right")) / float(valid_sorted.size)

    baseline = float(np.median(valid_series))
    sigma_c = max(1.0, 0.22 * L)  # stronger center preference than before

    best = None
    best_info = None

    for idx, p in enumerate(peaks.tolist()):
        if not layer_valid[p]:
            continue

        peak_val = float(smooth[p])

        # band threshold: adaptive (widen shoulders)
        height = max(peak_val - baseline, 1e-12)

        # 1) start threshold fraction: lower than 0.5 to include shoulders
        #    reuse relative_height (default=0.8) to control wideness:
        #    relative_height higher => wider band
        thr_frac = float(np.clip(0.15 + (1.0 - float(relative_height)), 0.15, 0.65))
        # default relative_height=0.8 -> thr_frac=0.35 (wider than 0.5)

        # 2) ensure band is not too short: target width scales with L
        target_width = max(3, int(np.ceil(0.32 * L)))  # e.g., L=36 -> 11 layers (close to 14-23)

        def _expand_band(thr_value: float):
            l = p
            while l - 1 >= 0 and layer_valid[l - 1] and float(smooth[l - 1]) >= thr_value:
                l -= 1
            r = p
            while r + 1 < L and layer_valid[r + 1] and float(smooth[r + 1]) >= thr_value:
                r += 1
            return l, r

        # progressively lower threshold until wide enough (or hit floor)
        while True:
            thr = baseline + thr_frac * height
            l, r = _expand_band(thr)
            width = int(r - l + 1)

            if width >= target_width or thr_frac <= 0.12:
                break
            thr_frac = max(0.12, thr_frac * 0.85)  # lower threshold -> wider band

        amp_rank = _pct_rank(peak_val)
        if prominences is not None and idx < len(prominences):
            prom_rank = float(np.searchsorted(np.sort(prominences), float(prominences[idx]), side="right")) / float(len(prominences))
        else:
            prom_rank = 0.5

        dist = float(p - center)
        center_w = float(np.exp(-0.5 * (dist / sigma_c) ** 2))

        width_w = min(1.0, width / max(3.0, 0.20 * L))

        score = center_w * (0.45 * amp_rank + 0.35 * prom_rank + 0.20 * width_w)

        cand = (score, -width, -amp_rank, p, l, r)
        if best is None or cand > best:
            best = cand
            best_info = {
                "peak": int(p),
                "band": [int(l), int(r)],
                "band_width": int(width),
                "baseline": float(baseline),
                "threshold": float(thr),
                "thr_frac": float(thr_frac),
                "target_width": int(target_width),
                "amp_rank": float(amp_rank),
                "prom_rank": float(prom_rank),
                "center_weight": float(center_w),
                "score": float(score),
            }

    if best is None:
        return _fallback_top_fraction()

    _, _, _, p, l, r = best

    try:
        thr_frac_used = float(best_info.get("thr_frac", 0.0)) if best_info else 0.0
    except Exception:
        thr_frac_used = 0.0

    # baseline/height computed on wins (more faithful for shoulders)
    wins_valid = wins[layer_valid].astype(np.float64)
    baseline_w = float(np.median(wins_valid))
    peak_w = float(wins[int(p)])
    height_w = max(peak_w - baseline_w, 1e-12)

    # slightly lower than band-building thr_frac; but don't go too low
    pad_thr_frac = max(0.02, thr_frac_used * 0.5)
    pad_thr_w = baseline_w + pad_thr_frac * height_w

    # only pad ONE layer on the right if it still looks like part of the hump
    rr = int(r)
    if rr + 1 < L and layer_valid[rr + 1]:
        nxt_w = float(wins[rr + 1])
        edge_w = float(wins[rr])

        # condition A: still above a low shoulder threshold
        cond_a = nxt_w >= pad_thr_w
        # condition B: monotonic-ish shoulder (do not drop after the band edge)
        cond_b = nxt_w >= edge_w

        # shape_ok = (rr + 2 >= L) or (not layer_valid[rr + 2]) or (float(wins[rr + 1]) >= float(wins[rr + 2]) - 1e-12)
        # if shape_ok:
        #     r = rr + 1
        if cond_a or cond_b:
            r = rr + 1
            if best_info is not None:
                best_info["padded_right"] = True
                best_info["pad_thr_frac"] = float(pad_thr_frac)
                best_info["pad_thr_w"] = float(pad_thr_w)
                best_info["band"] = [int(l), int(r)]
                best_info["band_width"] = int(r - l + 1)
        else:
            if best_info is not None:
                best_info["padded_right"] = False

    sel_layers = [int(i) for i in range(int(l), int(r) + 1) if layer_valid[i]]
    if not sel_layers:
        sel_layers = [int(p)]

    pad_layer = int(r) + 1
    if pad_layer < L and layer_valid[pad_layer]:
        sel_layers.append(pad_layer)

    selected = {int(lay): list(range(num_heads)) for lay in sel_layers}
    with open(json_path, "w") as f:
        json.dump(selected, f, indent=2)


    info = {
        "mode": "peak_auto",
        "L": int(L),
        "smooth_sigma": float(smooth_sigma),
        "min_prominence": float(prom_thr),
        "chosen": best_info,
        "num_selected_layers": int(len(sel_layers)),
    }
    return selected, info