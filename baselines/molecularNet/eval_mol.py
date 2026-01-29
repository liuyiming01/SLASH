import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from utils import (
    TASKS,
    build_graph_prompt,
    load_prompts,
    parse_yesno,
    resolve_csv_paths,
    row_extra,
    sample_shots_graph_from_df,
    smiles_to_graph_text,
    yesno_from_label,
    edge_aggre,
    edge_shuffle
)

REPO_ROOT = "SLASH"
_SLASH_IMPORTED = False
def _try_import_slash():
    global _SLASH_IMPORTED
    if _SLASH_IMPORTED:
        return True
    try:
        repo_root = REPO_ROOT
        src_path = os.path.join(repo_root, "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from slash.datasets import molecularnet_sample_indices_by_edge_range
        _SLASH_IMPORTED = True
        return True
    except Exception:
        return False
    
def _normalize_layers_heads(obj, num_heads: int) -> Optional[Dict[str, List[int]]]:
    if obj is None:
        return None

    if isinstance(obj, dict):
        if "layers_heads_to_modify" in obj and isinstance(obj["layers_heads_to_modify"], list):
            obj = obj["layers_heads_to_modify"]
        elif "selected_heads" in obj and isinstance(obj["selected_heads"], list):
            obj = obj["selected_heads"]

    if isinstance(obj, dict):
        out: Dict[str, List[int]] = {}
        for k, v in obj.items():
            if not isinstance(v, list):
                continue
            heads: List[int] = []
            for h in v:
                try:
                    heads.append(int(h))
                except Exception:
                    pass
            if heads:
                out[str(int(k))] = sorted(list(set(heads)))
        return out or None

    if isinstance(obj, list):
        out: Dict[str, List[int]] = {}
        for item in obj:
            if isinstance(item, dict) and "layer" in item and "heads" in item and isinstance(item["heads"], list):
                layer = str(int(item["layer"]))
                out.setdefault(layer, [])
                for h in item["heads"]:
                    try:
                        out[layer].append(int(h))
                    except Exception:
                        pass
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    layer = str(int(item[0]))
                    head = int(item[1])
                    out.setdefault(layer, []).append(head)
                except Exception:
                    pass
        for k in list(out.keys()):
            out[k] = sorted(list(set(out[k])))
            if not out[k]:
                out.pop(k, None)
        return out or None

    return None

def _compute_metrics_from_pred_jsonl(pred_path: str) -> Dict[str, float]:
    y_true: List[int] = []
    y_pred: List[int] = []
    invalid = 0

    if not os.path.exists(pred_path):
        return {"accuracy": 0.0, "f1": 0.0, "total": 0.0, "correct": 0.0, "invalid": 0.0}

    with open(pred_path, "r", encoding="utf-8") as rf:
        for line in rf:
            try:
                obj = json.loads(line)
            except Exception:
                continue

            gt = obj.get("ground_truth")
            pred = obj.get("prediction")

            if gt not in {"Yes", "No"}:
                continue
            if pred not in {"Yes", "No"}:
                y_true.append(1 if gt == "Yes" else 0)
                y_pred.append(0 if gt == "Yes" else 1)
                invalid += 1
                continue

            y_true.append(1 if gt == "Yes" else 0)
            y_pred.append(1 if pred == "Yes" else 0)

    if len(y_true) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "total": 0.0, "correct": 0.0, "invalid": float(invalid)}

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="binary"))
    correct = float(sum(int(a == b) for a, b in zip(y_true, y_pred)))
    total = float(len(y_true))
    return {"accuracy": acc, "f1": f1, "total": total, "correct": correct, "invalid": float(invalid)}
    
def MODIFICATION(model, layers_heads_to_modify, gamma, first_token_idx=0):
    import sys
    import types

    sys.path.insert(0, REPO_ROOT)

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    model_name = model.__class__.__name__.lower()

    # Prefer config.model_type; fallback to class name heuristic
    is_llama = (model_type in {"llama"}) or ("llama" in model_name)
    is_qwen3 = (model_type in {"qwen3"}) or ("qwen3" in model_name)

    if is_llama:
        from modeling import modeling_llama_attn_shift
        LlamaModel_forward, LlamaDecoderLayer_forward, LlamaAttention_forward = (
            modeling_llama_attn_shift.get_modified_forward_llama(
                layers_heads_to_modify=layers_heads_to_modify,
                gamma=gamma,
                first_token_idx=first_token_idx,
            )
        )
        model.model.forward = types.MethodType(LlamaModel_forward, model.model)
        for layer in model.model.layers:
            layer.forward = types.MethodType(LlamaDecoderLayer_forward, layer)
            layer.self_attn.forward = types.MethodType(LlamaAttention_forward, layer.self_attn)
        return

    if is_qwen3:
        from modeling import modeling_qwen3_attn_shift
        Qwen3Model_forward, Qwen3DecoderLayer_forward, Qwen3Attention_forward = (
            modeling_qwen3_attn_shift.get_modified_forward_qwen3(
                layers_heads_to_modify=layers_heads_to_modify,
                gamma=gamma,
                first_token_idx=first_token_idx,
            )
        )
        model.model.forward = types.MethodType(Qwen3Model_forward, model.model)
        for layer in model.model.layers:
            layer.forward = types.MethodType(Qwen3DecoderLayer_forward, layer)
            layer.self_attn.forward = types.MethodType(Qwen3Attention_forward, layer.self_attn)
        return

    raise ValueError(
        f"Unsupported model for MODIFICATION(): model_type={model_type}, class={model.__class__.__name__}. "
        "Currently supports: llama, qwen3."
    )


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int, temperature: float = 0.0) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature is not None and temperature > 0
    gen = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(temperature) if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    outs = tokenizer.batch_decode(gen, skip_special_tokens=True)

    out_texts: List[str] = []
    for p, full in zip(prompts, outs):
        out_texts.append(full[len(p):].strip() if full.startswith(p) else full.strip())
    return out_texts

def score_first_token_yesno(model, tokenizer, prompts: List[str]):
    if len(prompts) == 0:
        return []

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -1, :]  # (B, V)
    probs = torch.softmax(logits, dim=-1)

    # collect robust single-token variants for yes/no
    yes_cands = ["Yes", " yes", "yes", " yes.", "Yes."]
    no_cands = ["No", " no", "no", " no.", "No."]
    yes_ids = {tokenizer.encode(s, add_special_tokens=False)[0] for s in yes_cands
               if len(tokenizer.encode(s, add_special_tokens=False)) == 1}
    no_ids = {tokenizer.encode(s, add_special_tokens=False)[0] for s in no_cands
              if len(tokenizer.encode(s, add_special_tokens=False)) == 1}
    # if tokenizer provides no single-token variant, signal caller to fallback
    if (not yes_ids) and (not no_ids):
        return [None] * logits.size(0)

    yes_idx = list(sorted(yes_ids))
    no_idx = list(sorted(no_ids))
    # yes_prob = probs[:, yes_idx].sum(dim=-1).cpu().tolist() if yes_idx else [0.0] * probs.size(0)
    # no_prob = probs[:, no_idx].sum(dim=-1).cpu().tolist() if no_idx else [0.0] * probs.size(0)
    yes_prob = probs[:, yes_idx].max(dim=-1)[0].cpu().tolist() if yes_idx else [0.0] * probs.size(0)
    no_prob = probs[:, no_idx].max(dim=-1)[0].cpu().tolist() if no_idx else [0.0] * probs.size(0)

    out = []
    for y, n in zip(yes_prob, no_prob):
        if (y == 0.0) and (n == 0.0):
            pred = None
        else:
            pred = "Yes" if (y > n) else "No"
        out.append({"yes_prob": float(y), "no_prob": float(n), "pred": pred})
    return out

def eval_task(
    args,
    model,
    tokenizer,
    base_prompt: str,
    task: str,
    label_col: str,
    test_df: pd.DataFrame,
    save_dir: str,
    weighted_edges: bool,
    indices: Optional[List[int]] = None,
    edge_range_tag: Optional[str] = None,  
) -> Dict[str, float]:
    spec = TASKS[task]
    smiles_col = str(spec["smiles_col"])

    os.makedirs(save_dir, exist_ok=True)
    pred_path = os.path.join(
        save_dir,
        f"_pred_graph_{'weighted' if weighted_edges else 'unweighted'}_{task}_{label_col}_{args.split}.jsonl",
    )

    processed = set()
    if os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as rf:
            for line in rf:
                try:
                    obj = json.loads(line)
                    processed.add(int(obj["idx"]))
                except Exception:
                    continue

    if indices is None:
        indices = list(range(len(test_df)))
    else:
        indices = [int(i) for i in indices if 0 <= int(i) < len(test_df)]

    if args.limit is not None:
        indices = indices[: int(args.limit)]

    indices = [i for i in indices if i not in processed]

    if len(indices) == 0:
        m = _compute_metrics_from_pred_jsonl(pred_path)
        return {
            "task": task,
            "label_col": label_col,
            "split": args.split,
            "shot": float(args.shot),
            "graph_mode": "weighted" if weighted_edges else "unweighted",
            "edge_range_tag": (edge_range_tag or ""),
            **m,
        }

    with open(pred_path, "a", encoding="utf-8") as wf:
        bs = max(1, int(args.batch_size))
        for start in tqdm(range(0, len(indices), bs), desc=f"Generating [{task}]", unit="batch"):
            batch_idx = indices[start : start + bs]
            prompts: List[str] = []
            truths: List[Optional[str]] = []
            graphs: List[Optional[str]] = []
            extras: List[Dict[str, str]] = []

            for i in batch_idx:
                row = test_df.iloc[i]
                smi = str(row.get(smiles_col, "")).strip()
                extra = row_extra(spec, row)
                gt = yesno_from_label(row.get(label_col))

                g = smiles_to_graph_text(smi, weighted_edges=weighted_edges) if smi else None
                # g = edge_shuffle(g)
                if g and args.edge_agg:
                    g = edge_aggre(g)
                graphs.append(g)
                extras.append(extra)

                # few-shot still samples from the SAME eval dataframe (test_df), excluding current row i
                # shot_examples = sample_shots_graph_from_df(
                #     df=test_df,
                #     spec=spec,
                #     label_col=label_col,
                #     shot=int(args.shot),
                #     seed=int(args.seed),
                #     exclude_idx=int(i),
                #     weighted_edges=weighted_edges,
                # )
                shot_examples = []
                prompts.append(
                    build_graph_prompt(
                        base_prompt,
                        task,
                        label_col,
                        g or "Invalid SMILES",
                        shot_examples,
                        extra,
                    )
                )
                truths.append(gt if (g and gt is not None) else None)

            if getattr(args, "first_token_prob", False):
                scores = score_first_token_yesno(model, tokenizer, prompts)
                # if tokenizer lacked single-token variants, fall back to generation
                if any(s is None for s in scores):
                    outs = generate_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                    scores = [None] * len(outs)
                else:
                    # create placeholder raw outputs (we record probs separately)
                    outs = [""] * len(scores)
            else:
                outs = generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                scores = [None] * len(outs)

            for i, out, prompt, gt, g, extra in zip(batch_idx, outs, prompts, truths, graphs, extras):
                score = scores.pop(0)
                if score is not None:
                    pred = score["pred"]
                    yes_p = float(score["yes_prob"])
                    no_p = float(score["no_prob"])
                    scoring_mode = "first_token_prob"
                else:
                    pred = parse_yesno(out)
                    yes_p = None
                    no_p = None
                    scoring_mode = "generation"
                wf.write(
                    json.dumps(
                        {
                            "idx": int(i),
                            "task": task,
                            "label_col": label_col,
                            "graph_mode": "weighted" if weighted_edges else "unweighted",
                            "ground_truth": gt,
                            "prediction": pred,
                            "raw_output": out,
                            "scoring_mode": scoring_mode,
                            "yes_prob": yes_p,
                            "no_prob": no_p,
                            "edge_range_tag": (edge_range_tag or ""),
                            "prompt": prompt,
                            "extra": extra,
                            "shot": int(args.shot),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    m = _compute_metrics_from_pred_jsonl(pred_path)
    return {
        "task": task,
        "label_col": label_col,
        "split": args.split,
        "shot": float(args.shot),
        "graph_mode": "weighted" if weighted_edges else "unweighted",
        "edge_range_tag": (edge_range_tag or ""),
        **m,
    }


def main():
    parser = argparse.ArgumentParser("Property Prediction Eval (Graph Prompt)")
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASKS.keys()))
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./Results")
    parser.add_argument("--split", type=str, default="test", choices=["test", "sample"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--graph_mode",
        type=str,
        default="unweighted",
        choices=["unweighted", "weighted"],
    )

    parser.add_argument("--sample_num", type=int, default=None, help="If set, sample N examples within an auto-chosen edge range for eval.")
    parser.add_argument("--preferred_min_edges", type=int, default=60)
    parser.add_argument("--hard_max_edges", type=int, default=-1, help="-1 disables")
    parser.add_argument("--no_edge_range_sample", action="store_true", help="Disable edge-range sampling even if --sample_num is set.")

    parser.add_argument("--layer_head_config_path", type=str, default=None)
    parser.add_argument("--layers_to_modify", type=int, nargs="+", default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--edge_agg", action="store_true")
    parser.add_argument(
        "--first_token_prob",
        action="store_true",
        help="Decide Yes/No from next-token probability mass for 'Yes' vs 'No' (robust to CoT).",
    )
    parser.add_argument("--run_mode", type=str, default="test")

    args = parser.parse_args()
    weighted_edges = args.graph_mode == "weighted"

    prompts = load_prompts(args.prompt_path)
    if args.task not in prompts:
        raise KeyError(f"Missing prompt for task={args.task} in {args.prompt_path}. Found: {sorted(prompts.keys())}")
    base_prompt = prompts[args.task]

    original_csv, test_csv = resolve_csv_paths(args.data_dir, args.task)
    test_df = pd.read_csv(test_csv)

    original_df = pd.read_csv(original_csv) if original_csv is not None else test_df
    if args.split != "test":
        test_df = original_df

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    ensure_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
    )
    model.eval()

    mod_mode = "none"
    layers_heads_to_modify = None

    if args.layer_head_config_path is not None:
        with open(args.layer_head_config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        layers_heads_to_modify = _normalize_layers_heads(raw, num_heads=int(model.config.num_attention_heads))
        mod_mode = "config"
    elif args.layers_to_modify is not None:
        layers_heads_to_modify = {
            str(l): list(range(int(model.config.num_attention_heads)))
            for l in args.layers_to_modify
        }
        mod_mode = "list"

    if layers_heads_to_modify:
        print(f"Applying modifications for layers: {list(layers_heads_to_modify.keys())}")
        MODIFICATION(model, layers_heads_to_modify, float(args.gamma))

    pure_model = os.path.basename(args.model_path.rstrip("/"))
    shot_tag = f"shot{int(args.shot)}"
    graph_tag = f"graph_{args.graph_mode}"
    split_tag = f"{args.split}_{args.preferred_min_edges}min_{args.hard_max_edges}max" if args.split != "test" else "test"
    base_dir = os.path.join(args.output_dir, pure_model, args.task, shot_tag, graph_tag, split_tag)
    agg_edge_tag = "_edgeAgg" if args.edge_agg else ""

    if mod_mode == "none":
        save_dir = os.path.join(base_dir, f"test{agg_edge_tag}")
    elif mod_mode == "config":
        config_tag = args.layer_head_config_path.split('_')[-1].replace('.json', '')
        save_dir = os.path.join(base_dir, f"modified_gamma{args.gamma}{agg_edge_tag}{config_tag}")
    else:
        layer_tag = "_".join(str(l) for l in args.layers_to_modify)
        save_dir = os.path.join(base_dir, f"gamma{args.gamma}_{layer_tag}{agg_edge_tag}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, float]] = {}
    spec = TASKS[args.task]
    label_to_eval = spec["label_cols"][0]

    if args.task == "Tox21":
        # Build an "overall toxic" label: if ANY assay == 1 => 1; else if any known (0/1) and none 1 => 0; else NaN
        tox_any_col = "__TOX21_TOXIC_ANY__"
        tox_cols = [c for c in spec["label_cols"] if c in test_df.columns]

        if tox_any_col in test_df.columns:
            return
        if not tox_cols:
            test_df[tox_any_col] = np.nan
            return

        mat = test_df[tox_cols].apply(pd.to_numeric, errors="coerce")
        has_any = mat.notna().any(axis=1)
        any1 = (mat == 1).any(axis=1)
        test_df[tox_any_col] = np.where(any1, 1, np.where(has_any, 0, np.nan))

        label_to_eval = tox_any_col

    calibration_indices = None
    if args.run_mode == "calibration":
        # determine which indices the "test" run would use (may be a sampled subset or the full split)
        test_mode_set = None
        if (args.sample_num is not None) and (not args.no_edge_range_sample):
            ok = _try_import_slash()
            if not ok:
                raise ImportError("Failed to import SLASH for edge-range sampling (needed to compute non-overlap).")
            from slash.datasets import molecularnet_sample_indices_by_edge_range

            mn_cfg = {
                "root": args.data_dir,
                "task": args.task,
                "split": args.split,
                "weighted_edges": bool(weighted_edges),
            }
            # indices that *would* be used by the test run (sample_num from args)
            test_idx_for_sample, _rng, _stats = molecularnet_sample_indices_by_edge_range(
                molecularnet=mn_cfg,
                sample_num=int(args.sample_num),
                preferred_min_edges=int(args.preferred_min_edges),
                hard_max_edges=(None if int(args.hard_max_edges) < 0 else int(args.hard_max_edges)),
                seed=int(args.seed),
                require_label=True,
            )
            test_mode_set = set(test_idx_for_sample or [])

            pool_idx_for_range, _rng2, _stats2 = molecularnet_sample_indices_by_edge_range(
                molecularnet=mn_cfg,
                sample_num=int(args.sample_num+100),
                preferred_min_edges=int(args.preferred_min_edges),
                hard_max_edges=(None if int(args.hard_max_edges) < 0 else int(args.hard_max_edges)),
                seed=int(args.seed),
                require_label=True,
            )
            pool_set = set(pool_idx_for_range or [])
            print(f"[Calibration] edge-range pool has {len(pool_set)} examples")
        else:
            # "test" run would evaluate the entire available dataframe -> no safe disjoint set
            test_mode_set = set(range(len(test_df)))
            pool_set = set()  # no edge-range pool when test covers whole DF

        # candidates must (a) be in the same edge-range pool (if applicable), (b) not overlap test_mode_set,
        # and (c) have a valid label for label_to_eval.
        if pool_set:
            candidates = [i for i in pool_set if i not in test_mode_set]
        else:
            candidates = [i for i in range(len(test_df)) if i not in test_mode_set]
        if len(candidates) < 100:
            raise ValueError(
                f"Cannot build 100 disjoint calibration examples that also satisfy the edge-range: "
                f"only {len(candidates)} available. Try using a different --preferred_min_edges/--hard_max_edges, "
                "--sample_num (so test uses a subset), or change --split."
            )
        calibration_indices = candidates[:100]

        print(f"[Calibration] selected {len(calibration_indices)} disjoint indices (seed={args.seed})")

    # choose indices by edge-range + sampling (optional)
    eval_indices = None
    edge_range_tag = None
    # if (args.sample_num is not None) and (not args.no_edge_range_sample):
    # If calibration was requested, use the precomputed calibration_indices and skip edge-range sampling.
    if args.run_mode == "calibration":
        eval_indices = calibration_indices
        edge_range_tag = "calibration_n100"
    elif (args.sample_num is not None) and (not args.no_edge_range_sample):
        ok = _try_import_slash()
        if not ok:
            raise ImportError("Failed to import SLASH for edge-range sampling. Check SLASH/src is accessible.")
        from slash.datasets import molecularnet_sample_indices_by_edge_range

        mn_cfg = {
            "root": args.data_dir,
            "task": args.task,
            "split": args.split,
            "weighted_edges": bool(weighted_edges),
        }

        hard_cap = None if int(args.hard_max_edges) < 0 else int(args.hard_max_edges)
        eval_indices, (emin, emax), stats = molecularnet_sample_indices_by_edge_range(
            molecularnet=mn_cfg,
            sample_num=int(args.sample_num),
            preferred_min_edges=int(args.preferred_min_edges),
            hard_max_edges=hard_cap,
            seed=int(args.seed),
            require_label=True,
        )
        edge_range_tag = f"edges{int(emin)}-{int(emax)}_n{int(len(eval_indices))}"
        print(f"[EdgeRangeSample] {json.dumps(stats, ensure_ascii=False)}")

        if not eval_indices:
            print("[EdgeRangeSample] No indices selected; will fall back to full eval.")
            eval_indices = None
            edge_range_tag = None
    
    print(f"Evaluating task={args.task}, model={pure_model}, gamma={args.gamma}, split={args.split}")
    m = eval_task(
        args=args,
        model=model,
        tokenizer=tokenizer,
        base_prompt=base_prompt,
        task=args.task,
        label_col=label_to_eval,
        test_df=test_df,
        save_dir=save_dir,
        weighted_edges=weighted_edges,
        indices=eval_indices,
        edge_range_tag=edge_range_tag, 
    )
    metrics[str(label_to_eval)] = m

    vals = [v for v in metrics.values() if float(v.get("total", 0.0)) > 0]
    agg = {
        "task": args.task,
        "split": args.split,
        "shot": float(args.shot),
        "graph_mode": args.graph_mode,
        "mean_accuracy": float(np.mean([v["accuracy"] for v in vals])) if vals else 0.0,
        "mean_f1": float(np.mean([v["f1"] for v in vals])) if vals else 0.0,
    }

    out = {"aggregate": agg}
    out_path = os.path.join(save_dir, f"metrics_graph_{args.task}_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()