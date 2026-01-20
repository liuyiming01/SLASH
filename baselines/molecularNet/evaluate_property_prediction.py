import argparse
import json
import os
import random
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Prompts
# -----------------------------
def load_prompts(prompt_txt_path: str) -> Dict[str, str]:
    """
    Parse lines like:
      BACE: prompt = "...."
    Return dict: {"BACE": "...", ...}
    """
    prompts: Dict[str, str] = {}
    with open(prompt_txt_path, "r", encoding="utf-8") as f:
        txt = f.read()

    # Match: KEY: prompt = " .... "
    # (non-greedy for quoted string; handle escaped quotes roughly)
    pattern = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*:\s*prompt\s*=\s*\"([\s\S]*?)\"\s*$", re.MULTILINE)
    for m in pattern.finditer(txt):
        key = m.group(1).strip()
        prompt = m.group(2)
        # Unescape common \n and \t sequences if present
        prompt = prompt.replace("\\n", "\n").replace("\\t", "\t").replace("\\\"", "\"").strip()
        prompts[key] = prompt
    return prompts


def _yesno_from_label(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if s == "":
        return None
    # common encodings
    if s.lower() in {"yes", "y", "true"}:
        return "Yes"
    if s.lower() in {"no", "n", "false"}:
        return "No"
    # numeric 0/1
    try:
        iv = int(float(s))
        if iv == 1:
            return "Yes"
        if iv == 0:
            return "No"
    except Exception:
        pass
    return None


def _parse_yesno(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()
    # strict
    if re.search(r"\byes\b", t):
        return "Yes"
    if re.search(r"\bno\b", t):
        return "No"
    # fallback: first char
    if t[:1] == "y":
        return "Yes"
    if t[:1] == "n":
        return "No"
    return None


# -----------------------------
# Modification (same idea as GraphWiz/Graph-SST)
# -----------------------------
def parse_layer_head_config(path: str) -> List[Tuple[int, int]]:
    """
    Accept multiple JSON shapes and return [(layer, head), ...]
    Supported:
      - {"0":[1,2], "1":[0]} or {0:[...], ...}
      - [{"layer":0,"heads":[1,2]}, ...]
      - [[0,1],[0,2],...]
      - {"layers_heads_to_modify":[[0,1],...]} etc.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    def add_pair(out: List[Tuple[int, int]], l, h):
        try:
            out.append((int(l), int(h)))
        except Exception:
            return

    pairs: List[Tuple[int, int]] = []

    if isinstance(obj, dict):
        if "layers_heads_to_modify" in obj and isinstance(obj["layers_heads_to_modify"], list):
            obj = obj["layers_heads_to_modify"]
        elif "selected_heads" in obj and isinstance(obj["selected_heads"], list):
            obj = obj["selected_heads"]

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list):
                for head in v:
                    add_pair(pairs, k, head)
        return sorted(list(set(pairs)))

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and "layer" in item and "heads" in item and isinstance(item["heads"], list):
                for h in item["heads"]:
                    add_pair(pairs, item["layer"], h)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                add_pair(pairs, item[0], item[1])
        return sorted(list(set(pairs)))

    return []


def MODIFICATION(model, layers_heads_to_modify, delta_ratio: float, first_token_idx: int = 0):
    sys.path.insert(0, "/home/lym/LLM-Research/Attention/Graph_Attention/src/GraphLens")
    from modeling import modeling_llama_attn_shift

    LlamaModel_forward, LlamaDecoderLayer_forward, LlamaAttention_forward = (
        modeling_llama_attn_shift.get_modified_forward_llama(
            layers_heads_to_modify=layers_heads_to_modify,
            delta_ratio=delta_ratio,
            first_token_idx=first_token_idx,
        )
    )
    model.model.forward = types.MethodType(LlamaModel_forward, model.model)
    for idx, layer in enumerate(model.model.layers):
        layer.forward = types.MethodType(LlamaDecoderLayer_forward, layer)
        layer.self_attn.forward = types.MethodType(LlamaAttention_forward, layer.self_attn)


# -----------------------------
# Data specs
# -----------------------------
@dataclass
class TaskSpec:
    name: str
    smiles_col: str
    label_cols: List[str]  # evaluate each label col independently (binary)
    extra_cols: Optional[List[str]] = None


TOX21_LABEL_COLS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


TASKS: Dict[str, TaskSpec] = {
    "BACE": TaskSpec(name="BACE", smiles_col="mol", label_cols=["Class"]),
    "BBBP": TaskSpec(name="BBBP", smiles_col="smiles", label_cols=["p_np"]),
    "ClinTox": TaskSpec(name="ClinTox", smiles_col="smiles", label_cols=["FDA_APPROVED", "CT_TOX"]),
    "HIV": TaskSpec(name="HIV", smiles_col="smiles", label_cols=["HIV_active"], extra_cols=["activity"]),
    "Tox21": TaskSpec(name="Tox21", smiles_col="smiles", label_cols=TOX21_LABEL_COLS, extra_cols=["mol_id"]),
}


def resolve_csv_paths(data_dir: str, task: str) -> Tuple[Optional[str], str]:
    """
    Prefer: <task>_train.csv + <task>_test.csv
    But allow: only <task>_test.csv (no train).
    Return: (train_path_or_None, test_path)
    """
    candidates = [
        (f"{task}_train.csv", f"{task}_test.csv"),
        (f"{task.lower()}_train.csv", f"{task.lower()}_test.csv"),
    ]

    # 1) train+test both exist
    for tr, te in candidates:
        trp = os.path.join(data_dir, tr)
        tep = os.path.join(data_dir, te)
        if os.path.exists(trp) and os.path.exists(tep):
            return trp, tep

    # 2) only test exists
    for _, te in candidates:
        tep = os.path.join(data_dir, te)
        if os.path.exists(tep):
            return None, tep

    # 3) fallback: any *_test.csv that matches task name (case-insensitive)
    for fn in os.listdir(data_dir):
        if fn.lower() == f"{task.lower()}_test.csv":
            return None, os.path.join(data_dir, fn)

    raise FileNotFoundError(f"Cannot find *_test.csv for task={task} under {data_dir}")



def _format_example(task: str, label_col: str, smiles: str, label_yesno: str, extra: Optional[Dict[str, str]] = None) -> str:
    extra = extra or {}
    if task == "BACE":
        return f"SMILES: {smiles}\nBACE-1 Inhibit: {label_yesno}\n"
    if task == "BBBP":
        return f"SMILES: {smiles}\nBBBP Penetration: {label_yesno}\n"
    if task == "HIV":
        act = extra.get("activity", "")
        act_line = f"HIV activity test: {act}\n" if act else ""
        return f"SMILES: {smiles}\n{act_line}HIV Inhibit: {label_yesno}\n"
    if task == "ClinTox":
        if label_col == "FDA_APPROVED":
            return f"SMILES: {smiles}\nFDA Approved: {label_yesno}\n"
        return f"SMILES: {smiles}\nClinically-trial-toxic: {label_yesno}\n"
    if task == "Tox21":
        return f"SMILES: {smiles}\nAssay: {label_col}\nToxic: {label_yesno}\n"
    # fallback
    return f"SMILES: {smiles}\nLabel: {label_yesno}\n"


def _format_query(task: str, label_col: str, smiles: str, extra: Optional[Dict[str, str]] = None) -> str:
    extra = extra or {}
    if task == "BACE":
        return f"SMILES: {smiles}\nBACE-1 Inhibit:"
    if task == "BBBP":
        return f"SMILES: {smiles}\nBBBP Penetration:"
    if task == "HIV":
        act = extra.get("activity", "")
        act_line = f"HIV activity test: {act}\n" if act else ""
        return f"SMILES: {smiles}\n{act_line}HIV Inhibit:"
    if task == "ClinTox":
        if label_col == "FDA_APPROVED":
            return f"SMILES: {smiles}\nFDA Approved:"
        return f"SMILES: {smiles}\nClinically-trial-toxic:"
    if task == "Tox21":
        return f"SMILES: {smiles}\nAssay: {label_col}\nToxic:"
    return f"SMILES: {smiles}\nLabel:"


def build_prompt(
    base_prompt: str,
    task: str,
    label_col: str,
    smiles: str,
    shot_examples: List[Tuple[str, str, Dict[str, str]]],  # (smiles, Yes/No, extra)
    extra: Optional[Dict[str, str]] = None,
) -> str:
    parts = [base_prompt.strip()]
    for ex_smiles, ex_label, ex_extra in shot_examples:
        parts.append(_format_example(task, label_col, ex_smiles, ex_label, ex_extra).strip())
    parts.append(_format_query(task, label_col, smiles, extra).strip())
    return "\n".join(parts).strip() + "\n"


# -----------------------------
# Inference
# -----------------------------
@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float = 0.0,
) -> List[str]:
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

    # strip prompt prefix
    out_texts: List[str] = []
    for p, full in zip(prompts, outs):
        out_texts.append(full[len(p):].strip() if full.startswith(p) else full.strip())
    return out_texts


def sample_shots(
    train_df: pd.DataFrame,
    spec: TaskSpec,
    label_col: str,
    shot: int,
    seed: int,
) -> List[Tuple[str, str, Dict[str, str]]]:
    if shot <= 0:
        return []

    rng = random.Random(seed)
    df = train_df.copy()

    # keep only rows with valid binary labels
    y = []
    keep_idx = []
    for i, v in enumerate(df[label_col].tolist()):
        yn = _yesno_from_label(v)
        if yn is None:
            continue
        keep_idx.append(i)
        y.append(yn)
    if not keep_idx:
        return []

    df = df.iloc[keep_idx].reset_index(drop=True)

    # try balance Yes/No if possible
    yes_df = df[df[label_col].apply(lambda v: _yesno_from_label(v) == "Yes")]
    no_df = df[df[label_col].apply(lambda v: _yesno_from_label(v) == "No")]
    shots: List[pd.Series] = []

    if len(yes_df) > 0 and len(no_df) > 0 and shot >= 2:
        k_yes = shot // 2
        k_no = shot - k_yes
        shots += yes_df.sample(n=min(k_yes, len(yes_df)), random_state=seed).to_dict("records")
        shots += no_df.sample(n=min(k_no, len(no_df)), random_state=seed + 1).to_dict("records")
        rng.shuffle(shots)
    else:
        shots = df.sample(n=min(shot, len(df)), random_state=seed).to_dict("records")

    out: List[Tuple[str, str, Dict[str, str]]] = []
    for row in shots:
        smi = str(row.get(spec.smiles_col, "")).strip()
        yn = _yesno_from_label(row.get(label_col))
        if not smi or yn is None:
            continue
        extra: Dict[str, str] = {}
        if spec.extra_cols:
            for c in spec.extra_cols:
                if c in row and row[c] is not None and not (isinstance(row[c], float) and np.isnan(row[c])):
                    extra[c] = str(row[c])
        out.append((smi, yn, extra))
    return out


# -----------------------------
# Eval
# -----------------------------
def ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def eval_task_label(
    args,
    model,
    tokenizer,
    base_prompt: str,
    task: str,
    label_col: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: str,
) -> Dict[str, float]:
    spec = TASKS[task]

    pred_path = os.path.join(save_dir, f"_pred_{task}_{label_col}_{args.split}.jsonl")
    os.makedirs(save_dir, exist_ok=True)

    processed = set()
    if os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as rf:
            for line in rf:
                try:
                    obj = json.loads(line)
                    processed.add(int(obj["idx"]))
                except Exception:
                    continue

    # build indices (with optional limit), skip processed
    indices = list(range(len(test_df)))
    if args.limit is not None:
        indices = indices[: int(args.limit)]
    indices = [i for i in indices if i not in processed]
    if len(indices) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "total": 0.0, "correct": 0.0, "invalid": 0.0}

    shot_examples = sample_shots(train_df, spec, label_col, args.shot, args.seed)

    y_true: List[int] = []
    y_pred: List[int] = []
    invalid = 0

    def get_row_extra(row) -> Dict[str, str]:
        extra: Dict[str, str] = {}
        if spec.extra_cols:
            for c in spec.extra_cols:
                if c in row and row[c] is not None and not (isinstance(row[c], float) and np.isnan(row[c])):
                    extra[c] = str(row[c])
        return extra

    with open(pred_path, "a", encoding="utf-8") as wf:
        bs = max(1, int(args.batch_size))
        for start in range(0, len(indices), bs):
            batch_idx = indices[start:start + bs]
            prompts: List[str] = []
            truths: List[Optional[str]] = []
            smiles_list: List[str] = []
            extras_list: List[Dict[str, str]] = []

            for i in batch_idx:
                row = test_df.iloc[i]
                smiles = str(row.get(spec.smiles_col, "")).strip()
                extra = get_row_extra(row)

                gt = _yesno_from_label(row.get(label_col))
                if not smiles or gt is None:
                    # still write record (invalid ground truth), but skip scoring
                    prompts.append(build_prompt(base_prompt, task, label_col, smiles, shot_examples, extra))
                    truths.append(None)
                else:
                    prompts.append(build_prompt(base_prompt, task, label_col, smiles, shot_examples, extra))
                    truths.append(gt)

                smiles_list.append(smiles)
                extras_list.append(extra)

            outs = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            for i, prompt, out, gt, smiles, extra in zip(batch_idx, prompts, outs, truths, smiles_list, extras_list):
                pred = _parse_yesno(out)
                rec = {
                    "idx": int(i),
                    "task": task,
                    "label_col": label_col,
                    "smiles": smiles,
                    "extra": extra,
                    "shot": int(args.shot),
                    "ground_truth": gt,
                    "prediction": pred,
                    "raw_output": out,
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if gt is None:
                    continue
                if pred is None:
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

    return {
        "task": task,
        "label_col": label_col,
        "split": args.split,
        "shot": float(args.shot),
        "total": total,
        "correct": correct,
        "invalid": float(invalid),
        "accuracy": acc,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser("Property Prediction Eval (ChemLLMBench)")
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASKS.keys()))
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./Results_Property")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--layer_head_config_path",
        type=str,
        default=None,
        help="Path to a JSON file specifying which layer-heads to modify.",
    )
    parser.add_argument(
        "--layers_to_modify",
        type=int,
        nargs="+",
        default=None,
        help="List of layer indices to modify when no layer_head_config_path is provided. "
             "Do not pass to disable modification.",
    )
    parser.add_argument(
        "--delta_ratio",
        type=float,
        default=0.4,
        help="The delta ratio for attention redistribution (used when modification is enabled).",
    )

    args = parser.parse_args()

    prompts = load_prompts(args.prompt_path)
    if args.task not in prompts:
        raise KeyError(f"Missing prompt for task={args.task} in {args.prompt_path}. Found: {sorted(prompts.keys())}")
    base_prompt = prompts[args.task]

    train_csv, test_csv = resolve_csv_paths(args.data_dir, args.task)
    test_df = pd.read_csv(test_csv)
    # If no train split, fall back to using test as the shot pool (simplest workable behavior)
    train_df = pd.read_csv(train_csv) if train_csv is not None else test_df

    if args.split != "test":
        # No dedicated train split in your setting; keep behavior simple:
        test_df = train_df

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    ensure_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="eager",
    )
    model.eval()

    # -----------------------------
    # Modification mode (none / config / list)
    # -----------------------------
    mod_mode = "none"
    layers_heads_to_modify = None

    if args.layer_head_config_path is not None:
        # config mode: load json directly (same style as Graph-SST evaluate.py)
        print(f"Loading layer-head config from {args.layer_head_config_path}")
        with open(args.layer_head_config_path, "r", encoding="utf-8") as f:
            layers_heads_to_modify = json.load(f)
        mod_mode = "config"
    elif args.layers_to_modify is not None:
        # list mode: modify all heads in the selected layers
        print(f"Using layers_to_modify from args: {args.layers_to_modify}")
        layers_heads_to_modify = {
            str(l): list(range(model.config.num_attention_heads))
            for l in args.layers_to_modify
        }
        mod_mode = "list"
    else:
        print("No layer_head_config_path and no layers_to_modify; no modification will be applied.")
        layers_heads_to_modify = None
        mod_mode = "none"

    if layers_heads_to_modify:
        MODIFICATION(model, layers_heads_to_modify, float(args.delta_ratio))

    # -----------------------------
    # Output dir
    # -----------------------------
    pure_model = os.path.basename(args.model_path.rstrip("/"))
    shot_tag = f"shot{int(args.shot)}"
    base_dir = os.path.join(args.output_dir, pure_model, args.task, shot_tag)

    if mod_mode == "none":
        save_dir = os.path.join(base_dir, "test")
    elif mod_mode == "config":
        save_dir = os.path.join(base_dir, f"modified_delta{args.delta_ratio}")
    else:  # mod_mode == "list"
        layer_tag = "_".join(str(l) for l in args.layers_to_modify)
        save_dir = os.path.join(base_dir, f"delta{args.delta_ratio}_{layer_tag}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ...existing code...
    # evaluate each label col
    spec = TASKS[args.task]
    metrics: Dict[str, Dict[str, float]] = {}
    for label_col in spec.label_cols:
        m = eval_task_label(
            args=args,
            model=model,
            tokenizer=tokenizer,
            base_prompt=base_prompt,
            task=args.task,
            label_col=label_col,
            train_df=train_df,
            test_df=test_df,
            save_dir=save_dir,
        )
        metrics[label_col] = m

    # aggregate (mean over label columns that have non-zero total)
    vals = [v for v in metrics.values() if float(v.get("total", 0.0)) > 0]
    agg = {
        "task": args.task,
        "split": args.split,
        "shot": float(args.shot),
        "num_labels": float(len(spec.label_cols)),
        "mean_accuracy": float(np.mean([v["accuracy"] for v in vals])) if vals else 0.0,
        "mean_f1": float(np.mean([v["f1"] for v in vals])) if vals else 0.0,
    }

    out = {"aggregate": agg, "per_label": metrics}
    out_path = os.path.join(save_dir, f"metrics_{args.task}_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()