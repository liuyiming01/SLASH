import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from dig.xgraph.dataset import SentiGraphDataset

def load_graph_prompt_templates(path: str) -> Dict[str, str]:
    prompts = {}
    current_section = None
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                if current_section and lines:
                    prompts[current_section] = "\n".join(lines).strip()
                current_section = stripped[1:-1].strip()
                lines = []
                continue
            if current_section is None:
                continue
            lines.append(line)
        if current_section and lines:
            prompts[current_section] = "\n".join(lines).strip()
    return prompts

def get_label2text(data_name: str) -> Dict[int, str]:
    if "Twitter" in data_name:
        return {0: "negative", 1: "neutral", 2: "positive"}
    if data_name == "Graph-SST2":
        return {0: "negative", 1: "positive"}
    return {0: "very negative", 1: "negative", 2: "neutral", 3: "positive", 4: "very positive"}


def get_split_indices_from_supplement(dataset, split: str) -> List[int]:
    """
    DIG puts split info in dataset.supplement['split_indices'] (len == len(dataset)).
    Values are usually {0,1,2} or {1,2,3}. This function handles both + a fallback heuristic.
    """
    split = split.lower()
    if not hasattr(dataset, "supplement"):
        return list(range(len(dataset)))
    s = dataset.supplement.get("split_indices", None)
    if s is None:
        return list(range(len(dataset)))

    if torch.is_tensor(s):
        s = s.cpu().tolist()
    else:
        s = list(s)

    if len(s) != len(dataset):
        return list(range(len(dataset)))

    uniq = sorted({int(x) for x in s})
    if uniq == [0, 1, 2]:
        tag = {"train": 0, "val": 1, "valid": 1, "test": 2}[split]
    elif uniq == [1, 2, 3]:
        tag = {"train": 1, "val": 2, "valid": 2, "test": 3}[split]
    else:
        tag_map = {
            "train": uniq[0],
            "val": uniq[1] if len(uniq) >= 3 else uniq[0],
            "valid": uniq[1] if len(uniq) >= 3 else uniq[0],
            "test": uniq[-1],
        }
        tag = tag_map[split]

    return [i for i, t in enumerate(s) if int(t) == int(tag)]


def get_node_texts(data, dataset=None, idx: Optional[int] = None) -> List[str]:
    """
    Prefer dataset.supplement['sentence_tokens'][str(idx)] if available.
    Fallback: node_0, node_1, ...
    """
    num_nodes: Optional[int] = None
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        num_nodes = int(data.num_nodes)
    elif hasattr(data, "x") and torch.is_tensor(data.x):
        num_nodes = int(data.x.size(0))

    toks: Optional[List[str]] = None
    if dataset is not None and idx is not None and hasattr(dataset, "supplement"):
        st = getattr(dataset, "supplement", {}).get("sentence_tokens", None)
        if isinstance(st, dict):
            v = st.get(str(idx), None)
            if isinstance(v, list):
                toks = [str(x) for x in v]

    if num_nodes is None:
        return toks or []

    if toks is None:
        return [f"node_{i}" for i in range(num_nodes)]

    # align length to num_nodes
    if len(toks) < num_nodes:
        toks = toks + [f"node_{i}" for i in range(len(toks), num_nodes)]
    elif len(toks) > num_nodes:
        toks = toks[:num_nodes]
    return toks


def linearize_graph(
    data,
    node_texts: List[str],
    directed: bool = False,
) -> Tuple[str, str]:
    """
    Pure graph serialization components (no task text):
    - nodes_part: "[0, w0] [1, w1] ..."
    - edges_part: "(u, v) ..." or "(u->v) ..."
    """
    n = len(node_texts)

    nodes_part = " ".join(
        [f"[{i}, {str(node_texts[i]).replace(chr(10), ' ')}]" for i in range(n)]
    )

    edges: List[Tuple[int, int]] = []
    if hasattr(data, "edge_index") and torch.is_tensor(data.edge_index):
        ei = data.edge_index
        if ei.numel() > 0:
            ei = ei.detach().cpu()
            src = ei[0].tolist()
            dst = ei[1].tolist()

            if directed:
                # keep first seen direction per undirected pair {u,v}
                seen = set()
                for u, v in zip(src, dst):
                    u, v = int(u), int(v)
                    if u == v:
                        continue
                    key = (min(u, v), max(u, v))
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append((u, v))
                edges.sort(key=lambda x: (x[0], x[1]))
            else:
                uniq = set()
                for u, v in zip(src, dst):
                    u, v = int(u), int(v)
                    if u == v:
                        continue
                    a, b = (u, v) if u < v else (v, u)
                    uniq.add((a, b))
                edges = sorted(list(uniq), key=lambda x: (x[0], x[1]))

    if directed:
        edges_part = " ".join([f"({u}->{v})" for u, v in edges])
    else:
        edges_part = " ".join([f"({u}, {v})" for u, v in edges])

    if not edges_part:
        edges_part = "(empty)"

    return nodes_part, edges_part


def build_prompt(
    nodes_part: str,
    edges_part: str,
    label2text: Dict[int, str],
    num_nodes: int,
    prompt_templates: Dict[str, str],
    directed: bool = False,
) -> str:
    """
    Mimic the provided example style closely, while keeping graph serialization separate.
    """
    options = "\n".join([f"{k}: {v}." for k, v in label2text.items()])
    last_id = max(0, int(num_nodes) - 1)
    section = "directed" if directed else "undirected"
    template = prompt_templates.get(section)
    if template is None:
        raise KeyError(f"Missing prompt template for section={section}")
    return template.format(
        options=options,
        nodes_part=nodes_part,
        edges_part=edges_part,
        last_id=last_id,
    )

def _parse_first_int(text: str) -> Optional[int]:
    m = re.search(r"-?\d+", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


@torch.inference_mode()
def llm_predict_label_id_batch(
    prompts: List[str],
    tokenizer,
    model,
    max_new_tokens: int = 16,
    add_special_tokens: bool = True,
) -> List[Tuple[Optional[int], str]]:
    """
    Batched generation. Returns (optional_int, raw_generation) per prompt.
    Correct for LEFT padding by slicing with the padded input length.
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=add_special_tokens,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    in_len = enc["input_ids"].shape[1]
    gen_ids = out[:, in_len:]
    gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    results: List[Tuple[Optional[int], str]] = []
    for gen_text in gen_texts:
        raw = (gen_text or "").strip()
        results.append((_parse_first_int(raw), raw))
    return results


def MODIFICATION(model, layers_heads_to_modify, delta_ratio, first_token_idx=0):
    import sys
    import types

    sys.path.insert(0, "/home/lym/LLM-Research/Attention/Graph_Attention/GraphLens")

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
                delta_ratio=delta_ratio,
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
                delta_ratio=delta_ratio,
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

def evaluate(
    args,
    dataset: SentiGraphDataset,
    model_path: str,
    split: str = "test",
    batch_size: int = 1,
    max_tokens: int = 6,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    label2text = get_label2text(dataset.name)
    valid_ids = set(label2text.keys())

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    # keep model/config consistent
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # 记录修改模式：none / config / list
    mod_mode = "none"
    layers_heads_to_modify = None

    if args.layer_head_config_path is not None:
        # 情况 2: 使用外部 json 配置
        print(f"Loading layer-head config from {args.layer_head_config_path}")
        with open(args.layer_head_config_path, "r") as f:
            layers_heads_to_modify = json.load(f)
        mod_mode = "config"
    else:
        # 情况 3: 使用命令行列表指定要修改的 layer
        if args.layers_to_modify is not None:
            print(f"Using layers_to_modify from args: {args.layers_to_modify}")
            layers_to_modify = args.layers_to_modify
            layers_heads_to_modify = {
                str(l): list(range(model.config.num_attention_heads))
                for l in layers_to_modify
            }
            mod_mode = "list"
        else:
            # 情况 1: 不做任何修改
            print("No layer_head_config_path and no layers_to_modify; no modification will be applied.")
            layers_heads_to_modify = None
            mod_mode = "none"

    if layers_heads_to_modify:
        print(f"Applying modifications for layers: {list(layers_heads_to_modify.keys())}")
        MODIFICATION(model, layers_heads_to_modify, delta_ratio=args.delta_ratio)

    pure_model = model_path.split('/')[-1]
    prompt_tag = "only"
    if args.sys_prompt:
        prompt_tag += "_sys"

    base_dir = f"{args.output_dir}/{pure_model}/{prompt_tag}"
    if mod_mode == "none":
        save_dir = f"{base_dir}/test"
    elif mod_mode == "config":
        config_tag = args.layer_head_config_path.split('_')[-1].replace('.json', '')
        save_dir = f"{base_dir}/modified_{config_tag}_delta{args.delta_ratio}"
    elif mod_mode == "list":
        layer_tag = "_".join(str(l) for l in args.layers_to_modify)
        save_dir = f"{base_dir}/delta{args.delta_ratio}_{layer_tag}"
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------
    # JSONL append + resume
    # ---------------------------
    pred_path = os.path.join(save_dir, f"_pred_{dataset.name}_{split}.jsonl")

    processed = set()
    total = correct = invalid = 0

    if os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # tolerate a partially-written last line
                    continue
                idx0 = rec.get("idx", None)
                if idx0 is None:
                    continue
                processed.add(int(idx0))
                total += 1
                if rec.get("invalid", False):
                    invalid += 1
                elif rec.get("correct", False):
                    correct += 1

        print(f"Resume: found {len(processed)} processed samples in {pred_path}")


    indices = get_split_indices_from_supplement(dataset, split=split)
    if limit is not None:
        random.seed(args.seed)
        indices = random.sample(indices, min(limit, len(indices)))

    # skip processed
    indices = [i for i in indices if int(i) not in processed]
    if len(indices) == 0:
        result = {
            "dataset": dataset.name,
            "split": split,
            "total": float(total),
            "correct": float(correct),
            "invalid": float(invalid),
            "accuracy": float(correct / total) if total else 0.0,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    prompt_templates = load_graph_prompt_templates(args.prompt_path)
    bs = max(1, int(batch_size))

    with open(pred_path, "a", encoding="utf-8") as wf:
        for start in tqdm(range(0, len(indices), bs), desc=f"eval {dataset.name}/{split}"):
            batch_indices = indices[start : start + bs]

            batch_prompts: List[str] = []
            batch_labels: List[int] = []

            for idx in batch_indices:
                data = dataset[idx]
                y = int(data.y) if torch.is_tensor(data.y) else int(data.y)

                node_texts = get_node_texts(data, dataset=dataset, idx=idx)
                directed = args.directed  # keep consistent with linearize_graph default; expose as arg later if needed
                nodes_part, edges_part = linearize_graph(data, node_texts, directed=directed)
                prompt = build_prompt(
                    nodes_part=nodes_part,
                    edges_part=edges_part,
                    label2text=label2text,
                    num_nodes=len(node_texts),
                    prompt_templates=prompt_templates,
                    directed=directed,
                )

                batch_prompts.append(prompt)
                batch_labels.append(y)

            # --- sys_prompt: mimic evaluate_new.py ---
            if args.sys_prompt:
                if "qwen" in model_path.lower():
                    batch_prompts = [
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": t}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        for t in batch_prompts
                    ]
                else:
                    batch_prompts = [
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": t}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for t in batch_prompts
                    ]

            pred_and_text = llm_predict_label_id_batch(
                batch_prompts,
                tokenizer,
                model,
                max_new_tokens=max_tokens,
                add_special_tokens=False if (args.sys_prompt and "llama" in model_path.lower()) else True,
            )

            for idx, (pred, gen_text), y in zip(batch_indices, pred_and_text, batch_labels):
                is_invalid = (pred is None) or (pred not in valid_ids)
                is_correct = (not is_invalid) and (pred == y)

                total += 1
                if is_invalid:
                    invalid += 1
                elif is_correct:
                    correct += 1

                rec = {
                    "idx": int(idx),
                    "y": int(y),
                    "pred": None if pred is None else int(pred),
                    "gen_text": gen_text,
                    "invalid": bool(is_invalid),
                    "correct": bool(is_correct),
                    "input_prompt": batch_prompts[batch_indices.index(idx)],
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wf.flush()

    result = {
        "dataset": dataset.name,
        "split": split,
        "total": float(total),
        "correct": float(correct),
        "invalid": float(invalid),
        "accuracy": float(correct / total) if total else 0.0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    out_path = os.path.join(save_dir, f"metrics_{dataset.name}_{split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

def _parse_data_names(data_name_arg: str) -> List[str]:
    v = (data_name_arg or "").strip()
    if not v or v.lower() == "all":
        return ["Graph-SST2", "Graph-SST5", "Graph-Twitter"]
    if "," in v:
        return [x.strip() for x in v.split(",") if x.strip()]
    return [v]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="HF model name or local path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write metrics json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=16, help="max_new_tokens for generation")
    parser.add_argument("--data_name", type=str, default="all", help="Graph-SST2 / Graph-SST5 / Graph-Twitter / all / comma-separated")
    parser.add_argument("--split", type=str, default="test", help="train/val/valid/test")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to evaluate")
    parser.add_argument("--directed", action="store_true", help="Whether to treat graphs as directed")
    parser.add_argument(
        "--sys_prompt",
        action="store_true",
        help="whether to add system prompt",
        default=False,
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Path to the graph prompt template file (defaults to baselines/Graph-SST/prompt/graph_prompt.txt).",
    )
    parser.add_argument(
        '--layer_head_config_path',
        type=str,
        default=None,
        help="Path to a JSON file specifying which layer-heads to modify."
    )
    parser.add_argument(
        '--layers_to_modify',
        type=int,
        nargs='+',
        default=None,
        help="List of layer indices to modify when no layer_head_config_path is provided; "
             "set to None (do not pass this argument) to disable modification."
    )
    parser.add_argument(
        '--delta_ratio',
        type=float,
        default=0.4,
        help="The delta ratio for attention redistribution."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling evaluation subset"
    )
    args = parser.parse_args()

    data_root = "/home/lym/data1/Datasets/Graph-SST"
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for data_name in _parse_data_names(args.data_name):
        ds = SentiGraphDataset(root=data_root, name=data_name)
        metrics = evaluate(
            args,
            ds,
            model_path=args.model_path,
            split=args.split,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            limit=args.limit,
        )
        results.append(metrics)


if __name__ == "__main__":
    main()