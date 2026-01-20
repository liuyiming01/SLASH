import os
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch

from .utils import count_edges_in_prompt, standardize_prompt_edges


# ============================================================
# MolecularNet helpers (reuse baselines/molecularNet/utils.py)
# ============================================================

_MOL_UTILS = None

def _import_molecularnet_utils():
    """
    Load baselines/molecularNet/utils.py via file path to avoid packaging issues.
    """
    global _MOL_UTILS
    if _MOL_UTILS is not None:
        return _MOL_UTILS

    import importlib.util

    util_path = "/home/lym/LLM-Research/Attention/Graph_Attention/GraphLens/baselines/molecularNet/utils.py"
    if not os.path.exists(util_path):
        raise FileNotFoundError(f"Cannot find MolecularNet utils.py at: {util_path}")

    spec = importlib.util.spec_from_file_location("graphlens_molecularnet_utils", util_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import MolecularNet utils from: {util_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MOL_UTILS = mod
    return _MOL_UTILS


def molecularnet_iter_task_label_ids(task_spec: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """
    UPDATED (match baselines/molecularNet/eval_mol.py):
      - Each MolecularNet task uses ONLY ONE label for evaluation.
      - For Tox21, use an aggregated label: "__TOX21_TOXIC_ANY__" (any assay == 1 => toxic).

    Returns list of (task_id, task, label_col).
    task_id naming rule (updated): task_id == task

    task_spec examples:
      - None / ""      -> all MolecularNet tasks
      - "BACE"         -> only BACE
      - "Tox21__NR-AR" -> (backward-compatible) treated as "Tox21" (label part ignored)
    """
    m = _import_molecularnet_utils()
    tasks = m.TASKS  # dict

    want_task = None
    if task_spec:
        # backward-compatible: ignore "__label" suffix
        want_task = task_spec.split("__", 1)[0]

    out: List[Tuple[str, str, str]] = []
    for tname, spec in tasks.items():
        if want_task and str(tname) != str(want_task):
            continue

        tname = str(tname)
        if tname == "Tox21":
            label_col = "__TOX21_TOXIC_ANY__"
        else:
            lcs = spec.get("label_cols", [])
            label_col = str(lcs[0]) if lcs else ""

        task_id = tname
        out.append((task_id, tname, label_col))
    return out


def _molecularnet_load_and_build_prompts(
    molecularnet: Dict[str, Any],
    input_column: str,
) -> pd.DataFrame:
    """
    Build a DataFrame with [input_column], plus helper cols:
      - __idx
      - __num_edges

    UPDATED to match baselines/molecularNet/eval_mol.py:
      - Each task uses a single evaluation label.
      - Tox21 label is aggregated: "__TOX21_TOXIC_ANY__"
    """
    m = _import_molecularnet_utils()

    root = molecularnet["root"]
    task = str(molecularnet["task"])
    split = str(molecularnet.get("split", "test"))
    prompt_path = molecularnet.get("prompt_path", None)
    shot = int(molecularnet.get("shot", 0))
    seed = int(molecularnet.get("seed", 42))
    weighted_edges = bool(molecularnet.get("weighted_edges", False))

    # base prompt (optional)
    base_prompt = ""
    if prompt_path:
        prompts = m.load_prompts(prompt_path)
        base_prompt = str(prompts.get(task, "")).strip()

    tr_csv, te_csv = m.resolve_csv_paths(root, task)
    test_df = pd.read_csv(te_csv)
    original_df = pd.read_csv(tr_csv) if tr_csv is not None else test_df
    df = test_df if split == "test" else original_df

    spec = m.TASKS[task]
    smiles_col = str(spec["smiles_col"])

    # --- choose evaluation label (single label per task) ---
    if task == "Tox21":
        # Build an "overall toxic" label: if ANY assay == 1 => 1; else if any known (0/1) and none 1 => 0; else NaN
        tox_any_col = "__TOX21_TOXIC_ANY__"
        tox_cols = [c for c in (spec.get("label_cols") or []) if c in df.columns]
        if tox_any_col not in df.columns:
            if tox_cols:
                mat = df[tox_cols].apply(pd.to_numeric, errors="coerce")
                has_any = mat.notna().any(axis=1)
                any1 = (mat == 1).any(axis=1)
                df[tox_any_col] = np.where(any1, 1, np.where(has_any, 0, np.nan))
            else:
                df[tox_any_col] = np.nan
        label_col = tox_any_col
    else:
        lcs = spec.get("label_cols", [])
        label_col = str(lcs[0]) if lcs else ""

    records = []
    for i in range(len(df)):
        row = df.iloc[i]
        smi = str(row.get(smiles_col, "")).strip()
        if not smi:
            continue

        g = m.smiles_to_graph_text(smi, weighted_edges=weighted_edges)
        if not g:
            continue

        extra = m.row_extra(spec, row)

        # few-shot is optional; sample from SAME df being evaluated (match eval_mol.py)
        if shot > 0 and label_col:
            shot_examples = m.sample_shots_graph_from_df(
                df=df,
                spec=spec,
                label_col=label_col,
                shot=shot,
                seed=seed,
                exclude_idx=int(i),
                weighted_edges=weighted_edges,
            )
        else:
            shot_examples = []

        prompt = m.build_graph_prompt(
            base_prompt=base_prompt,
            task=task,
            label_col=label_col,
            graph_text=g,
            shot_examples=shot_examples,
            extra=extra,
        )
        records.append({input_column: prompt, "__idx": int(i)})

    out_df = pd.DataFrame(records)
    if len(out_df) == 0:
        return out_df

    # robust edge counting
    out_df["__num_edges"] = out_df[input_column].map(
        lambda s: count_edges_in_prompt(standardize_prompt_edges(s))
    ).astype(np.int32)

    return out_df

def _molecularnet_get_eval_label_col(df: pd.DataFrame, spec: Dict[str, Any], task: str) -> str:
    """
    Determine the single evaluation label column for MolecularNet.
    For Tox21, create/use the aggregated label "__TOX21_TOXIC_ANY__".
    Mutates df for Tox21 (adds the aggregated label col if missing).
    """
    task = str(task)
    if task == "Tox21":
        tox_any_col = "__TOX21_TOXIC_ANY__"
        tox_cols = [c for c in (spec.get("label_cols") or []) if c in df.columns]
        if tox_any_col not in df.columns:
            if tox_cols:
                mat = df[tox_cols].apply(pd.to_numeric, errors="coerce")
                has_any = mat.notna().any(axis=1)
                any1 = (mat == 1).any(axis=1)
                df[tox_any_col] = np.where(any1, 1, np.where(has_any, 0, np.nan))
            else:
                df[tox_any_col] = np.nan
        return tox_any_col

    lcs = spec.get("label_cols", [])
    return str(lcs[0]) if lcs else ""


def _molecularnet_load_split_df(molecularnet: Dict[str, Any]) -> Tuple[Any, pd.DataFrame, Dict[str, Any], str, str, str]:
    """
    Returns: (m_utils_module, df, spec, task, split, label_col)
    """
    m = _import_molecularnet_utils()

    root = molecularnet["root"]
    task = str(molecularnet["task"])
    split = str(molecularnet.get("split", "test"))

    tr_csv, te_csv = m.resolve_csv_paths(root, task)
    test_df = pd.read_csv(te_csv)
    original_df = pd.read_csv(tr_csv) if tr_csv is not None else test_df
    df = test_df if split == "test" else original_df

    spec = m.TASKS[task]
    label_col = _molecularnet_get_eval_label_col(df=df, spec=spec, task=task)
    return m, df, spec, task, split, label_col


def molecularnet_build_prompts_from_indices(
    molecularnet: Dict[str, Any],
    input_column: str,
    indices: List[int],
) -> pd.DataFrame:
    """
    Build prompts ONLY for selected row indices (df.iloc[idx]).
    Returns DataFrame with columns: [input_column, "__idx", "__num_edges"].
    """
    m, df, spec, task, split, label_col = _molecularnet_load_split_df(molecularnet=molecularnet)

    prompt_path = molecularnet.get("prompt_path", None)
    shot = int(molecularnet.get("shot", 0))
    seed = int(molecularnet.get("seed", 42))
    weighted_edges = bool(molecularnet.get("weighted_edges", False))

    # base prompt (optional)
    base_prompt = ""
    if prompt_path:
        prompts = m.load_prompts(prompt_path)
        base_prompt = str(prompts.get(task, "")).strip()

    smiles_col = str(spec["smiles_col"])

    records: List[Dict[str, Any]] = []
    for i in indices:
        i = int(i)
        if i < 0 or i >= len(df):
            continue

        row = df.iloc[i]
        smi = str(row.get(smiles_col, "")).strip()
        if not smi:
            continue

        g = m.smiles_to_graph_text(smi, weighted_edges=weighted_edges)
        if not g:
            continue

        extra = m.row_extra(spec, row)

        # few-shot optional; sample from SAME df being evaluated (match eval_mol.py)
        if shot > 0 and label_col:
            shot_examples = m.sample_shots_graph_from_df(
                df=df,
                spec=spec,
                label_col=label_col,
                shot=shot,
                seed=seed,
                exclude_idx=int(i),
                weighted_edges=weighted_edges,
            )
        else:
            shot_examples = []

        prompt = m.build_graph_prompt(
            base_prompt=base_prompt,
            task=task,
            label_col=label_col,
            graph_text=g,
            shot_examples=shot_examples,
            extra=extra,
        )
        records.append({input_column: prompt, "__idx": int(i)})

    out_df = pd.DataFrame(records)
    if len(out_df) == 0:
        return out_df

    out_df["__num_edges"] = out_df[input_column].map(
        lambda s: count_edges_in_prompt(standardize_prompt_edges(str(s)))
    ).astype(np.int32)

    return out_df


def molecularnet_sample_prompts_by_edge_range(
    molecularnet: Dict[str, Any],
    input_column: str,
    sample_num: int,
    preferred_min_edges: int = 60,
    hard_max_edges: Optional[int] = None,
    seed: int = 42,
    require_label: bool = True,
) -> Tuple[pd.DataFrame, Tuple[int, int], Dict[str, Any]]:
    """
    One-stop helper for scoring/entropy:
      1) sample row indices by an auto-chosen edge range
      2) build prompts only for those indices
    """
    indices, (chosen_min, chosen_max), stats = molecularnet_sample_indices_by_edge_range(
        molecularnet=molecularnet,
        sample_num=int(sample_num),
        preferred_min_edges=int(preferred_min_edges),
        hard_max_edges=hard_max_edges,
        seed=int(seed),
        require_label=bool(require_label),
    )
    if not indices:
        return pd.DataFrame([]), (int(chosen_min), int(chosen_max)), stats

    df_prompts = molecularnet_build_prompts_from_indices(
        molecularnet=molecularnet,
        input_column=input_column,
        indices=indices,
    )

    # Keep size consistent (in rare cases, some indices may fail prompt build)
    if len(df_prompts) > int(sample_num):
        df_prompts = df_prompts.sample(n=int(sample_num), random_state=int(seed)).reset_index(drop=True)
    else:
        df_prompts = df_prompts.reset_index(drop=True)

    stats = {**stats, "built_prompts_n": int(len(df_prompts))}
    return df_prompts, (int(chosen_min), int(chosen_max)), stats


def molecularnet_sample_indices_by_edge_range(
    molecularnet: Dict[str, Any],
    sample_num: int,
    preferred_min_edges: int = 60,
    hard_max_edges: Optional[int] = None,
    seed: int = 42,
    require_label: bool = True,
) -> Tuple[List[int], Tuple[int, int], Dict[str, Any]]:
    """
    MolecularNet adaptation:
      - Load the evaluation dataframe for the requested split (test/sample).
      - For each row, build graph_text (smiles_to_graph_text) and count edges.
      - Auto choose [min_edges, max_edges] via choose_edge_range().
      - Filter indices within the chosen range and sample sample_num indices.

    Returns:
      (indices, (chosen_min, chosen_max), stats)

    Notes:
      - indices are row indices into the selected dataframe (same indexing as df.iloc[i]).
      - If require_label=True, only keep rows whose label is usable (yes/no not None).
        (For Tox21, it will use the aggregated label "__TOX21_TOXIC_ANY__".)
    """
    m = _import_molecularnet_utils()

    root = molecularnet["root"]
    task = str(molecularnet["task"])
    split = str(molecularnet.get("split", "test"))
    weighted_edges = bool(molecularnet.get("weighted_edges", False))

    tr_csv, te_csv = m.resolve_csv_paths(root, task)
    test_df = pd.read_csv(te_csv)
    original_df = pd.read_csv(tr_csv) if tr_csv is not None else test_df
    df = test_df if split == "test" else original_df

    spec = m.TASKS[task]
    smiles_col = str(spec["smiles_col"])

    # determine label_col (single label; Tox21 aggregated)
    label_col = _molecularnet_get_eval_label_col(df=df, spec=spec, task=task)

    valid_indices: List[int] = []
    edge_counts: List[int] = []

    for i in range(len(df)):
        row = df.iloc[i]
        smi = str(row.get(smiles_col, "")).strip()
        if not smi:
            continue

        if require_label and label_col:
            gt = m.yesno_from_label(row.get(label_col))
            if gt is None:
                continue

        g = m.smiles_to_graph_text(smi, weighted_edges=weighted_edges)
        if not g:
            continue

        g_std = standardize_prompt_edges(str(g))
        c = int(count_edges_in_prompt(g_std))
        if c <= 0:
            continue

        valid_indices.append(int(i))
        edge_counts.append(int(c))

    if len(valid_indices) == 0:
        return [], (0, 0), {"reason": "no_valid_rows_after_graph_and_label_filter"}

    # NEW: global edge stats on valid pool
    ec_arr = np.asarray(edge_counts, dtype=np.int32)
    global_stats = {
        "true_min_edges": int(ec_arr.min()),
        "true_max_edges": int(ec_arr.max()),
        "true_median_edges": float(np.median(ec_arr)),
        "true_valid_pool_n": int(ec_arr.size),
    }

    chosen_min, chosen_max, stats = choose_edge_range(
        edge_counts=ec_arr,
        sample_num=int(sample_num),
        preferred_min_edges=int(preferred_min_edges),
        hard_max_edges=hard_max_edges,
    )
    if chosen_min is None or chosen_max is None:
        return [], (0, 0), {"reason": "choose_edge_range_failed", **stats, **global_stats}

    in_range = [
        idx for idx, ec in zip(valid_indices, edge_counts)
        if int(ec) >= int(chosen_min) and int(ec) <= int(chosen_max)
    ]

    pool = in_range if len(in_range) >= 1 else valid_indices
    if len(pool) < int(sample_num):
        chosen = pool
        stats = {**stats, "warning": "pool_smaller_than_sample_num", "pool_n": int(len(pool))}
    else:
        rng = np.random.RandomState(int(seed))
        chosen = rng.choice(pool, size=int(sample_num), replace=False).tolist()

    stats = {
        **stats,
        **global_stats,
        "task": task,
        "split": split,
        "weighted_edges": bool(weighted_edges),
        "label_col": label_col,
        "valid_n": int(len(valid_indices)),
        "in_range_n": int(len(in_range)),
        "chosen_n": int(len(chosen)),
        "chosen_min_edges": int(chosen_min),
        "chosen_max_edges": int(chosen_max),
    }
    return [int(x) for x in chosen], (int(chosen_min), int(chosen_max)), stats

# ============================================================
# Graph-SST helpers (MUST match baselines/Graph-SST/evaluate.py)
# ============================================================

def _graphsst_label2text(data_name: str) -> Dict[int, str]:
    if "Twitter" in data_name:
        return {0: "negative", 1: "neutral", 2: "positive"}
    if data_name == "Graph-SST2":
        return {0: "negative", 1: "positive"}
    return {0: "very negative", 1: "negative", 2: "neutral", 3: "positive", 4: "very positive"}


def _graphsst_split_indices(dataset, split: str) -> List[int]:
    """
    Same logic as evaluate.py:get_split_indices_from_supplement
    """
    split = (split or "test").lower()
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


def _graphsst_get_node_texts(data, dataset=None, idx: Optional[int] = None) -> List[str]:
    """
    Same logic as evaluate.py:get_node_texts
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

    if len(toks) < num_nodes:
        toks = toks + [f"node_{i}" for i in range(len(toks), num_nodes)]
    elif len(toks) > num_nodes:
        toks = toks[:num_nodes]
    return toks


def _graphsst_linearize_graph(
    data,
    node_texts: List[str],
    directed: bool = False,
) -> Tuple[str, str]:
    """
    Same logic as evaluate.py:linearize_graph
    Returns (nodes_part, edges_part).
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
        # IMPORTANT: match evaluate.py formatting with comma+space
        edges_part = " ".join([f"({u}, {v})" for u, v in edges])

    if not edges_part:
        edges_part = "(empty)"

    return nodes_part, edges_part

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

def _graphsst_build_prompt(
    nodes_part: str,
    edges_part: str,
    label2text: Dict[int, str],
    num_nodes: int,
    prompt_templates: Dict[str, str],
    directed: bool = False,
) -> str:
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

# ============================================================
# Unified loader
# ============================================================

def load_and_filter_samples(
    data_path: Optional[str],
    input_column: str,
    min_edges: int,
    max_edges: int,
    sample_num: int,
    graphsst: Optional[Dict[str, Any]] = None,
    molecularnet: Optional[Dict[str, Any]] = None,
):
    """
    Returns a DataFrame with at least: [input_column], plus helper cols:
      - __idx (Graph-SST / MolecularNet)
      - __num_edges

    Graph-SST prompt construction MUST match baselines/Graph-SST/evaluate.py.
    MolecularNet prompt construction reuses baselines/molecularNet/utils.py.
    """
    # -------- MolecularNet family --------
    if molecularnet is not None:
        df = _molecularnet_load_and_build_prompts(molecularnet=molecularnet, input_column=input_column)
        if df is None or len(df) == 0:
            print(f"Warning: no MolecularNet samples for task={molecularnet.get('task')} label={molecularnet.get('label_col')}")
            return None

        df_filt = df[(df["__num_edges"] >= int(min_edges)) & (df["__num_edges"] <= int(max_edges))]
        if len(df_filt) == 0:
            print(f"Warning: no MolecularNet samples with {min_edges}-{max_edges} edges.")
            return None

        n_total = len(df_filt)
        n_req = int(sample_num)
        if n_req >= n_total:
            return df_filt.reset_index(drop=True)
        return df_filt.sample(n=n_req, random_state=42).reset_index(drop=True)

    # -------- Graph-SST family --------
    if graphsst is not None:
        try:
            from dig.xgraph.dataset import SentiGraphDataset
        except Exception as e:
            raise ImportError(
                "Graph-SST requires DIG (`dig.xgraph`). Please ensure it is installed. "
                f"Original error: {e}"
            )

        root = graphsst["root"]
        name = graphsst["name"]
        split = graphsst.get("split", "test")
        directed = bool(graphsst.get("directed", False))

        ds = SentiGraphDataset(root=root, name=name)
        indices = _graphsst_split_indices(ds, split=split)

        label2text = _graphsst_label2text(name)
        prompt_templates = load_graph_prompt_templates(graphsst["prompt_path"])
        records = []
        for idx in indices:
            data = ds[int(idx)]
            node_texts = _graphsst_get_node_texts(data, dataset=ds, idx=int(idx))
            nodes_part, edges_part = _graphsst_linearize_graph(data, node_texts, directed=directed)
            prompt = _graphsst_build_prompt(
                nodes_part=nodes_part,
                edges_part=edges_part,
                label2text=label2text,
                num_nodes=len(node_texts),
                prompt_templates=prompt_templates,
                directed=directed,
            )
            records.append({input_column: prompt, "__idx": int(idx)})

        df = pd.DataFrame(records)

        df["__num_edges"] = df[input_column].map(lambda s: count_edges_in_prompt(standardize_prompt_edges(s))).astype(np.int32)

        df_filt = df[(df["__num_edges"] >= int(min_edges)) & (df["__num_edges"] <= int(max_edges))]
        if len(df_filt) == 0:
            print(f"Warning: no samples with {min_edges}-{max_edges} edges in Graph-SST:{name}/{split}.")
            return None

        n_total = len(df_filt)
        n_req = int(sample_num)

        if n_req >= n_total:
            return df_filt.reset_index(drop=True)

        return df_filt.sample(n=n_req, random_state=42).reset_index(drop=True)

    # -------- GraphWiz / jsonl --------
    if not data_path or not os.path.exists(data_path):
        print(f"Warning: data file not found: {data_path}")
        return None

    ext = os.path.splitext(data_path)[1].lower()
    if ext in [".json", ".jsonl"]:
        df = pd.read_json(data_path, lines=True)
    else:
        raise ValueError(f"Unsupported data file extension: {ext}, path={data_path}")

    if input_column not in df.columns:
        raise ValueError(f"Input column '{input_column}' not found in data file: {data_path}")

    df["__num_edges"] = df[input_column].map(count_edges_in_prompt).astype(np.int32)
    df_filt = df[(df["__num_edges"] >= int(min_edges)) & (df["__num_edges"] <= int(max_edges))]
    print("Filtered samples count:", len(df_filt))
    if len(df_filt) == 0:
        print(f"Warning: no samples with {min_edges}-{max_edges} edges in {data_path}.")
        return None

    n_total = len(df_filt)
    n_req = int(sample_num)
    if n_req >= n_total:
        return df_filt.reset_index(drop=True)

    return df_filt.sample(n=n_req, random_state=42).reset_index(drop=True)


def choose_edge_range(
    edge_counts: np.ndarray,
    sample_num: int,
    preferred_min_edges: int = 50,
    hard_max_edges: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int], Dict[str, Any]]:
    """
    Unified edge-range chooser for scoring/plot/entropy.

    Strategy:
      1) Optionally cap edges by hard_max_edges (to control compute).
      2) Prefer samples with edges >= preferred_min_edges if enough to cover sample_num:
         choose a compact window centered around the median.
      3) Otherwise fallback (Graph-SST common): include the TOP edge-count samples until >= sample_num
         (i.e., pick threshold as the k-th largest edge count).
    """
    ec = np.asarray(edge_counts, dtype=np.int32)
    if ec.size == 0:
        return None, None, {"reason": "empty_edge_counts"}

    # apply hard cap
    if hard_max_edges is not None and int(hard_max_edges) >= 0:
        cap = int(hard_max_edges)
        ec_cap = ec[ec <= cap]
    else:
        cap = None
        ec_cap = ec

    if ec_cap.size == 0:
        return None, None, {"reason": "empty_after_hard_cap", "hard_max_edges": cap}

    k = int(max(1, sample_num))
    v_all = np.sort(ec_cap.astype(np.int32))
    n_all = int(v_all.size)

    # helper: median-centered compact window (used only when preferred_min is feasible)
    def _pick_median_window(vals: np.ndarray, k_: int) -> Tuple[int, int, Dict[str, Any]]:
        v = np.sort(vals.astype(np.int32))
        n = int(v.size)
        if k_ >= n:
            return int(v[0]), int(v[-1]), {"window_n": n, "k": k_, "mode": "all"}
        mid = n // 2
        left = max(0, mid - (k_ // 2))
        right = min(n, left + k_)
        left = max(0, right - k_)
        return int(v[left]), int(v[right - 1]), {
            "window_n": n,
            "k": k_,
            "mode": "median_center_window",
            "median": float(np.median(v)),
        }

    pref = int(max(0, preferred_min_edges))
    ec_pref = ec_cap[ec_cap >= pref]
    used_pref = (ec_pref.size >= k)

    if used_pref:
        chosen_min, chosen_max, detail = _pick_median_window(ec_pref, k)
    else:
        # Fallback: take "top edges" until covering k samples
        if k >= n_all:
            chosen_min, chosen_max = int(v_all[0]), int(v_all[-1])
            detail = {"window_n": n_all, "k": k, "mode": "all_fallback_top_edges"}
        else:
            # k-th largest is at index n_all - k in ascending array
            chosen_min = int(v_all[n_all - k])
            chosen_max = int(v_all[-1])
            detail = {
                "window_n": n_all,
                "k": k,
                "mode": "top_edges_threshold_fallback",
                "kth_largest_threshold": chosen_min,
                "max_edges": chosen_max,
            }

    stats = {
        "preferred_min_edges": pref,
        "hard_max_edges": cap,
        "used_preferred_min": bool(used_pref),
        "available_n": int(ec.size),
        "available_after_cap_n": int(ec_cap.size),
        "eligible_after_pref_n": int(ec_pref.size),
        **detail,
    }
    return chosen_min, chosen_max, stats
