import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Prompts
# -----------------------------
def load_prompts(prompt_txt_path: str) -> Dict[str, str]:
    """
    Parse lines like:
      BBBP: prompt = "...."
    Return dict: {"BBBP": "...", ...}
    """
    prompts: Dict[str, str] = {}
    with open(prompt_txt_path, "r", encoding="utf-8") as f:
        txt = f.read()

    pattern = re.compile(
        r"^\s*([A-Za-z0-9_-]+)\s*:\s*prompt\s*=\s*\"([\s\S]*?)\"\s*$",
        re.MULTILINE,
    )
    for m in pattern.finditer(txt):
        key = m.group(1).strip()
        prompt = m.group(2)
        prompt = prompt.replace("\\n", "\n").replace("\\t", "\t").replace("\\\"", "\"").strip()
        prompts[key] = prompt
    return prompts


# -----------------------------
# Labels
# -----------------------------
def yesno_from_label(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if s == "":
        return None
    if s.lower() in {"yes", "y", "true"}:
        return "Yes"
    if s.lower() in {"no", "n", "false"}:
        return "No"
    try:
        iv = int(float(s))
        if iv == 1:
            return "Yes"
        if iv == 0:
            return "No"
    except Exception:
        pass
    return None


def parse_yesno(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()
    if re.search(r"\byes\b", t):
        return "Yes"
    if re.search(r"\bno\b", t):
        return "No"
    if t[:1] == "y":
        return "Yes"
    if t[:1] == "n":
        return "No"
    return None


# -----------------------------
# Data specs (NO class)
# -----------------------------
TOX21_LABEL_COLS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

# Each spec is a plain dict to keep things simple.
# Keys: name, smiles_col, label_cols, extra_cols(optional)
TASKS: Dict[str, Dict[str, object]] = {
    "BACE": {"name": "BACE", "smiles_col": "mol", "label_cols": ["Class"]},
    "BBBP": {"name": "BBBP", "smiles_col": "smiles", "label_cols": ["p_np"]},
    "ClinTox": {
        "name": "ClinTox",
        "smiles_col": "smiles",
        "label_cols": ["CT_TOX"],
        "extra_cols": ["FDA_APPROVED"],
    },
    "HIV": {"name": "HIV", "smiles_col": "smiles", "label_cols": ["HIV_active"], "extra_cols": ["activity"]},
    "Tox21": {"name": "Tox21", "smiles_col": "smiles", "label_cols": TOX21_LABEL_COLS, "extra_cols": ["mol_id"]},
}

def resolve_csv_paths(data_dir: str, task: str) -> Tuple[Optional[str], str]:
    candidates = [
        (f"{task}.csv", f"{task}_test.csv"),
        (f"{task.lower()}.csv", f"{task.lower()}_test.csv"),
    ]

    for tr, te in candidates:
        trp = os.path.join(data_dir, tr)
        tep = os.path.join(data_dir, te)
        if os.path.exists(trp) and os.path.exists(tep):
            return trp, tep

    for _, te in candidates:
        tep = os.path.join(data_dir, te)
        if os.path.exists(tep):
            return None, tep

    for fn in os.listdir(data_dir):
        if fn.lower() == f"{task.lower()}_test.csv":
            return None, os.path.join(data_dir, fn)

    raise FileNotFoundError(f"Cannot find *_test.csv for task={task} under {data_dir}")

# -----------------------------
# Prompt formatting (graph)
# -----------------------------
def _format_example_graph(
    task: str,
    label_col: str,
    graph_text: str,
    label_yesno: str,
    extra: Optional[Dict[str, str]] = None,
) -> str:
    extra = extra or {}

    if task == "BACE":
        return f"Graph: {graph_text}\nBACE-1 Inhibit: {label_yesno}\n"
    if task == "BBBP":
        return f"Graph: {graph_text}\nBBBP Penetration: {label_yesno}\n"
    if task == "HIV":
        act = extra.get("activity", "")
        act_line = f"Activity test result: {act}\n" if act else ""
        return f"Graph: {graph_text}\n{act_line}HIV Inhibit: {label_yesno}\n"
    if task == "ClinTox":
        fda = yesno_from_label(extra.get("FDA_APPROVED"))
        fda_line = f"FDA Approved: {fda}\n" if fda in {"Yes", "No"} else ""
        return f"Graph: {graph_text}\n{fda_line}Clinically-trial-toxic: {label_yesno}\n"
    if task == "Tox21":
        return f"Graph: {graph_text}\nToxic: {label_yesno}\n"

    return f"Graph: {graph_text}\nLabel: {label_yesno}\n"


def _format_query_graph(
    task: str,
    graph_text: str,
    extra: Optional[Dict[str, str]] = None,
) -> str:
    extra = extra or {}

    if task == "BACE":
        return f"Molecular Graph: {graph_text}\nPlease answer with only Yes or No.\nBACE-1 Inhibit:"

    if task == "BBBP":
        return f"Molecular Graph: {graph_text}\nPlease answer with only Yes or No.\nPenetration:"
    
    if task == "HIV":
        act = extra.get("activity", "")
        # act_line = f"Activity test result: {act}\n" if act else ""
        act_line = ""
        return f"Molecular Graph: {graph_text}\n{act_line}Please answer with only Yes or No.\nHIV Inhibit:"

    if task == "ClinTox":
        fda = yesno_from_label(extra.get("FDA_APPROVED"))
        # fda_line = f"FDA Approved: {fda}\n" if fda in {"Yes", "No"} else ""
        fda_line = ""
        return f"Molecular Graph: {graph_text}\n{fda_line}Clinically-trial-toxic:"

    if task == "Tox21":
        return f"Molecular Graph: {graph_text}\nToxic:"

    raise ValueError(f"Unknown task: {task}")


def build_graph_prompt(
    base_prompt: str,
    task: str,
    label_col: str,
    graph_text: str,
    shot_examples: List[Tuple[str, str, Dict[str, str]]],  # (graph_text, Yes/No, extra)
    extra: Optional[Dict[str, str]] = None,
) -> str:
    # parts = [base_prompt.strip()]
    # for ex_graph, ex_label, ex_extra in shot_examples:
    #     parts.append(_format_example_graph(task, ex_graph, ex_label, ex_extra).strip())

    # parts.append(_format_query_graph(task, graph_text, extra).strip())
    # return "\n".join(parts).strip() + "\n"
    return base_prompt + _format_query_graph(task, graph_text, extra).strip()


def row_extra(spec: Dict[str, object], row: pd.Series) -> Dict[str, str]:
    extra: Dict[str, str] = {}
    for c in (spec.get("extra_cols") or []):
        v = row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        extra[str(c)] = str(v)
    return extra


def smiles_to_graph_text(smiles: str, weighted_edges: bool = False) -> Optional[str]:
    """
    Convert SMILES to the custom graph description string.
    weighted_edges=False: edges as (u, v)
    weighted_edges=True : edges as (u, v, bond_order)
    Return None if SMILES invalid / RDKit unavailable.
    """
    try:
        from rdkit import Chem
    except Exception:
        return None

    smi = (smiles or "").strip()
    if not smi:
        return None

    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None

    node_parts: List[str] = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        w = atom.GetAtomicNum()
        node_parts.append(f"[{idx}, {w}]")

    edge_parts: List[str] = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        if weighted_edges:
            bo = float(bond.GetBondTypeAsDouble())
            edge_parts.append(f"({u}, {v}, {bo})")
        else:
            edge_parts.append(f"({u}, {v})")

    n = mol.GetNumAtoms()
    if weighted_edges:
        return (
            f"Nodes are numbered from 0 to {n - 1} with atomic numbers:\n"
            f"{' '.join(node_parts)}\n"
            f"Edges (with bond order) are:\n"
            f"{' '.join(edge_parts)}"
        )
    return (
        f"Nodes are numbered from 0 to {n - 1} with atomic numbers:\n"
        f"{' '.join(node_parts)}\n"
        f"Edges are:\n"
        f"{' '.join(edge_parts)}"
    )

def sample_shots_graph_from_df(
    df: pd.DataFrame,
    spec: Dict[str, object],
    label_col: str,
    shot: int,
    seed: int,
    exclude_idx: Optional[int] = None,
    weighted_edges: bool = False,
) -> List[Tuple[str, str, Dict[str, str]]]:
    """
    Sample few-shot examples ONLY from the provided dataframe `df` (current eval split),
    and exclude the current query row (exclude_idx) to avoid leakage.

    Returns: List[(graph_text, "Yes"/"No", extra_dict)]
    """
    raise NotImplementedError("sample_shots_graph_from_df is disabled.")
    # if shot <= 0 or df is None or len(df) == 0:
    #     return []

    # smiles_col = str(spec["smiles_col"])
    # rng = random.Random(int(seed) + (int(exclude_idx) if exclude_idx is not None else 0))

    # # collect valid candidates (by label + smiles), excluding current idx
    # yes_candidates: List[int] = []
    # no_candidates: List[int] = []

    # for i in range(len(df)):
    #     if exclude_idx is not None and int(i) == int(exclude_idx):
    #         continue
    #     row = df.iloc[i]
    #     yn = yesno_from_label(row.get(label_col))
    #     smi = str(row.get(smiles_col, "")).strip()
    #     if yn is None or not smi:
    #         continue
    #     if yn == "Yes":
    #         yes_candidates.append(i)
    #     else:
    #         no_candidates.append(i)

    # if not yes_candidates and not no_candidates:
    #     return []

    # # choose indices with a simple class-balance heuristic when possible
    # chosen: List[int] = []
    # if shot >= 2 and yes_candidates and no_candidates:
    #     k_yes = shot // 2
    #     k_no = shot - k_yes
    #     rng.shuffle(yes_candidates)
    #     rng.shuffle(no_candidates)
    #     chosen = yes_candidates[:k_yes] + no_candidates[:k_no]
    #     rng.shuffle(chosen)
    # else:
    #     all_candidates = yes_candidates + no_candidates
    #     rng.shuffle(all_candidates)
    #     chosen = all_candidates[:shot]

    # # build outputs (skip invalid RDKit conversions)
    # out: List[Tuple[str, str, Dict[str, str]]] = []
    # for i in chosen:
    #     row = df.iloc[i]
    #     smi = str(row.get(smiles_col, "")).strip()
    #     yn = yesno_from_label(row.get(label_col))
    #     if yn is None or not smi:
    #         continue

    #     g = smiles_to_graph_text(smi, weighted_edges=weighted_edges)
    #     if not g:
    #         continue

    #     extra = row_extra(spec, row)
    #     out.append((g, yn, extra))

    # return out