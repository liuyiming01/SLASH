import argparse
import json
import os
from typing import Dict, Set, Tuple, Optional

import numpy as np

from .utils import select_layers_middle_peak_entropy2


def _load_scoring_selection(path: str) -> Dict[int, Set[int]]:
    with open(path, "r") as f:
        raw = json.load(f)
    out: Dict[int, Set[int]] = {}
    for k, v in raw.items():
        layer = int(k)
        if isinstance(v, list):
            heads = {int(h) for h in v}
        else:
            heads = {int(v)}
        if heads:
            out[layer] = heads
    return out


def _infer_num_heads_from_scoring(sel: Dict[int, Set[int]]) -> int:
    mx = -1
    for hs in sel.values():
        if hs:
            mx = max(mx, max(hs))
    return int(mx + 1) if mx >= 0 else 0


def _entropy_select_from_npz(
    npz_path: str,
    mode: str,
    num_heads: int,
    tmp_json_path: str,
) -> Tuple[Dict[int, Set[int]], dict]:
    data = np.load(npz_path)
    avg_entropy = data["avg_entropy"]
    count = data["count"]

    valid_mask = count > 0

    # select_layers_middle_peak_entropy expects json_path; write to a temp location then reload
    selected, info = select_layers_middle_peak_entropy2(
        scores=avg_entropy,
        valid_mask=valid_mask,
        score_mode=mode,
        num_heads=num_heads,
        json_path=tmp_json_path,
    )

    # convert to set-form for intersection
    sel_set: Dict[int, Set[int]] = {int(l): {int(h) for h in hs} for l, hs in selected.items()}
    return sel_set, info


def _intersect(ent_sel: Dict[int, Set[int]], scoring_sel: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for layer, hs_e in ent_sel.items():
        hs_s = scoring_sel.get(layer)
        if not hs_s:
            continue
        inter = hs_e & hs_s
        if inter:
            out[int(layer)] = set(sorted(inter))
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Refresh *_intersection.json from cached entropy .npz + auto scoring JSON (no model run)."
    )
    p.add_argument("--entropy_dir", type=str, required=True, help="Directory containing entropy3 outputs (entropy_results).")
    p.add_argument("--scoring_dir", type=str, required=True, help="Directory containing *_selected_layers_auto_scoring.json files.")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to write *_intersection.json files.")
    p.add_argument("--mode", type=str, choices=["per_head", "per_layer"], required=True)
    p.add_argument("--alpha", type=float, default=1.0, help="Alpha used in entropy3 directory naming (default: 1.0).")
    p.add_argument("--dry_run", action="store_true", help="Only print planned work; do not write outputs.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    alpha_str = str(args.alpha)
    suffix_dir = f"_entropy_{args.mode}_alpha{alpha_str}"

    # scan entropy task dirs
    task_dirs = []
    for name in os.listdir(args.entropy_dir):
        full = os.path.join(args.entropy_dir, name)
        if os.path.isdir(full) and name.endswith(suffix_dir) and ("_entropy_" in name):
            task_dirs.append(full)

    if not task_dirs:
        print(f"[refresh] No entropy dirs found under {args.entropy_dir} matching '*{suffix_dir}'.")
        return

    task_dirs.sort()

    for td in task_dirs:
        base = os.path.basename(td)
        # "<task>_entropy_<mode>_alphaX"
        task = base.split("_entropy_", 1)[0]

        npz_path = os.path.join(td, f"{task}_entropy_{args.mode}.npz")
        scoring_json = os.path.join(args.scoring_dir, f"{task}_selected_layers_auto_scoring.json")
        out_json = os.path.join(args.out_dir, f"{task}_{args.mode}_intersection.json")

        if not os.path.exists(npz_path):
            print(f"[refresh] Skip {task}: missing npz {npz_path}")
            continue
        if not os.path.exists(scoring_json):
            print(f"[refresh] Skip {task}: missing scoring json {scoring_json}")
            continue

        scoring_sel = _load_scoring_selection(scoring_json)
        num_heads = _infer_num_heads_from_scoring(scoring_sel)

        tmp_json = os.path.join(args.out_dir, f".tmp_entropy_{task}_{args.mode}.json")
        ent_sel, info = _entropy_select_from_npz(npz_path=npz_path, mode=args.mode, num_heads=num_heads, tmp_json_path=tmp_json)

        inter = _intersect(ent_sel, scoring_sel)
        out = {str(l): sorted(list(hs)) for l, hs in sorted(inter.items())}

        print(f"[refresh] {task}: entropy_mode={info.get('mode')} -> intersection layers={len(out)} -> {out_json}")

        if not args.dry_run:
            with open(out_json, "w") as f:
                json.dump(out, f, indent=2)
            # cleanup tmp
            try:
                os.remove(tmp_json)
            except OSError:
                pass


if __name__ == "__main__":
    main()