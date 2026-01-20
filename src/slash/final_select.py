import argparse
import json
from typing import Dict, List, Set


def _load_selection(path: str) -> Dict[int, Set[int]]:
    with open(path, "r") as f:
        raw = json.load(f)

    sel: Dict[int, Set[int]] = {}
    for k, v in raw.items():
        layer = int(k)
        if isinstance(v, list):
            heads = {int(h) for h in v}
        else:
            # 兼容万一写成单个数字的情况
            heads = {int(v)}
        if heads:
            sel[layer] = heads
    return sel


def _intersect_selections(a: Dict[int, Set[int]],
                          b: Dict[int, Set[int]]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for layer, heads_a in a.items():
        heads_b = b.get(layer)
        if not heads_b:
            continue
        inter = heads_a & heads_b
        if not inter:
            continue
        out[layer] = sorted(inter)
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Intersect layer/head selections from scoring.py and entropy.py"
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["per_head", "per_layer"],
        required=True,
        help="Selection mode; affects upstream file naming, "
             "but intersection is always done on (layer, head) level.",
    )
    p.add_argument(
        "--json_scoring",
        type=str,
        required=True,
        help="Path to JSON produced by scoring.py (selected layers/heads).",
    )
    p.add_argument(
        "--json_entropy",
        type=str,
        required=True,
        help="Path to JSON produced by entropy.py (selected layers/heads).",
    )
    p.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save the intersection JSON.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    sel_scoring = _load_selection(args.json_scoring)
    sel_entropy = _load_selection(args.json_entropy)

    inter = _intersect_selections(sel_scoring, sel_entropy)

    # JSON 要求 key 为字符串
    out = {str(layer): heads for layer, heads in sorted(inter.items())}

    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved intersection to {args.output_json}")


if __name__ == "__main__":
    main()