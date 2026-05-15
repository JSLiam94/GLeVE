from __future__ import annotations

import argparse
import json
import os
from typing import List

import pandas as pd

from utils_nii import load_nii


LESION_IDS = [3, 4, 6, 8]


def graph_has_lesion(graph_path: str) -> bool:
    if not os.path.exists(graph_path):
        return False
    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except Exception:
        return False

    nodes = graph.get("nodes", []) or []
    return any(str(node.get("type", "")).lower() == "lesion" for node in nodes)


def mask_has_lesion_voxels(mask_path: str) -> bool:
    if not os.path.exists(mask_path):
        return False
    try:
        mask = load_nii(mask_path)
    except Exception:
        return False
    for lesion_id in LESION_IDS:
        if (mask == lesion_id).any():
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default="data/train.csv")
    ap.add_argument("--graph_root", type=str, default="data/gleve_offline/graphs")
    ap.add_argument("--mask_root", type=str, default="data/altas/combined_labels")
    ap.add_argument("--output_csv", type=str, default="data/IID_train_has_lesion.csv")
    ap.add_argument("--removed_ids_txt", type=str, default="data/IID_train_no_lesion.txt", help="Optional path to save removed case ids")
    ap.add_argument("--id_col", type=str, default="BDMAP ID")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.id_col not in df.columns:
        raise ValueError(f"Missing id column: {args.id_col}")

    keep_rows: List[bool] = []
    removed_ids: List[str] = []
    removed_no_graph_lesion = 0
    removed_zero_gt_vox = 0

    for raw_case_id in df[args.id_col].tolist():
        case_id = str(raw_case_id).strip()
        graph_path = os.path.join(args.graph_root, f"{case_id}.graph.json")
        mask_path = os.path.join(args.mask_root, case_id, "combined_labels.nii.gz")

        keep = graph_has_lesion(graph_path)
        if not keep:
            removed_no_graph_lesion += 1
        else:
            keep = mask_has_lesion_voxels(mask_path)
            if not keep:
                removed_zero_gt_vox += 1
        keep_rows.append(keep)
        if not keep:
            removed_ids.append(case_id)

    out_df = df.loc[keep_rows].reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    if args.removed_ids_txt:
        os.makedirs(os.path.dirname(os.path.abspath(args.removed_ids_txt)), exist_ok=True)
        with open(args.removed_ids_txt, "w", encoding="utf-8") as f:
            for case_id in removed_ids:
                f.write(case_id + "\n")

    print(f"Input rows: {len(df)}")
    print(f"Kept rows: {len(out_df)}")
    print(f"Removed rows: {len(removed_ids)}")
    print(f"Removed for missing graph lesion: {removed_no_graph_lesion}")
    print(f"Removed for zero GT lesion voxels: {removed_zero_gt_vox}")


if __name__ == "__main__":
    main()
