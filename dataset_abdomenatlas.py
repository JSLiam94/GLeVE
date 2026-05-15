# dataset_abdomenatlas.py
from __future__ import annotations
import os
import random
import json
import signal
import sys
from collections import deque
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils_nii import HU_MAX, HU_MIN, load_nii, load_nii_with_spacing, hu_window_and_normalize

try:
    from scipy import ndimage as scipy_ndimage
except ImportError:
    scipy_ndimage = None

LABELS_N = [
    'background',                  # 0
    'kidney_right',                # 1
    'kidney_left',                 # 2
    'kidney_lesion_kidney_right',  # 3
    'kidney_lesion_kidney_left',   # 4
    'pancreas',                    # 5
    'pancreatic_lesion',           # 6
    'liver',                       # 7
    'liver_lesion',                # 8
]

ORGAN_IDS = {
    "kidney": [1, 2],   # merge left/right
    "kidney_right": [1],
    "kidney_left": [2],
    "pancreas": [5],
    "liver": [7],
}
LESION_IDS = {
    "kidney": [3, 4],
    "kidney_right": [3],
    "kidney_left": [4],
    "pancreas": [6],
    "liver": [8],
}

def _make_binary(mask: np.ndarray, ids: List[int]) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in ids:
        out |= (mask == i).astype(np.uint8)
    return out


def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    mask_bin = mask.astype(bool)
    if not mask_bin.any():
        return []

    if scipy_ndimage is not None:
        structure = scipy_ndimage.generate_binary_structure(rank=3, connectivity=1)
        labeled, num = scipy_ndimage.label(mask_bin, structure=structure)
        if num == 0:
            return []
        counts = np.bincount(labeled.ravel())[1:]
        if counts.size == 0:
            return []
        labels = np.argsort(counts)[::-1] + 1
        return [(labeled == lab).astype(np.uint8) for lab in labels]

    H, W, D = mask_bin.shape
    visited = np.zeros_like(mask_bin, dtype=bool)
    neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    comps: List[List[Tuple[int, int, int]]] = []

    fg = np.argwhere(mask_bin)
    for h, w, d in fg:
        if visited[h, w, d]:
            continue
        q = deque([(int(h), int(w), int(d))])
        visited[h, w, d] = True
        coords: List[Tuple[int, int, int]] = []
        while q:
            ch, cw, cd = q.popleft()
            coords.append((ch, cw, cd))
            for dh, dw, dd in neighbors:
                nh, nw, nd = ch + dh, cw + dw, cd + dd
                if nh < 0 or nh >= H or nw < 0 or nw >= W or nd < 0 or nd >= D:
                    continue
                if visited[nh, nw, nd] or not mask_bin[nh, nw, nd]:
                    continue
                visited[nh, nw, nd] = True
                q.append((nh, nw, nd))
        comps.append(coords)

    comps.sort(key=len, reverse=True)
    out: List[np.ndarray] = []
    for coords in comps:
        comp = np.zeros_like(mask, dtype=np.uint8)
        hh, ww, dd = zip(*coords)
        comp[list(hh), list(ww), list(dd)] = 1
        out.append(comp)
    return out


def _lesion_support_key(node: Dict[str, Any]) -> str:
    organ = str(node.get("organ", "unknown"))
    payload = node.get("payload") or {}
    sub_location = str(payload.get("sub_location") or "").lower()
    if organ == "kidney":
        if "right" in sub_location:
            return "kidney_right"
        if "left" in sub_location:
            return "kidney_left"
    return organ


def _node_volume(node: Dict[str, Any]) -> float:
    payload = node.get("payload") or {}
    volume = payload.get("volume_cc", None)
    if volume is None:
        return float("-inf")
    try:
        value = float(volume)
    except Exception:
        return float("-inf")
    if not np.isfinite(value):
        return float("-inf")
    return value


def _load_text_embedding_npz(path: str) -> np.ndarray:
    # The offline builder writes `text_emb` in the same order as `graph["nodes"]`
    # and also stores `node_ids` as an auxiliary object array. Some archives were
    # produced under numpy>=2 and older numpy versions can hard-crash when trying
    # to unpickle `node_ids`. Only load the dense embedding matrix here and use
    # graph order as the canonical row order.
    sys.modules.setdefault("numpy._core", np.core)
    emb = np.load(path, allow_pickle=True)
    try:
        text_emb = emb["text_emb"].astype(np.float32)
    finally:
        emb.close()
    return text_emb


@lru_cache(maxsize=1024)
def _load_graph_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1024)
def _load_text_embedding_cached(path: str) -> np.ndarray:
    return _load_text_embedding_npz(path)


class _SampleTimeoutError(TimeoutError):
    pass


class _SampleTimeout:
    def __init__(self, seconds: float):
        self.seconds = max(float(seconds), 0.0)
        self.enabled = False
        self.prev_handler = None
        self.prev_timer: Optional[Tuple[float, float]] = None

    def _handle_timeout(self, signum, frame) -> None:
        raise _SampleTimeoutError(f"sample loading exceeded {self.seconds:.1f}s")

    def __enter__(self):
        if (
            self.seconds <= 0.0
            or os.name == "nt"
            or not hasattr(signal, "SIGALRM")
            or not hasattr(signal, "setitimer")
        ):
            return self

        self.enabled = True
        self.prev_handler = signal.getsignal(signal.SIGALRM)
        try:
            self.prev_timer = signal.getitimer(signal.ITIMER_REAL)
        except Exception:
            self.prev_timer = None
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False

        signal.setitimer(signal.ITIMER_REAL, 0.0)
        if self.prev_handler is not None:
            signal.signal(signal.SIGALRM, self.prev_handler)
        if self.prev_timer is not None and (self.prev_timer[0] > 0.0 or self.prev_timer[1] > 0.0):
            signal.setitimer(signal.ITIMER_REAL, self.prev_timer[0], self.prev_timer[1])
        return False


def _filter_graph_to_crop(
    graph: Dict[str, Any],
    text_emb: np.ndarray,
    mask_crop: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray]:
    nodes = graph["nodes"]

    kidney_right_present = bool(_make_binary(mask_crop, LESION_IDS["kidney_right"]).any())
    kidney_left_present = bool(_make_binary(mask_crop, LESION_IDS["kidney_left"]).any())
    kidney_union_components = _connected_components(_make_binary(mask_crop, LESION_IDS["kidney"]))
    support_slots = {
        "kidney_right": 1 if kidney_right_present else 0,
        "kidney_left": 1 if kidney_left_present else 0,
        "kidney": len(kidney_union_components),
        "pancreas": len(_connected_components(_make_binary(mask_crop, LESION_IDS["pancreas"]))),
        "liver": len(_connected_components(_make_binary(mask_crop, LESION_IDS["liver"]))),
    }

    lesion_nodes = [n for n in nodes if n.get("type") == "lesion"]
    keep_lesion_ids: set[str] = set()
    selected_side_specific = 0

    for support_key in ("kidney_right", "kidney_left"):
        candidates = sorted(
            [n for n in lesion_nodes if _lesion_support_key(n) == support_key],
            key=_node_volume,
            reverse=True,
        )
        keep_n = min(len(candidates), support_slots[support_key])
        for node in candidates[:keep_n]:
            keep_lesion_ids.add(str(node.get("lesion_id") or node["id"]))
        selected_side_specific += keep_n

    support_slots["kidney"] = max(support_slots["kidney"] - selected_side_specific, 0)
    for support_key in ("kidney", "pancreas", "liver"):
        candidates = sorted(
            [
                n for n in lesion_nodes
                if _lesion_support_key(n) == support_key
                and str(n.get("lesion_id") or n["id"]) not in keep_lesion_ids
            ],
            key=_node_volume,
            reverse=True,
        )
        keep_n = min(len(candidates), support_slots[support_key])
        for node in candidates[:keep_n]:
            keep_lesion_ids.add(str(node.get("lesion_id") or node["id"]))

    keep_supports = {
        _lesion_support_key(n)
        for n in lesion_nodes
        if str(n.get("lesion_id") or n["id"]) in keep_lesion_ids
    }
    keep_organs = {support.split("_")[0] for support in keep_supports}
    keep_node_ids: set[str] = set()
    keep_indices: List[int] = []
    for idx, node in enumerate(nodes):
        node_id = str(node["id"])
        node_type = str(node.get("type"))
        lesion_id = str(node.get("lesion_id") or node_id)
        if node_type == "lesion" and lesion_id in keep_lesion_ids:
            keep_node_ids.add(node_id)
            keep_indices.append(idx)
        elif node_type == "attr" and lesion_id in keep_lesion_ids:
            keep_node_ids.add(node_id)
            keep_indices.append(idx)
        elif node_type == "organ" and str(node.get("organ")) in keep_organs:
            keep_node_ids.add(node_id)
            keep_indices.append(idx)

    filtered_nodes = [nodes[i] for i in keep_indices]
    filtered_edges = [
        e for e in graph.get("edges", [])
        if str(e.get("source")) in keep_node_ids and str(e.get("target")) in keep_node_ids
    ]
    filtered_graph = {
        "case_id": graph.get("case_id"),
        "nodes": filtered_nodes,
        "edges": filtered_edges,
        "meta": graph.get("meta", {}),
    }
    filtered_emb = text_emb[keep_indices] if keep_indices else text_emb[:0]
    if any(n.get("type") == "lesion" for n in filtered_nodes):
        return filtered_graph, filtered_emb
    # Filtering is intentionally conservative; if it removes every lesion node
    # while the crop still contains lesion voxels, fall back to the original
    # graph instead of silently dropping the sample during training.
    if _make_binary(mask_crop, LESION_IDS["kidney"] + LESION_IDS["pancreas"] + LESION_IDS["liver"]).any():
        return graph, text_emb
    return filtered_graph, filtered_emb


def _limit_graph_lesions(
    graph: Dict[str, Any],
    text_emb: np.ndarray,
    max_lesions: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    max_lesions = int(max_lesions)
    if max_lesions <= 0:
        return graph, text_emb

    nodes = graph["nodes"]
    lesion_entries = [
        (idx, node)
        for idx, node in enumerate(nodes)
        if str(node.get("type")) == "lesion"
    ]
    if len(lesion_entries) <= max_lesions:
        return graph, text_emb

    lesion_entries = sorted(
        lesion_entries,
        key=lambda item: (_node_volume(item[1]), -item[0]),
        reverse=True,
    )
    keep_lesion_ids = {
        str(node.get("lesion_id") or node["id"])
        for _, node in lesion_entries[:max_lesions]
    }
    keep_organs = {
        str(node.get("organ"))
        for _, node in lesion_entries[:max_lesions]
        if node.get("organ") is not None
    }

    keep_node_ids: set[str] = set()
    keep_indices: List[int] = []
    for idx, node in enumerate(nodes):
        node_id = str(node["id"])
        node_type = str(node.get("type"))
        lesion_id = str(node.get("lesion_id") or node_id)
        if node_type == "lesion" and lesion_id in keep_lesion_ids:
            keep_node_ids.add(node_id)
            keep_indices.append(idx)
        elif node_type == "attr" and lesion_id in keep_lesion_ids:
            keep_node_ids.add(node_id)
            keep_indices.append(idx)
        elif node_type == "organ" and str(node.get("organ")) in keep_organs:
            keep_node_ids.add(node_id)
            keep_indices.append(idx)

    filtered_nodes = [nodes[i] for i in keep_indices]
    filtered_edges = [
        e for e in graph.get("edges", [])
        if str(e.get("source")) in keep_node_ids and str(e.get("target")) in keep_node_ids
    ]
    filtered_meta = dict(graph.get("meta", {}))
    filtered_meta["num_lesions_before_limit"] = int(len(lesion_entries))
    filtered_meta["num_lesions_after_limit"] = int(sum(1 for n in filtered_nodes if str(n.get("type")) == "lesion"))
    filtered_meta["max_lesions_per_case"] = int(max_lesions)
    filtered_graph = {
        "case_id": graph.get("case_id"),
        "nodes": filtered_nodes,
        "edges": filtered_edges,
        "meta": filtered_meta,
    }
    filtered_emb = text_emb[keep_indices] if keep_indices else text_emb[:0]
    return filtered_graph, filtered_emb


def estimate_physical_stats(
    train_csv: str,
    image_root: str,
    mask_root: str,
    graph_root: Optional[str] = None,
    max_cases: int = 64,
    seed: int = 0,
) -> Dict[str, Any]:
    df = pd.read_csv(train_csv)
    if "BDMAP ID" not in df.columns:
        raise ValueError("train.csv must contain column 'BDMAP ID'")
    ids = [str(x).strip() for x in df["BDMAP ID"].tolist() if str(x).strip()]
    if not ids:
        return {
            "num_cases_scanned": 0,
            "num_cases_valid": 0,
            "voxel_cc_median": 1.0,
            "spacing_mm_median": [1.0, 1.0, 1.0],
            "lesion_volume_cc_median": 1.0,
            "report_volume_cc_median": 1.0,
            "report_hu_scale": max(HU_MAX - HU_MIN, 1.0),
            "hu_min": HU_MIN,
            "hu_max": HU_MAX,
        }

    rnd = random.Random(seed)
    ids_shuf = ids[:]
    rnd.shuffle(ids_shuf)
    chosen_ids = ids_shuf[: max(1, min(int(max_cases), len(ids_shuf)))]

    voxel_cc_values: List[float] = []
    spacing_values: List[Tuple[float, float, float]] = []
    lesion_volume_cc_values: List[float] = []
    report_volume_cc_values: List[float] = []
    report_hu_values: List[float] = []
    lesion_label_ids = sorted({lab for ids_one in LESION_IDS.values() for lab in ids_one})

    for case_id in chosen_ids:
        ct_path = os.path.join(image_root, case_id, "ct.nii.gz")
        mk_path = os.path.join(mask_root, case_id, "combined_labels.nii.gz")
        if not (os.path.exists(ct_path) and os.path.exists(mk_path)):
            continue
        try:
            _, spacing = load_nii_with_spacing(ct_path)
            mask = np.asarray(load_nii(mk_path, dtype=None), dtype=np.int16)
        except Exception:
            continue

        spacing_tuple = tuple(float(v) for v in spacing)
        voxel_cc = float(np.prod(np.asarray(spacing_tuple, dtype=np.float64)) / 1000.0)
        if not np.isfinite(voxel_cc) or voxel_cc <= 0.0:
            continue

        spacing_values.append(spacing_tuple)
        voxel_cc_values.append(voxel_cc)
        for lesion_id in lesion_label_ids:
            vox = int(np.count_nonzero(mask == lesion_id))
            if vox > 0:
                lesion_volume_cc_values.append(float(vox) * voxel_cc)

        if graph_root:
            graph_path = os.path.join(graph_root, f"{case_id}.graph.json")
            if os.path.exists(graph_path):
                try:
                    graph = _load_graph_json(graph_path)
                except Exception:
                    graph = None
                if graph is not None:
                    for node in graph.get("nodes", []):
                        if node.get("type") != "lesion":
                            continue
                        payload = node.get("payload") or {}
                        report_v = payload.get("volume_cc", None)
                        report_mu = (payload.get("hu") or {}).get("mean", None)
                        try:
                            report_v = float(report_v)
                            if np.isfinite(report_v) and report_v > 0.0:
                                report_volume_cc_values.append(report_v)
                        except Exception:
                            pass
                        try:
                            report_mu = float(report_mu)
                            if np.isfinite(report_mu):
                                report_hu_values.append(float(np.clip(report_mu, HU_MIN, HU_MAX)))
                        except Exception:
                            pass

    def _median_or(values: List[float], default: float) -> float:
        if not values:
            return float(default)
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float(default)
        return float(np.median(arr))

    def _percentile_gap(values: List[float], q_lo: float, q_hi: float, default: float) -> float:
        if not values:
            return float(default)
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float(default)
        return float(max(np.percentile(arr, q_hi) - np.percentile(arr, q_lo), default))

    spacing_median = [1.0, 1.0, 1.0]
    if spacing_values:
        spacing_arr = np.asarray(spacing_values, dtype=np.float64)
        spacing_median = [float(v) for v in np.median(spacing_arr, axis=0).tolist()]

    return {
        "num_cases_scanned": int(len(chosen_ids)),
        "num_cases_valid": int(len(voxel_cc_values)),
        "voxel_cc_median": _median_or(voxel_cc_values, 1e-3),
        "spacing_mm_median": spacing_median,
        "lesion_volume_cc_median": _median_or(lesion_volume_cc_values, 3.0),
        "report_volume_cc_median": _median_or(report_volume_cc_values, max(_median_or(lesion_volume_cc_values, 3.0), 1.0)),
        "report_hu_scale": _percentile_gap(report_hu_values, 10.0, 90.0, 80.0),
        "hu_min": float(HU_MIN),
        "hu_max": float(HU_MAX),
    }

class AbdomenAtlasGLeVEDataset(Dataset):
    def __init__(
        self,
        train_csv: str,
        image_root: str,
        mask_root: str,
        graph_root: str,
        emb_root: str,
        exclude_ids_path: Optional[str] = None,
        mask_ratio: float = 1.0,
        seed: int = 0,
        patch_size: Tuple[int,int,int] = (160,160,160),
        lesion_crop_prob: float = 1.0,
        deterministic_crop: bool = False,
        sample_timeout_sec: float = 0.0,
        max_lesions_per_case: int = 15,
        supervised_gt_crop_epochs: int = 50,
    ):
        super().__init__()
        self.df = pd.read_csv(train_csv)
        if "BDMAP ID" not in self.df.columns:
            raise ValueError("train.csv must contain column 'BDMAP ID'")
        self.ids: List[str] = [str(x).strip() for x in self.df["BDMAP ID"].tolist() if str(x).strip()]

        if exclude_ids_path:
            with open(exclude_ids_path, "r", encoding="utf-8") as f:
                exclude_ids = {line.strip() for line in f if line.strip()}
            self.ids = [case_id for case_id in self.ids if case_id not in exclude_ids]

        self.image_root = image_root
        self.mask_root = mask_root
        self.graph_root = graph_root
        self.emb_root = emb_root
        self.patch_size = patch_size
        self.lesion_crop_prob = float(lesion_crop_prob)
        self.deterministic_crop = bool(deterministic_crop)
        self.sample_timeout_sec = max(float(sample_timeout_sec), 0.0)
        self.max_lesions_per_case = max(int(max_lesions_per_case), 0)
        self.supervised_gt_crop_epochs = max(int(supervised_gt_crop_epochs), 0)
        self.current_epoch = 1

        # choose supervised subset by ratio (fixed)
        rnd = random.Random(seed)
        ids_shuf = self.ids[:]
        rnd.shuffle(ids_shuf)
        sup_n = int(round(len(ids_shuf) * float(mask_ratio)))
        self.sup_set = set(ids_shuf[:sup_n])

    def __len__(self) -> int:
        return len(self.ids)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(int(epoch), 1)

    def _build_crop_meta(
        self,
        original_shape: Tuple[int, int, int],
        patch_shape: Tuple[int, int, int],
        full_start: Tuple[int, int, int],
        full_end: Tuple[int, int, int],
        patch_start: Tuple[int, int, int],
        patch_end: Tuple[int, int, int],
        mode: str,
    ) -> Dict[str, Any]:
        return {
            "mode": str(mode),
            "original_shape": [int(v) for v in original_shape],
            "patch_shape": [int(v) for v in patch_shape],
            "full_start": [int(v) for v in full_start],
            "full_end": [int(v) for v in full_end],
            "patch_start": [int(v) for v in patch_start],
            "patch_end": [int(v) for v in patch_end],
        }

    def _center_crop_or_pad_with_meta(
        self,
        x: np.ndarray,
        ps: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # x: (H,W,D)
        H,W,D = x.shape
        ph,pw,pd = ps
        out = np.zeros(ps, dtype=x.dtype)

        sh = max((H - ph)//2, 0)
        sw = max((W - pw)//2, 0)
        sd = max((D - pd)//2, 0)
        eh = min(sh + ph, H)
        ew = min(sw + pw, W)
        ed = min(sd + pd, D)

        th = max((ph - H)//2, 0)
        tw = max((pw - W)//2, 0)
        td = max((pd - D)//2, 0)

        out[th:th+(eh-sh), tw:tw+(ew-sw), td:td+(ed-sd)] = x[sh:eh, sw:ew, sd:ed]
        meta = self._build_crop_meta(
            original_shape=(H, W, D),
            patch_shape=ps,
            full_start=(sh, sw, sd),
            full_end=(eh, ew, ed),
            patch_start=(th, tw, td),
            patch_end=(th + (eh - sh), tw + (ew - sw), td + (ed - sd)),
            mode="center",
        )
        return out, meta

    def _center_crop_or_pad(self, x: np.ndarray, ps: Tuple[int,int,int]) -> np.ndarray:
        out, _ = self._center_crop_or_pad_with_meta(x, ps)
        return out

    def _crop_or_pad_from_start_with_meta(
        self,
        x: np.ndarray,
        start: Tuple[int, int, int],
        ps: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        H,W,D = x.shape
        ph,pw,pd = ps
        sh = min(max(start[0], 0), max(H - ph, 0))
        sw = min(max(start[1], 0), max(W - pw, 0))
        sd = min(max(start[2], 0), max(D - pd, 0))

        eh = min(sh + ph, H)
        ew = min(sw + pw, W)
        ed = min(sd + pd, D)

        out = np.zeros(ps, dtype=x.dtype)
        out[:eh-sh, :ew-sw, :ed-sd] = x[sh:eh, sw:ew, sd:ed]
        meta = self._build_crop_meta(
            original_shape=(H, W, D),
            patch_shape=ps,
            full_start=(sh, sw, sd),
            full_end=(eh, ew, ed),
            patch_start=(0, 0, 0),
            patch_end=(eh - sh, ew - sw, ed - sd),
            mode="start",
        )
        return out, meta

    def _crop_or_pad_from_start(self, x: np.ndarray, start: Tuple[int,int,int], ps: Tuple[int,int,int]) -> np.ndarray:
        out, _ = self._crop_or_pad_from_start_with_meta(x, start, ps)
        return out

    def _sample_crop_start_from_mask(self, mask: np.ndarray, ps: Tuple[int,int,int]) -> Optional[Tuple[int,int,int]]:
        idx = np.argwhere(mask > 0)
        if idx.size == 0:
            return None

        H,W,D = mask.shape
        ph,pw,pd = ps
        if self.deterministic_crop:
            center = ((idx.min(axis=0) + idx.max(axis=0)) // 2).astype(np.int64)
        else:
            center = idx[random.randrange(len(idx))]

        def sample_axis(c: int, crop: int, size: int) -> int:
            low = max(c - crop + 1, 0)
            high = min(c, max(size - crop, 0))
            if high < low:
                return max(min(c - crop // 2, size - crop), 0)
            if self.deterministic_crop:
                return max(min(c - crop // 2, size - crop), 0)
            return random.randint(low, high)

        return (
            sample_axis(int(center[0]), ph, H),
            sample_axis(int(center[1]), pw, W),
            sample_axis(int(center[2]), pd, D),
        )

    def _sample_crop_start_covering_mask(self, mask: np.ndarray, ps: Tuple[int,int,int]) -> Optional[Tuple[int,int,int]]:
        idx = np.argwhere(mask > 0)
        if idx.size == 0:
            return None

        H,W,D = mask.shape
        ph,pw,pd = ps
        mins = idx.min(axis=0)
        maxs = idx.max(axis=0) + 1

        def sample_axis(lo: int, hi: int, crop: int, size: int) -> int:
            span = hi - lo
            max_start = max(size - crop, 0)
            if span >= crop:
                center = (lo + hi) // 2
                return max(min(center - crop // 2, max_start), 0)

            start_low = max(hi - crop, 0)
            start_high = min(lo, max_start)
            if start_high < start_low:
                center = (lo + hi) // 2
                return max(min(center - crop // 2, max_start), 0)
            if self.deterministic_crop:
                center = (lo + hi) // 2
                return max(min(center - crop // 2, max_start), 0)
            return random.randint(start_low, start_high)

        return (
            sample_axis(int(mins[0]), int(maxs[0]), ph, H),
            sample_axis(int(mins[1]), int(maxs[1]), pw, W),
            sample_axis(int(mins[2]), int(maxs[2]), pd, D),
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id = self.ids[idx]
        try:
            with _SampleTimeout(self.sample_timeout_sec):
                ct_path = os.path.join(self.image_root, case_id, "ct.nii.gz")
                mk_path = os.path.join(self.mask_root, case_id, "combined_labels.nii.gz")

                graph_path = os.path.join(self.graph_root, f"{case_id}.graph.json")
                emb_path = os.path.join(self.emb_root, f"{case_id}.text_emb.npz")

                ct, spacing = load_nii_with_spacing(ct_path)  # (H,W,D)
                ct = hu_window_and_normalize(ct)
                mask = np.asarray(load_nii(mk_path, dtype=None), dtype=np.int16)
                has_mask = case_id in self.sup_set

                lesion_union_full = _make_binary(mask, LESION_IDS["kidney"] + LESION_IDS["pancreas"] + LESION_IDS["liver"])
                organ_union_full = _make_binary(mask, ORGAN_IDS["kidney"] + ORGAN_IDS["pancreas"] + ORGAN_IDS["liver"])

                crop_start = None
                crop_mode = "inference_like"
                use_gt_guided_crop = (
                    has_mask
                    and self.current_epoch <= self.supervised_gt_crop_epochs
                    and lesion_union_full.any()
                    and random.random() < self.lesion_crop_prob
                )
                if use_gt_guided_crop:
                    crop_start = self._sample_crop_start_covering_mask(lesion_union_full, self.patch_size)
                    if crop_start is None:
                        crop_start = self._sample_crop_start_from_mask(lesion_union_full, self.patch_size)
                    if crop_start is not None:
                        crop_mode = "gt_lesion"
                if crop_start is None:
                    crop_start = self._sample_crop_start_covering_mask(organ_union_full, self.patch_size)
                if crop_start is None:
                    crop_start = self._sample_crop_start_from_mask(organ_union_full, self.patch_size)

                # crop/pad (simple, deterministic). 你也可以换成随机 crop
                if crop_start is not None:
                    ct, crop_meta = self._crop_or_pad_from_start_with_meta(ct, crop_start, self.patch_size)
                    mask, _ = self._crop_or_pad_from_start_with_meta(mask, crop_start, self.patch_size)
                else:
                    ct, crop_meta = self._center_crop_or_pad_with_meta(ct, self.patch_size)
                    mask, _ = self._center_crop_or_pad_with_meta(mask, self.patch_size)

                # organ priors (from GT organ labels; used as anatomy prior)
                organ_masks = {
                    "kidney": _make_binary(mask, ORGAN_IDS["kidney"]),
                    "kidney_right": _make_binary(mask, ORGAN_IDS["kidney_right"]),
                    "kidney_left": _make_binary(mask, ORGAN_IDS["kidney_left"]),
                    "pancreas": _make_binary(mask, ORGAN_IDS["pancreas"]),
                    "liver": _make_binary(mask, ORGAN_IDS["liver"]),
                }
                # lesion GT per organ / side
                lesion_masks = {
                    "kidney": _make_binary(mask, LESION_IDS["kidney"]),
                    "kidney_right": _make_binary(mask, LESION_IDS["kidney_right"]),
                    "kidney_left": _make_binary(mask, LESION_IDS["kidney_left"]),
                    "pancreas": _make_binary(mask, LESION_IDS["pancreas"]),
                    "liver": _make_binary(mask, LESION_IDS["liver"]),
                }

                graph = _load_graph_json(graph_path)

                graph_node_ids = [str(n["id"]) for n in graph["nodes"]]
                text_emb = _load_text_embedding_cached(emb_path)
                if text_emb.shape[0] != len(graph_node_ids):
                    raise ValueError(
                        f"Embedding rows ({text_emb.shape[0]}) do not match graph nodes ({len(graph_node_ids)}) "
                        f"for case {case_id}."
                    )
                graph, text_emb = _filter_graph_to_crop(graph, text_emb, mask)
                graph, text_emb = _limit_graph_lesions(graph, text_emb, self.max_lesions_per_case)
                graph_node_ids = [str(n["id"]) for n in graph["nodes"]]

                # torch tensors
                ct_t = torch.from_numpy(ct[None, ...])  # (1,H,W,D)
                mask_t = torch.from_numpy(mask[None, ...].astype(np.int64))  # (1,H,W,D)

                organ_t = {k: torch.from_numpy(v[None, ...].astype(np.float32)) for k,v in organ_masks.items()}
                lesion_t = {k: torch.from_numpy(v[None, ...].astype(np.float32)) for k,v in lesion_masks.items()}

                return {
                    "case_id": case_id,
                    "crop_mode": crop_mode,
                    "ct_path": ct_path,
                    "mask_path": mk_path,
                    "crop_meta": crop_meta,
                    "ct": ct_t,
                    "mask_all": mask_t,
                    "organ_masks": organ_t,
                    "lesion_masks": lesion_t,
                    "graph": graph,
                    "node_ids": graph_node_ids,
                    "text_emb": torch.from_numpy(text_emb),
                    "has_mask": torch.tensor([1 if has_mask else 0], dtype=torch.float32),
                    "spacing": torch.tensor(spacing, dtype=torch.float32),
                    "skip_sample": False,
                }
        except _SampleTimeoutError as e:
            return {
                "case_id": case_id,
                "skip_sample": True,
                "skip_reason": str(e),
                "valid": False,
            }
