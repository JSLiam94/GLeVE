# utils_nii.py
from __future__ import annotations
import os
import numpy as np
import nibabel as nib

HU_MIN = -100.0
HU_MAX = 400.0

def load_nii(path: str, dtype: np.dtype | None = np.float32) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    nii = nib.load(path)
    if dtype is None:
        arr = np.asanyarray(nii.dataobj)
    else:
        arr = nii.get_fdata(dtype=dtype)
    return arr  # (H,W,D) or (H,W,D,C)


def load_nii_with_spacing(path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    nii = nib.load(path)
    arr = nii.get_fdata(dtype=np.float32)
    spacing = tuple(float(x) for x in nii.header.get_zooms()[:3])
    return arr, spacing

def save_npz(path: str, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **kwargs)

def hu_window_and_normalize(x: np.ndarray, hu_min: float=HU_MIN, hu_max: float=HU_MAX) -> np.ndarray:
    x = np.clip(x, hu_min, hu_max)
    x = (x - hu_min) / (hu_max - hu_min + 1e-8)
    return x.astype(np.float32)


def normalized_to_hu(x: np.ndarray | float, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> np.ndarray | float:
    return np.asarray(x) * (hu_max - hu_min) + hu_min


def collate_fn(batch):
    # Each DDP process still trains with local batch_size=1.
    assert len(batch) == 1, "This training code assumes per-process batch_size=1."
    item = batch[0]
    if item.get("skip_sample", False):
        return dict(item)

    out = dict(item)
    out["ct"] = item["ct"].unsqueeze(0)  # (B=1,C,H,W,D)
    out["mask_all"] = item["mask_all"].unsqueeze(0)
    out["has_mask"] = item["has_mask"].unsqueeze(0)
    out["spacing"] = item["spacing"].unsqueeze(0)
    out["text_emb"] = item["text_emb"]
    out["organ_masks"] = {k: v.unsqueeze(0) for k, v in item["organ_masks"].items()}
    out["lesion_masks"] = {k: v.unsqueeze(0) for k, v in item["lesion_masks"].items()}
    return out