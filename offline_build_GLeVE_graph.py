# filename: # filename: offline_build_GLeVE_graph.py
# -*- coding: utf-8 -*-
"""
Offline pipeline (paper-aligned):
1) Read (case_id, structured_report) from CSV
2) Use Qwen3-8B to extract:
   - organ summaries: size_status, volume_cc, mean_hu(value,std)
   - lesions: organ, sub_location, size_mm, volume_cc, image_index, hu(mean,std), attenuation, appearance
   - relations: size_ranking (within organ), spatial (within organ), optional contrast polarity
3) Build a paper-aligned lesion semantic graph G_i=(V_i,E_i):
   - Nodes: organ nodes, lesion nodes, attribute nodes (compact per-lesion attr node)
   - Edges:
       lesion_organ            (attribution)
       lesion_attr             (lesion->attribute association)
       lesion_lesion_size      (size ranking constraint)
       lesion_lesion_spatial   (relative orientation constraint)
       lesion_organ_contrast   (lesion vs organ HU contrast)
4) Encode each node with CLIP Text Encoder offline and save embeddings to a separate file.
5) Save graph JSON (nodes/edges + all quantitative fields needed for losses).

Outputs per case:
- {out_dir}/graphs/{case_id}.graph.json               (nodes/edges + payloads for losses)
- {out_dir}/embeddings/{case_id}.text_emb.npz         (node_text_emb matrix + node_ids)
- {out_dir}/failed/{case_id}.txt                      (LLM output debug if JSON repair failed)
- {out_dir}/failed/{case_id}_final.txt                (final failure note)

Notes:
- This file focuses on "graph construction + node encoding + saving" aligned with the paper.
- Multi-GPU multiprocessing can be added externally (mp.spawn), but not required for correctness here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_HOME'] = 'data/LLM_cache'
# os.environ['TRANSFORMERS_OFFLINE'] = '0'
# os.environ['HF_DATASETS_OFFLINE'] = '0'
# os.environ['HF_EVALUATE_OFFLINE'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Qwen3 via ModelScope (as you used)
from modelscope import AutoModelForCausalLM, AutoTokenizer

# CLIP text encoder (offline embeddings)
from transformers import CLIPTextModel, CLIPTokenizer


# ============================================================
# 0) Config & Utilities
# ============================================================

TARGET_ORGANS = ["liver", "pancreas", "kidney"]

THINK_END_TOKEN_ID = 151668  # Qwen3: </think>

MAX_REPORT_CHARS = 8000

MAX_NEW_TOKENS_INIT = 1600
MAX_NEW_TOKENS_CAP = 12288
REGEN_GROWTH = 2
MAX_REGEN_ROUNDS = 3

SLEEP_TIME = 0.0


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(round(float(x)))
    except Exception:
        return None


def normalize_organ(s: str) -> str:
    if not s:
        return "unknown"
    t = s.strip().lower()
    # conservative mapping for this dataset
    if "liver" in t or "hepatic" in t:
        return "liver"
    if "pancreas" in t or "pancreatic" in t:
        return "pancreas"
    if "kidney" in t or "renal" in t:
        return "kidney"
    return "unknown"


def normalize_size_status(s: Any) -> str:
    t = str(s or "unknown").strip().lower()
    if t in ["normal", "enlarged", "shrunken"]:
        return t
    return "unknown"


def normalize_attenuation(s: Any) -> str:
    t = str(s or "unknown").strip().lower()
    # accept common set used in your paper
    for k in ["hypoattenuating", "hyperattenuating", "isoattenuating", "heterogeneous"]:
        if k in t:
            return k
    return "unknown"


def looks_truncated(text: str) -> bool:
    open_curly = text.count("{")
    close_curly = text.count("}")
    open_sq = text.count("[")
    close_sq = text.count("]")
    if open_curly > close_curly or open_sq > close_sq:
        return True
    tail = text.strip()[-50:]
    if tail.endswith((",", ":", "{", "[", "\"")):
        return True
    if "{" in text and "}" not in text:
        return True
    return False


def robust_json_loads(text: str, *, case_id: str, dump_dir: str) -> Tuple[Dict[str, Any], str]:
    """
    Parse JSON with common LLM repair rules. Dump failure for inspection.
    Returns: (parsed_obj, repaired_text_used)
    """
    try:
        return json.loads(text), text
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("No JSON object found")

    s = m.group(0)

    repairs = [
        # remove units attached to numbers
        (r'(\d+(?:\.\d+)?)\s*(cc|mm|cm|hu|HU)\b', r'\1'),
        # NaN / Infinity
        (r'\bNaN\b', 'null'),
        (r'\bInfinity\b', 'null'),
        # missing commas between string keys (best-effort)
        (r'(\]|\}|\d)"\s*\n\s*"', r'\1,\n"'),
        # trailing commas
        (r',\s*([\]}])', r'\1'),
    ]
    for pat, rep in repairs:
        s = re.sub(pat, rep, s)

    try:
        return json.loads(s), s
    except Exception as e:
        ensure_dir(dump_dir)
        dump_path = os.path.join(dump_dir, f"{case_id}.txt")
        try:
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write("===== ATTEMPTED JSON (AFTER REPAIR) =====\n")
                f.write(s)
                f.write("\n\n===== ORIGINAL LLM OUTPUT =====\n")
                f.write(text)
        except Exception:
            pass
        raise ValueError("JSON_REPAIR_FAILED") from e


# ============================================================
# 1) CSV Reader (two columns)
# ============================================================

def read_csv_two_cols(csv_path: str, id_col: str, report_col: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if id_col not in (reader.fieldnames or []) or report_col not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV missing columns. Need id_col='{id_col}', report_col='{report_col}'. "
                f"Got: {reader.fieldnames}"
            )
        for row in reader:
            cid = str(row.get(id_col, "")).strip()
            rpt = row.get(report_col, "")
            if not cid or not isinstance(rpt, str) or not rpt.strip():
                continue
            out.append((cid, rpt))
    return out


# ============================================================
# 2) Qwen3 Adaptive Parser (non-thinking -> thinking; dynamic max_new_tokens)
# ============================================================

def _gen_cfg(enable_thinking: bool) -> Dict[str, Any]:
    if enable_thinking:
        return dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    return dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)


@torch.no_grad()
def run_qwen3(prompt: str, tokenizer, model, enable_thinking: bool, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        **_gen_cfg(enable_thinking),
    )
    new_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # strip thinking content
    if enable_thinking:
        try:
            idx = len(new_ids) - new_ids[::-1].index(THINK_END_TOKEN_ID)
        except ValueError:
            idx = 0
        new_ids = new_ids[idx:]

    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def build_prompt(report_text: str) -> str:
    """
    Paper-aligned extraction prompt:
    - organs: size_status, volume_cc, mean_hu(value,std)
    - lesions: organ/sub_location/size_mm/volume_cc/image_index/hu/attenuation/appearance
    - relations: size_ranking, spatial
    """
    report_text = report_text.strip()
    if len(report_text) > MAX_REPORT_CHARS:
        report_text = report_text[:MAX_REPORT_CHARS] + "\n[TRUNCATED]"

    schema = {
        "organs": [
            {"organ": "liver|pancreas|kidney|unknown",
             "size_status": "normal|enlarged|shrunken|unknown",
             "volume_cc": None,
             "mean_hu": {"value": None, "std": None}}
        ],
        "lesions": [
            {"lesion_id": "L1",
             "organ": "liver|pancreas|kidney|unknown",
             "sub_location": "free text",
             "size_mm": [None, None],
             "volume_cc": None,
             "image_index": None,
             "hu": {"mean": None, "std": None},
             "attenuation": "hypoattenuating|hyperattenuating|isoattenuating|heterogeneous|unknown",
             "appearance": "short phrase"}
        ],
        "relations": {
            "size_ranking": [{"organ": "liver|pancreas|kidney|unknown", "larger": "L1", "smaller": "L2"}],
            "spatial": [{"organ": "liver|pancreas|kidney|unknown", "a": "L1", "b": "L2",
                         "relation": "anterior|posterior|superior|inferior|left|right|medial|lateral|proximal|distal|unknown"}]
        }
    }

    return f"""
You are a medical information extraction system.
Convert the radiology report into STRICT JSON for lesion grounding.

Requirements:
1) Extract organ summaries for {TARGET_ORGANS}: size_status, volume_cc (cc), and mean HU (value +/- std) if present.
2) Extract ALL explicitly described lesions (do not merge). Keep "appearance" concise.
3) Extract intra-organ relations: size_ranking and spatial relations if stated.
4) Output strictly valid JSON only. No markdown. No extra text.

Schema example (format only):
{json.dumps(schema, indent=2)}

Report:
\"\"\"{report_text}\"\"\"
""".strip()


def adaptive_parse_report(
    report_text: str,
    case_id: str,
    tokenizer,
    model,
    failed_dir: str,
) -> Dict[str, Any]:
    """
    1) non-thinking try_one with dynamic output length
    2) thinking fallback try_one with dynamic output length
    """

    def try_one(enable_thinking: bool) -> Dict[str, Any]:
        max_tok = MAX_NEW_TOKENS_INIT
        last_err: Optional[Exception] = None
        last_text: str = ""

        for _ in range(MAX_REGEN_ROUNDS):
            prompt = build_prompt(report_text)
            last_text = run_qwen3(prompt, tokenizer, model, enable_thinking, max_tok)
            try:
                obj, _ = robust_json_loads(last_text, case_id=case_id, dump_dir=failed_dir)
                return obj
            except Exception as e:
                last_err = e
                if looks_truncated(last_text):
                    max_tok = min(max_tok * REGEN_GROWTH, MAX_NEW_TOKENS_CAP)
                    continue
                break

        if last_err is not None:
            raise last_err
        raise RuntimeError("Unknown parsing failure")

    # non-thinking
    try:
        return try_one(enable_thinking=False)
    except Exception:
        pass

    # thinking fallback
    try:
        return try_one(enable_thinking=True)
    except Exception as e:
        ensure_dir(failed_dir)
        with open(os.path.join(failed_dir, f"{case_id}_final.txt"), "w", encoding="utf-8") as f:
            f.write("Failed after adaptive retries.\n")
            f.write(f"Error: {repr(e)}\n")
        raise


# ============================================================
# 3) Safety-net backfill (optional, for organ summaries)
# ============================================================

def _find_section_text(report: str, organ: str) -> str:
    pat = rf"\b{re.escape(organ)}\b\s*:\s*(.*?)(?=\n[A-Z][A-Za-z ]{{2,}}:\s|\nIMPRESSION:|\Z)"
    m = re.search(pat, report, re.S | re.I)
    return m.group(0) if m else ""


def backfill_organ_summary_from_text(report: str, organ: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    sec = _find_section_text(report, organ)
    if not sec:
        return summary

    # size_status
    if summary.get("size_status") in [None, "unknown"]:
        if re.search(r"\benlarged\b", sec, re.I):
            summary["size_status"] = "enlarged"
        elif re.search(r"\bNormal size\b", sec, re.I):
            summary["size_status"] = "normal"

    # volume_cc
    if summary.get("volume_cc") is None:
        mv = re.search(r"\bvolume\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*cc\b", sec, re.I)
        if mv:
            summary["volume_cc"] = safe_float(mv.group(1))

    # mean HU
    mh = summary.get("mean_hu") or {"value": None, "std": None}
    if mh.get("value") is None:
        mm = re.search(r"Mean HU value\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*\+/-\s*([0-9]+(?:\.[0-9]+)?)", sec, re.I)
        if mm:
            mh["value"] = safe_float(mm.group(1))
            mh["std"] = safe_float(mm.group(2))
            summary["mean_hu"] = mh

    return summary


# ============================================================
# 4) Paper-aligned graph data structures
# ============================================================

@dataclass
class GraphNode:
    id: str
    type: str  # "organ" | "lesion" | "attr"
    name: str
    organ: Optional[str] = None
    lesion_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    # Embeddings are saved separately; keep reference info here
    emb_index: Optional[int] = None  # row index in .npz matrix


@dataclass
class GraphEdge:
    source: str
    target: str
    rel: str  # relation type name (must match training-time rel2id)
    payload: Optional[Dict[str, Any]] = None
    bidirectional: bool = False


class PaperAlignedGraphBuilder:
    """
    Strictly align node/edge semantics with your paper description.
    """

    REL_LESION_ORGAN = "lesion_organ"
    REL_LESION_ATTR = "lesion_attr"
    REL_SIZE = "lesion_lesion_size"
    REL_SPATIAL = "lesion_lesion_spatial"
    REL_CONTRAST = "lesion_organ_contrast"

    def __init__(self, create_attr_nodes: bool = True):
        self.create_attr_nodes = create_attr_nodes

    def normalize_extraction(self, raw: Dict[str, Any], report_text: str) -> Dict[str, Any]:
        """
        Normalize LLM output to a stable schema used downstream.
        Keeps all quantitative fields needed for losses.
        """
        organs_out: Dict[str, Any] = {}
        organs_raw = raw.get("organs", []) or []
        # support both list and dict variants
        if isinstance(organs_raw, dict):
            # if model returned dict organ->summary
            for k, v in organs_raw.items():
                organs_out[normalize_organ(k)] = v
        elif isinstance(organs_raw, list):
            for o in organs_raw:
                organ = normalize_organ(str(o.get("organ", "unknown")))
                if organ not in TARGET_ORGANS:
                    continue
                size_status = normalize_size_status(o.get("size_status"))
                volume_cc = safe_float(o.get("volume_cc"))
                mh = o.get("mean_hu") or {}
                mean_val = safe_float(mh.get("value"))
                mean_std = safe_float(mh.get("std"))
                organs_out[organ] = {
                    "size_status": size_status,
                    "volume_cc": volume_cc,
                    "mean_hu": {"value": mean_val, "std": mean_std},
                }

        # backfill summaries (safety net)
        for organ in TARGET_ORGANS:
            organs_out.setdefault(organ, {"size_status": "unknown", "volume_cc": None, "mean_hu": {"value": None, "std": None}})
            organs_out[organ] = backfill_organ_summary_from_text(report_text, organ, organs_out[organ])

        # lesions
        lesions_norm: List[Dict[str, Any]] = []
        lesions_raw = raw.get("lesions", []) or []
        for i, l in enumerate(lesions_raw):
            organ = normalize_organ(str(l.get("organ", "unknown")))
            if organ not in TARGET_ORGANS:
                continue

            lid = str(l.get("lesion_id") or f"L{i+1}")
            subloc = str(l.get("sub_location") or "unknown")
            size_mm = l.get("size_mm", None)
            if isinstance(size_mm, list):
                size_mm = [safe_float(x) for x in size_mm][:3]
                size_mm = [x for x in size_mm if x is not None]
            else:
                size_mm = None

            volume_cc = safe_float(l.get("volume_cc"))
            image_index = safe_int(l.get("image_index"))
            hu = l.get("hu") or {}
            hu_mean = safe_float(hu.get("mean"))
            hu_std = safe_float(hu.get("std"))
            atten = normalize_attenuation(l.get("attenuation"))
            app = str(l.get("appearance") or "")

            lesions_norm.append({
                "lesion_id": lid,
                "organ": organ,
                "sub_location": subloc,
                "size_mm": size_mm,
                "volume_cc": volume_cc,          # used as V_i^r for weak loss
                "image_index": image_index,
                "hu": {"mean": hu_mean, "std": hu_std},  # used as mu_i^r for weak loss
                "attenuation": atten,
                "appearance": app,
            })

        # relations
        rels = raw.get("relations", {}) or {}
        size_ranking = rels.get("size_ranking", []) or []
        spatial = rels.get("spatial", []) or []

        # normalize relation entries
        size_ranking_n = []
        for r in size_ranking:
            organ = normalize_organ(str(r.get("organ", "unknown")))
            if organ not in TARGET_ORGANS:
                continue
            larger = str(r.get("larger", "")).strip()
            smaller = str(r.get("smaller", "")).strip()
            if larger and smaller:
                size_ranking_n.append({"organ": organ, "larger": larger, "smaller": smaller})

        spatial_n = []
        for r in spatial:
            organ = normalize_organ(str(r.get("organ", "unknown")))
            if organ not in TARGET_ORGANS:
                continue
            a = str(r.get("a", "")).strip()
            b = str(r.get("b", "")).strip()
            rel = str(r.get("relation", "unknown")).strip().lower()
            if a and b:
                spatial_n.append({"organ": organ, "a": a, "b": b, "relation": rel or "unknown"})

        return {
            "organs": organs_out,
            "lesions": lesions_norm,
            "relations": {"size_ranking": size_ranking_n, "spatial": spatial_n},
        }

    def build(self, case_id: str, extracted: Dict[str, Any]) -> Dict[str, Any]:
        organs_info: Dict[str, Any] = extracted["organs"]
        lesions: List[Dict[str, Any]] = extracted["lesions"]
        relations: Dict[str, Any] = extracted["relations"]

        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        # Organ nodes
        organ_node_id: Dict[str, str] = {}
        for organ in TARGET_ORGANS:
            nid = f"org_{organ}"
            organ_node_id[organ] = nid
            nodes.append(
                GraphNode(
                    id=nid,
                    type="organ",
                    name=organ,
                    organ=organ,
                    payload={"summary": organs_info.get(organ, {})},
                )
            )

        # Lesion nodes + optional attr nodes
        lesion_node_id: Dict[str, str] = {}
        attr_node_id: Dict[str, str] = {}

        for l in lesions:
            lid = l["lesion_id"]
            organ = l["organ"]
            lnid = f"les_{lid}"
            lesion_node_id[lid] = lnid

            nodes.append(
                GraphNode(
                    id=lnid,
                    type="lesion",
                    name=lid,
                    organ=organ,
                    lesion_id=lid,
                    payload={
                        # Keep all loss-related quantities in JSON:
                        "sub_location": l.get("sub_location"),
                        "size_mm": l.get("size_mm"),
                        "volume_cc": l.get("volume_cc"),  # V_i^r
                        "image_index": l.get("image_index"),
                        "hu": l.get("hu"),               # mu_i^r
                        "attenuation": l.get("attenuation"),
                        "appearance": l.get("appearance"),
                    },
                )
            )

            # lesion -> organ attribution (paper)
            edges.append(GraphEdge(
                source=lnid,
                target=organ_node_id[organ],
                rel=self.REL_LESION_ORGAN,
                payload={"organ": organ},
            ))

            # lesion -> organ contrast (paper)
            org_mean = (organs_info.get(organ, {}).get("mean_hu") or {}).get("value", None)
            les_mean = (l.get("hu") or {}).get("mean", None)
            if org_mean is not None and les_mean is not None:
                edges.append(GraphEdge(
                    source=lnid,
                    target=organ_node_id[organ],
                    rel=self.REL_CONTRAST,
                    payload={
                        "organ": organ,
                        "lesion_hu_mean": les_mean,
                        "organ_mean_hu": org_mean,
                        "delta": float(les_mean) - float(org_mean),
                    },
                ))

            if self.create_attr_nodes:
                anid = f"attr_{lid}"
                attr_node_id[lid] = anid
                nodes.append(
                    GraphNode(
                        id=anid,
                        type="attr",
                        name=f"attr_{lid}",
                        organ=organ,
                        lesion_id=lid,
                        payload={
                            # attribute pack (paper: lesion-attribute association)
                            "sub_location": l.get("sub_location"),
                            "size_mm": l.get("size_mm"),
                            "volume_cc": l.get("volume_cc"),
                            "hu": l.get("hu"),
                            "attenuation": l.get("attenuation"),
                            "appearance": l.get("appearance"),
                        },
                    )
                )
                edges.append(GraphEdge(
                    source=lnid,
                    target=anid,
                    rel=self.REL_LESION_ATTR,
                    payload={"lesion_id": lid},
                ))

        # Size ranking edges (paper: explicit comparative constraint)
        for r in relations.get("size_ranking", []) or []:
            larger = r["larger"]
            smaller = r["smaller"]
            if larger in lesion_node_id and smaller in lesion_node_id:
                edges.append(GraphEdge(
                    source=lesion_node_id[larger],
                    target=lesion_node_id[smaller],
                    rel=self.REL_SIZE,
                    payload=r,
                ))

        # Spatial edges (paper: relative orientation constraint)
        for r in relations.get("spatial", []) or []:
            a = r["a"]
            b = r["b"]
            if a in lesion_node_id and b in lesion_node_id:
                edges.append(GraphEdge(
                    source=lesion_node_id[a],
                    target=lesion_node_id[b],
                    rel=self.REL_SPATIAL,
                    payload=r,
                    bidirectional=True,
                ))

        graph = {
            "case_id": case_id,
            "nodes": [asdict(n) for n in nodes],
            "edges": [asdict(e) for e in edges],
            "meta": {
                "target_organs": TARGET_ORGANS,
                "create_attr_nodes": self.create_attr_nodes,
                # training-time: make sure rel2id matches these rel names
                "rel_types": sorted(list({e.rel for e in edges})),
            }
        }
        return graph


# ============================================================
# 5) CLIP Node Encoder (write embeddings separately)
# ============================================================

class CLIPNodeEmbedder:
    """
    Encode node texts with CLIP Text Encoder and save to separate file.

    Saved NPZ keys:
      - node_ids: (N,) object array of node ids
      - text_emb: (N, D) float32 array (L2-normalized)
    """
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        self.device = device
        self.tok = CLIPTokenizer.from_pretrained(model_name)
        self.enc = CLIPTextModel.from_pretrained(model_name).to(device)
        self.enc.eval()

    @torch.inference_mode()
    def encode_texts(self, texts: List[str], max_length: int = 77, batch_size: int = 64) -> np.ndarray:
        outs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch_text = texts[i:i + batch_size]
            batch = self.tok(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.enc(**batch)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                emb = out.last_hidden_state[:, 0]
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
            outs.append(emb.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)

    def node_to_text(self, node: Dict[str, Any]) -> str:
        """
        Paper-aligned node textualization:
        - organ nodes: include size_status, volume, mean HU
        - lesion nodes: include organ, location, size, volume, HU, attenuation, appearance, image index
        - attr nodes: compact attribute summary (mirrors lesion fields)
        """
        ntype = node["type"]
        organ = node.get("organ", "unknown")
        name = node.get("name", node.get("id", ""))

        payload = node.get("payload") or {}
        if ntype == "organ":
            summ = payload.get("summary") or {}
            ss = summ.get("size_status", "unknown")
            vol = summ.get("volume_cc", None)
            mh = (summ.get("mean_hu") or {}).get("value", None)
            sh = (summ.get("mean_hu") or {}).get("std", None)
            return f"organ {organ}. size_status: {ss}. volume_cc: {vol}. mean_hu: {mh} +/- {sh}."
        if ntype == "lesion":
            hu = payload.get("hu") or {}
            return (
                f"lesion {name}. organ: {organ}. location: {payload.get('sub_location')}. "
                f"size_mm: {payload.get('size_mm')}. volume_cc: {payload.get('volume_cc')}. "
                f"image_index: {payload.get('image_index')}. "
                f"HU mean/std: {hu.get('mean')}/{hu.get('std')}. "
                f"attenuation: {payload.get('attenuation')}. appearance: {payload.get('appearance')}."
            )
        # attr
        hu = (payload.get("hu") or {})
        return (
            f"attributes for lesion {node.get('lesion_id')}. organ: {organ}. "
            f"location: {payload.get('sub_location')}. size_mm: {payload.get('size_mm')}. "
            f"volume_cc: {payload.get('volume_cc')}. HU mean/std: {hu.get('mean')}/{hu.get('std')}. "
            f"attenuation: {payload.get('attenuation')}. appearance: {payload.get('appearance')}."
        )

    def encode_and_save(self, graph: Dict[str, Any], emb_path: str) -> Dict[str, Any]:
        nodes = graph["nodes"]
        node_ids = [n["id"] for n in nodes]
        texts = [self.node_to_text(n) for n in nodes]
        emb = self.encode_texts(texts)

        ensure_dir(os.path.dirname(emb_path))
        np.savez_compressed(emb_path, node_ids=np.array(node_ids, dtype=object), text_emb=emb)

        # write emb_index back to graph nodes
        for i, n in enumerate(nodes):
            n["emb_index"] = int(i)

        graph["meta"]["text_emb_path"] = emb_path
        graph["meta"]["text_emb_dim"] = int(emb.shape[1])
        return graph


# ============================================================
# 6) End-to-end per-case processing
# ============================================================

def process_one_case(
    case_id: str,
    report: str,
    tokenizer,
    model,
    builder: PaperAlignedGraphBuilder,
    embedder: Optional[CLIPNodeEmbedder],
    out_graph_dir: str,
    out_emb_dir: str,
    failed_dir: str,
    keep_extracted_in_json: bool = True,
) -> None:
    raw = adaptive_parse_report(report, case_id, tokenizer, model, failed_dir=failed_dir)
    extracted = builder.normalize_extraction(raw, report_text=report)

    graph = builder.build(case_id, extracted)

    # keep extracted (with all quantities for loss), optional but useful
    if keep_extracted_in_json:
        graph["meta"]["extracted"] = extracted

    # embeddings
    if embedder is not None:
        emb_path = os.path.join(out_emb_dir, f"{case_id}.text_emb.npz")
        graph = embedder.encode_and_save(graph, emb_path=emb_path)

    ensure_dir(out_graph_dir)
    out_path = os.path.join(out_graph_dir, f"{case_id}.graph.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)


# ============================================================
# 7) Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default="/data/AbdomenAtlas3.0MiniWithMeta.csv")
    ap.add_argument("--id_col", type=str, default="BDMAP ID")
    ap.add_argument("--report_col", type=str, default="structured report")

    ap.add_argument("--model_name_or_path", type=str, default="/data/Qwen3-8B", help="Local Qwen3-8B path")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--out_dir", type=str, default="gleve_offline")
    ap.add_argument("--no_clip", action="store_true")
    ap.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")

    ap.add_argument("--create_attr_nodes", action="store_true",default=True, help="Create per-lesion attribute nodes")
    ap.add_argument("--keep_extracted_in_json", action="store_true",default=True, help="Store extracted fields in graph JSON meta")
    args = ap.parse_args()

    out_graph_dir = os.path.join(args.out_dir, "graphs")
    out_emb_dir = os.path.join(args.out_dir, "embeddings")
    failed_dir = os.path.join(args.out_dir, "failed")
    ensure_dir(out_graph_dir)
    ensure_dir(out_emb_dir)
    ensure_dir(failed_dir)

    # Load Qwen3
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).eval()

    builder = PaperAlignedGraphBuilder(create_attr_nodes=args.create_attr_nodes)

    embedder = None
    if not args.no_clip:
        dev = args.device if args.device in ["cuda", "cpu"] else "cuda"
        embedder = CLIPNodeEmbedder(model_name=args.clip_model, device=dev)

    # Read data
    rows = read_csv_two_cols(args.input_csv, args.id_col, args.report_col)

    failed_ids: List[str] = []
    for case_id, report in rows:
        try:
            process_one_case(
                case_id=case_id,
                report=report,
                tokenizer=tokenizer,
                model=model,
                builder=builder,
                embedder=embedder,
                out_graph_dir=out_graph_dir,
                out_emb_dir=out_emb_dir,
                failed_dir=failed_dir,
                keep_extracted_in_json=args.keep_extracted_in_json,
            )
            print(f"OK {case_id}")
        except Exception:
            failed_ids.append(case_id)
            print(f"FAIL {case_id}")
        time.sleep(SLEEP_TIME)

    if failed_ids:
        with open(os.path.join(args.out_dir, "failed_ids.txt"), "w", encoding="utf-8") as f:
            for cid in failed_ids:
                f.write(cid + "\n")


if __name__ == "__main__":
    main()
