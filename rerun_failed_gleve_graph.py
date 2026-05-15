from __future__ import annotations

import argparse
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import offline_build_GLeVE_graph as base
from modelscope import AutoModelForCausalLM, AutoTokenizer


RETRY_MAX_REPORT_CHARS = 20000
RETRY_MAX_NEW_TOKENS_INIT = 3200
RETRY_MAX_NEW_TOKENS_CAP = 16384
RETRY_MAX_REGEN_ROUNDS = 4
RETRY_SLEEP_TIME = 0.0


def read_failed_ids(failed_ids_path: Optional[str], failed_dir: str) -> List[str]:
    ids: List[str] = []
    seen = set()

    if failed_ids_path and os.path.exists(failed_ids_path):
        with open(failed_ids_path, "r", encoding="utf-8") as f:
            for line in f:
                cid = line.strip()
                if cid and cid not in seen:
                    seen.add(cid)
                    ids.append(cid)

    if os.path.isdir(failed_dir):
        for name in sorted(os.listdir(failed_dir)):
            if not name.endswith(".txt"):
                continue
            cid = re.sub(r"(_final)?\.txt$", "", name)
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)

    return ids


def extract_priority_sections(report_text: str) -> str:
    patterns = [
        r"IMPRESSION\s*:.*?(?=\n[A-Z][A-Z /()-]{2,}\s*:|\Z)",
        r"FINDINGS\s*:.*?(?=\n[A-Z][A-Z /()-]{2,}\s*:|\Z)",
        r"LIVER\s*:.*?(?=\n[A-Z][A-Z /()-]{2,}\s*:|\Z)",
        r"PANCREAS\s*:.*?(?=\n[A-Z][A-Z /()-]{2,}\s*:|\Z)",
        r"KIDNEY(?:S)?\s*:.*?(?=\n[A-Z][A-Z /()-]{2,}\s*:|\Z)",
    ]
    chunks: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, report_text, re.I | re.S):
            txt = m.group(0).strip()
            if txt and txt not in chunks:
                chunks.append(txt)
    return "\n\n".join(chunks)


def build_retry_prompt(report_text: str, mode: str) -> str:
    report_text = report_text.strip()
    if len(report_text) > RETRY_MAX_REPORT_CHARS:
        report_text = report_text[:RETRY_MAX_REPORT_CHARS] + "\n[TRUNCATED]"

    priority = extract_priority_sections(report_text)
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
             "size_mm": [None, None, None],
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

    if mode == "priority":
        prompt_intro = (
            "You are repairing a previous failed radiology extraction.\n"
            "Use the priority sections first, then use the full report only to fill missing values.\n"
            "Extract every explicit lesion separately. Do not merge lesions.\n"
            "Return strict JSON only."
        )
        body = f'Priority sections:\n"""\n{priority}\n"""\n\nFull report:\n"""\n{report_text}\n"""'
    else:
        prompt_intro = (
            "You are repairing a previous failed radiology extraction.\n"
            "The previous attempt likely failed due to truncation or malformed JSON.\n"
            "Read the entire report carefully and return strict JSON only.\n"
            "Extract every explicit lesion separately. Do not merge lesions."
        )
        body = f'Full report:\n"""\n{report_text}\n"""'

    return (
        f"{prompt_intro}\n\n"
        "Requirements:\n"
        "1) Extract organ summaries for liver, pancreas, kidney.\n"
        "2) Extract ALL explicitly described lesions.\n"
        "3) Keep lesion appearance concise.\n"
        "4) If a value is missing, use null.\n"
        "5) Output strictly valid JSON only. No markdown. No explanation.\n\n"
        f"Schema example:\n{base.json.dumps(schema, indent=2)}\n\n"
        f"{body}"
    ).strip()


def adaptive_parse_report_retry(
    report_text: str,
    case_id: str,
    tokenizer,
    model,
    failed_dir: str,
) -> Dict[str, Any]:
    prompt_modes: Sequence[str] = ("full", "priority")

    def try_prompt(enable_thinking: bool, prompt_mode: str) -> Dict[str, Any]:
        max_tok = RETRY_MAX_NEW_TOKENS_INIT
        last_err: Optional[Exception] = None
        last_text = ""

        for _ in range(RETRY_MAX_REGEN_ROUNDS):
            prompt = build_retry_prompt(report_text, mode=prompt_mode)
            last_text = base.run_qwen3(prompt, tokenizer, model, enable_thinking, max_tok)
            try:
                obj, _ = base.robust_json_loads(last_text, case_id=case_id, dump_dir=failed_dir)
                return obj
            except Exception as e:
                last_err = e
                if base.looks_truncated(last_text):
                    max_tok = min(max_tok * base.REGEN_GROWTH, RETRY_MAX_NEW_TOKENS_CAP)
                    continue
                break

        if last_err is not None:
            raise last_err
        raise RuntimeError("Unknown retry parsing failure")

    last_exception: Optional[Exception] = None
    for prompt_mode in prompt_modes:
        for enable_thinking in (False, True):
            try:
                return try_prompt(enable_thinking=enable_thinking, prompt_mode=prompt_mode)
            except Exception as e:
                last_exception = e

    base.ensure_dir(failed_dir)
    with open(os.path.join(failed_dir, f"{case_id}_retry_final.txt"), "w", encoding="utf-8") as f:
        f.write("Retry failed after expanded-context attempts.\n")
        if last_exception is not None:
            f.write(f"Error: {repr(last_exception)}\n")
    raise last_exception if last_exception is not None else RuntimeError("Retry failed")


def process_one_case_retry(
    case_id: str,
    report: str,
    tokenizer,
    model,
    builder: base.PaperAlignedGraphBuilder,
    embedder: Optional[base.CLIPNodeEmbedder],
    out_graph_dir: str,
    out_emb_dir: str,
    failed_dir: str,
    keep_extracted_in_json: bool = True,
) -> None:
    raw = adaptive_parse_report_retry(report, case_id, tokenizer, model, failed_dir=failed_dir)
    extracted = builder.normalize_extraction(raw, report_text=report)
    graph = builder.build(case_id, extracted)

    if keep_extracted_in_json:
        graph["meta"]["extracted"] = extracted
        graph["meta"]["retry_strategy"] = {
            "expanded_context": True,
            "max_report_chars": RETRY_MAX_REPORT_CHARS,
            "max_new_tokens_init": RETRY_MAX_NEW_TOKENS_INIT,
            "max_new_tokens_cap": RETRY_MAX_NEW_TOKENS_CAP,
        }

    if embedder is not None:
        emb_path = os.path.join(out_emb_dir, f"{case_id}.text_emb.npz")
        graph = embedder.encode_and_save(graph, emb_path=emb_path)

    base.ensure_dir(out_graph_dir)
    out_path = os.path.join(out_graph_dir, f"{case_id}.graph.json")
    with open(out_path, "w", encoding="utf-8") as f:
        base.json.dump(graph, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default="/data/AbdomenAtlas3.0MiniWithMeta.csv")
    ap.add_argument("--id_col", type=str, default="BDMAP ID")
    ap.add_argument("--report_col", type=str, default="structured report")
    ap.add_argument("--failed_ids", type=str, default=None, help="Path to failed_ids.txt from the original offline run")
    ap.add_argument("--out_dir", type=str, default="gleve_offline")
    ap.add_argument("--model_name_or_path", type=str, default="/data/Qwen3-8B")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_clip", action="store_true")
    ap.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    ap.add_argument("--create_attr_nodes", action="store_true", default=True)
    ap.add_argument("--keep_extracted_in_json", action="store_true", default=True)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing graph/embedding outputs")
    args = ap.parse_args()

    out_graph_dir = os.path.join(args.out_dir, "graphs")
    out_emb_dir = os.path.join(args.out_dir, "embeddings")
    failed_dir = os.path.join(args.out_dir, "failed")
    retry_failed_dir = os.path.join(args.out_dir, "retry_failed")
    base.ensure_dir(out_graph_dir)
    base.ensure_dir(out_emb_dir)
    base.ensure_dir(failed_dir)
    base.ensure_dir(retry_failed_dir)

    failed_ids_path = args.failed_ids or os.path.join(args.out_dir, "failed_ids.txt")
    target_ids = set(read_failed_ids(failed_ids_path, failed_dir))
    if not target_ids:
        print("No failed cases found.")
        return

    rows = base.read_csv_two_cols(args.input_csv, args.id_col, args.report_col)
    rows = [(case_id, report) for case_id, report in rows if case_id in target_ids]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    builder = base.PaperAlignedGraphBuilder(create_attr_nodes=args.create_attr_nodes)
    embedder = None
    if not args.no_clip:
        dev = args.device if args.device in ["cuda", "cpu"] else "cuda"
        embedder = base.CLIPNodeEmbedder(model_name=args.clip_model, device=dev)

    still_failed: List[str] = []
    rerun_ok: List[str] = []

    for case_id, report in rows:
        graph_path = os.path.join(out_graph_dir, f"{case_id}.graph.json")
        emb_path = os.path.join(out_emb_dir, f"{case_id}.text_emb.npz")
        if (not args.overwrite) and os.path.exists(graph_path) and (args.no_clip or os.path.exists(emb_path)):
            print(f"SKIP {case_id} existing output")
            continue

        try:
            process_one_case_retry(
                case_id=case_id,
                report=report,
                tokenizer=tokenizer,
                model=model,
                builder=builder,
                embedder=embedder,
                out_graph_dir=out_graph_dir,
                out_emb_dir=out_emb_dir,
                failed_dir=retry_failed_dir,
                keep_extracted_in_json=args.keep_extracted_in_json,
            )
            rerun_ok.append(case_id)
            print(f"RETRY_OK {case_id}")
        except Exception as e:
            still_failed.append(case_id)
            with open(os.path.join(retry_failed_dir, f"{case_id}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Retry failed.\nError: {repr(e)}\n")
            print(f"RETRY_FAIL {case_id}")
        time.sleep(RETRY_SLEEP_TIME)

    if rerun_ok:
        with open(os.path.join(args.out_dir, "retry_success_ids.txt"), "w", encoding="utf-8") as f:
            for cid in rerun_ok:
                f.write(cid + "\n")

    if still_failed:
        with open(os.path.join(args.out_dir, "retry_failed_ids.txt"), "w", encoding="utf-8") as f:
            for cid in still_failed:
                f.write(cid + "\n")


if __name__ == "__main__":
    main()
