import os
import json
import time
import re
import csv
import ast
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from google import genai

PROMPTS_DIR = Path("ml1m_dmf_prompts")
PROMPT_GLOB = "prompt_user_*.txt"
TOPK = 10
BATCH_INPUT_FILE = Path("batch_requests_ml1m_dmf_rerank.jsonl")
BATCH_RESULTS_JSONL = Path("batch_results_ml1m_dmf_rerank.jsonl")
DMF_RECS_FILE = Path("DMF_top50_seed2020.csv")
OUT_RECS_CSV = Path("reranked_ml1m_dmf_top10_recbole_like.csv")

MODEL_CANDIDATES = ["gemini-2.0-flash", "models/gemini-2.0-flash"]
GENERATION_CONFIG = {"max_output_tokens": 4096, "temperature": 0.2}


def extract_user_id_from_filename(stem: str) -> Optional[str]:
    m = re.search(r"prompt_user_(.+)$", stem)
    return m.group(1) if m else None


def pick_prompt_files(folder: Path, pattern: str) -> Dict[str, Path]:
    mp: Dict[str, Path] = {}
    for p in folder.glob(pattern):
        uid = extract_user_id_from_filename(p.stem)
        if uid is not None:
            mp[uid] = p
    return mp


def extract_candidates_from_prompt(prompt_text: str) -> List[dict]:
    matches = re.findall(r"#List of Recommended Items\s*\n.*?\n(\[.*?\])\s*\n#Instructions", prompt_text, re.DOTALL)
    if matches:
        try:
            obj = json.loads(matches[-1])
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
        except Exception:
            pass

    names = re.findall(
        r'\{\s*"rank"\s*:\s*(\d+)\s*,\s*"item_id"\s*:\s*(\d+)\s*,\s*"item_name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        prompt_text
    )
    out = []
    for r, iid, name in names:
        try:
            name_dec = bytes(name, "utf-8").decode("unicode_escape").strip()
        except Exception:
            name_dec = name.strip()
        out.append({"rank": int(r), "item_id": int(iid), "item_name": name_dec})
    return out


def build_batch_jsonl(prompt_map: Dict[str, Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for uid in sorted(prompt_map, key=lambda x: int(x)):
            prompt = prompt_map[uid].read_text(encoding="utf-8")
            line = {
                "key": prompt_map[uid].stem,
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generation_config": GENERATION_CONFIG,
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def get_state(job):
    return getattr(job, "state", None) or (job.get("state") if isinstance(job, dict) else None)


def job_to_dict(job) -> dict:
    if isinstance(job, dict):
        return job
    if hasattr(job, "model_dump"):
        return job.model_dump()
    if hasattr(job, "dict"):
        return job.dict()
    return json.loads(str(job))


def find_output_file_name(job_dict: dict) -> str:
    for a, b in [
        ("dest", "fileName"),
        ("dest", "file_name"),
        ("output", "fileName"),
        ("output", "file_name"),
        ("output_file", "fileName"),
        ("output_file", "file_name"),
        ("result", "fileName"),
        ("result", "file_name"),
    ]:
        block = job_dict.get(a)
        if isinstance(block, dict) and block.get(b):
            return block[b]
    raw = json.dumps(job_dict)
    m = re.search(r"(files/[A-Za-z0-9_\-]+)", raw)
    if m:
        return m.group(1)
    raise RuntimeError("Non trovo fileName dell'output nel batch job.")


def extract_text_from_batch_result(obj: dict) -> str:
    resp = obj.get("response") or {}
    candidates = resp.get("candidates") or []
    if candidates:
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
        if texts:
            return "".join(texts).strip()
    if isinstance(resp.get("text"), str):
        return resp["text"].strip()
    return ""


def parse_json_list(text: str):
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else None
    except Exception:
        pass
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, list) else None
        except Exception:
            pass
    return None


def salvage_item_ids(text: str) -> List[int]:
    ids = re.findall(r'"item_id"\s*:\s*(\d+)', text or "")
    return [int(x) for x in ids]


def load_dmf_recs(path: Path) -> Dict[int, List[int]]:
    df = pd.read_csv(path)
    needed = {"user_id", "recs"}
    if not needed.issubset(df.columns):
        raise RuntimeError(f"{path} non ha le colonne attese {needed}. Trovate: {list(df.columns)}")
    out: Dict[int, List[int]] = {}
    for _, row in df.iterrows():
        uid = int(row["user_id"])
        recs = ast.literal_eval(row["recs"])
        out[uid] = [int(x) for x in recs]
    return out


def run_gemini_batch(client, prompt_map: Dict[str, Path]) -> None:
    build_batch_jsonl(prompt_map, BATCH_INPUT_FILE)
    print(f"Creato input batch: {BATCH_INPUT_FILE} ({len(prompt_map)} prompt)")
    uploaded = client.files.upload(file=str(BATCH_INPUT_FILE), config={"mime_type": "application/jsonl"})
    print("Upload ok. File name:", uploaded.name)

    batch_job = None
    last_err = None
    for model_name in MODEL_CANDIDATES:
        try:
            print("Provo model:", model_name)
            batch_job = client.batches.create(
                model=model_name,
                src=uploaded.name,
                config={"display_name": f"ml1m-dmf-{len(prompt_map)}-users-rerank"},
            )
            print("Batch job creato:", batch_job.name)
            break
        except Exception as e:
            last_err = e
            print("Fallito:", e)
    if batch_job is None:
        raise RuntimeError(f"Impossibile creare batch. Ultimo errore: {last_err}")

    terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_PAUSED"}
    while True:
        batch_job = client.batches.get(name=batch_job.name)
        state = get_state(batch_job)
        print("Stato:", state)
        if state in terminal:
            break
        time.sleep(30)

    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch fallito: state={state}")

    output_file = find_output_file_name(job_to_dict(batch_job))
    print("Results file:", output_file)
    file_bytes = client.files.download(file=output_file)
    BATCH_RESULTS_JSONL.write_bytes(file_bytes)
    print("Salvato:", BATCH_RESULTS_JSONL)


def main():
    for p, label in [(PROMPTS_DIR, "Cartella prompt"), (DMF_RECS_FILE, "DMF_top50_seed2020.csv")]:
        if not p.exists():
            raise RuntimeError(f"{label} non trovato: {p.resolve()}")

    prompt_map = pick_prompt_files(PROMPTS_DIR, PROMPT_GLOB)
    if not prompt_map:
        raise RuntimeError(f"Nessun prompt trovato in {PROMPTS_DIR.resolve()}")
    print(f"Prompt trovati: {len(prompt_map)} utenti")

    print("Estraggo candidati dai prompt...")
    prompt_candidates: Dict[str, List[dict]] = {}
    for uid, path in prompt_map.items():
        prompt_candidates[uid] = extract_candidates_from_prompt(path.read_text(encoding="utf-8"))
    avg_cands = sum(len(v) for v in prompt_candidates.values()) / max(len(prompt_candidates), 1)
    print(f"Candidati medi per prompt: {avg_cands:.1f}")

    print("Carico DMF top50...")
    dmf_user_to_recs = load_dmf_recs(DMF_RECS_FILE)
    print(f"Utenti nel DMF csv: {len(dmf_user_to_recs)}")

    if not BATCH_RESULTS_JSONL.exists():
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Manca GEMINI_API_KEY nelle variabili d'ambiente.")
        client = genai.Client(api_key=api_key)
        run_gemini_batch(client, prompt_map)
    else:
        print(f"Risultati già presenti, salto Gemini: {BATCH_RESULTS_JSONL}")

    print("Parsing risultati batch...")
    rerank_item_ids_by_uid: Dict[str, List[int]] = {}
    with BATCH_RESULTS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("key", "")
            uid = extract_user_id_from_filename(key)
            if uid is None:
                continue
            text = extract_text_from_batch_result(obj)
            parsed = parse_json_list(text)
            item_ids: List[int] = []
            if parsed is not None:
                for it in parsed:
                    if isinstance(it, dict) and it.get("item_id") is not None:
                        try:
                            item_ids.append(int(it["item_id"]))
                        except Exception:
                            pass
            else:
                item_ids = salvage_item_ids(text)
            seen = set()
            clean = []
            for iid in item_ids:
                if iid in seen:
                    continue
                seen.add(iid)
                clean.append(iid)
            rerank_item_ids_by_uid[uid] = clean
    print(f"Utenti con output Gemini: {len(rerank_item_ids_by_uid)}")

    missing_batch = zero_matches = partial_matches = total_matches = fallback_used = skipped_users = 0
    invalid_ids = non_candidate_ids = 0
    match_counts: List[int] = []

    OUT_RECS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RECS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "item_id", "score"])

        for uid_str in sorted(prompt_map.keys(), key=lambda x: int(x)):
            try:
                uid_inner = int(uid_str)
            except ValueError:
                skipped_users += 1
                continue

            candidate_objs = prompt_candidates.get(uid_str, [])
            prompt_candidate_ids = [int(x["item_id"]) for x in candidate_objs if x.get("item_id") is not None]
            dmf_item_ids = dmf_user_to_recs.get(uid_inner, [])
            if not dmf_item_ids:
                skipped_users += 1
                continue

            candidate_ids = prompt_candidate_ids if prompt_candidate_ids else dmf_item_ids
            candidate_set = set(candidate_ids)
            fallback_source = dmf_item_ids

            rr_ids_raw = rerank_item_ids_by_uid.get(uid_str, [])
            if uid_str not in rerank_item_ids_by_uid:
                missing_batch += 1

            rr_ids: List[int] = []
            seen = set()
            for iid in rr_ids_raw:
                if not isinstance(iid, int):
                    invalid_ids += 1
                    continue
                if iid not in candidate_set:
                    non_candidate_ids += 1
                    continue
                if iid in seen:
                    continue
                seen.add(iid)
                rr_ids.append(iid)

            match_counts.append(len(rr_ids))
            total_matches += len(rr_ids)
            if len(rr_ids) == 0:
                zero_matches += 1
            elif len(rr_ids) < TOPK:
                partial_matches += 1

            final_ids: List[int] = []
            used = set()
            for iid in rr_ids:
                if len(final_ids) >= TOPK:
                    break
                if iid in used:
                    continue
                used.add(iid)
                final_ids.append(iid)

            if len(final_ids) < TOPK:
                fallback_used += 1
                for iid in fallback_source:
                    if len(final_ids) >= TOPK:
                        break
                    if iid in used:
                        continue
                    used.add(iid)
                    final_ids.append(iid)

            for rank, iid_inner in enumerate(final_ids[:TOPK], start=1):
                score = float(TOPK - rank + 1)
                w.writerow([uid_inner, iid_inner, score])

    n_written = len(prompt_map) - skipped_users
    print("\n=== DONE ===")
    print(f"CSV:                                 {OUT_RECS_CSV.resolve()}")
    print(f"Utenti scritti:                      {n_written}")
    print(f"Utenti skippati:                     {skipped_users}")
    print(f"Righe totali teoriche max:           {n_written * TOPK}")
    print(f"Utenti senza output Gemini:          {missing_batch}")
    print(f"Utenti con zero match ID:            {zero_matches}")
    print(f"Utenti con match parziale (<{TOPK}): {partial_matches}")
    print(f"Utenti con fallback prompt:          {fallback_used}")
    print(f"Invalid item_id da Gemini:           {invalid_ids}")
    print(f"Non-candidate item_id da Gemini:     {non_candidate_ids}")
    print(f"Avg match Gemini/utente:             {total_matches / max(n_written, 1):.2f}")
    if match_counts:
        print(f"Match min/max:                       {min(match_counts)}/{max(match_counts)}")
    print("\nCSV compatibile con eval_definitivo_ml1m_dmf.py (inner RecBole IDs)")
    print("Nessun item fuori dalla top50 DMF può entrare nel CSV finale.")


if __name__ == "__main__":
    main()