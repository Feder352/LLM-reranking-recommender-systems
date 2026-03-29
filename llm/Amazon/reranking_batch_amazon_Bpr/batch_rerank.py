import os
import json
import time
import re
import csv
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from google import genai

# Se vuoi ancora usare load_data_and_model per controlli extra, puoi lasciarlo.
# In questa versione non serve per il mapping finale.
# from recbole.quick_start import load_data_and_model

# ==========================================================
# Amazon Books: prompt_user_*.txt -> Gemini Batch -> CSV
#
# VERSIONE DEFINITIVA:
#   1. I prompt hanno nome prompt_user_<INNER_UID>.txt
#   2. Il mapping finale usa BPR_top50_seed2020.csv come verità
#   3. I titoli del prompt sono associati agli item_id BPR per POSIZIONE
#   4. Nessun item fuori dalla top50 può entrare nel CSV finale
# ==========================================================

# -------------------------
# CONFIG
# -------------------------
PROMPTS_DIR         = Path("amazon_bpr_prompts")
PROMPT_GLOB         = "prompt_user_*.txt"

TOPK                = 10

BATCH_INPUT_FILE    = Path("batch_requests_amazon_rerank.jsonl")
BATCH_RESULTS_JSONL = Path("batch_results_amazon_rerank.jsonl")

BPR_RECS_FILE       = Path("BPR_top50_seed2020.csv")
OUT_RECS_CSV        = Path("reranked_amazon_top10_recbole_like.csv")

MODEL_CANDIDATES = [
    "gemini-2.0-flash",
    "models/gemini-2.0-flash",
]
GENERATION_CONFIG = {
    "max_output_tokens": 4096,
    "temperature": 0.2,
}


# -------------------------
# user_id dal nome file
# -------------------------
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


# -------------------------
# normalizzazione titoli
# -------------------------
def normalize_title(t: str) -> str:
    t = (t or "").strip().strip('"\'` ')
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t.casefold()


def title_variants(title: str) -> List[str]:
    title = (title or "").strip()
    variants = {title}

    # "Something, The" -> "The Something"
    m = re.match(r"^(.*),\s*(The|A|An)\s*$", title)
    if m:
        variants.add(f"{m.group(2)} {m.group(1)}")

    return list(variants)


# -------------------------
# parsing candidati dal prompt
# -------------------------
def extract_candidates_from_prompt(prompt_text: str) -> List[str]:
    candidates: List[str] = []

    matches = re.findall(
        r"#List of Recommended Items\s*\n.*?\n(\[.*?\])\s*\n#Instructions",
        prompt_text,
        re.DOTALL,
    )

    if matches:
        try:
            items = json.loads(matches[-1])
            for it in items:
                name = (it.get("item_name") if isinstance(it, dict) else "") or ""
                name = name.strip()
                if name:
                    candidates.append(name)
            return candidates
        except Exception:
            pass

    idx = prompt_text.rfind("#List of Recommended Items")
    if idx != -1:
        section = prompt_text[idx:]
        names = re.findall(r'"item_name"\s*:\s*"((?:[^"\\]|\\.)*)"', section)
        for n in names:
            try:
                candidates.append(bytes(n, "utf-8").decode("unicode_escape").strip())
            except Exception:
                candidates.append(n.strip())

    return [c for c in candidates if c]


# -------------------------
# batch helpers
# -------------------------
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
        ("dest", "fileName"), ("dest", "file_name"),
        ("output", "fileName"), ("output", "file_name"),
        ("output_file", "fileName"), ("output_file", "file_name"),
        ("result", "fileName"), ("result", "file_name"),
    ]:
        block = job_dict.get(a)
        if isinstance(block, dict) and block.get(b):
            return block[b]
    raw = json.dumps(job_dict)
    m = re.search(r"(files/[A-Za-z0-9_\-]+)", raw)
    if m:
        return m.group(1)
    raise RuntimeError("Non trovo fileName dell'output nel batch job.")


# -------------------------
# parsing Gemini robusto
# -------------------------
def extract_text_from_batch_result(obj: dict) -> str:
    resp = obj.get("response") or {}
    candidates = resp.get("candidates") or []
    if candidates:
        content = (candidates[0].get("content") or {})
        parts = content.get("parts") or []
        texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
        if texts:
            return "".join(texts).strip()
    if isinstance(resp.get("text"), str):
        return resp["text"].strip()

    error = resp.get("error") or obj.get("error")
    if error:
        print(f"  [WARN] Gemini error: {error}")
    return ""


def parse_json_list(text: str) -> Optional[List[dict]]:
    if not text:
        return None

    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else None
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, list) else None
        except Exception:
            return None

    return None


def salvage_item_names(text: str) -> List[str]:
    if not text:
        return []

    names = re.findall(r'"item_name"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    out = []
    for n in names:
        try:
            out.append(bytes(n, "utf-8").decode("unicode_escape").strip())
        except Exception:
            out.append(n.strip())
    if out:
        return [x for x in out if x]

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        ln = re.sub(r"^[-*•]\s*", "", ln)
        ln = re.sub(r"^\d+[\.)]\s*", "", ln)
        ln = ln.strip().strip('"\'` ')
        if ln and len(ln) <= 300:
            cleaned.append(ln)
    return cleaned


# -------------------------
# BPR top50 loader
# -------------------------
def load_bpr_recs(path: Path) -> Dict[int, List[int]]:
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


# -------------------------
# mapping locale DEFINITIVO
# -------------------------
def build_local_candidate_maps_from_prompt_and_bpr(
    candidate_titles: List[str],
    bpr_item_ids: List[int],
) -> Tuple[Dict[str, List[int]], List[int]]:
    """
    Costruisce la mappa locale usando ESATTAMENTE:
      titolo in posizione i del prompt <-> item_id in posizione i della top50 BPR
    """
    local_title_to_ids: Dict[str, List[int]] = {}
    ordered_candidate_ids: List[int] = []

    n = min(len(candidate_titles), len(bpr_item_ids))
    for title, iid in zip(candidate_titles[:n], bpr_item_ids[:n]):
        ordered_candidate_ids.append(iid)

        for v in title_variants(title):
            key = normalize_title(v)
            if not key:
                continue
            if key not in local_title_to_ids:
                local_title_to_ids[key] = []
            local_title_to_ids[key].append(iid)

    return local_title_to_ids, ordered_candidate_ids


def local_title_to_inner_id_from_bpr(
    item_name: str,
    local_title_to_ids: Dict[str, List[int]],
    used_ids: set,
) -> Optional[int]:
    if not item_name:
        return None

    key = normalize_title(item_name)
    candidates = local_title_to_ids.get(key, [])
    for iid in candidates:
        if iid not in used_ids:
            return iid
    return None


# -------------------------
# Gemini batch: submit + poll + download
# -------------------------
def run_gemini_batch(client, prompt_map: Dict[str, Path]) -> None:
    build_batch_jsonl(prompt_map, BATCH_INPUT_FILE)
    print(f"Creato input batch: {BATCH_INPUT_FILE} ({len(prompt_map)} prompt)")

    uploaded = client.files.upload(
        file=str(BATCH_INPUT_FILE),
        config={"mime_type": "application/jsonl"},
    )
    print("Upload ok. File name:", uploaded.name)

    batch_job = None
    last_err = None
    for model_name in MODEL_CANDIDATES:
        try:
            print("Provo model:", model_name)
            batch_job = client.batches.create(
                model=model_name,
                src=uploaded.name,
                config={"display_name": f"amazon-{len(prompt_map)}-users-rerank"},
            )
            print("Batch job creato:", batch_job.name)
            break
        except Exception as e:
            last_err = e
            print("Fallito:", e)

    if batch_job is None:
        raise RuntimeError(f"Impossibile creare batch. Ultimo errore: {last_err}")

    terminal = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_PAUSED",
    }

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


# -------------------------
# MAIN
# -------------------------
def main():
    for p, label in [
        (PROMPTS_DIR, "Cartella prompt"),
        (BPR_RECS_FILE, "BPR_top50_seed2020.csv"),
    ]:
        if not p.exists():
            raise RuntimeError(f"{label} non trovato: {p.resolve()}")

    prompt_map = pick_prompt_files(PROMPTS_DIR, PROMPT_GLOB)
    if not prompt_map:
        raise RuntimeError(f"Nessun prompt trovato in {PROMPTS_DIR.resolve()}")
    print(f"Prompt trovati: {len(prompt_map)} utenti")

    print("Estraggo candidati dai prompt...")
    prompt_candidates: Dict[str, List[str]] = {}
    for uid, path in prompt_map.items():
        text = path.read_text(encoding="utf-8")
        prompt_candidates[uid] = extract_candidates_from_prompt(text)

    avg_cands = sum(len(v) for v in prompt_candidates.values()) / max(len(prompt_candidates), 1)
    print(f"Candidati medi per prompt: {avg_cands:.1f}")

    print("Carico BPR top50...")
    bpr_user_to_recs = load_bpr_recs(BPR_RECS_FILE)
    print(f"Utenti nel BPR csv: {len(bpr_user_to_recs)}")

    if not BATCH_RESULTS_JSONL.exists():
        API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
        if not API_KEY:
            raise RuntimeError("Manca GEMINI_API_KEY nelle variabili d'ambiente.")
        client = genai.Client(api_key=API_KEY)
        run_gemini_batch(client, prompt_map)
    else:
        print(f"Risultati già presenti, salto Gemini: {BATCH_RESULTS_JSONL}")

    print("Parsing risultati batch...")
    rerank_titles_by_uid: Dict[str, List[str]] = {}

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

            titles: List[str] = []
            if parsed is not None:
                for it in parsed:
                    name = (it.get("item_name") if isinstance(it, dict) else "") or ""
                    name = name.strip()
                    if name:
                        titles.append(name)
            else:
                titles = salvage_item_names(text)

            seen, clean = set(), []
            for t in titles:
                k = normalize_title(t)
                if not k or k in seen:
                    continue
                seen.add(k)
                clean.append(t.strip())

            rerank_titles_by_uid[uid] = clean

    print(f"Utenti con output Gemini: {len(rerank_titles_by_uid)}")

    missing_batch = 0
    zero_matches = 0
    partial_matches = 0
    total_matches = 0
    fallback_used = 0
    skipped_users = 0
    match_counts: List[int] = []
    short_prompt_pairs = 0

    OUT_RECS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RECS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "item_id", "score"])

        for uid_str in sorted(prompt_map.keys(), key=lambda x: int(x)):
            try:
                uid_inner = int(uid_str)
            except ValueError:
                print(f"  WARN: user '{uid_str}' non è un inner user id valido, skip")
                skipped_users += 1
                continue

            candidate_titles = prompt_candidates.get(uid_str, [])
            bpr_item_ids = bpr_user_to_recs.get(uid_inner, [])

            if not bpr_item_ids:
                print(f"  WARN: user '{uid_inner}' non trovato in BPR_top50_seed2020.csv, skip")
                skipped_users += 1
                continue

            if len(candidate_titles) != len(bpr_item_ids):
                short_prompt_pairs += 1
                # Non blocchiamo: usiamo min(len(prompt), len(bpr))

            local_title_to_ids, ordered_candidate_ids = build_local_candidate_maps_from_prompt_and_bpr(
                candidate_titles,
                bpr_item_ids
            )

            rr_titles = rerank_titles_by_uid.get(uid_str, [])
            if uid_str not in rerank_titles_by_uid:
                missing_batch += 1

            rr_ids: List[int] = []
            seen_ids = set()
            for t in rr_titles:
                iid_inner = local_title_to_inner_id_from_bpr(t, local_title_to_ids, seen_ids)
                if iid_inner is None:
                    continue
                seen_ids.add(iid_inner)
                rr_ids.append(iid_inner)

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
                for iid in ordered_candidate_ids:
                    if len(final_ids) >= TOPK:
                        break
                    if iid in used:
                        continue
                    used.add(iid)
                    final_ids.append(iid)

            if len(final_ids) < TOPK:
                print(f"  WARN: user '{uid_str}' -> {len(final_ids)}/{TOPK} item dopo fallback prompt")

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
    print(f"Utenti con zero match titoli:        {zero_matches}")
    print(f"Utenti con match parziale (<{TOPK}): {partial_matches}")
    print(f"Utenti con fallback prompt:          {fallback_used}")
    print(f"Prompt/BPR con lunghezza diversa:    {short_prompt_pairs}")
    print(f"Avg match Gemini/utente:             {total_matches / max(n_written, 1):.2f}")
    if match_counts:
        print(f"Match min/max:                       {min(match_counts)}/{max(match_counts)}")
    print("\nCSV compatibile con eval_definitivo.py (inner RecBole IDs)")
    print("Nessun item fuori dalla top50 BPR può entrare nel CSV finale.")


if __name__ == "__main__":
    main()