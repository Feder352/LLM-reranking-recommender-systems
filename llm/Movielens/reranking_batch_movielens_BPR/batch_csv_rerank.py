import ast
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from recbole.quick_start import load_data_and_model

# ==========================================================
# Rebuild reranked_recbole_inner.csv in a FAIR way:
# - candidate set source of truth = BPR_top50_seed2020.csv
# - reranked items must be a subset/permutation of each user's BPR top50
# - NO fallback outside candidate set
# - if Gemini returns too few valid items, complete with remaining BPR items
#
# Output CSV format:
#   user_id,item_id,score
# where both user_id and item_id are RecBole INNER ids.
# ==========================================================

# -------------------------
# CONFIG
# -------------------------
BPR_CANDIDATES_CSV = Path("BPR_top50_seed2020.csv")
BATCH_RESULTS_JSONL = Path("batch_results_movielens_rerank.jsonl")
ITEM_FILE = Path("ml-1m.item")
CKPT_PATH = Path("BPR-Jan-28-2026_20-29-07.pth")
OUT_RECS_CSV = Path("reranked_recbole_inner_fixed.csv")

TOPK = 10
STRICT_REQUIRE_TOP50 = True  # if True, each user must have exactly 50 candidates in BPR csv


# -------------------------
# Parsing helpers
# -------------------------

def extract_user_id_from_filename(stem: str) -> Optional[int]:
    m = re.search(r"prompt_user_(\d+)$", stem)
    return int(m.group(1)) if m else None


def extract_text_from_batch_result(obj: dict) -> str:
    resp = obj.get("response") or {}
    candidates = resp.get("candidates") or []
    if candidates:
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts = [p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
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

    out: List[str] = []

    # JSON-ish item_name fields
    names = re.findall(r'"item_name"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    for n in names:
        try:
            out.append(bytes(n, "utf-8").decode("unicode_escape").strip())
        except Exception:
            out.append(n.strip())

    if out:
        return [x for x in out if x]

    # Fallback: numbered or bulleted lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned: List[str] = []
    for ln in lines:
        ln = re.sub(r"^[-*•]\s*", "", ln)
        ln = re.sub(r"^\d+[\.)]\s*", "", ln)
        ln = ln.strip().strip('"\'` ')
        if ln and len(ln) <= 200:
            cleaned.append(ln)
    return cleaned


def normalize_title(t: str) -> str:
    t = (t or "").strip().strip('"\'` ')
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t.casefold()


def movielens_article_variants(title: str) -> List[str]:
    title = (title or "").strip()
    variants = {title}
    m = re.match(r"^(.*),\s*(The|A|An)\s*$", title)
    if m:
        variants.add(f"{m.group(2)} {m.group(1)}")
    return list(variants)


# -------------------------
# Item metadata / mapping
# -------------------------

def load_item_metadata(item_path: Path) -> Tuple[Dict[int, str], Dict[int, Set[str]], Dict[str, int]]:
    """
    Returns:
      item_id_to_canonical_title: ML item_id -> canonical display title
      item_id_to_norm_variants:   ML item_id -> set(normalized acceptable variants)
      global_norm_title_to_id:    normalized title -> ML item_id
    """
    item_id_to_title: Dict[int, str] = {}
    item_id_to_variants: Dict[int, Set[str]] = {}
    global_title_to_id: Dict[str, int] = {}

    with item_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_id = row.get("item_id:token") or row.get("item_id") or row.get("item_id:float")
            if raw_id is None:
                continue
            try:
                item_id = int(raw_id)
            except ValueError:
                continue

            title = (row.get("movie_title:token_seq") or row.get("movie_title") or row.get("item_name") or "").strip()
            year = (row.get("release_year:token") or row.get("release_year") or "").strip()
            if not title:
                continue

            item_id_to_title[item_id] = title
            variants: Set[str] = set()
            for v in movielens_article_variants(title):
                key = normalize_title(v)
                if key:
                    variants.add(key)
                    global_title_to_id.setdefault(key, item_id)
                if year:
                    keyy = normalize_title(f"{v} ({year})")
                    if keyy:
                        variants.add(keyy)
                        global_title_to_id.setdefault(keyy, item_id)

            item_id_to_variants[item_id] = variants

    return item_id_to_title, item_id_to_variants, global_title_to_id


def item_name_to_candidate_id(
    item_name: str,
    candidate_local_map: Dict[str, int],
    global_title_to_id: Dict[str, int],
    candidate_set: Set[int],
) -> Optional[int]:
    if not item_name:
        return None

    s = item_name.strip()
    m = re.fullmatch(r"ITEM_(\d+)", s)
    if m:
        iid = int(m.group(1))
        return iid if iid in candidate_set else None

    key = normalize_title(s)
    if not key:
        return None

    iid = candidate_local_map.get(key)
    if iid is not None:
        return iid

    iid = global_title_to_id.get(key)
    if iid is not None and iid in candidate_set:
        return iid

    return None


# -------------------------
# BPR candidates / Gemini outputs
# -------------------------

def load_bpr_candidates(path: Path) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = int(row["user_id"])
            recs = ast.literal_eval(row["recs"])
            recs = [int(x) for x in recs]
            if STRICT_REQUIRE_TOP50 and len(recs) != 50:
                raise RuntimeError(f"User {uid} has {len(recs)} candidates in {path}, expected 50")
            out[uid] = recs
    return out


def load_rerank_titles(batch_results_jsonl: Path) -> Dict[int, List[str]]:
    rerank_titles_by_uid: Dict[int, List[str]] = {}

    with batch_results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("key", "")
            uid = extract_user_id_from_filename(key)
            if uid is None:
                continue

            # Some previous runs had prompt_user_0... / off-by-one issues.
            if uid == 0:
                uid = 1

            text = extract_text_from_batch_result(obj)
            parsed = parse_json_list(text)

            titles: List[str] = []
            if parsed is not None:
                for it in parsed:
                    if isinstance(it, dict):
                        name = (it.get("item_name") or "").strip()
                        if name:
                            titles.append(name)
            else:
                titles = salvage_item_names(text)

            seen = set()
            clean: List[str] = []
            for t in titles:
                k = normalize_title(t)
                if not k or k in seen:
                    continue
                seen.add(k)
                clean.append(t.strip())

            rerank_titles_by_uid[uid] = clean

    return rerank_titles_by_uid


# -------------------------
# Main rebuild
# -------------------------

def main():
    if not BPR_CANDIDATES_CSV.exists():
        raise RuntimeError(f"BPR candidates file not found: {BPR_CANDIDATES_CSV.resolve()}")
    if not BATCH_RESULTS_JSONL.exists():
        raise RuntimeError(f"Batch results file not found: {BATCH_RESULTS_JSONL.resolve()}")
    if not ITEM_FILE.exists():
        raise RuntimeError(f"Item file not found: {ITEM_FILE.resolve()}")
    if not CKPT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found: {CKPT_PATH.resolve()}")

    print("Loading RecBole checkpoint and dataset mapping...")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=str(CKPT_PATH)
    )
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    user_inner_map = dataset.field2token_id[uid_field]
    item_inner_map = dataset.field2token_id[iid_field]

    print(f"RecBole users in mapping: {len(user_inner_map)}")
    print(f"RecBole items in mapping: {len(item_inner_map)}")

    print("Loading official BPR candidate sets...")
    candidates_by_uid = load_bpr_candidates(BPR_CANDIDATES_CSV)
    n_users = len(candidates_by_uid)
    print(f"Loaded BPR candidate lists for {n_users} users")

    print("Loading item metadata...")
    item_id_to_title, item_id_to_variants, global_title_to_id = load_item_metadata(ITEM_FILE)
    print(f"Loaded metadata for {len(item_id_to_title)} items")

    print("Loading Gemini batch outputs...")
    rerank_titles_by_uid = load_rerank_titles(BATCH_RESULTS_JSONL)
    print(f"Loaded rerank outputs for {len(rerank_titles_by_uid)} users")

    stats = {
        "users_total": 0,
        "users_written": 0,
        "users_missing_batch_output": 0,
        "users_with_zero_valid_gemini_matches": 0,
        "users_with_partial_candidate_filtering": 0,
        "users_with_short_valid_candidates": 0,
        "total_gemini_titles": 0,
        "total_valid_gemini_matches": 0,
        "sum_top10_overlap_with_baseline": 0,
    }

    OUT_RECS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RECS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "item_id", "score"])

        for uid_ml in sorted(candidates_by_uid):
            stats["users_total"] += 1

            uid_token = str(uid_ml)
            if uid_token not in user_inner_map:
                raise RuntimeError(f"User {uid_ml} not found in RecBole user mapping")
            uid_inner = int(user_inner_map[uid_token])

            candidate_ml_all = candidates_by_uid[uid_ml]
            candidate_ml_valid = [iid for iid in candidate_ml_all if str(iid) in item_inner_map]

            if len(candidate_ml_valid) < len(candidate_ml_all):
                stats["users_with_partial_candidate_filtering"] += 1
            if len(candidate_ml_valid) < TOPK:
                stats["users_with_short_valid_candidates"] += 1
                raise RuntimeError(
                    f"User {uid_ml} has only {len(candidate_ml_valid)} valid candidates in RecBole after filtering"
                )

            candidate_set = set(candidate_ml_valid)
            local_title_to_id: Dict[str, int] = {}
            for iid in candidate_ml_valid:
                for key in item_id_to_variants.get(iid, set()):
                    local_title_to_id[key] = iid

            rr_titles = rerank_titles_by_uid.get(uid_ml, [])
            if uid_ml not in rerank_titles_by_uid:
                stats["users_missing_batch_output"] += 1
            stats["total_gemini_titles"] += len(rr_titles)

            rr_ids_ml: List[int] = []
            seen_rr: Set[int] = set()
            for t in rr_titles:
                iid_ml = item_name_to_candidate_id(t, local_title_to_id, global_title_to_id, candidate_set)
                if iid_ml is None or iid_ml in seen_rr:
                    continue
                seen_rr.add(iid_ml)
                rr_ids_ml.append(iid_ml)

            stats["total_valid_gemini_matches"] += len(rr_ids_ml)
            if len(rr_ids_ml) == 0:
                stats["users_with_zero_valid_gemini_matches"] += 1

            # FAIR completion: only with remaining items from the same BPR top50
            final_ids_ml: List[int] = []
            used: Set[int] = set()

            for iid_ml in rr_ids_ml:
                if len(final_ids_ml) >= TOPK:
                    break
                if iid_ml in used:
                    continue
                used.add(iid_ml)
                final_ids_ml.append(iid_ml)

            for iid_ml in candidate_ml_valid:
                if len(final_ids_ml) >= TOPK:
                    break
                if iid_ml in used:
                    continue
                used.add(iid_ml)
                final_ids_ml.append(iid_ml)

            if len(final_ids_ml) != TOPK:
                raise RuntimeError(f"User {uid_ml}: final top{TOPK} has {len(final_ids_ml)} items instead of {TOPK}")

            if not set(final_ids_ml).issubset(candidate_set):
                raise RuntimeError(f"User {uid_ml}: final reranked list contains items outside candidate set")

            baseline_topk = candidate_ml_valid[:TOPK]
            overlap = len(set(final_ids_ml).intersection(baseline_topk))
            stats["sum_top10_overlap_with_baseline"] += overlap

            for rank, iid_ml in enumerate(final_ids_ml, start=1):
                iid_inner = int(item_inner_map[str(iid_ml)])
                score = float(TOPK - rank + 1)
                w.writerow([uid_inner, iid_inner, score])

            stats["users_written"] += 1

    expected_rows = stats["users_written"] * TOPK
    avg_valid_matches = (
        stats["total_valid_gemini_matches"] / stats["users_total"] if stats["users_total"] else 0.0
    )
    avg_overlap = (
        stats["sum_top10_overlap_with_baseline"] / stats["users_total"] if stats["users_total"] else 0.0
    )

    print("\n=== DONE ===")
    print(f"Output CSV: {OUT_RECS_CSV.resolve()}")
    print(f"Users written: {stats['users_written']} / {stats['users_total']}")
    print(f"Expected rows: {expected_rows}")
    print(f"Users missing batch output: {stats['users_missing_batch_output']}")
    print(f"Users with zero valid Gemini matches: {stats['users_with_zero_valid_gemini_matches']}")
    print(f"Users with candidate filtering (<50 valid in RecBole): {stats['users_with_partial_candidate_filtering']}")
    print(f"Average valid Gemini matches per user: {avg_valid_matches:.3f}")
    print(f"Average top{TOPK} overlap with baseline top{TOPK}: {avg_overlap:.3f}")
    print("Guarantees:")
    print("  - every final item belongs to the user's BPR top50")
    print("  - no global catalog fallback")
    print("  - comparison with baseline is experimentally fair")


if __name__ == "__main__":
    main()