import json
import ast
import argparse
from pathlib import Path
import zipfile

import pandas as pd


BASE_PROMPT_TEMPLATE = """Prompt for Creative Re-ranking

# Role and Task

You are an expert recommender system whose goal is to rank the creativity of recommendations.

Given a user’s historical interactions and a list of recommended items, your task is to re-rank the items by placing the most creative items first, according to the definition of creativity provided below.

Creativity must always be evaluated relative to the user’s history.

# User Profile

The user profile is represented by the user’s historical interactions that includes each previously liked item.

User History

{user_history_json}

#Definition of Creativity

Creativity is defined as a balanced combination of unexpectedness, relevance, and novelty.

•Unexpectedness: The degree to which a candidate item deviates from what would typically be recommended to the user based on their historical preferences.

•Relevance: The degree to which the candidate item aligns with the user’s interests inferred from their historical data.

•Novelty: The degree to which the candidate item introduces new or not popular items compared to the user’s past interactions.

Use this definition of creativity consistently when evaluating all items.

#List of Recommended Items

You are given a list of items to re-rank:

{candidate_items_json}

#Instructions

1.Assess the creativity of each item according to the provided definition.

2.Re-rank the list from most creative to least creative.

Items that are highly relevant but predictable should rank lower than items that achieve a better balance between relevance, novelty, and unexpectedness.

#Output Format

Return only the re-ranked list, without explanations, using the following format:

[
  {{
    "rank": 1,
    "item_name": "..."
  }},
  {{
    "rank": 2,
    "item_name": "..."
  }}
]

IMPORTANT:
- Return exactly 10 items (rank 1..10).
- Output must be ONLY the JSON list. No extra text.
"""


def strip_recbole_types(cols):
    # "user_id:token" -> "user_id"
    return [c.split(":")[0] for c in cols]


def safe_user_id(uid: int) -> str:
    return str(uid).replace("/", "_").replace("\\", "_").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--inter_path", required=True, help="ml-1m.inter (TSV RecBole)")
    ap.add_argument("--item_path", required=True, help="ml-1m.item (TSV RecBole)")
    ap.add_argument("--user_path", required=True, help="ml-1m.user (TSV RecBole) - per lista completa utenti")
    ap.add_argument("--bpr_recs_path", required=True, help="BPR_top50_seed2020.csv (user_id, recs)")

    ap.add_argument("--out_dir", required=True, help="Cartella output prompt .txt")
    ap.add_argument("--zip_path", default="", help="Se valorizzato, crea anche uno zip con tutti i prompt")
    ap.add_argument("--k_candidates", type=int, default=25, help="Quanti item candidati far re-rankare (default 25)")
    ap.add_argument("--max_hist", type=int, default=30, help="Massimo numero di item in history (default 30)")
    ap.add_argument("--min_pos_rating", type=float, default=4.0, help="Soglia positivi (default 4.0 => rating 4 e 5)")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load RecBole tsv ----------
    inter = pd.read_csv(args.inter_path, sep="\t")
    inter.columns = strip_recbole_types(inter.columns)

    items = pd.read_csv(args.item_path, sep="\t")
    items.columns = strip_recbole_types(items.columns)

    users = pd.read_csv(args.user_path, sep="\t")
    users.columns = strip_recbole_types(users.columns)

    # ---------- Load BPR recs ----------
    bpr = pd.read_csv(args.bpr_recs_path)

    # ---------- Checks ----------
    needed_inter = {"user_id", "item_id", "rating", "timestamp"}
    needed_item = {"item_id", "movie_title"}
    needed_user = {"user_id"}
    needed_bpr = {"user_id", "recs"}

    if not needed_inter.issubset(inter.columns):
        raise ValueError(f"ml-1m.inter colonne mancanti. Trovate: {list(inter.columns)}")
    if not needed_item.issubset(items.columns):
        raise ValueError(f"ml-1m.item colonne mancanti. Trovate: {list(items.columns)}")
    if not needed_user.issubset(users.columns):
        raise ValueError(f"ml-1m.user colonne mancanti. Trovate: {list(users.columns)}")
    if not needed_bpr.issubset(bpr.columns):
        raise ValueError(f"BPR_top50_seed2020.csv colonne mancanti. Trovate: {list(bpr.columns)}")

    # item_id -> title
    item_id_to_title = dict(zip(items["item_id"].astype(int), items["movie_title"].astype(str)))

    # full user list (tutti)
    all_users = users["user_id"].astype(int).unique().tolist()

    bpr_users = set(bpr["user_id"].astype(int).unique().tolist())
    missing = [u for u in all_users if u not in bpr_users]
    if missing:
        raise RuntimeError(
            f"ERRORE: BPR recs non contengono tutti gli utenti. Mancano {len(missing)} utenti (es: {missing[:10]})."
        )

    # positives
    inter["rating"] = pd.to_numeric(inter["rating"], errors="coerce")
    inter["timestamp"] = pd.to_numeric(inter["timestamp"], errors="coerce")
    inter_pos = inter[inter["rating"] >= args.min_pos_rating].copy()
    inter_pos.sort_values(["user_id", "timestamp"], inplace=True)

    pos_groups = {uid: df for uid, df in inter_pos.groupby("user_id")}

    # BPR rec dict
    user_to_recs = {}
    for _, row in bpr.iterrows():
        uid = int(row["user_id"])
        rec_list = ast.literal_eval(row["recs"]) if isinstance(row["recs"], str) else list(row["recs"])
        user_to_recs[uid] = [int(x) for x in rec_list]

    # optional zip
    make_zip = bool(args.zip_path.strip())
    zip_path = Path(args.zip_path) if make_zip else None
    zipf = zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) if make_zip else None

    written = 0
    empty_hist = 0

    try:
        for uid in all_users:
            # history (ultimi max_hist positivi)
            dfh = pos_groups.get(uid, None)
            if dfh is None or dfh.empty:
                empty_hist += 1
                hist_titles = []
            else:
                last_ids = dfh["item_id"].astype(int).tolist()[-args.max_hist:]
                hist_titles = [item_id_to_title.get(i, f"ITEM_{i}") for i in last_ids]

            user_history = [{"item_name": t} for t in hist_titles]

            # candidates (top K) con rank 1..K
            rec_ids = user_to_recs[uid][: args.k_candidates]
            cand_titles = [item_id_to_title.get(i, f"ITEM_{i}") for i in rec_ids]
            candidates = [{"rank": r, "item_name": t} for r, t in enumerate(cand_titles, start=1)]

            prompt = BASE_PROMPT_TEMPLATE.format(
                user_history_json=json.dumps(user_history, ensure_ascii=False, indent=2),
                candidate_items_json=json.dumps(candidates, ensure_ascii=False, indent=2),
            )

            fname = f"prompt_user_{safe_user_id(uid)}.txt"
            fpath = out_dir / fname
            fpath.write_text(prompt, encoding="utf-8")

            if zipf is not None:
                zipf.write(fpath, arcname=fname)

            written += 1

    finally:
        if zipf is not None:
            zipf.close()

    print(f"OK: generati {written} prompt in {out_dir}")
    if make_zip:
        print(f"OK: creato zip in {zip_path}")
    if empty_hist:
        print(f"ATTENZIONE: {empty_hist} utenti senza interazioni positive (rating >= {args.min_pos_rating}). Prompt con history vuota.")
    print("Done.")


if __name__ == "__main__":
    main()
