# Fix compatibilità scipy / RecBole
try:
    import scipy.sparse as _sp
    if not hasattr(_sp.dok_matrix, "_update"):
        def _dok_update(self, data_dict):
            for k, v in data_dict.items():
                self[k] = v
            return self
        _sp.dok_matrix._update = _dok_update
except Exception:
    pass

import json
import ast
import argparse
from pathlib import Path
import zipfile

import pandas as pd
from recbole.quick_start import load_data_and_model

BASE_PROMPT_TEMPLATE = """
# Role and Task

You are an expert recommender system whose goal is to rank the creativity of recommendations.

Given a user's historical interactions and a list of recommended items, your task is to re-rank the items by placing the most creative items first, according to the definition of creativity provided below.

Creativity must always be evaluated relative to the user's history.

# User Profile

The user profile is represented by the user's historical interactions that includes each previously liked item.

User History

{user_history_json}

#Definition of Creativity

Creativity is defined as a balanced combination of unexpectedness, relevance, and novelty.

•Unexpectedness: The degree to which a candidate item deviates from what would typically be recommended to the user based on their historical preferences.

•Relevance: The degree to which the candidate item aligns with the user's interests inferred from their historical data.

•Novelty: The degree to which the candidate item introduces new or not popular items compared to the user's past interactions.

Use this definition of creativity consistently when evaluating all items.

#List of Recommended Items

You are given a list of items to re-rank:

{candidate_items_json}

#Instructions

1. Assess the creativity of each item according to the provided definition.
2. Re-rank the list from most creative to least creative.

Items that are highly relevant but predictable should rank lower than items that achieve a better balance between relevance, novelty, and unexpectedness.

#Output Format

Return only the re-ranked list, without explanations, using the following format:

[
  {{
    "rank": 1,
    "item_id": 123,
    "item_name": "..."
  }},
  {{
    "rank": 2,
    "item_id": 456,
    "item_name": "..."
  }}
]

IMPORTANT:
- Return exactly {k_candidates} items (rank 1..{k_candidates}).
- item_id MUST be copied exactly from the candidate list above.
- Do not invent new item_id values.
- Output must be ONLY the JSON list. No extra text.
"""

TITLE_CANDIDATES = ["movie_title", "title", "item_title", "product_title", "name", "item_name"]
TIME_CANDIDATES = ["timestamp", "time", "datetime"]
RATING_CANDIDATES = ["rating", "score", "label"]


def strip_recbole_types(cols):
    return [c.split(":")[0] for c in cols]


def safe_user_id(uid) -> str:
    return str(uid).replace("/", "_").replace("\\", "_").replace(" ", "_")


def find_first_existing(columns, candidates, required=False, what="column"):
    for c in candidates:
        if c in columns:
            return c
    if required:
        raise ValueError(f"Impossibile trovare {what}. Colonne disponibili: {list(columns)}")
    return None


def parse_rec_list(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    if pd.isna(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [int(x) for x in parsed]
        raise ValueError(f"Formato recs non valido: {value[:100]}")
    raise ValueError(f"Formato recs non supportato: {type(value)}")


def load_recbole_mappings(ckpt_path: str):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=ckpt_path)
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    return {
        "dataset": dataset,
        "uid_field": uid_field,
        "iid_field": iid_field,
        "inner2user": dataset.field2id_token[uid_field],
        "inner2item": dataset.field2id_token[iid_field],
    }


def main():
    ap = argparse.ArgumentParser()
    # KGCN usa il dataset KG per le interazioni utente-item
    ap.add_argument("--inter_path", default="MovieLens-KG.inter")
    # I titoli però li leggiamo dal file MovieLens standard
    ap.add_argument("--item_path", default="ml-1m.item")
    ap.add_argument("--kgcn_recs_path", default="KGCN_top50_seed2020.csv")
    ap.add_argument("--ckpt_path", default="KGCN-Jan-29-2026_15-38-49.pth")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--zip_path", default="")
    ap.add_argument("--k_candidates", type=int, default=25)
    ap.add_argument("--max_hist", type=int, default=30)
    ap.add_argument("--min_pos_rating", type=float, default=4.0)
    ap.add_argument("--title_col", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inter = pd.read_csv(args.inter_path, sep="\t")
    inter.columns = strip_recbole_types(inter.columns)

    items = pd.read_csv(args.item_path, sep="\t")
    items.columns = strip_recbole_types(items.columns)

    kgcn = pd.read_csv(args.kgcn_recs_path)

    needed_inter = {"user_id", "item_id"}
    needed_item = {"item_id"}
    needed_recs = {"user_id", "recs"}

    if not needed_inter.issubset(inter.columns):
        raise ValueError(f"{args.inter_path} colonne mancanti: {list(inter.columns)}")
    if not needed_item.issubset(items.columns):
        raise ValueError(f"{args.item_path} colonne mancanti: {list(items.columns)}")
    if not needed_recs.issubset(kgcn.columns):
        raise ValueError(f"{args.kgcn_recs_path} colonne mancanti: {list(kgcn.columns)}")

    title_col = args.title_col.strip() or find_first_existing(
        items.columns, TITLE_CANDIDATES, required=True, what="la colonna titolo"
    )
    time_col = find_first_existing(inter.columns, TIME_CANDIDATES, required=False)
    rating_col = find_first_existing(inter.columns, RATING_CANDIDATES, required=False)

    print(f"Colonna titolo rilevata: {title_col}")
    print(f"Colonna timestamp rilevata: {time_col}")
    print(f"Colonna rating rilevata: {rating_col}")

    print(f"Carico mapping dal checkpoint: {args.ckpt_path}")
    maps = load_recbole_mappings(args.ckpt_path)
    inner2user = maps["inner2user"]
    inner2item = maps["inner2item"]

    # MovieLens standard: item_id raw numerico, ma trattato come stringa per uniformità
    items["item_id"] = items["item_id"].astype(str)
    item_token_to_title = dict(zip(items["item_id"], items[title_col].astype(str)))

    # Dataset KG: history utente-item da MovieLens-KG.inter
    inter_hist = inter.copy()
    inter_hist["user_id"] = inter_hist["user_id"].astype(str)
    inter_hist["item_id"] = inter_hist["item_id"].astype(str)

    if rating_col is not None and args.min_pos_rating is not None:
        inter_hist[rating_col] = pd.to_numeric(inter_hist[rating_col], errors="coerce")
        inter_hist = inter_hist[inter_hist[rating_col] >= args.min_pos_rating].copy()
        print(f"Uso solo interazioni positive con {rating_col} >= {args.min_pos_rating}")
    else:
        print("Uso tutte le interazioni presenti come history positiva.")

    if time_col is not None:
        inter_hist[time_col] = pd.to_numeric(inter_hist[time_col], errors="coerce")
        inter_hist.sort_values(["user_id", time_col], inplace=True)
    else:
        inter_hist = inter_hist.reset_index(drop=False).rename(columns={"index": "__row_order__"})
        inter_hist.sort_values(["user_id", "__row_order__"], inplace=True)

    user_token_to_hist_items = {
        uid: df["item_id"].astype(str).tolist()
        for uid, df in inter_hist.groupby("user_id")
    }

    user_to_recs = {}
    bad_len = []
    for _, row in kgcn.iterrows():
        uid_inner = int(row["user_id"])
        rec_list_inner = parse_rec_list(row["recs"])
        user_to_recs[uid_inner] = rec_list_inner
        if len(rec_list_inner) < args.k_candidates:
            bad_len.append((uid_inner, len(rec_list_inner)))

    if bad_len:
        raise RuntimeError(
            f"Alcuni utenti hanno meno di {args.k_candidates} raccomandazioni. Esempi: {bad_len[:10]}"
        )

    make_zip = bool(args.zip_path.strip())
    zip_path = Path(args.zip_path) if make_zip else None
    zipf = zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) if make_zip else None

    written = 0
    empty_hist = 0
    missing_titles_hist = 0
    missing_titles_cand = 0
    missing_user_mapping = 0
    missing_item_mapping = 0

    try:
        for uid_inner in sorted(user_to_recs.keys()):
            if uid_inner >= len(inner2user):
                missing_user_mapping += 1
                continue

            user_token = inner2user[uid_inner]
            if user_token == "[PAD]":
                missing_user_mapping += 1
                continue

            hist_item_tokens = user_token_to_hist_items.get(str(user_token), [])
            if not hist_item_tokens:
                empty_hist += 1

            hist_titles = []
            for item_token in hist_item_tokens[-args.max_hist:]:
                title = item_token_to_title.get(str(item_token))
                if title is None or title == "nan":
                    missing_titles_hist += 1
                    title = f"ITEM_{item_token}"
                hist_titles.append(title)

            user_history = [{"item_name": t} for t in hist_titles]

            rec_inner_ids = user_to_recs[uid_inner][:args.k_candidates]
            candidates = []
            for rank, iid_inner in enumerate(rec_inner_ids, start=1):
                if iid_inner >= len(inner2item):
                    missing_item_mapping += 1
                    title = f"ITEM_INNER_{iid_inner}"
                else:
                    item_token = inner2item[iid_inner]
                    if item_token == "[PAD]":
                        missing_item_mapping += 1
                        title = f"ITEM_INNER_{iid_inner}"
                    else:
                        title = item_token_to_title.get(str(item_token))
                        if title is None or title == "nan":
                            missing_titles_cand += 1
                            title = f"ITEM_{item_token}"

                candidates.append({
                    "rank": rank,
                    "item_id": int(iid_inner),
                    "item_name": title
                })

            prompt = BASE_PROMPT_TEMPLATE.format(
                user_history_json=json.dumps(user_history, ensure_ascii=False, indent=2),
                candidate_items_json=json.dumps(candidates, ensure_ascii=False, indent=2),
                k_candidates=args.k_candidates,
            )

            fname = f"prompt_user_{safe_user_id(uid_inner)}.txt"
            fpath = out_dir / fname
            fpath.write_text(prompt, encoding="utf-8")

            if zipf is not None:
                zipf.write(fpath, arcname=fname)

            written += 1

    finally:
        if zipf is not None:
            zipf.close()

    print(f"\nOK: generati {written} prompt in {out_dir}")
    if make_zip:
        print(f"OK: creato zip in {zip_path}")
    if empty_hist:
        print(f"ATTENZIONE: {empty_hist} utenti senza history utile. Prompt con history vuota.")
    if missing_titles_hist:
        print(f"ATTENZIONE: {missing_titles_hist} item della history senza titolo nei metadata.")
    if missing_titles_cand:
        print(f"ATTENZIONE: {missing_titles_cand} item candidati senza titolo nei metadata.")
    if missing_user_mapping:
        print(f"ATTENZIONE: {missing_user_mapping} user inner id senza mapping valido.")
    if missing_item_mapping:
        print(f"ATTENZIONE: {missing_item_mapping} item inner id senza mapping valido.")
    print("Done.")


if __name__ == "__main__":
    main()