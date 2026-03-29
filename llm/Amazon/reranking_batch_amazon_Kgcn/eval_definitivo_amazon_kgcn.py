"""
Valutazione baseline e reranked per Amazon con KGCN.

Note importanti:
- baseline: Trainer.evaluate() di RecBole
- ground truth: positiva esatta da FullSortEvalDataLoader
- reranked CSV: atteso in inner RecBole ids (user_id, item_id, score)
- include una diagnostica sulla top50 KGCN per controllare coerenza col checkpoint
"""

# Fix compatibilità scipy
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

import ast
import copy
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from recbole.evaluator import Evaluator
from recbole.evaluator.collector import DataStruct
from recbole.quick_start import load_data_and_model
from recbole.utils import get_trainer

CKPT_PATH = Path("KGCN-Jan-27-2026_21-15-12.pth")
TOP50_CSV = Path("KGCN_top50_seed2020.csv")
RERANKED_CSV = Path("reranked_amazon_kgcn_top10_recbole_like.csv")
TOPKS = [5, 10]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_CSV = Path("metrics_amazon_kgcn.csv")


def get_train_dataset(train_data):
    if hasattr(train_data, "dataset"):
        return train_data.dataset
    if hasattr(train_data, "_dataset"):
        return train_data._dataset
    raise AttributeError("Impossibile trovare il dataset dentro train_data")


def extract_ground_truth_from_recbole(test_data, model, dataset):
    uid_field = dataset.uid_field
    n_items = dataset.item_num
    gt = defaultdict(set)

    model.eval()
    with torch.no_grad():
        for batched_data in test_data:
            interaction, history_index, positive_u, positive_i = batched_data
            batch_uids = interaction[uid_field].cpu().numpy()
            pos_u_np = positive_u.cpu().numpy()
            pos_i_np = positive_i.cpu().numpy()

            for row_idx, iid in zip(pos_u_np, pos_i_np):
                uid = int(batch_uids[row_idx])
                gt[uid].add(int(iid))

    print(f"Ground truth estratta: {len(gt)} utenti")
    avg_gt = np.mean([len(v) for v in gt.values()]) if gt else 0.0
    print(f"Positivi per utente (media): {avg_gt:.2f}")
    return gt, n_items


def evaluate_baseline(config, model, train_data, test_data, topks, metrics):
    cfg = copy.deepcopy(config)
    cfg["topk"] = topks

    # come nel file KGCN che hai usato, evitiamo averagepopularity nel passaggio baseline
    safe_metrics = [m for m in metrics if m != "averagepopularity"]
    cfg["metrics"] = safe_metrics

    trainer_cls = get_trainer(cfg["MODEL_TYPE"], cfg["model"])
    trainer = trainer_cls(cfg, model)

    if not hasattr(train_data, "dataset") and hasattr(train_data, "_dataset"):
        train_data.dataset = train_data._dataset

    trainer.eval_collector.data_collect(train_data)
    result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)

    # ricostruzione averagepopularity dal top50 benchmark-style
    # verrà poi riallineata dallo STEP 2b se la top50 è coerente
    return result


def read_csv_preserve_order(path: Path, topk_max: int):
    per_user = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = int(row["user_id"])
            iid = int(row["item_id"])
            per_user[uid].append(iid)
    return {uid: lst[:topk_max] for uid, lst in per_user.items()}


def read_top50_csv(path: Path, topk_max: int):
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        uid = int(row["user_id"])
        recs = ast.literal_eval(row["recs"]) if isinstance(row["recs"], str) else list(row["recs"])
        out[uid] = [int(x) for x in recs[:topk_max]]
    return out


def build_datastruct_from_csv(user_recs, ground_truth, train_dataset, topks):
    max_k = max(topks)
    users = [
        uid for uid in user_recs
        if uid in ground_truth and len(ground_truth[uid]) > 0 and len(user_recs[uid]) >= max_k
    ]
    if not users:
        raise RuntimeError(f"Nessun utente comune tra CSV e ground truth con >= {max_k} item.")

    print(f"Utenti valutati nel DataStruct: {len(users)}")

    item_matrix = torch.zeros((len(users), max_k), dtype=torch.long)
    pos_matrix = torch.zeros((len(users), max_k), dtype=torch.int)
    pos_len_list = torch.zeros((len(users), 1), dtype=torch.int)

    for row_idx, uid in enumerate(users):
        recs = [int(x) for x in user_recs[uid][:max_k]]
        gt = ground_truth[uid]
        item_matrix[row_idx] = torch.tensor(recs, dtype=torch.long)
        pos_matrix[row_idx] = torch.tensor([1 if iid in gt else 0 for iid in recs], dtype=torch.int)
        pos_len_list[row_idx, 0] = len(gt)

    ds = DataStruct()
    ds.set("rec.items", item_matrix)
    ds.set("rec.topk", torch.cat([pos_matrix, pos_len_list], dim=1))
    ds.set("data.num_items", train_dataset.item_num)
    ds.set("data.num_users", train_dataset.user_num)
    ds.set("data.count_items", train_dataset.item_counter)
    ds.set("data.count_users", train_dataset.user_counter)
    return ds


def build_pop_counter(inter_feat, iid_field):
    pop = Counter()
    for iid in inter_feat[iid_field].numpy():
        pop[int(iid)] += 1
    return pop


def get_embeddings(model):
    """
    Estrae embedding confrontabili per KGCN.
    In KGCN spesso gli item sono rappresentati da entity_embedding.
    """
    def extract(obj):
        if isinstance(obj, torch.nn.Embedding):
            return obj.weight.detach().cpu()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        return None

    u_emb = i_emb = None

    for a in ["user_embedding", "user_embeddings", "user_embedding_layer"]:
        if hasattr(model, a):
            u_emb = extract(getattr(model, a))
            if u_emb is not None:
                break

    for a in ["item_embedding", "item_embeddings", "entity_embedding", "entity_embedding_layer"]:
        if hasattr(model, a):
            i_emb = extract(getattr(model, a))
            if i_emb is not None:
                break

    return u_emb, i_emb


def serendipity_ge(user_recs, gt, pop_counter, topk):
    pm = set(i for i, _ in pop_counter.most_common(topk))
    tot, n = 0.0, 0
    for uid, recs in user_recs.items():
        if uid not in gt:
            continue
        hits = [i for i in recs[:topk] if i in gt[uid] and i not in pm]
        tot += len(hits) / topk
        n += 1
    return tot / n if n else 0.0


def serendipity_yan_unexpectedness(user_recs, gt, u_embs, i_embs, topk):
    if u_embs is None or i_embs is None:
        return 0.0, 0.0

    try:
        u_n = F.normalize(u_embs, p=2, dim=1).cpu()
        i_n = F.normalize(i_embs, p=2, dim=1).cpu()
    except Exception:
        return 0.0, 0.0

    if u_n.dim() != 2 or i_n.dim() != 2:
        return 0.0, 0.0
    if u_n.size(1) != i_n.size(1):
        return 0.0, 0.0

    ts, tu, n = 0.0, 0.0, 0

    for uid, recs in user_recs.items():
        if uid not in gt or uid >= len(u_n):
            continue

        uv = u_n[uid]
        ss = su = 0.0

        for iid in recs[:topk]:
            if iid >= len(i_n):
                continue

            sim = max(0.0, min(1.0, (torch.dot(uv, i_n[iid]).item() + 1.0) / 2.0))
            unex = 1.0 - sim
            su += unex
            if iid in gt[uid]:
                ss += unex

        ts += ss / topk
        tu += su / topk
        n += 1

    return (ts / n if n else 0.0), (tu / n if n else 0.0)


def novelty(user_recs, pop_counter, topk):
    tot, n = 0.0, 0
    for recs in user_recs.values():
        nov = sum(
            1.0 / np.log1p(pop_counter[i]) if pop_counter.get(i, 0) > 0 else 1.0
            for i in recs[:topk]
        )
        tot += nov / topk
        n += 1
    return tot / n if n else 0.0


def item_coverage(user_recs, n_items_total, topk):
    return len({i for r in user_recs.values() for i in r[:topk]}) / n_items_total


def compute_extra_metrics(user_recs, gt, pop_counter, u_embs, i_embs, n_items, topks):
    extra = {}
    for k in topks:
        kr = {u: r[:k] for u, r in user_recs.items()}
        extra[f"Serendipity_Ge@{k}"] = serendipity_ge(kr, gt, pop_counter, k)
        sy, un = serendipity_yan_unexpectedness(kr, gt, u_embs, i_embs, k)
        extra[f"Serendipity_Yan@{k}"] = sy
        extra[f"Unexpectedness@{k}"] = un
        extra[f"Novelty@{k}"] = novelty(kr, pop_counter, k)
        extra[f"ItemCoverage@{k}"] = item_coverage(user_recs, n_items, k)
    return extra


def save_csv(out_path, base_result, base_extra, rer_result, rer_extra):
    recbole_metrics = {"ndcg", "recall", "precision", "averagepopularity", "giniindex", "shannonentropy"}

    def sort_key(s):
        base = s.split("@")[0] if "@" in s else s
        k = int(s.split("@")[1]) if "@" in s else 999
        return (0 if base.lower() in recbole_metrics else 1, base.lower(), k)

    all_keys = sorted(
        set(list(base_result) + list(base_extra) + list(rer_result) + list(rer_extra)),
        key=sort_key
    )

    rows = []
    for m in all_keys:
        b = float(base_result.get(m, base_extra.get(m, float("nan"))))
        r = float(rer_result.get(m, rer_extra.get(m, float("nan"))))
        rows.append({
            "metric": m,
            "baseline": b,
            "reranked": r,
            "delta": r - b if not (np.isnan(b) or np.isnan(r)) else float("nan")
        })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n✅ CSV salvato: {out_path.resolve()}")


def main():
    for p, label in [(CKPT_PATH, "Checkpoint"), (TOP50_CSV, "Top50 CSV"), (RERANKED_CSV, "Reranked CSV")]:
        if not p.exists():
            raise RuntimeError(f"{label} non trovato: {p.resolve()}")

    print(f"Carico checkpoint: {CKPT_PATH.name}")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=str(CKPT_PATH))
    model = model.to(DEVICE)
    model.eval()

    train_dataset = get_train_dataset(train_data)

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    eval_config = {
        "topk": TOPKS,
        "metrics": ["ndcg", "recall", "precision", "averagepopularity", "giniindex", "shannonentropy"],
        "metric_decimal_place": 6,
    }

    print("\n" + "=" * 60)
    print("STEP 1 — BASELINE (Trainer.evaluate)")
    print("=" * 60)
    base_result = evaluate_baseline(
        config, model, train_data, test_data, TOPKS,
        ["ndcg", "recall", "precision", "averagepopularity", "giniindex", "shannonentropy"]
    )

    # averagepopularity baseline calcolata dal top50 benchmark-style se disponibile
    top50_for_ap = read_top50_csv(TOP50_CSV, topk_max=max(TOPKS))
    ds_top50_tmp = build_datastruct_from_csv(top50_for_ap, defaultdict(set), train_dataset, TOPKS) if False else None

    print("\nMetriche baseline:")
    for k in sorted(base_result):
        print(f"  {k}: {base_result[k]:.6f}")

    print("\n" + "=" * 60)
    print("STEP 2 — Estrazione ground truth corretta da RecBole")
    print("=" * 60)
    gt, n_items = extract_ground_truth_from_recbole(test_data, model, dataset)

    print("\n" + "=" * 60)
    print("STEP 2b — DIAGNOSTICA top50 KGCN")
    print("=" * 60)
    top50_recs = read_top50_csv(TOP50_CSV, topk_max=max(TOPKS))
    ds_top50 = build_datastruct_from_csv(top50_recs, gt, train_dataset, TOPKS)
    diag_result = Evaluator(eval_config).evaluate(ds_top50)

    # aggiungiamo averagepopularity dalla diagnostica top50, che è benchmark-style
    if "averagepopularity@5" in diag_result:
        base_result["averagepopularity@5"] = diag_result["averagepopularity@5"]
    if "averagepopularity@10" in diag_result:
        base_result["averagepopularity@10"] = diag_result["averagepopularity@10"]

    print("\nMetriche top10 dal CSV top50:")
    for k in sorted(diag_result):
        print(f"  {k}: {diag_result[k]:.6f}")

    if "ndcg@10" in base_result and "ndcg@10" in diag_result:
        diff = abs(diag_result["ndcg@10"] - base_result["ndcg@10"])
        print(f"\nConfronto ndcg@10 baseline vs top50 CSV: diff={diff:.6f}")
        if diff <= 0.02:
            print("  ✅ Top50 coerente col checkpoint")
        else:
            print("  ❌ Top50 molto diversa dalla baseline: controlla mapping/checkpoint")

    print("\n" + "=" * 60)
    print("STEP 3 — RERANKED")
    print("=" * 60)
    user_recs = read_csv_preserve_order(RERANKED_CSV, topk_max=max(TOPKS))
    print(f"Utenti nel CSV reranked: {len(user_recs)}")
    ds = build_datastruct_from_csv(user_recs, gt, train_dataset, TOPKS)
    rer_result = Evaluator(eval_config).evaluate(ds)

    # averagepopularity reranked dal DataStruct con evaluator separato
    eval_config_ap = {
        "topk": TOPKS,
        "metrics": ["averagepopularity"],
        "metric_decimal_place": 6,
    }
    rer_ap = Evaluator(eval_config_ap).evaluate(ds)
    rer_result.update(rer_ap)

    print("\nMetriche reranked:")
    for k in sorted(rer_result):
        print(f"  {k}: {rer_result[k]:.6f}")

    print("\n" + "=" * 60)
    print("STEP 4 — Metriche creatività")
    print("=" * 60)
    pop_counter = build_pop_counter(train_dataset.inter_feat, iid_field)
    u_embs, i_embs = get_embeddings(model)

    baseline_recs = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batched_data in test_data:
            interaction, history_index, positive_u, positive_i = batched_data
            interaction = interaction.to(DEVICE)
            scores = model.full_sort_predict(interaction).view(-1, n_items)
            scores[:, 0] = -np.inf
            if history_index is not None:
                scores[history_index] = -np.inf
            _, topk_items = torch.topk(scores, max(TOPKS), dim=-1)
            batch_uids = interaction[uid_field].cpu().numpy()
            for row_idx, uid in enumerate(batch_uids):
                baseline_recs[int(uid)] = topk_items[row_idx].cpu().numpy().tolist()

    base_extra = compute_extra_metrics(baseline_recs, gt, pop_counter, u_embs, i_embs, n_items, TOPKS)
    rer_extra = compute_extra_metrics(user_recs, gt, pop_counter, u_embs, i_embs, n_items, TOPKS)

    print("\nMetriche creatività BASELINE:")
    for k in TOPKS:
        print(
            f"  @{k}: Ser_Ge={base_extra[f'Serendipity_Ge@{k}']:.6f}  "
            f"Ser_Yan={base_extra[f'Serendipity_Yan@{k}']:.6f}  "
            f"Unexp={base_extra[f'Unexpectedness@{k}']:.6f}  "
            f"Nov={base_extra[f'Novelty@{k}']:.6f}  "
            f"Cov={base_extra[f'ItemCoverage@{k}']:.6f}"
        )

    print("\nMetriche creatività RERANKED:")
    for k in TOPKS:
        print(
            f"  @{k}: Ser_Ge={rer_extra[f'Serendipity_Ge@{k}']:.6f}  "
            f"Ser_Yan={rer_extra[f'Serendipity_Yan@{k}']:.6f}  "
            f"Unexp={rer_extra[f'Unexpectedness@{k}']:.6f}  "
            f"Nov={rer_extra[f'Novelty@{k}']:.6f}  "
            f"Cov={rer_extra[f'ItemCoverage@{k}']:.6f}"
        )

    print("\n" + "=" * 60)
    print("DELTA (reranked - baseline)")
    print("=" * 60)
    for k in sorted(base_result):
        if k in rer_result:
            print(f"  {k}: {rer_result[k] - base_result[k]:+.6f}")

    save_csv(OUT_CSV, base_result, base_extra, rer_result, rer_extra)


if __name__ == "__main__":
    main()