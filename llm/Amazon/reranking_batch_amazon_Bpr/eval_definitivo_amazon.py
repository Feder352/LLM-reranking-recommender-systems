"""
eval_amazon_bpr.py
==================
Script di valutazione per BPR su Amazon_Books_cut500.
Adattato da eval_definitivo.py (versione ml-1m).

Baseline attesa dal benchmark (BPR-Jan-27-2026_18-10-05.pth):
  ndcg@5=0.0930   ndcg@10=0.0843
  recall@5=0.0114  recall@10=0.0189
  precision@5=0.0880  precision@10=0.0767
  averagepopularity@5=45.8421  averagepopularity@10=41.4982
  giniindex@5=0.9863  giniindex@10=0.9782
  shannonentropy@5=0.0028  shannonentropy@10=0.0018

COME USARE:
  python eval_amazon_bpr.py
  (assicurarsi che BPR-Jan-27-2026_18-10-05.pth e
   reranked_amazon_top10_recbole_like.csv siano nella stessa cartella)
"""

import csv
import copy
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.utils import get_trainer
from recbole.evaluator import Evaluator
from recbole.evaluator.collector import DataStruct


# ===================== CONFIG =====================
CKPT_PATH    = Path("BPR-Jan-27-2026_18-10-05.pth")
RERANKED_CSV = Path("reranked_amazon_secondo.csv")
TOPKS        = [5, 10]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_CSV      = Path("metrics_amazon_bpr.csv")


# ===================== Step 1: Ground truth corretta =====================

def extract_ground_truth_from_recbole(test_data, model, dataset, device):
    """
    Esegue il loop interno di RecBole su test_data e raccoglie
    la ground truth ESATTA (positive_u, positive_i) che RecBole usa
    per calcolare le metriche. Garantisce coerenza col benchmark.
    """
    uid_field = dataset.uid_field
    n_items   = dataset.item_num

    gt = defaultdict(set)

    model.eval()
    with torch.no_grad():
        for batched_data in test_data:
            interaction, history_index, positive_u, positive_i = batched_data

            batch_uids = interaction[uid_field].cpu().numpy()
            pos_u_np   = positive_u.cpu().numpy()
            pos_i_np   = positive_i.cpu().numpy()

            for row_idx, iid in zip(pos_u_np, pos_i_np):
                uid = int(batch_uids[row_idx])
                gt[uid].add(int(iid))

    print(f"Ground truth estratta: {len(gt)} utenti")
    avg_gt = np.mean([len(v) for v in gt.values()]) if gt else 0.0
    print(f"Positivi per utente (media): {avg_gt:.2f}")
    return gt, n_items


# ===================== Step 2: Baseline con Trainer =====================

def evaluate_baseline(config, model, train_data, test_data, topks, metrics):
    """
    Usa Trainer.evaluate() di RecBole direttamente.
    Produce metriche IDENTICHE al benchmark originale.
    """
    cfg = copy.deepcopy(config)
    cfg["topk"]    = topks
    cfg["metrics"] = metrics

    trainer_cls = get_trainer(cfg["MODEL_TYPE"], cfg["model"])
    trainer     = trainer_cls(cfg, model)

    # FIX fondamentale: registra le statistiche necessarie alle metriche
    trainer.eval_collector.data_collect(train_data)

    result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=False
    )
    return result


# ===================== Step 3: DataStruct da CSV =====================

def read_csv_preserve_order(path: Path, topk_max: int) -> dict:
    """
    Legge CSV (user_id, item_id, score) preservando l'ordine delle righe.
    NON riordina per score: il CSV deve già essere ordinato rank1 -> rankK.
    """
    per_user = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = int(row["user_id"])
            iid = int(row["item_id"])
            per_user[uid].append(iid)
    return {uid: lst[:topk_max] for uid, lst in per_user.items()}


def build_datastruct_from_csv(user_recs: dict, ground_truth: dict,
                              train_dataset, topks: list):
    """
    Costruisce il DataStruct RecBole usando la ground truth CORRETTA.
    Solo gli utenti presenti in ENTRAMBI csv e ground_truth vengono valutati.
    """
    max_k = max(topks)

    users = [
        uid for uid in user_recs
        if uid in ground_truth
        and len(ground_truth[uid]) > 0
        and len(user_recs[uid]) >= max_k
    ]

    if not users:
        raise RuntimeError(
            f"Nessun utente comune tra CSV e ground truth con >= {max_k} item."
        )

    print(f"Utenti valutati nel DataStruct: {len(users)}")

    item_matrix  = torch.zeros((len(users), max_k), dtype=torch.long)
    pos_matrix   = torch.zeros((len(users), max_k), dtype=torch.int)
    pos_len_list = torch.zeros((len(users), 1),     dtype=torch.int)

    for row_idx, uid in enumerate(users):
        recs = user_recs[uid][:max_k]
        item_matrix[row_idx] = torch.tensor(recs, dtype=torch.long)

        gt = ground_truth[uid]
        pos_len_list[row_idx, 0] = len(gt)
        pos_matrix[row_idx] = torch.tensor(
            [1 if iid in gt else 0 for iid in recs],
            dtype=torch.int
        )

    ds = DataStruct()
    ds.set("rec.items",       item_matrix)
    ds.set("rec.topk",        torch.cat([pos_matrix, pos_len_list], dim=1))
    ds.set("data.num_items",  train_dataset.item_num)
    ds.set("data.num_users",  train_dataset.user_num)
    ds.set("data.count_items", train_dataset.item_counter)
    ds.set("data.count_users", train_dataset.user_counter)
    return ds


# ===================== Metriche creatività =====================

def build_pop_counter(inter_feat, iid_field):
    pop = Counter()
    for iid in inter_feat[iid_field].numpy():
        pop[int(iid)] += 1
    return pop


def get_embeddings(model):
    def extract(obj):
        if isinstance(obj, torch.nn.Embedding):
            return obj.weight.detach()
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        return None

    u_emb = i_emb = None

    for a in ["user_embedding", "user_embeddings", "user_embeddings_lookup"]:
        if hasattr(model, a):
            u_emb = extract(getattr(model, a))
            break

    for a in ["item_embedding", "item_embeddings", "item_embeddings_lookup", "entity_embedding"]:
        if hasattr(model, a):
            i_emb = extract(getattr(model, a))
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
        n   += 1

    return tot / n if n else 0.0


def serendipity_yan_unexpectedness(user_recs, gt, u_embs, i_embs, topk):
    if u_embs is None or i_embs is None:
        return 0.0, 0.0

    u_n = F.normalize(u_embs, p=2, dim=1).cpu()
    i_n = F.normalize(i_embs, p=2, dim=1).cpu()

    ts, tu, n = 0.0, 0.0, 0
    for uid, recs in user_recs.items():
        if uid not in gt or uid >= len(u_n):
            continue

        uv = u_n[uid]
        ss = su = 0.0

        for iid in recs[:topk]:
            if iid >= len(i_n):
                continue
            sim  = max(0.0, min(1.0, (torch.dot(uv, i_n[iid]).item() + 1.0) / 2.0))
            unex = 1.0 - sim
            su  += unex
            if iid in gt[uid]:
                ss += unex

        ts += ss / topk
        tu += su / topk
        n  += 1

    return (ts / n if n else 0.0), (tu / n if n else 0.0)


def novelty(user_recs, pop_counter, topk):
    tot, n = 0.0, 0
    for recs in user_recs.values():
        nov = sum(
            1.0 / np.log1p(pop_counter[i]) if pop_counter.get(i, 0) > 0 else 1.0
            for i in recs[:topk]
        )
        tot += nov / topk
        n   += 1
    return tot / n if n else 0.0


def item_coverage(user_recs, n_items_total, topk):
    return len({i for r in user_recs.values() for i in r[:topk]}) / n_items_total


def compute_extra_metrics(user_recs, gt, pop_counter, u_embs, i_embs, n_items, topks):
    extra = {}
    for k in topks:
        kr = {u: r[:k] for u, r in user_recs.items()}
        extra[f"Serendipity_Ge@{k}"]    = serendipity_ge(kr, gt, pop_counter, k)
        sy, un                           = serendipity_yan_unexpectedness(kr, gt, u_embs, i_embs, k)
        extra[f"Serendipity_Yan@{k}"]   = sy
        extra[f"Unexpectedness@{k}"]    = un
        extra[f"Novelty@{k}"]           = novelty(kr, pop_counter, k)
        extra[f"ItemCoverage@{k}"]      = item_coverage(user_recs, n_items, k)
    return extra


# ===================== Salvataggio CSV =====================

def save_csv(out_path, base_result, base_extra, rer_result, rer_extra):
    RECBOLE = {"ndcg", "recall", "precision", "averagepopularity", "giniindex", "shannonentropy"}

    def sort_key(s):
        base = s.split("@")[0] if "@" in s else s
        k    = int(s.split("@")[1]) if "@" in s else 999
        return (0 if base in RECBOLE else 1, base, k)

    all_keys = sorted(
        set(list(base_result) + list(base_extra) + list(rer_result) + list(rer_extra)),
        key=sort_key
    )

    rows = []
    for m in all_keys:
        b = float(base_result.get(m, base_extra.get(m, float("nan"))))
        r = float(rer_result.get(m, rer_extra.get(m, float("nan"))))
        rows.append({
            "metric":   m,
            "baseline": b,
            "reranked": r,
            "delta":    r - b if not (np.isnan(b) or np.isnan(r)) else float("nan")
        })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n✅ CSV salvato: {out_path.resolve()}")


# ===================== MAIN =====================

def main():
    for p, label in [(CKPT_PATH, "Checkpoint"), (RERANKED_CSV, "Reranked CSV")]:
        if not p.exists():
            raise RuntimeError(f"{label} non trovato: {p.resolve()}")

    print(f"Carico checkpoint: {CKPT_PATH.name}")
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=str(CKPT_PATH)
    )
    model = model.to(DEVICE)
    model.eval()

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    # Config separato per Evaluator custom (solo per reranked)
    eval_config = {
        "topk":                 TOPKS,
        "metrics":              ["ndcg", "recall", "precision",
                                 "averagepopularity", "giniindex", "shannonentropy"],
        "metric_decimal_place": 6,
    }

    # ─────────────────────────────────────────
    # STEP 1: Metriche BASELINE con Trainer
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 — BASELINE (Trainer.evaluate, uguale al benchmark)")
    print("=" * 60)

    base_result = evaluate_baseline(
        config  = config,
        model   = model,
        train_data = train_data,
        test_data  = test_data,
        topks   = TOPKS,
        metrics = ["ndcg", "recall", "precision",
                   "averagepopularity", "giniindex", "shannonentropy"]
    )

    print("\nMetriche baseline calcolate:")
    for k in sorted(base_result):
        print(f"  {k}: {base_result[k]:.6f}")

    print("\nBenchmark atteso (BPR Amazon_Books_cut500):")
    print("  ndcg@5=0.093000    ndcg@10=0.084300")
    print("  recall@5=0.011400  recall@10=0.018900")
    print("  precision@5=0.088000  precision@10=0.076700")
    print("  averagepopularity@5=45.842100  averagepopularity@10=41.498200")
    print("  giniindex@5=0.986300  giniindex@10=0.978200")
    print("  shannonentropy@5=0.002800  shannonentropy@10=0.001800")

    # Verifica automatica corrispondenza benchmark
    benchmark = {
        "ndcg@5": 0.0930, "ndcg@10": 0.0843,
        "recall@5": 0.0114, "recall@10": 0.0189,
        "precision@5": 0.0880, "precision@10": 0.0767,
    }
    print("\nCheck vs benchmark (tolleranza 0.001):")
    all_ok = True
    for key, expected in benchmark.items():
        got  = base_result.get(key, float("nan"))
        diff = abs(got - expected)
        ok   = "✅" if diff <= 0.001 else "❌"
        if diff > 0.001:
            all_ok = False
        print(f"  {ok} {key}: got={got:.6f}  expected={expected:.4f}  diff={diff:.6f}")
    if all_ok:
        print("  → Baseline CONFERMATA ✅")
    else:
        print("  → ATTENZIONE: alcune metriche si discostano dal benchmark ❌")

    # ─────────────────────────────────────────
    # STEP 2: Estrai ground truth CORRETTA
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Estrazione ground truth corretta da RecBole")
    print("=" * 60)

    gt, n_items = extract_ground_truth_from_recbole(test_data, model, dataset, DEVICE)

    # ─────────────────────────────────────────
    # STEP 3: Metriche RERANKED
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — RERANKED (DataStruct con ground truth corretta)")
    print("=" * 60)

    user_recs = read_csv_preserve_order(RERANKED_CSV, topk_max=max(TOPKS))
    print(f"Utenti nel CSV reranked: {len(user_recs)}")

    ds = build_datastruct_from_csv(user_recs, gt, train_data.dataset, TOPKS)

    evaluator  = Evaluator(eval_config)
    rer_result = evaluator.evaluate(ds)

    print("\nMetriche reranked:")
    for k in sorted(rer_result):
        print(f"  {k}: {rer_result[k]:.6f}")

    # ─────────────────────────────────────────
    # STEP 4: Metriche creatività per entrambi
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Metriche creatività")
    print("=" * 60)

    pop_counter = build_pop_counter(train_data.dataset.inter_feat, iid_field)
    u_embs, i_embs = get_embeddings(model)

    # Baseline recommendations dal full-sort
    baseline_recs = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batched_data in test_data:
            interaction, history_index, positive_u, positive_i = batched_data
            interaction = interaction.to(DEVICE)

            scores = model.full_sort_predict(interaction).view(-1, n_items)
            scores[:, 0] = -np.inf  # padding item

            if history_index is not None:
                scores[history_index] = -np.inf

            _, topk_items = torch.topk(scores, max(TOPKS), dim=-1)
            batch_uids   = interaction[uid_field].cpu().numpy()

            for row_idx, uid in enumerate(batch_uids):
                baseline_recs[int(uid)] = topk_items[row_idx].cpu().numpy().tolist()

    base_extra = compute_extra_metrics(
        baseline_recs, gt, pop_counter, u_embs, i_embs, n_items, TOPKS
    )
    rer_extra = compute_extra_metrics(
        user_recs, gt, pop_counter, u_embs, i_embs, n_items, TOPKS
    )

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

    # ─────────────────────────────────────────
    # STEP 5: Delta e salvataggio
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DELTA (reranked - baseline)")
    print("=" * 60)

    for k in sorted(base_result):
        if k in rer_result:
            print(f"  {k}: {rer_result[k] - base_result[k]:+.6f}")

    save_csv(OUT_CSV, base_result, base_extra, rer_result, rer_extra)


if __name__ == "__main__":
    main()