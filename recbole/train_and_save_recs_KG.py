import os
import sys
import gc
import json
import time
import traceback
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.utils import create_dataset
from recbole.utils import init_seed, get_model, get_trainer
from recbole.trainer import Trainer
from recbole.data.dataloader.knowledge_dataloader import KnowledgeBasedDataLoader

# ==========================================
# PATCH PER BUG RECBOLE 1.2.0
# ==========================================
# Il KnowledgeBasedDataLoader non espone l'attributo 'dataset' richiesto dal Collector
if not hasattr(KnowledgeBasedDataLoader, 'dataset'):
    print("🔧 Applicazione patch: KnowledgeBasedDataLoader.dataset")
    setattr(KnowledgeBasedDataLoader, 'dataset', property(lambda self: self._dataset))
# GPU check
if not torch.cuda.is_available():
    print("CUDA non disponibile: serve una GPU NVIDIA.")
    sys.exit(1)

# SciPy shim
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

TOPKS = [5, 10, 50, 100]
SAVE_TOPK = [50, 100]
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 10000))
RESULTS_FILE = "benchmark_train_and_recs.csv"
RECS_DIR = "saved_recs"
SEED = 2020

# Config base (adatta dataset/model qui sotto)
GLOBAL_CONFIG = {
    "metrics": ["NDCG", "Recall", "Precision", "AveragePopularity", "GiniIndex", "ShannonEntropy"],
    "valid_metric": "ndcg@10",
    "topk": TOPKS,
    "eval_setting": "TO_RS, 80_10_10",
    "device": "cuda",
    "num_workers": 0,
    "show_progress": True,
    "save_model": True,
    "seed": SEED,
    "reproducibility": True,
    "gpu_id": 0,
    "stopping_step": 3,
    "eval_step": 1,
    "load_col": {
        "inter": ["user_id", "item_id", "rating"],
        "kg": ["head_id", "relation_id", "tail_id"],
        "link": ["item_id", "entity_id"],
    },
    "threshold": {"rating": 3},
}

DATASETS = {
    "MovieLens-KG": {
        "dataset": "MovieLens-KG",
        "data_path": "./dataset",
        "train_batch_size": 256,
        "eval_batch_size": 256,
    },
    #"Amazon-book-KG-500": {
     #   "dataset": "Amazon-book-KG-500",
      #  "data_path": "./dataset",
       # "train_batch_size": 256,
        #"eval_batch_size": 256,
    #},
}

MODELS = {
    "CFKG": {
        "epochs": 20,
        "learning_rate": 0.001,
        "embedding_size": 64,
        "loss_type": "BPR",
    },
    "CKE": {
        "epochs": 20,
        "learning_rate": 0.001,
        "embedding_size": 64,
        "kg_embedding_size": 64,
    },
    "KGCN": {
        "epochs": 20,
        "learning_rate": 0.001,
        "embedding_size": 64,
        "kg_embedding_size": 64,
        "neighbor_sample_size": 4,
        "n_iter": 2,
        "aggregator_type": "sum",
    },
    "KGNNLS": {
        "epochs": 20,
        "learning_rate": 0.001,
        "embedding_size": 64,
        "kg_embedding_size": 64,
        "neighbor_sample_size": 4,
        "n_iter": 2,
        "ls_weight": 0.5,
    },
    "MKR": {
        "epochs": 15,
        "learning_rate": 0.001,
        "embedding_size": 64,
        "kg_embedding_size": 64,
        "low_layer_num": 1,
        "high_layer_num": 1,
        "use_inner_product": True,
    },
}

# Modelli con componenti KG: vanno eseguiti solo su dataset KG
KG_MODELS = {"CFKG", "CKE", "KGCN", "KGNNLS", "MKR"}


def is_kg_dataset(name: str) -> bool:
    return "KG" in name or "kg" in name


def build_pop_counter(inter_feat, iid_field):
    pop = Counter()
    for iid in inter_feat[iid_field].numpy():
        pop[int(iid)] += 1
    return pop


def build_ground_truth(inter_feat, uid_field, iid_field, label_field=None):
    """
    Costruisce il ground truth dalle interazioni di test.
    
    Se label_field è presente (da threshold), filtra solo label=1.
    Altrimenti, usa tutte le interazioni.
    """
    truth = defaultdict(set)
    uids = inter_feat[uid_field].numpy()
    iids = inter_feat[iid_field].numpy()
    
    if label_field and label_field in inter_feat:
        # Filtra per label=1 (solo rating >= threshold)
        labels = inter_feat[label_field].numpy()
        for u, i, label in zip(uids, iids, labels):
            if label == 1:
                truth[int(u)].add(int(i))
    else:
        # Usa tutte le interazioni (threshold=None)
        for u, i in zip(uids, iids):
            truth[int(u)].add(int(i))
    
    return truth


def build_user_history_cpu(dataset):
    inter_feat = dataset.inter_feat
    users = inter_feat[dataset.uid_field].numpy()
    items = inter_feat[dataset.iid_field].numpy()
    history = defaultdict(list)
    for u, i in zip(users, items):
        history[u].append(i)
    out = defaultdict(lambda: torch.LongTensor([]))
    for u, arr in history.items():
        out[u] = torch.LongTensor(arr)
    return out


def get_vectors_gpu(model, dataset):
    u_emb, i_emb = None, None
    device = next(model.parameters()).device

    def extract(obj):
        if isinstance(obj, torch.nn.Embedding):
            return obj.weight.detach()
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        return None

    name = model.__class__.__name__
    if name == "DMF":
        try:
            batch_size = 256
            u_list, i_list = [], []
            for i in range(0, dataset.user_num, batch_size):
                end = min(i + batch_size, dataset.user_num)
                users = torch.arange(i, end, device=device)
                emb = model.get_user_embedding(users)
                emb = model.user_fc_layers(emb)
                u_list.append(emb.detach())
            for i in range(0, dataset.item_num, batch_size):
                end = min(i + batch_size, dataset.item_num)
                items = torch.arange(i, end, device=device)
                col_indices = model.history_user_id[items].flatten()
                row_indices = torch.arange(items.shape[0], device=device).repeat_interleave(model.history_user_id.shape[1], dim=0)
                matrix_01 = torch.zeros(items.shape[0], model.n_users, device=device)
                matrix_01.index_put_((row_indices, col_indices), model.history_user_value[items].flatten())
                emb = model.item_linear(matrix_01)
                emb = model.item_fc_layers(emb)
                i_list.append(emb.detach())
            u_emb = torch.cat(u_list, dim=0)
            i_emb = torch.cat(i_list, dim=0)
            return u_emb, i_emb
        except Exception:
            return None, None

    if hasattr(model, "user_embeddings_lookup"):
        u_emb = extract(model.user_embeddings_lookup)
    elif hasattr(model, "user_embedding"):
        u_emb = extract(model.user_embedding)
    elif hasattr(model, "user_embeddings"):
        u_emb = extract(model.user_embeddings)

    if hasattr(model, "item_embeddings_lookup"):
        i_emb = extract(model.item_embeddings_lookup)
    elif hasattr(model, "item_embedding"):
        i_emb = extract(model.item_embedding)
    elif hasattr(model, "item_embeddings"):
        i_emb = extract(model.item_embeddings)
    elif hasattr(model, "entity_embedding"):
        i_emb = extract(model.entity_embedding)

    return u_emb, i_emb


def generate_recommendations(model, test_data, dataset, topk, device, user_history, u_embs=None, i_embs=None):
    model.eval()
    user_recs = {}
    model = model.to(device)
    use_emb = (u_embs is not None) and (i_embs is not None)
    uid_field = dataset.uid_field

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Generazione raccomandazioni", leave=False):
            interaction = batch[0] if isinstance(batch, (list, tuple)) else batch
            interaction = interaction.to(device)
            batch_users = interaction[uid_field]
            bsz = batch_users.size(0)

            global_vals = torch.full((bsz, topk), -float("inf"), device=device)
            global_inds = torch.zeros((bsz, topk), dtype=torch.long, device=device)

            if use_emb:
                batch_u = u_embs[batch_users]
                for start in range(0, dataset.item_num, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, dataset.item_num)
                    chunk = i_embs[start:end]
                    scores = torch.matmul(batch_u, chunk.t())

                    mask = torch.zeros((bsz, end - start), dtype=torch.bool, device=device)
                    bu_list = batch_users.tolist()
                    for i, u in enumerate(bu_list):
                        seen = user_history[u]
                        if len(seen) == 0:
                            continue
                        in_chunk = (seen >= start) & (seen < end)
                        if in_chunk.any():
                            idx = (seen[in_chunk] - start).to(device)
                            mask[i, idx] = True
                    scores.masked_fill_(mask, -float("inf"))

                    k_local = min(topk, end - start)
                    vals, inds = torch.topk(scores, k_local, dim=1)
                    inds = inds + start
                    cand_vals = torch.cat([global_vals, vals], dim=1)
                    cand_inds = torch.cat([global_inds, inds], dim=1)
                    global_vals, top_idx = torch.topk(cand_vals, topk, dim=1)
                    global_inds = torch.gather(cand_inds, 1, top_idx)

            else:
                scores = model.full_sort_predict(interaction)
                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)
                mask = torch.zeros_like(scores, dtype=torch.bool)
                bu_list = batch_users.tolist()
                for i, u in enumerate(bu_list):
                    seen = user_history[u]
                    if len(seen) == 0:
                        continue
                    valid = seen < scores.size(1)
                    seen = seen[valid]
                    if len(seen) > 0:
                        mask[i, seen] = True
                scores.masked_fill_(mask.to(device), -float("inf"))
                global_vals, global_inds = torch.topk(scores, topk, dim=1)

            u_cpu = batch_users.cpu().numpy()
            inds_cpu = global_inds.cpu().numpy()
            for uid, items in zip(u_cpu, inds_cpu):
                user_recs[int(uid)] = [int(x) for x in items]

    return user_recs


def serendipity_ge_binary(user_recs, ground_truth, pop_counter, topk):
    if not user_recs:
        return 0.0
    pm_set = set([item for item, _ in pop_counter.most_common(topk)])
    total = 0.0
    users = 0
    for uid, recs in user_recs.items():
        if uid not in ground_truth:
            continue
        l_u = recs[:topk]
        t_u = ground_truth[uid]
        hits = [i for i in l_u if i in t_u]
        ser_items = [i for i in hits if i not in pm_set]
        total += len(ser_items) / float(topk)
        users += 1
    return total / users if users else 0.0


def calc_serendipity_and_unexpectedness_yan_gpu(user_recs, ground_truth, u_embs, i_embs, topk):
    if not user_recs or u_embs is None or i_embs is None:
        return 0.0, 0.0
    total_ser = 0.0
    total_unexp = 0.0
    users = 0
    u_norm = F.normalize(u_embs, p=2, dim=1).cpu()
    i_norm = F.normalize(i_embs, p=2, dim=1).cpu()
    for uid, recs in user_recs.items():
        if uid not in ground_truth:
            continue
        if uid >= len(u_norm):
            continue
        t_u = ground_truth[uid]
        l_u = recs[:topk]
        score_ser_u = 0.0
        score_unexp_u = 0.0
        user_vec = u_norm[uid]
        for iid in l_u:
            if iid >= len(i_norm):
                continue
            item_vec = i_norm[iid]
            sim = torch.dot(user_vec, item_vec).item()
            sim = (sim + 1.0) / 2.0
            sim = max(0.0, min(1.0, sim))
            unexp = 1.0 - sim
            score_unexp_u += unexp
            if iid in t_u:
                score_ser_u += unexp
        total_ser += score_ser_u / float(topk)
        total_unexp += score_unexp_u / float(topk)
        users += 1
    return (total_ser / users if users else 0.0), (total_unexp / users if users else 0.0)


def _gini(array):
    arr = np.array(array, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * arr) / (n * np.sum(arr))) - (n + 1) / n


def _shannon(array):
    arr = np.array(array, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def compute_offline_metrics(user_recs, ground_truth, pop_counter, u_embs, i_embs, k):
    rec_list = []
    for recs in user_recs.values():
        rec_list.extend(recs[:k])
    recall_sum = 0.0
    prec_sum = 0.0
    ndcg_sum = 0.0
    users = 0
    for uid, recs in user_recs.items():
        if uid not in ground_truth:
            continue
        gt = ground_truth[uid]
        if len(gt) == 0:
            continue
        topk = recs[:k]
        hits = [i for i in topk if i in gt]
        recall_sum += len(hits) / float(len(gt))
        prec_sum += len(hits) / float(k)
        if hits:
            dcg = 0.0
            for rank, item in enumerate(topk, start=1):
                if item in gt:
                    dcg += 1.0 / np.log2(rank + 1)
            ideal = min(len(gt), k)
            idcg = sum([1.0 / np.log2(i + 1) for i in range(1, ideal + 1)])
            ndcg_sum += dcg / idcg if idcg > 0 else 0.0
        users += 1
    avg_recall = recall_sum / users if users else 0.0
    avg_prec = prec_sum / users if users else 0.0
    avg_ndcg = ndcg_sum / users if users else 0.0

    pop_vals = [pop_counter.get(i, 0) for i in rec_list]
    avg_pop = float(np.mean(pop_vals)) if pop_vals else 0.0
    novelty = 1.0 / np.log1p(avg_pop) if avg_pop > 0 else 0.0
    gini = _gini(pop_vals)
    shannon = _shannon(rec_list)

    ser_ge = serendipity_ge_binary(user_recs, ground_truth, pop_counter, k)
    ser_yan, unexp = calc_serendipity_and_unexpectedness_yan_gpu(user_recs, ground_truth, u_embs, i_embs, k)

    return {
        f"NDCG@{k}": avg_ndcg,
        f"Recall@{k}": avg_recall,
        f"Precision@{k}": avg_prec,
        f"AvgPop@{k}": avg_pop,
        f"Novelty@{k}": novelty,
        f"Gini@{k}": gini,
        f"Shannon@{k}": shannon,
        f"Serendipity_Ge@{k}": ser_ge,
        f"Serendipity_Yan@{k}": ser_yan,
        f"Unexpectedness@{k}": unexp,
    }


def save_recs(dataset_name, data_path, model_name, seed, uid_field, iid_field, user_recs, topk, u_embs=None, i_embs=None):
    os.makedirs(RECS_DIR, exist_ok=True)
    ds_dir = os.path.join(RECS_DIR, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)
    base = f"{model_name}_top{SAVE_TOPK}_seed{seed}"
    rec_path = os.path.join(ds_dir, base + ".parquet")
    meta_path = os.path.join(ds_dir, base + "_meta.json")

    df = pd.DataFrame({"user_id": list(user_recs.keys()), "recs": list(user_recs.values())})
    rec_format = "parquet"
    try:
        df.to_parquet(rec_path, index=False, compression="gzip")
    except Exception:
        # Fallback se manca pyarrow/fastparquet
        rec_format = "csv"
        rec_path = os.path.join(ds_dir, base + ".csv.gz")
        df.to_csv(rec_path, index=False, compression="gzip")

    u_path = None
    i_path = None
    if u_embs is not None:
        u_path = os.path.join(ds_dir, base + "_u.pt")
        torch.save(u_embs.cpu(), u_path)
    if i_embs is not None:
        i_path = os.path.join(ds_dir, base + "_i.pt")
        torch.save(i_embs.cpu(), i_path)

    meta = {
        "dataset": dataset_name,
        "data_path": data_path,
        "uid_field": uid_field,
        "iid_field": iid_field,
        "seed": seed,
        "topk_saved": SAVE_TOPK,
        "model": model_name,
        "u_emb_path": u_path,
        "i_emb_path": i_path,
        "rec_format": rec_format,
        "rec_path": rec_path,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return rec_path, meta_path






def save_recs_multiple_k(
    dataset_name,
    data_path,
    model_name,
    seed,
    uid_field,
    iid_field,
    user_recs,
    ks,
    u_embs=None,
    i_embs=None
):
    saved = []
    for k in ks:
        user_recs_k = {u: r[:k] for u, r in user_recs.items()}
        rec_path, meta_path = save_recs(
            dataset_name,
            data_path,
            model_name,
            seed,
            uid_field,
            iid_field,
            user_recs_k,
            topk=k,
            u_embs=u_embs,
            i_embs=i_embs,
        )
        saved.append((k, rec_path, meta_path))
    return saved








def rec_already_saved(dataset_name, model_name, seed=SEED, ks=None):
    if ks is None:
        ks = SAVE_TOPK
    ds_dir = os.path.join(RECS_DIR, dataset_name)
    for k in ks:
        base = f"{model_name}_top{k}_seed{seed}"
        parquet_path = os.path.join(ds_dir, base + ".parquet")
        csv_path = os.path.join(ds_dir, base + ".csv.gz")
        if not (os.path.exists(parquet_path) or os.path.exists(csv_path)):
            return False
    return True


def run_one(dataset_label, dataset_cfg, model_name, model_params):
    print(f"\n=== {dataset_label} | {model_name} ===")
    cfg_dict = GLOBAL_CONFIG.copy()
    cfg_dict.update(dataset_cfg)
    cfg_dict["dataset"] = dataset_cfg["dataset"]
    cfg_dict["data_path"] = dataset_cfg["data_path"]
    cfg_dict["model"] = model_name
    cfg_dict.update(model_params)

    init_seed(SEED, True)

    config = Config(model=model_name, dataset=cfg_dict["dataset"], config_dict=cfg_dict)
    init_seed(config["seed"], config["reproducibility"])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model_cls = get_model(config["model"])
    model = model_cls(config, train_data.dataset).to(config["device"])

    trainer_cls = get_trainer(config['MODEL_TYPE'], config['model'])
    trainer = trainer_cls(config, model)
    train_output = trainer.fit(train_data, valid_data)

    base_metrics = trainer.evaluate(test_data, load_best_model=False, show_progress=False)

    user_history = build_user_history_cpu(train_data.dataset)
    label_field = dataset.label_field if hasattr(dataset, 'label_field') else None
    pop_counter = build_pop_counter(train_data.dataset.inter_feat, dataset.iid_field)
    truth = build_ground_truth(test_data.dataset.inter_feat, dataset.uid_field, dataset.iid_field, label_field=label_field)
    u_embs, i_embs = get_vectors_gpu(model, train_data.dataset)

    # genera almeno fino al massimo richiesto (100)
    max_k = max(SAVE_TOPK)
    user_recs = generate_recommendations(
        model, test_data, dataset, max_k,
        config["device"], user_history, u_embs, i_embs
    )

    saved_files = save_recs_multiple_k(
        dataset_cfg["dataset"],
        dataset_cfg["data_path"],
        model_name,
        config["seed"],
        dataset.uid_field,
        dataset.iid_field,
        user_recs,
        ks=SAVE_TOPK,
        u_embs=u_embs,
        i_embs=i_embs,
    )

    # nel CSV risultati, metti il path del top100 (cioè max_k)
    rec_path = [p for (k, p, m) in saved_files if k == max_k][0]

    result = {
        "Dataset": dataset_label,
        "Model": model_name,
        "Checkpoint": "in-memory",
        "RecFile": rec_path,
    }

    for k in TOPKS:
        metrics_off = compute_offline_metrics({u: r[:k] for u, r in user_recs.items()}, truth, pop_counter, u_embs, i_embs, k)
        result.update(metrics_off)
        # Merge RecBole metrics if present
        rb_map = {
            f"ndcg@{k}": f"NDCG@{k}",
            f"recall@{k}": f"Recall@{k}",
            f"precision@{k}": f"Precision@{k}",
            f"averagepopularity@{k}": f"AvgPop@{k}",
            f"giniindex@{k}": f"Gini@{k}",
            f"shannonentropy@{k}": f"Shannon@{k}",
        }
        for raw, pretty in rb_map.items():
            if raw in base_metrics:
                result[pretty] = base_metrics[raw]
                if pretty == f"AvgPop@{k}" and base_metrics[raw] > 0:
                    result[f"Novelty@{k}"] = 1.0 / np.log1p(base_metrics[raw])

    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    results = []
    for ds_label, ds_cfg in DATASETS.items():
        ds_folder = os.path.join(ds_cfg["data_path"], ds_cfg["dataset"])
        if not os.path.exists(ds_folder):
            print(f"[Skip] Dataset mancante: {ds_folder}")
            continue

        for model_name, model_params in MODELS.items():
            # KG models solo su dataset KG; non-KG solo su dataset non-KG
            if model_name in KG_MODELS and not is_kg_dataset(ds_cfg["dataset"]):
                continue
            if model_name not in KG_MODELS and is_kg_dataset(ds_cfg["dataset"]):
                continue

            if rec_already_saved(ds_cfg["dataset"], model_name, SEED):
                print(f"[Skip] {ds_label} - {model_name}: raccomandazioni già presenti in {RECS_DIR}/{ds_cfg['dataset']}")
                continue
            try:
                res = run_one(ds_label, ds_cfg, model_name, model_params)
                results.append(res)
                pd.DataFrame(results).to_csv(RESULTS_FILE, index=False, float_format="%.6f")
            except Exception:
                traceback.print_exc()
                continue

    if results:
        df = pd.DataFrame(results)
        cols_show = [
            "Dataset", "Model", "RecFile",
            "NDCG@10", "Recall@10", "Serendipity_Ge@10", "Serendipity_Yan@10", "Unexpectedness@10",
            "NDCG@5", "Recall@5", "Serendipity_Ge@5", "Serendipity_Yan@5", "Unexpectedness@5",
        ]
        cols_show = [c for c in cols_show if c in df.columns]
        print(df[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
