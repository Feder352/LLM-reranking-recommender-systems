import os
import sys
import glob
import gc
import traceback
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.data.dataloader.knowledge_dataloader import KnowledgeBasedDataLoader
from recbole.data.interaction import Interaction
from recbole.evaluator import Evaluator
from recbole.evaluator.collector import DataStruct

# Patch for KnowledgeBasedDataLoader missing 'dataset' attribute
if not hasattr(KnowledgeBasedDataLoader, 'dataset'):
    setattr(KnowledgeBasedDataLoader, 'dataset', property(lambda self: self._dataset))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPKS = [5, 10, 50, 100]  # N finale su cui calcolare le metriche
CANDIDATE_KS = [50, 100]  # K candidati da cui fare il reranking
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 128))
GEN_BATCH_SIZE = int(os.environ.get("GEN_BATCH_SIZE", 64))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 10000))
RESULTS_FILE = "benchmark_creativity_score_reranking.csv"
LOG_FILE = "eval_creativity_score_reranking.log"
CKPT_ROOT = "C:/Universita/Tesi/recbole_prog/saved"

# Pesi per il creativity score: (0.33 * relevance) + (0.33 * novelty) + (0.33 * unexpectedness)
WEIGHT_RELEVANCE = 0.25
WEIGHT_NOVELTY = 0.25
WEIGHT_UNEXPECTEDNESS = 0.50

# Lista checkpoint da valutare (include tutti i modelli, commentati e non)
TARGET_CKPTS = {
    # ========= MOVIELENS / AMAZON (stessi modelli, run diversi) =========

    # BPR
    "BPR-Jan-27-2026_18-10-05.pth",
    "BPR-Jan-28-2026_20-29-07.pth",

    # CFKG
    "CFKG-Jan-27-2026_21-09-15.pth",
    "CFKG-Jan-29-2026_14-02-32.pth",

    # CKE
    "CKE-Jan-27-2026_21-12-15.pth",
    "CKE-Jan-29-2026_14-50-12.pth",

    # DMF
    "DMF-Jan-27-2026_18-03-10.pth",
    "DMF-Jan-28-2026_20-21-41.pth",

    # ENMF
    "ENMF-Jan-27-2026_19-14-49.pth",
    "ENMF-Jan-28-2026_21-37-37.pth",

    # ItemKNN
    "ItemKNN-Jan-27-2026_18-00-08.pth",
    "ItemKNN-Jan-28-2026_20-16-05.pth",

    # KGCN
    "KGCN-Jan-27-2026_21-15-12.pth",
    "KGCN-Jan-29-2026_15-38-49.pth",

    # KGNNLS
    "KGNNLS-Jan-27-2026_21-21-02.pth",
    "KGNNLS-Jan-29-2026_16-40-12.pth",

    # LightGCN
    "LightGCN-Jan-27-2026_18-16-05.pth",
    "LightGCN-Jan-28-2026_20-38-21.pth",

    # MKR
    "MKR-Jan-27-2026_21-26-38.pth",
    "MKR-Jan-29-2026_17-37-12.pth",

    # MultiVAE
    "MultiVAE-Jan-27-2026_18-12-36.pth",
    "MultiVAE-Jan-28-2026_20-33-02.pth",

    # Pop
    "Pop-Jan-27-2026_17-58-20.pth",
    "Pop-Jan-28-2026_20-15-01.pth",
}


USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

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


class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


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


def get_vectors(model, dataset):
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


def _predict_full_set(model, batch_users, dataset, device):
    num_items = dataset.item_num
    batch_size = batch_users.size(0)
    all_items = torch.arange(num_items, device=device)
    scores_matrix = torch.zeros(batch_size, num_items, device=device)
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    for i, user_id in enumerate(batch_users):
        item_chunk_size = 10000
        for start_idx in range(0, num_items, item_chunk_size):
            end_idx = min(start_idx + item_chunk_size, num_items)
            current_items = all_items[start_idx:end_idx]
            current_len = len(current_items)
            u_ids = user_id.repeat(current_len)
            interaction_dict = {uid_field: u_ids, iid_field: current_items}
            interaction = Interaction(interaction_dict).to(device)
            with torch.no_grad():
                pred = model.predict(interaction)
            scores_matrix[i, start_idx:end_idx] = pred
    return scores_matrix


def generate_recommendations_with_scores_gpu(model, test_data, dataset, topk, device, user_history):
    """
    Genera raccomandazioni con i relativi score di rilevanza.
    Restituisce: dict[uid] = [(item_id, score), ...]
    """
    model.eval()
    user_field = dataset.uid_field
    user_recs = {}
    model = model.to(device)
    model_name = model.__class__.__name__
    is_lightgcn = model_name == 'LightGCN'
    is_graph_model = model_name in ['LightGCN', 'NGCF', 'GCN', 'KGCN']
    cached_user_emb = None
    cached_item_emb = None

    if is_lightgcn:
        with torch.no_grad():
            if hasattr(model, 'restore_user_e'):
                model.restore_user_e = None
            if hasattr(model, 'restore_item_e'):
                model.restore_item_e = None
            try:
                cached_user_emb, cached_item_emb = model.computer()
                cached_user_emb = cached_user_emb.detach()
                cached_item_emb = cached_item_emb.detach()
            except AttributeError:
                print("⚠️ model.computer() non trovato, provo model.forward()...")
                cached_user_emb, cached_item_emb = model.forward()
                cached_user_emb = cached_user_emb.detach()
                cached_item_emb = cached_item_emb.detach()

    has_embed = hasattr(model, 'user_embedding') and hasattr(model, 'item_embedding') and not is_graph_model
    if has_embed:
        item_embed = model.item_embedding.weight.to(device)
    else:
        item_embed = None

    with torch.no_grad():
        for batch in tqdm(test_data, desc="Gen recs with scores", leave=False):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            batch_users = batch[user_field]
            batch_size = batch_users.size(0)

            if is_lightgcn:
                batch_u_emb = cached_user_emb[batch_users]
                scores = torch.matmul(batch_u_emb, cached_item_emb.t())
                batch_users_list = batch_users.tolist()
                n_items = scores.size(1)
                for i, u_id in enumerate(batch_users_list):
                    seen = user_history[int(u_id)]
                    if len(seen) == 0:
                        continue
                    valid_mask = seen < n_items
                    valid_seen = seen[valid_mask]
                    if len(valid_seen) > 0:
                        scores[i, valid_seen.to(device)] = -float("inf")
                topk_scores, topk_idx = torch.topk(scores, topk, dim=1)
            elif has_embed and dataset.item_num > 50000:
                user_e = model.user_embedding(batch_users).to(device)
                n_items = item_embed.size(0)
                vals = torch.empty((batch_size, 0), device=device)
                idxs = torch.empty((batch_size, 0), device=device, dtype=torch.long)
                for start in range(0, n_items, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, n_items)
                    scores_chunk = torch.matmul(user_e, item_embed[start:end].t())
                    batch_users_list = batch_users.tolist()
                    for i, u_id in enumerate(batch_users_list):
                        seen = user_history[int(u_id)]
                        if len(seen) == 0:
                            continue
                        mask_in_chunk = (seen >= start) & (seen < end)
                        if mask_in_chunk.any():
                            local_indices = (seen[mask_in_chunk] - start).to(device)
                            scores_chunk[i, local_indices] = -float("inf")
                    v, i_idx = torch.topk(scores_chunk, k=min(topk, end - start), dim=1)
                    i_idx = i_idx + start
                    vals = torch.cat([vals, v], dim=1)
                    idxs = torch.cat([idxs, i_idx], dim=1)
                    vals, top_idx = torch.topk(vals, k=topk, dim=1)
                    idxs = torch.gather(idxs, 1, top_idx)
                    del scores_chunk, v, i_idx
                topk_scores = vals
                topk_idx = idxs
            else:
                try:
                    scores = model.full_sort_predict(batch)
                except NotImplementedError:
                    scores = _predict_full_set(model, batch_users, dataset, device)
                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)
                if scores.size(0) != batch_size:
                    scores = scores.view(batch_size, -1)
                n_items = scores.size(1)
                batch_users_list = batch_users.tolist()
                for i, u_id in enumerate(batch_users_list):
                    seen = user_history[int(u_id)]
                    if len(seen) == 0:
                        continue
                    valid_mask = seen < n_items
                    valid_seen = seen[valid_mask]
                    if len(valid_seen) > 0:
                        scores[i, valid_seen.to(device)] = -float("inf")
                topk_scores, topk_idx = torch.topk(scores, topk, dim=1)

            uids = batch[user_field].cpu().numpy()
            topk_idx = topk_idx.cpu().numpy()
            topk_scores = topk_scores.cpu().numpy()
            for i, uid in enumerate(uids):
                user_recs[int(uid)] = [(int(topk_idx[i][j]), float(topk_scores[i][j])) for j in range(topk)]
            if 'scores' in locals():
                del scores
            del batch, topk_idx, topk_scores

    if is_lightgcn:
        if hasattr(model, 'restore_user_e'):
            model.restore_user_e = None
        if hasattr(model, 'restore_item_e'):
            model.restore_item_e = None
        del cached_user_emb, cached_item_emb

    return user_recs


def calc_item_novelty(item_id, pop_counter):
    """
    Calcola la novelty per un singolo item.
    Novelty(item) = 1 / log(1 + pop(item))
    """
    pop = pop_counter.get(int(item_id), 0)
    if pop > 0:
        return 1.0 / np.log1p(pop)
    else:
        return 1.0  # item mai visto = massima novelty


def calc_item_unexpectedness(item_id, user_history, i_embs_norm, device):
    """
    Calcola l'unexpectedness per un singolo item rispetto alla storia dell'utente.
    Unexpectedness(item) = 1 - avg_similarity(item, user_history)
    
    Se la storia è vuota, restituisce 0.5 (neutro).
    """
    if len(user_history) == 0:
        return 0.5
    
    valid_history = user_history[user_history < i_embs_norm.shape[0]]
    if len(valid_history) == 0:
        return 0.5
    
    valid_history = valid_history.to(device).long()
    if item_id >= i_embs_norm.shape[0]:
        return 0.5
    
    item_emb = i_embs_norm[item_id].unsqueeze(0)
    history_embs = i_embs_norm[valid_history]
    
    # Similarità coseno media tra item e storia utente
    sims = F.cosine_similarity(item_emb, history_embs, dim=1)
    avg_sim = sims.mean().item()
    
    # Normalizza similarità in [0, 1]
    avg_sim = (avg_sim + 1.0) / 2.0
    avg_sim = max(0.0, min(1.0, avg_sim))
    
    return 1.0 - avg_sim


def normalize_scores(scores):
    """
    Normalizza una lista di score in [0, 1] usando min-max normalization.
    """
    if len(scores) == 0:
        return []
    scores = np.array(scores)
    min_s = scores.min()
    max_s = scores.max()
    if max_s - min_s > 1e-8:
        return ((scores - min_s) / (max_s - min_s)).tolist()
    else:
        return [0.5] * len(scores)


def rerank_creativity_score(user_recs_with_scores, user_history, i_embs, pop_counter, 
                            num_candidates, topk, device):
    """
    Reranking basato su Creativity Score:
    creativity_score = (0.33 * relevance) + (0.33 * novelty) + (0.33 * unexpectedness)
    
    Args:
        user_recs_with_scores: dict[uid] = [(item_id, score), ...]
        user_history: dict[uid] = torch.LongTensor([item_ids])
        i_embs: item embeddings tensor
        pop_counter: Counter con popolarità items
        num_candidates: numero di candidati da considerare (K)
        topk: numero di item da restituire dopo reranking (N)
        device: torch device
    
    Returns:
        dict[uid] = [item_id, ...] (top-N rerankati)
    """
    if i_embs is None:
        print("⚠️ Embedding item non disponibili, uso solo relevance per reranking")
        # Fallback: usa solo la relevance
        user_recs_reranked = {}
        for uid, candidates_with_scores in user_recs_with_scores.items():
            C = candidates_with_scores[:num_candidates]
            # Ordina per score decrescente
            C_sorted = sorted(C, key=lambda x: x[1], reverse=True)
            user_recs_reranked[uid] = [item_id for item_id, _ in C_sorted[:topk]]
        return user_recs_reranked
    
    i_embs_norm = F.normalize(i_embs, p=2, dim=1)
    user_recs_reranked = {}
    
    for uid, candidates_with_scores in tqdm(user_recs_with_scores.items(), 
                                             desc=f"Reranking creativity (K={num_candidates})", 
                                             leave=False):
        C = candidates_with_scores[:num_candidates]
        if len(C) < 2:
            user_recs_reranked[uid] = [item_id for item_id, _ in C[:topk]]
            continue
        
        history = user_history.get(int(uid), torch.LongTensor([]))
        
        # Estrai item_ids e scores
        item_ids = [item_id for item_id, _ in C]
        relevance_scores = [score for _, score in C]
        
        # Calcola novelty per ogni item
        novelty_scores = [calc_item_novelty(iid, pop_counter) for iid in item_ids]
        
        # Calcola unexpectedness per ogni item
        unexpectedness_scores = [
            calc_item_unexpectedness(iid, history, i_embs_norm, device) 
            for iid in item_ids
        ]
        
        # Normalizza tutti gli score in [0, 1]
        relevance_norm = normalize_scores(relevance_scores)
        novelty_norm = normalize_scores(novelty_scores)
        unexpectedness_norm = normalize_scores(unexpectedness_scores)
        
        # Calcola creativity score per ogni item
        creativity_scores = []
        for i in range(len(item_ids)):
            cs = (WEIGHT_RELEVANCE * relevance_norm[i] + 
                  WEIGHT_NOVELTY * novelty_norm[i] + 
                  WEIGHT_UNEXPECTEDNESS * unexpectedness_norm[i])
            creativity_scores.append(cs)
        
        # Ordina per creativity score decrescente
        sorted_indices = np.argsort(creativity_scores)[::-1]
        
        user_recs_reranked[uid] = [item_ids[i] for i in sorted_indices[:topk]]
    
    return user_recs_reranked


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


def build_recbole_datastruct_from_reranked(user_recs, ground_truth, train_dataset, topks):
    """Costruisce un DataStruct RecBole dai risultati rerankati."""
    if not user_recs:
        return None

    users = [uid for uid in user_recs.keys() if uid in ground_truth and len(ground_truth[uid]) > 0]
    if not users:
        return None

    max_k = max(topks)
    item_matrix = torch.zeros((len(users), max_k), dtype=torch.long)
    pos_matrix = torch.zeros((len(users), max_k), dtype=torch.int)
    pos_len_list = torch.zeros((len(users), 1), dtype=torch.int)

    for row_idx, uid in enumerate(users):
        recs = user_recs[uid][:max_k]
        if len(recs) < max_k:
            recs = recs + ([recs[-1]] * (max_k - len(recs))) if recs else [0] * max_k
        item_matrix[row_idx] = torch.tensor(recs, dtype=torch.long)
        gt = ground_truth[uid]
        pos_len_list[row_idx, 0] = len(gt)
        hit_row = [1 if iid in gt else 0 for iid in recs[:max_k]]
        pos_matrix[row_idx] = torch.tensor(hit_row, dtype=torch.int)

    data_struct = DataStruct()
    data_struct.set("rec.items", item_matrix)
    data_struct.set("rec.topk", torch.cat([pos_matrix, pos_len_list], dim=1))
    data_struct.set("data.num_items", train_dataset.item_num)
    data_struct.set("data.num_users", train_dataset.user_num)
    data_struct.set("data.count_items", train_dataset.item_counter)
    data_struct.set("data.count_users", train_dataset.user_counter)
    return data_struct


# -------------------- Valutazione --------------------

def evaluate_checkpoint(model_path):
    print(f"\nProcessing: {model_path}")
    gc.collect()
    if USE_GPU:
        torch.cuda.empty_cache()
    try:
        config, model, dataset, train_data, _, test_data = load_data_and_model(model_file=model_path)
        config['use_gpu'] = USE_GPU
        config['device'] = DEVICE
        config['gpu_id'] = 0 if USE_GPU else ''
        config['eval_batch_size'] = EVAL_BATCH_SIZE
        config['topk'] = TOPKS
        if hasattr(test_data, 'config'):
            test_data.config['eval_batch_size'] = EVAL_BATCH_SIZE
            test_data.config['device'] = DEVICE
        object.__setattr__(test_data, 'batch_size', EVAL_BATCH_SIZE)
        if hasattr(test_data, 'step'):
            object.__setattr__(test_data, 'step', EVAL_BATCH_SIZE)
        model = model.to(DEVICE)
        trainer = Trainer(config, model)
        trainer.save_model = False
        
        print("   Metriche RecBole (originali)...")
        trainer.eval_collector.data_collect(train_data)
        base_metrics = trainer.evaluate(test_data, load_best_model=False, show_progress=False)

        uid_field = dataset.uid_field
        iid_field = dataset.iid_field
        label_field = dataset.label_field if hasattr(dataset, 'label_field') else None
        pop_counter = build_pop_counter(train_data.dataset.inter_feat, iid_field)
        test_truth = build_ground_truth(test_data.dataset.inter_feat, uid_field, iid_field, label_field=label_field)
        u_embs, i_embs = get_vectors(model, train_data.dataset)
        user_history = build_user_history_cpu(train_data.dataset)

        object.__setattr__(test_data, 'batch_size', GEN_BATCH_SIZE)
        if hasattr(test_data, 'step'):
            object.__setattr__(test_data, 'step', GEN_BATCH_SIZE)
        if hasattr(test_data, 'config'):
            test_data.config['eval_batch_size'] = GEN_BATCH_SIZE
            test_data.config['device'] = DEVICE

        try:
            dataset_name = config['dataset']
        except KeyError:
            dataset_name = dataset.name if hasattr(dataset, 'name') else ''
        try:
            model_name = config['model']
        except KeyError:
            model_name = model.__class__.__name__

        result = {
            'Checkpoint': os.path.basename(model_path),
            'Dataset': dataset_name,
            'Model': model_name,
        }

        # Metriche RecBole originali (valori di base)
        for key, val in base_metrics.items():
            result[key] = val

        # Genera raccomandazioni con score per il massimo K candidati
        max_candidates = max(CANDIDATE_KS)
        print(f"   Generando raccomandazioni (K={max_candidates} candidati)...")
        user_recs_with_scores = generate_recommendations_with_scores_gpu(
            model, test_data, dataset, max_candidates, DEVICE, user_history
        )

        # Converti a formato senza score per le metriche originali
        user_recs_original = {uid: [iid for iid, _ in recs] for uid, recs in user_recs_with_scores.items()}

        # Per ogni combinazione di K candidati
        for num_candidates in CANDIDATE_KS:
            print(f"   Applicando reranking creativity score (K={num_candidates})...")
            
            # Applica reranking con creativity score
            user_recs_reranked = rerank_creativity_score(
                user_recs_with_scores,
                user_history,
                i_embs,
                pop_counter,
                num_candidates=num_candidates,
                topk=max(TOPKS),
                device=DEVICE
            )

            # Metriche RecBole calcolate sul ranking rerankato
            rerank_data_struct = build_recbole_datastruct_from_reranked(
                user_recs_reranked, test_truth, train_data.dataset, TOPKS
            )
            if rerank_data_struct is not None:
                rerank_config = {
                    'topk': TOPKS,
                    'metrics': [m.lower() for m in config['metrics']],
                    'metric_decimal_place': config['metric_decimal_place'] if 'metric_decimal_place' in config else 4,
                }
                evaluator = Evaluator(rerank_config)
                rerank_metrics = evaluator.evaluate(rerank_data_struct)
                for key, val in rerank_metrics.items():
                    result[f'{key}_reranked_K{num_candidates}'] = val
                
                # Delta per tutte le metriche RecBole
                for key, val in base_metrics.items():
                    if key in rerank_metrics:
                        result[f'Delta_{key}_K{num_candidates}'] = rerank_metrics[key] - val

            # Metriche di serendipità/inaspettatezza
            for k in TOPKS:
                # Metriche originali (solo per K=max_candidates per evitare duplicati)
                if num_candidates == max(CANDIDATE_KS):
                    k_recs_orig = {u: r[:k] for u, r in user_recs_original.items()}
                    ser_ge_orig = serendipity_ge_binary(k_recs_orig, test_truth, pop_counter, k)
                    ser_yan_orig, unexp_orig = calc_serendipity_and_unexpectedness_yan_gpu(
                        k_recs_orig, test_truth, u_embs, i_embs, k
                    )
                    result[f'Serendipity_Ge_Original@{k}'] = ser_ge_orig
                    result[f'Serendipity_Yan_Original@{k}'] = ser_yan_orig
                    result[f'Unexpectedness_Original@{k}'] = unexp_orig

                # Metriche rerankate
                k_recs_reranked = {u: r[:k] for u, r in user_recs_reranked.items()}
                ser_ge_reranked = serendipity_ge_binary(k_recs_reranked, test_truth, pop_counter, k)
                ser_yan_reranked, unexp_reranked = calc_serendipity_and_unexpectedness_yan_gpu(
                    k_recs_reranked, test_truth, u_embs, i_embs, k
                )
                
                result[f'Serendipity_Ge_Reranked@{k}_K{num_candidates}'] = ser_ge_reranked
                result[f'Serendipity_Yan_Reranked@{k}_K{num_candidates}'] = ser_yan_reranked
                result[f'Unexpectedness_Reranked@{k}_K{num_candidates}'] = unexp_reranked

                # Delta (solo se abbiamo i valori originali)
                if f'Serendipity_Ge_Original@{k}' in result:
                    result[f'Delta_Serendipity_Ge@{k}_K{num_candidates}'] = ser_ge_reranked - result[f'Serendipity_Ge_Original@{k}']
                    result[f'Delta_Serendipity_Yan@{k}_K{num_candidates}'] = ser_yan_reranked - result[f'Serendipity_Yan_Original@{k}']
                    result[f'Delta_Unexpectedness@{k}_K{num_candidates}'] = unexp_reranked - result[f'Unexpectedness_Original@{k}']

        return result
    except Exception as e:
        print(f"Errore su {model_path}: {e}")
        traceback.print_exc()
        return None


def filter_checkpoints(ckpts):
    if not TARGET_CKPTS:
        return ckpts
    filtered = [p for p in ckpts if os.path.basename(p) in TARGET_CKPTS]
    missing = sorted(list(TARGET_CKPTS - {os.path.basename(p) for p in ckpts}))
    if missing:
        print("Checkpoint richiesti mancanti:", ", ".join(missing))
    return filtered


def main():
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    tee = TeeLogger(LOG_FILE)
    sys.stdout = tee
    sys.stderr = tee
    try:
        print("GPU:", USE_GPU, "Device:", DEVICE)
        print("=" * 80)
        print(f"VALUTAZIONE RERANKING CREATIVITY SCORE")
        print(f"  Pesi: relevance={WEIGHT_RELEVANCE}, novelty={WEIGHT_NOVELTY}, unexpectedness={WEIGHT_UNEXPECTEDNESS}")
        print(f"  Candidati K: {CANDIDATE_KS}")
        print(f"  Top-N finale: {TOPKS}")
        print("=" * 80)
        ckpts = glob.glob(os.path.join(CKPT_ROOT, "**", "*.pth"), recursive=True)
        ckpts = filter_checkpoints(ckpts)
        if not ckpts:
            print("Nessun checkpoint trovato.")
            return
        results = []
        header_written = False
        for i, ck in enumerate(ckpts, 1):
            print(f"[{i}/{len(ckpts)}]")
            res = evaluate_checkpoint(ck)
            if res:
                results.append(res)
                pd.DataFrame([res]).to_csv(RESULTS_FILE, mode='a', header=not header_written, index=False, float_format='%.6f')
                header_written = True
            gc.collect()
            if USE_GPU:
                torch.cuda.empty_cache()
        if results:
            df = pd.DataFrame(results)
            cols = [c for c in ['Checkpoint', 'Dataset', 'Model'] if c in df.columns]
            other_cols = [c for c in df.columns if c not in cols]
            cols += other_cols
            try:
                print(df[cols].to_markdown(index=False, floatfmt=".4f"))
            except Exception:
                print(df[cols])
    finally:
        sys.stdout = tee.terminal
        sys.stderr = tee.terminal
        tee.close()


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
