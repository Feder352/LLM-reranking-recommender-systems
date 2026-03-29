"""
Genera diagrammi di Kiviat (radar chart) per RQ1.
3 assi: Relevance (NDCG@10), Novelty, Unexpectedness
Normalizzazione: min=0 fisso, max=massimo globale per metrica.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

MODELS = ["LightGCN", "BPR", "DMF", "KGCN", "CKE"]
METRICS = ["Relevance\n(NDCG@10)", "Novelty", "Unexpectedness"]

DATA = {
    "Amazon": {
        "baseline": {
            "LightGCN": [0.1050, 0.325028, 0.218823],
            "BPR":      [0.0843, 0.310179, 0.166155],
            "DMF":      [0.0432, 0.239999, 0.237073],
            "KGCN":     [0.0400, 0.325761, 0.503099],
            "CKE":      [0.0756, 0.310543, 0.193944],
        },
        "reranked": {
            "LightGCN": [0.061321, 0.343853, 0.220760],
            "BPR":      [0.026527, 0.333083, 0.179374],
            "DMF":      [0.029297, 0.256607, 0.222835],
            "KGCN":     [0.022747, 0.343058, 0.500228],
            "CKE":      [0.036062, 0.328188, 0.204903],
        },
    },
    "MovieLens": {
        "baseline": {
            "LightGCN": [0.2496, 0.204984, 0.190288],
            "BPR":      [0.2369, 0.146033, 0.173870],
            "DMF":      [0.1990, 0.138968, 0.195325],
            "KGCN":     [0.2213, 0.148888, 0.513393],
            "CKE":      [0.2339, 0.147923, 0.174873],
        },
        "reranked": {
            "LightGCN": [0.114005, 0.150326, 0.408208],
            "BPR":      [0.145840, 0.147924, 0.185420],
            "DMF":      [0.088753, 0.142623, 0.200679],
            "KGCN":     [0.100804, 0.154292, 0.514297],
            "CKE":      [0.103925, 0.153117, 0.193115],
        },
    },
}

MODEL_COLORS = {
    "LightGCN": "#e6194b",
    "BPR":      "#3cb44b",
    "DMF":      "#4363d8",
    "KGCN":     "#f58231",
    "CKE":      "#911eb4",
}

def get_global_max(ds_data):
    """max globale per metrica (baseline + reranked), min fisso a 0"""
    n = len(METRICS)
    maxs = [float("-inf")] * n
    for split in ("baseline", "reranked"):
        for vals in ds_data[split].values():
            for i, v in enumerate(vals):
                maxs[i] = max(maxs[i], v)
    return maxs

def normalize(values, maxs):
    """normalizza con min=0 fisso e max globale"""
    return [v / maxs[i] if maxs[i] > 0 else 0.0 for i, v in enumerate(values)]

os.makedirs("output", exist_ok=True)

N = len(METRICS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for dataset_name, ds_data in DATA.items():
    maxs = get_global_max(ds_data)

    for split, split_label in [("baseline", "Baseline"), ("reranked", "Re-ranking (Gemini)")]:

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(METRICS, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="grey")
        ax.grid(color="grey", linestyle="-", linewidth=0.6, alpha=0.4)
        ax.spines["polar"].set_color("grey")

        for model in MODELS:
            norm = normalize(ds_data[split][model], maxs)
            v = norm + norm[:1]
            ax.plot(angles, v, color=MODEL_COLORS[model], linewidth=2.2,
                    marker="o", markersize=5, label=model)
            ax.fill(angles, v, color=MODEL_COLORS[model], alpha=0.07)

        ax.set_title(f"{dataset_name}  —  {split_label}",
                     fontsize=14, fontweight="bold", pad=24)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
                  ncol=len(MODELS), fontsize=11, frameon=True, framealpha=0.9)

        plt.tight_layout()
        out = f"output/kiviat_{dataset_name.lower()}_{split}.png"
        plt.savefig(out, bbox_inches="tight", dpi=200, facecolor="white")
        plt.close()
        print(f"✅ {out}")

print("\nDone!")
