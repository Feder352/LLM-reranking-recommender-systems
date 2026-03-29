# Tecniche di Re-Ranking basate su Large Language Models per Recommender Systems Orientati alla Creatività

Questa tesi si propone di superare limitazioni quali il Filter Bubble integrando nel processo di raccomandazione strategie di re-ranking basate su **Large Language Models (LLM)**. L'obiettivo è sfruttare le capacità semantiche di **Gemini** per riordinare le liste di raccomandazione, bilanciando la rilevanza dei suggerimenti con metriche di creatività quali Novelty, Unexpectedness e Serendipity.

---

## 🎯 Obiettivi

- **Valutazione della creatività di base**: valutare le performance dei modelli di raccomandazione standard (senza re-ranking) per stabilire una linea base sulle metriche beyond-accuracy.
- **Re-ranking LLM-based**: utilizzare **Gemini** per riordinare semanticamente le liste di candidati, iniettando creatività nelle raccomandazioni finali.
- **Identificazione del modello più creativo**: determinare quale tra i modelli considerati raggiunge il miglior livello di creatività nelle raccomandazioni prodotte, valutando le performance su metriche beyond-accuracy quali Novelty, Unexpectedness e Serendipity.
- **Analisi del trade-off**: indagare sperimentalmente la relazione tra il miglioramento delle metriche creative e la stabilità delle metriche di accuratezza (NDCG, Recall, Precision), identificando il punto di equilibrio ottimale.

---

## 💻 Informazioni sull'Hardware

Di seguito sono riportate le specifiche tecniche della macchina utilizzata:
• Processore (CPU): Intel Core i5-12450H (12a generazione, 2.00 GHz). Si tratta di
una CPU di ultima generazione dotata di architettura ibrida ad alte prestazioni.
• Memoria (RAM): 16 GB di memoria di sistema (15,7 GB utilizzabili).
• Acceleratore Grafico (GPU): NVIDIA GeForce RTX 3050 Ti Laptop GPU, utilizzata
per accelerare il training dei modelli tramite CUDA.

Questa configurazione ha permesso di contenere i tempi di addestramento, sfruttando
l’accelerazione hardware CUDA per le fasi computazionalmente più intensive.

---

## 🛠️ Requisiti

L'intera sperimentazione è stata eseguita su una macchina locale con sistema operativo **Windows 11**.

- Python 3.10
- GPU CUDA-compatible (CUDA 11.7)

### Installazione

```bash
# Clona il repository
git clone https://github.com/Feder352/LLM-reranking-recommender-systems.git
cd LLM-reranking-recommender-systems

# Crea l'ambiente virtuale
python -m venv .venv
source .venv/bin/activate  # Su Windows: .venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt
```

---

## 📂 Struttura del Progetto

```
root/
│
├── recbole/                        # Addestramento e valutazione modelli
│  
│
├── llm/                            # Re-ranking LLM-based con Gemini
│   ├── Amazon/                     # Script e prompt per dataset Amazon
│   │   ├── reranking_batch_amazon_Bpr/
│   │   ├── reranking_batch_amazon_Cke/
│   │   ├── reranking_batch_amazon_Dmf/
│   │   ├── reranking_batch_amazon_Kgcn/
│   │   └── reranking_batch_amazon_Lightgcn/
│   └── Movielens/                  # Script e prompt per dataset MovieLens
│       ├── reranking_batch_movielens_BPR/
│       ├── reranking_batch_movielens_CKE/
│       ├── reranking_batch_movielens_DMF/
│       ├── reranking_batch_movielens_KGCN/
│       └── reranking_batch_movielens_LIGHTGCN/
│
├── results/                        # Risultati sperimentali
│   ├── Amazon_results.xlsx
│   └── MovieLens_results.xlsx
│
├── graphs/                         # Grafici e codici per generarli
│
├── requirements.txt
├── LICENSE
└── README.md
``` 

---

## 📦 Dataset

| Dataset | Utenti | Item | Interazioni | Tipo |
|---|---|---|---|---|
| **MovieLens-1M** | 6.040 | 3.952 | ~1M | Rating espliciti (1–5) |
| **Amazon Books (ridotto)** | 1760 | 78.142 | 900K | Rating espliciti, filtrati ≥500 interazioni/utente |

Entrambi i dataset sono gestiti tramite **RecBole** nel formato Atomic Files (`.inter`, `.item`, `.kg`, `.link`).

---

## 🤖 Modelli

I seguenti modelli di raccomandazione sono stati addestrati e valutati tramite il framework [RecBole](https://recbole.io/):

| Modello | Tipo |
|---|---|
| **BPR** | Collaborative Filtering |
| **DMF** | Collaborative Filtering |
| **LightGCN** | Graph-based |
| **KGCN** | Knowledge-Aware |
| **CKE** | Knowledge-Aware |

---

## ⚙️ Metodologia

L'approccio adottato segue un'architettura **Two-Stage Retrieval & Re-ranking**:

1. **Stage 1 — Retrieval**: i modelli RecBole generano una lista di candidati (top-50) per ogni utente
2. **Stage 2 — Re-ranking LLM**: **Gemini** riceve in input la lista di candidati con i relativi metadati e la riordina per massimizzare la creatività delle raccomandazioni, mantenendo un livello accettabile di accuratezza

---

## 📏 Metriche di Valutazione

### Accuratezza
| Metrica | Descrizione |
|---|---|
| **Precision@K** | Frazione di item raccomandati effettivamente rilevanti |
| **Recall@K** | Frazione di item rilevanti effettivamente raccomandati |
| **NDCG@K** | Qualità del ranking con penalizzazione posizionale |

### Beyond-Accuracy (Creatività)
| Metrica | Descrizione | Valore ideale |
|---|---|---|
| **AveragePopularity@K** | Popolarità media degli item raccomandati | ↓ minore = meglio |
| **Gini Index** | Equità della copertura del catalogo | ↓ minore = meglio |
| **Shannon Entropy** | Contenuto informativo della distribuzione | ↑ maggiore = meglio |
| **Novelty@K** | Popolarità inversa logaritmica  | ↑ maggiore = meglio |
| **Unexpectedness@K** | Distanza coseno media dal profilo utente | ↑ maggiore = meglio |
| **Serendipity (Ge)@K** | Item rilevanti non presenti nella lista Most Popular | ↑ maggiore = meglio |
| **Serendipity (Yan)@K** | Continua: rilevanza × inaspettatezza | ↑ maggiore = meglio |
| **ItemCoverage@K** | Frazione del catalogo coperta dalle raccomandazioni | ↑ maggiore = meglio |

---

## 📈 Domande di Ricerca e Risultati Principali

### RQ1 — Quale modello raggiunge il miglior livello di creatività dopo il re-ranking?

**KGCN** si distingue nettamente su entrambi i dataset, registrando i valori più elevati di Unexpectedness e ItemCoverage dopo il re-ranking. L'integrazione di grafi della conoscenza favorisce la generazione di raccomandazioni più creative e diversificate, amplificando ulteriormente le proprietà creative nelle liste finali prodotte da Gemini.

### RQ2 — Quale modello offre il miglior trade-off tra creatività e accuratezza?

Su **Amazon**, **KGCN** presenta il miglior bilanciamento: subisce la penalità di accuratezza più contenuta (ΔNDCG@10 = −0.017) ottenendo al contempo i guadagni più consistenti su ItemCoverage (+0.042). Su **MovieLens**, **BPR** emerge come scelta ottimale, con la perdita di accuratezza più bassa (ΔNDCG@10 = −0.091) e guadagni positivi su Novelty e Unexpectedness.

---



## 🔮 Sviluppi Futuri

L'evoluzione naturale di questo lavoro prevede la personalizzazione del concetto di
creatività all'interno della strategia di re-ranking. Anziché affidarsi a una definizione
fissa e uniforme per tutti gli utenti, il sistema potrebbe adattarla dinamicamente in base
al profilo di ciascuno — calcolando il grado di eterogeneità dello storico tramite la
distanza coseno media tra le rappresentazioni vettoriali degli item interagiti. Questo
consentirebbe il passaggio da una creatività **generica e statica** a una creatività
**adattiva e personalizzata**, in grado di massimizzare il valore percepito della
raccomandazione per ogni singolo utente.


---




## 👨‍🎓 Autore

> **Federico Martinelli**

Tesi:
> *Tecniche di Re-Ranking basate su Large Language Models per Recommender Systems Orientati alla Creatività*

Corso di Laurea in Informatica e Tecnologie per la produzione del software
Università degli Studi di Bari "Aldo Moro"

📫 Contatti: federicomartinelli35@gmail.com

🔗 GitHub: https://github.com/Feder352

> **Relatore:** Prof. Cataldo Musto **Correlatrice:** Prof.ssa Allegra De Filippo
> **Anno accademico:** 2024–2025

---

## 📄 Licenza

Questo progetto è distribuito sotto licenza MIT — vedi il file [LICENSE](./LICENSE) per i dettagli.
