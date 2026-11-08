<h1 align="center">🧬 MidnightStar</h1>

<p align="center"><strong>Find hidden gene–disease connections with graph machine learning.</strong></p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch 2.x" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="PyTorch Geometric 2.5+" src="https://img.shields.io/badge/PyG-2.5%2B-3C2179?logo=pyg&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Modal cloud GPU" src="https://img.shields.io/badge/Modal-cloud%20GPU-7C3AED?logo=modal&logoColor=white">
  <img alt="Tested with pytest" src="https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white">
  <img alt="Platform macOS | Linux" src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey">
</p>

<p align="center">
  <img alt="Built with Python, PyTorch, scikit-learn, SQLite, Linux and Git" src="https://skillicons.dev/icons?i=python,pytorch,sklearn,sqlite,linux,git">
  <br>
  <sub>Python · PyTorch · PyTorch Geometric · scikit-learn · SQLite · Streamlit · Modal</sub>
</p>

MidnightStar is a local-first Streamlit app that downloads five public genomics
databases, assembles them into a gene interaction network, trains a graph
neural network to predict missing links, and lets you explore the results
without writing code. Everything runs on your machine; the cloud GPU option is
opt-in.

## 🚀 What it does

1. **Get data** – one-click download of GWAS Catalog, GTEx v8, Human Protein
   Atlas, STRING v12 and AlphaFold confidence scores into a local SQLite
   database (`.datasets/datasets.db`, about 2.5 GB when complete).
2. **Search and explore** – look up a gene (e.g. `TP53`) to see its disease
   associations, tissue expression, protein interactions and predicted 3D
   structure, and browse its interaction neighbourhood as a graph.
3. **Build a network** – either the full STRING network (~20K genes, edge count
   set by the confidence threshold) or the neighbourhood of one gene. Nodes
   carry expression, structure and GWAS features; edges carry the STRING
   evidence channels.
4. **Train a model** – choose a GNN, a Graph Transformer or a Graph VAE and
   train it for link prediction on CPU, Apple GPU (MPS) or a Modal cloud GPU.
5. **Predict** – rank candidate genes for a disease, find genes with similar
   learned profiles, or score a specific gene pair.

## 📦 Requirements

- Python 3.11 or newer
- macOS or Linux (on Apple Silicon the GPU is offered as a training target)
- Roughly 3 GB of free disk for the datasets and saved networks
- Optional: a [Modal](https://modal.com) account for cloud GPU training

## ⚙️ Setup

```bash
git clone <this repo> midnightstar && cd midnightstar
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"        # add ",gpu" inside the brackets for Modal support
streamlit run app.py
```

Open http://localhost:8501. On first use go to **Download Datasets**; the
bulk files take 5–10 minutes, the AlphaFold step queries the public API per
gene and takes longer. Downloads are one-time and stored under `.datasets/`,
which is git-ignored.

## 🧠 Training

The **Train Model** page converts the current network to a PyTorch Geometric
graph and trains a link predictor.

**Data split.** Unique undirected edges are shuffled with a fixed seed and
split 80 / 10 / 10 by default (configurable). Training edges are further split
85 / 15 into message-passing edges and supervision edges, so the model never
scores an edge it can also see in the graph. Negative edges are sampled against
all known edges. Node degree is recomputed from the message-passing graph for
each split so held-out edges cannot leak through the features. The sliders
cannot produce an empty test split.

**Models.**

| Model | Encoder | Notes |
|---|---|---|
| GNN (default) | GraphSAGE layers | Fast; good baseline |
| Graph Transformer | Pre-norm `TransformerConv` blocks with feed-forward paths | Uses the STRING evidence channels as edge features, optional random-walk positional encoding (RWSE), AdamW with warmup |
| Graph VAE | GraphSAGE encoder with a Gaussian latent space | KL-regularised; useful for similarity search |

All three score a pair as the dot product of the two node embeddings and are
trained with binary cross-entropy on logits.

**Training controls.** Epochs, learning rate, early stopping (restores the
best-validation weights), random seed (same seed means the same split,
negatives and initialisation, so models are comparable), and an optional
mini-batch mode that chunks the edge scoring and applies edge dropout. Note
that mini-batch mode does not reduce encoder memory; the encoder always sees
the whole message-passing graph.

**Metrics.** AUC-ROC and average precision on the held-out test edges. Each run
is recorded in the database and saved as a checkpoint in
`.datasets/trained_models/<model>_<timestamp>_checkpoint.pt` together with its
constructor arguments, metrics and training configuration. The **Predictions**
page can reload any checkpoint.

**Where to train.**

- *This computer (CPU)* – always available.
- *This Mac's GPU* – shown when PyTorch detects MPS. A 20K-gene network takes
  roughly 30–50 s per epoch.
- *Cloud GPU (Modal)* – T4, A10G or A100. Requires `pip install modal` and
  `modal token new`. Modal only schedules GPU functions for workspaces that
  have a payment method; if you see "Please add a payment method", check that
  the active profile in `modal profile list` is the workspace you added the
  card to, and restart the app after changing tokens. The source tree is
  shipped with each job, so no deployment step is needed.

## 🔍 Interpreting predictions

The training objective is protein–protein link prediction on the STRING
network. The **Predict Disease Genes** tab ranks genes by guilt-by-association:
the hypergeometric enrichment of the trait's known genes among a gene's STRING
interaction partners, with cosine similarity of the learned embeddings to the
known genes' mean embedding breaking ties. In a held-out check that hides 20 %
of each trait's known genes, this recovers them about six times more often
than ranking by embedding similarity alone, which mostly surfaces hub genes.
It is a network heuristic, not a supervised disease classifier. The **Export
discoveries** section at the bottom of the page writes the full ranking for
every trait as a CSV bundle with a manifest, ready to hand to Claude Science
or any other tool. A heterogeneous gene–disease graph that trains on disease
links directly is specified in
`docs/superpowers/specs/2026-03-30-gene-disease-hetero-graph-design.md`.

The Transformer's positional encoding is a randomised estimate; it is
deterministic for a given graph, but raising **Position encoding accuracy**
(the number of probes) reduces its noise at the cost of preprocessing time.

### 📏 Held-out disease-gene benchmark

`scripts/benchmark_disease_genes.py` measures how useful a scorer is for
disease-gene ranking, independently of the protein–protein AUC that training
reports. For every trait of category *disease* with at least 10 known GWAS
genes (219 on `full_s800_gwas`), it hides 20 % of the known genes with a fixed
seed, scores every gene from the remaining 80 %, and reports where the hidden
genes land among all candidate genes: recall@50, recall@200, median rank and
MRR (mean over diseases of the reciprocal rank of the best-ranked hidden
gene). Chance recall@50 is about 0.0025.

```bash
python scripts/benchmark_disease_genes.py --graph full_s800_gwas \
    --checkpoint .datasets/trained_models/<run>_checkpoint.pt --json benchmark.json
```

| Scorer | recall@50 | recall@200 | median rank | MRR | distinct genes in top-10 | max rank-1 |
|---|---|---|---|---|---|---|
| Enrichment (production ranking) | 0.042 | 0.083 | 5740 | 0.086 | 1470 | 6 |
| Centroid cosine, mean-centred embeddings (`gnn_20261030_012253`) | 0.022 | 0.063 | 5135 | 0.043 | 662 | 13 |
| Centroid cosine, raw embeddings (`gnn_20261030_012253`) | 0.007 | 0.028 | 5580 | 0.012 | 287 | 34 |
| Hetero gene–disease model alone (`hetero_gnn_20261108_013654`) | 0.002 | 0.011 | 4252 | 0.003 | 982 | 10 |
| Enrichment + hetero model, z-score sum (`hetero+enrichment-z`) | **0.046** | **0.092** | 4264 | **0.094** | 1457 | 7 |
| Enrichment + hetero model, rank average (`hetero+enrichment`) | 0.035 | 0.085 | 4204 | 0.057 | 1349 | 3 |
| Enrichment, hetero model for unscored genes (`enrichment>hetero`) | 0.042 | 0.083 | 4794 | 0.086 | 1480 | 6 |

The last two columns measure hub concentration: how many distinct genes fill
the 219 top-10 lists, and the most diseases any single gene ranks first for.
Raw cosine is hub-dominated (287 genes, one of them first for 34 diseases); the
hetero model is not (982 genes, none first for more than 10).

The heterogeneous model's checkpoint (dot decoder, learning rate 1e-3,
early-stopped at epoch 42 with 1 optimizer step per epoch, so undertrained)
does not beat enrichment on its own, and **enrichment stays the default
ranking**. Summed as z-scores with enrichment it lifts every metric
(recall@50 0.042 → 0.046, MRR 0.086 → 0.094), so the combination is worth
revisiting once a longer-trained checkpoint exists; the training command below
reproduces the run that was stopped at epoch 283 of 1500 (loss 0.37, falling).

The same machinery is available in code: `src.utils.benchmark.run_benchmark`
accepts any function with the signature of `predict_gene_disease_links` (it is
handed the masked index, so it cannot see the hidden genes) or a precomputed
genes × diseases score matrix, and `make_holdout` returns the split so a model
can be trained with exactly the same gene–disease pairs held out. Every result
also reports hub concentration: how many distinct genes fill the top-10 lists
across diseases and the most diseases any one gene ranks first for.

### 🧬 Heterogeneous gene–disease model

The protein–protein models learn who interacts with whom, which is not the
question the Predictions page asks. `scripts/train_hetero_disease_model.py`
trains a model on the question itself: genes and **all** GWAS traits become
nodes of one heterogeneous graph (`src/utils/hetero_graph_features.py`), STRING
interactions and GWAS gene–trait associations become edges, and a
HeteroConv/SAGE encoder with a dot-product decoder (`src/models/hetero_gnn.py`)
is trained to score (gene, disease) pairs (`src/models/hetero_trainer.py`).
Before training, the benchmark's hidden gene–disease pairs are removed from the
graph, so the model is scored on pairs it never saw; the split is stored in the
checkpoint and the benchmark CLI reuses it.

```bash
python scripts/train_hetero_disease_model.py --graph full_s800_gwas --lr 0.003 \
    --epochs 1500 --eval-every 10 --patience 15 --json run.json
python scripts/benchmark_disease_genes.py --graph full_s800_gwas \
    --hetero-checkpoint .datasets/trained_models/hetero_gnn_<stamp>_checkpoint.pt
```

Training is full-batch (one optimizer step per epoch, about a second on CPU at
full size; Apple MPS was several times slower for this model), validates on a
second seeded hold-out of known disease genes scored the benchmark way, and
keeps the best epoch. Options cover
the decoder (`dot`, `bilinear`, `mlp`), degree-aware negatives, and dropping
traits with few genes; `--no-holdout` trains a production model on every known
pair once the honest one has been validated.

When a hetero checkpoint exists in `.datasets/trained_models`, the Predictions
page offers **Ranking score: Network enrichment (default) / Gene–disease
model** in the Predict Disease Genes tab and the export form; the model option
is labelled *experimental, below enrichment on the benchmark* until a checkpoint
beats enrichment. The export's `manifest.json` names the scorer that produced
the bundle, and the predicted links gain a trailing `model_score` column (empty
when enrichment ranked); the existing columns are unchanged.

## 🛠️ Development

```bash
pytest                      # 158 tests, CPU only, about 15 s
pytest tests/models         # model and trainer tests, including the training regression suite
```

`tests/models/test_training_regressions.py` pins the behaviours that matter
for trustworthy metrics: edge attributes stay attached to their edges through
the split, supervision edges are disjoint from the message-passing graph, the
degree feature comes from the message-passing graph, early stopping restores
the best weights, the mini-batch step replays the same stochastic encoder pass
and matches the full-batch gradient, and the RWSE cache is deterministic.

### 🗂️ Layout

```
app.py                     Streamlit entry point and navigation
pages/                     One file per page (download, search, explorer, training, predictions, data browsers)
src/data/                  Bulk downloaders, SQLite manager, live API clients (GWAS, GTEx, HPA, STRING, AlphaFold, mygene)
src/models/                GNN, Graph Transformer, Graph VAE, Trainer, Modal app; hetero gene–disease model and trainer
src/utils/                 NetworkX -> PyG conversion (homogeneous and heterogeneous), graph builder, discovery scoring,
                           benchmark, hetero checkpoint scorer, session helpers, text explainer
scripts/                   Command-line tools: held-out disease-gene benchmark, hetero gene–disease model training
tests/                     pytest suite
docs/                      Graph Transformer tutorial, design specs, write-ups
lib/                       Vendored JS assets for the network visualisation
.datasets/                 Local database, saved networks and checkpoints (git-ignored)
```

### 📚 Further reading

- `docs/graph-transformer-tutorial.md` – how the Graph Transformer link
  predictor works, step by step.
- `docs/superpowers/specs/` – design documents.
