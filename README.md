# MidnightStar

Find hidden gene–disease connections with graph machine learning.

MidnightStar is a local-first Streamlit app that downloads five public genomics
databases, assembles them into a gene interaction network, trains a graph
neural network to predict missing links, and lets you explore the results
without writing code. Everything runs on your machine; the cloud GPU option is
opt-in.

## What it does

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

## Requirements

- Python 3.11 or newer
- macOS or Linux (on Apple Silicon the GPU is offered as a training target)
- Roughly 3 GB of free disk for the datasets and saved networks
- Optional: a [Modal](https://modal.com) account for cloud GPU training

## Setup

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

## Training

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

## Interpreting predictions

The training objective is protein–protein link prediction on the STRING
network. The **Predict Disease Genes** tab ranks genes by cosine similarity of
their learned embeddings to the embeddings of genes already associated with
the chosen trait; it is a similarity heuristic, not a supervised disease
classifier. A heterogeneous gene–disease graph that trains on disease links
directly is specified in
`docs/superpowers/specs/2026-03-30-gene-disease-hetero-graph-design.md`.

The Transformer's positional encoding is a randomised estimate; it is
deterministic for a given graph, but raising **Position encoding accuracy**
(the number of probes) reduces its noise at the cost of preprocessing time.

## Development

```bash
pytest                      # 80 tests, CPU only, about 15 s
pytest tests/models         # model and trainer tests, including the training regression suite
```

`tests/models/test_training_regressions.py` pins the behaviours that matter
for trustworthy metrics: edge attributes stay attached to their edges through
the split, supervision edges are disjoint from the message-passing graph, the
degree feature comes from the message-passing graph, early stopping restores
the best weights, the mini-batch step replays the same stochastic encoder pass
and matches the full-batch gradient, and the RWSE cache is deterministic.

### Layout

```
app.py                     Streamlit entry point and navigation
pages/                     One file per page (download, search, explorer, training, predictions, data browsers)
src/data/                  Bulk downloaders, SQLite manager, live API clients (GWAS, GTEx, HPA, STRING, AlphaFold, mygene)
src/models/                GNN, Graph Transformer, Graph VAE, Trainer, Modal app
src/utils/                 NetworkX -> PyG conversion, graph builder, session helpers, text explainer
tests/                     pytest suite
docs/                      Graph Transformer tutorial, design specs, write-ups
lib/                       Vendored JS assets for the network visualisation
.datasets/                 Local database, saved networks and checkpoints (git-ignored)
```

### Further reading

- `docs/graph-transformer-tutorial.md` – how the Graph Transformer link
  predictor works, step by step.
- `docs/superpowers/specs/` – design documents.
