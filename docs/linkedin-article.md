# I Built a Gene-Disease Discovery Platform Using Deep Learning — Here's What I Learned

**How Graph Neural Networks can find hidden biological connections that traditional methods miss.**

---

Last week I built MidnightStar — an open-source platform that downloads real genomic databases, builds gene interaction networks, and trains deep learning models to predict novel gene-disease correlations.

The result? A GNN model that scores **0.88 AUC-ROC** on link prediction across the human protein interaction network.

Here's the full story.

---

## The Problem

Modern biomedical research generates massive datasets:
- **1 million+** genetic variant-disease associations (GWAS Catalog)
- **56,000 genes** with expression data across 54 human tissues (GTEx)
- **20,000 proteins** with subcellular location and tissue data (Human Protein Atlas)
- **13.7 million** protein-protein interactions (STRING)
- **20,000+** predicted protein structures (AlphaFold)

No human can connect dots across all of this. But a graph neural network can.

## The Approach

I treated biology as a graph problem:

**Nodes** = genes and diseases (~32,000)
**Edges** = known interactions and associations (~470,000 at high confidence)
**Node features** = tissue expression (54 tissues), structural confidence (AlphaFold pLDDT), protein disorder fraction, network degree

Then I trained three different models to predict **missing edges** — connections that likely exist but aren't yet in any database:

| Model | How it works | What it finds |
|-------|-------------|---------------|
| **GNN (GraphSAGE)** | Averages neighbor features through message passing | Genes with similar neighborhoods |
| **Graph Transformer** | Attention-weighted neighbor aggregation + structural encoding | Subtle long-range patterns |
| **VAE** | Compresses genes to latent space, reconstructs the network | Hidden clusters and groupings |

## The Architecture

Built with Python and Streamlit. The stack:

```
Data Layer:     GWAS + GTEx + HPA + STRING + AlphaFold → SQLite (2.2 GB local)
Graph Layer:    NetworkX → PyTorch Geometric
Model Layer:    GNN / Graph Transformer / VAE → Link prediction
UI Layer:       Streamlit multi-page app with Pyvis network visualization
```

Key design decisions:

**1. Download everything locally.** All 5 databases are bulk-downloaded into SQLite. No API calls during training or exploration. The full human interactome sits in a 2.2 GB database on your machine.

**2. Guided ML configuration.** Users pick a model type and adjust hyperparameters through sliders — no code required. The platform was designed for researchers who don't have deep biology or ML backgrounds.

**3. Multi-source graph building.** STRING protein interactions form the backbone. GTEx expression becomes node features. GWAS associations add disease nodes. HPA provides subcellular location. AlphaFold adds structural confidence. All merged into one graph.

## Results

Training a 2-layer GNN (GraphSAGE, hidden dim 64) on the full human protein interaction network:

- **AUC-ROC: 0.88** — the model correctly ranks real connections above non-connections 88% of the time
- **Average Precision: 0.85** — 85% of top-ranked predictions are real connections
- Training time: ~2 minutes on CPU for the full graph

What this means: when the model predicts a new gene-gene connection with high confidence, there's a strong chance it reflects a real biological relationship not yet captured in existing databases.

## What I Learned

**Graph neural networks are remarkably effective on biological networks.** The message-passing paradigm maps naturally onto how biology works — genes influence their interaction partners, who influence their partners, creating pathway-level patterns that GNNs capture inherently.

**Feature engineering matters more than model complexity.** The GNN with simple mean aggregation outperformed the Graph Transformer on this dataset. The expression profiles from GTEx (54-dimensional feature vector per gene) carried more signal than attention-weighted aggregation could add. Start simple.

**The data integration challenge is the real bottleneck.** Building the platform took 3,600+ lines of code. The ML models are ~200 lines. The data clients, caching, graph building, and visualization are the other 3,400. In bioinformatics, getting the data right is 90% of the work.

**AlphaFold adds a new dimension.** Protein structure confidence (pLDDT) and disorder fraction provide information that expression data alone can't capture. Disordered proteins are often interaction hubs — knowing this helps the model understand network topology.

## The Stack

For anyone wanting to build something similar:

- **Streamlit** — Rapid UI prototyping with built-in widgets for ML parameter tuning
- **PyTorch Geometric** — Graph neural networks with SAGEConv, TransformerConv, GCNConv
- **NetworkX + Pyvis** — Graph construction and interactive browser-based visualization
- **SQLite** — Local bulk data storage (no cloud database needed)
- **3Dmol.js** — Protein structure viewer embedded in the browser

Total: **3,800+ lines of Python**, 54 tests, 50 files, 5 data sources, 3 ML models, 10 Streamlit pages.

## What's Next

- **GNNExplainer integration** — Model-interpretable "why" explanations for each prediction
- **Multiprocessing training** — Non-blocking UI during large model training runs
- **Cross-species analysis** — Expand beyond human to model organisms
- **Drug target scoring** — Combine predicted interactions with druggability databases

---

The code is open source. If you're working on computational biology, drug discovery, or just curious about how GNNs work on real biological data — check it out.

MidnightStar shows that you don't need a supercomputer or a PhD in bioinformatics to find novel gene-disease connections. You need good data, the right graph structure, and a message-passing algorithm that lets biology speak for itself.

---

#MachineLearning #GraphNeuralNetworks #Bioinformatics #DeepLearning #DrugDiscovery #ComputationalBiology #Python #OpenSource #GeneticResearch #AlphaFold #Streamlit #PyTorch
