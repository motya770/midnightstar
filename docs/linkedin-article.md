# My Deep Learning Model Predicted a Gene-Alzheimer's Link That Researchers Are Already Investigating

**How I built a platform that finds hidden gene-disease connections using only open-source data — and why the simplest model won.**

---

I built MidnightStar — a platform that downloads real genomic databases, builds gene interaction networks, and trains deep learning models to predict novel gene-disease connections.

Then I asked it: which genes might be linked to Alzheimer's disease that aren't yet in the GWAS database?

Its #1 prediction was **NR5A1** (Nuclear Receptor Subfamily 5 Group A Member 1) with a 0.98 confidence score. I searched PubMed — and found that independent researchers are already investigating the NR5A1-Alzheimer's link in published studies.

The model surfaced a real, active research lead on its own. Using only publicly available open-source data. No proprietary datasets, no private knowledge. Just five open genomic databases and a neural network.

---

## The Data

Five major open genomic databases, all downloaded locally:

- **STRING** — 13.7 million protein-protein interactions
- **GWAS Catalog** — 1 million+ genetic variant-disease associations
- **GTEx** — gene expression across 54 human tissues
- **Human Protein Atlas** — 20,000 proteins with subcellular location and tissue data
- **AlphaFold** — 20,000+ predicted protein structures

Everything runs locally in SQLite. No API calls during training. No cloud dependency.

## The Approach

I treated biology as a graph problem:

- **Nodes** = 20,000 genes
- **Edges** = 230,000 high-confidence protein interactions
- **Node features** = tissue expression (54 tissues) + AlphaFold structure (pLDDT, disorder, sequence length) + GWAS disease/trait associations + network degree

Then I trained models to predict **missing edges** — connections that likely exist but aren't yet in any database.

## Head-to-Head: Three Architectures

I compared three models on the same 20K-node graph:

| Model | AUC-ROC | What it does |
|-------|---------|-------------|
| **GNN (GraphSAGE)** | **0.92** | Averages neighbor features through message passing |
| **Graph Transformer** | 0.71 | Attention-weighted aggregation + positional encoding |

The GNN won by a wide margin. A 2-layer GraphSAGE with 256 hidden dimensions and mean aggregation — the simplest architecture — outperformed every Transformer configuration I tried, including 6-layer models with 512 hidden dimensions and 8 attention heads on A100 GPUs.

## Why Simple Won

**Biology is already a graph.** Genes interact with their neighbors, who interact with their neighbors, creating pathway-level patterns. GraphSAGE's message-passing paradigm maps directly onto this structure. It doesn't need attention mechanisms to decide which neighbors matter — the interaction network already encodes that.

**The Transformer struggled** because its local attention (TransformerConv) doesn't provide true global context, and the positional encoding (Random Walk Structural Encoding) adds noise rather than signal on large graphs. More parameters didn't help — they just made training unstable.

**Feature engineering mattered more than model complexity.** The 54-dimensional tissue expression profile from GTEx, combined with AlphaFold structural features and GWAS disease associations, gave the GNN everything it needed. The signal was in the data, not the architecture.

## The Stack

```
Data Layer:     GWAS + GTEx + HPA + STRING + AlphaFold -> SQLite
Graph Layer:    NetworkX -> PyTorch Geometric
Model Layer:    GNN / Graph Transformer / VAE -> Link prediction
Compute:        Local (CPU/MPS) or remote GPU via Modal (T4/A10G/A100)
UI Layer:       Streamlit multi-page app with interactive network visualization
```

5,600+ lines of Python. Runs on a laptop or scales to A100 GPUs.

## What I Learned

**Start with the simplest model.** I spent hours tuning Transformers and VAEs. The 2-layer GNN I trained as a baseline turned out to be the best model. Don't assume complexity equals performance.

**Data integration is 90% of the work.** The ML models are ~200 lines of code. The data clients, caching, graph construction, feature engineering, and visualization are the other 5,400. In bioinformatics, getting the data right is the real challenge.

**Open data is enough.** Every dataset used here is freely available. No institutional access required. No proprietary knowledge. A neural network trained on public data predicted a gene-disease link that active research is investigating. The tools to do meaningful computational biology are accessible to anyone.

---

If you're working on computational biology, drug discovery, or applied ML — I'm happy to share more about the approach.

MidnightStar shows that you don't need a supercomputer or a PhD in bioinformatics to find novel gene-disease connections. You need good data, the right graph structure, and a message-passing algorithm that lets biology speak for itself.

---

#MachineLearning #GraphNeuralNetworks #Bioinformatics #DeepLearning #DrugDiscovery #ComputationalBiology #Python #OpenSource #GeneticResearch #AlphaFold #Streamlit #PyTorch
