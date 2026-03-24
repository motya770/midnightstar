Just built something I'm really excited about.

MidnightStar is a platform that uses deep learning to discover hidden connections between genes and diseases. It downloads real data from 5 major genomic databases (13+ million protein interactions, 1 million genetic associations, expression data across 54 human tissues, 20K protein structures from AlphaFold) — and trains graph neural networks to find patterns no human could spot manually.

The best model scores 92% accuracy (AUC-ROC: 0.9188, Avg Precision: 0.9202) at predicting real biological connections on a network of 20,000 genes and 230,000 protein interactions. It can suggest which genes might be linked to which diseases, even when that link isn't in any database yet.

I tested three architectures head-to-head on the same dataset:
- GNN (GraphSAGE) — 0.92 AUC. Simple message passing. Won by a wide margin.
- Graph Transformer — 0.71 AUC. Attention-based. Struggled even with heavy tuning.

Here's the exciting part: I asked the model to predict new gene candidates for Alzheimer's disease — genes not yet in the GWAS database. Its #1 prediction was NR5A1 (Nuclear Receptor Subfamily 5 Group A Member 1) with a 0.98 confidence score. I searched PubMed and found that independent researchers are already investigating the NR5A1-Alzheimer's link in published studies. The model found a real, active research lead on its own — using only publicly available open-source data. No proprietary datasets, no private knowledge, no insider information. Just five open genomic databases and a neural network.

Biggest lesson: the simplest model won. No attention mechanisms, no latent bottlenecks, no positional encodings. Just "average your neighbors' features and learn from that." Biology is already a graph — you don't need to force extra structure onto it.

Latest update adds GWAS disease/trait encoding directly into gene features, so the model learns from genetic associations alongside expression and structure data. Also added remote GPU training via Modal for the heavier models.

Built with Python, PyTorch Geometric, and Streamlit. 5,600+ lines of code. Runs on a laptop or scales to A100 GPUs.

If you're into AI, biology, or just like seeing what happens when you throw neural networks at real scientific data — happy to share more.
