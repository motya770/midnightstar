Trained three different AI models on the full human protein interaction network — 20,000 genes, 470,000 connections, data from 5 genomic databases — using cloud GPUs through Modal.

The results surprised me.

I compared three deep learning architectures on the same biological graph:
- GNN (simple neighbor averaging): 0.90 AUC-ROC
- VAE (compresses genes to latent space): 0.69
- Graph Transformer (attention + structural encoding): 0.63

The simplest model crushed the others. By a lot.

Why? Protein interaction networks are fundamentally about "do these two genes have similar neighborhoods?" A basic GNN answers exactly that question. The Transformer's attention mechanism tries to learn which neighbors matter more — but in biology, most neighbors matter roughly equally. More parameters, worse results.

The VAE is interesting for a different reason. Its 0.69 score doesn't tell the full story — it's not optimizing for prediction accuracy. It's learning a compressed map of biology. Genes that land close together in its 32-dimensional latent space likely share pathways, even if they're not directly connected. It's a discovery tool, not a prediction tool.

The Modal part: I went from "runs on my laptop" to "trains on an A100 GPU" by adding ~100 lines of Python. No Docker. No AWS. No CUDA driver installs. You decorate a function with @app.function(gpu="T4") and it just works. Each model trained in under 2 minutes.

Sometimes the straightforward approach wins — in both the model architecture and the infrastructure.

#MachineLearning #DeepLearning #Modal #GraphNeuralNetworks #Bioinformatics #Python
