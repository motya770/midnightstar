# How I Trained Gene-Disease Discovery Models on Cloud GPUs Using Modal

**Serverless GPU training for graph neural networks — and what three different architectures taught me about biological networks.**

---

Training deep learning models on biological networks with 20,000+ genes and 470,000+ protein interactions can be slow on a laptop CPU. I needed GPU acceleration without setting up AWS instances, managing Docker containers, or dealing with CUDA driver installs.

Enter **Modal** — a serverless GPU platform that lets you run Python functions on cloud GPUs with zero infrastructure.

Here's how I wired it up, what happened when I trained three different models on the full human protein interaction network, and why the simplest model crushed the others.

## The Problem

MidnightStar is a platform that downloads real genomic databases (GWAS, GTEx, Human Protein Atlas, STRING, AlphaFold), builds gene interaction graphs, and trains deep learning models to predict novel gene-disease connections. The full human interactome graph has **20K+ nodes** and **470K+ edges** with 60-dimensional feature vectors per node (54 tissue expression values from GTEx + AlphaFold structural features + network degree + node type).

I wanted to compare three architectures on this graph:

| Model | Architecture | How it works |
|-------|-------------|--------------|
| **GNN** | GraphSAGE (SAGEConv) | Mean-aggregates neighbor features through message passing |
| **Graph Transformer** | TransformerConv + RWSE | Attention-weighted neighbors with Random Walk Structural Encoding |
| **VAE** | Variational Autoencoder (SAGEConv encoder) | Compresses to latent space, reconstructs the network |

Training all three locally on CPU was doable but slow. With Modal, each model trains in under 2 minutes on a T4 GPU.

## How I Integrated Modal

### 1. Define the Remote Environment

```python
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "torch-geometric>=2.5.0",
    "scikit-learn>=1.4.0",
    "networkx>=3.2",
)

app = modal.App("midnightstar-training", image=image)
```

No Dockerfile. No `apt-get`. Modal handles CUDA, cuDNN, and GPU drivers automatically.

### 2. Ship the Source Code

The biggest challenge with serverless functions: your remote environment doesn't have your source code. I collect all Python source files and ship them alongside the training data:

```python
def collect_source_code() -> dict:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    files = {}
    for dirpath, _, filenames in os.walk(os.path.join(base, "src")):
        for fname in filenames:
            if fname.endswith(".py"):
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, base)
                with open(full) as f:
                    files[rel] = f.read()
    return files
```

On the remote side, these files get written to a temp directory and added to `sys.path`. The model classes, trainer, and data utilities all import normally — the remote function doesn't know it's running in the cloud.

### 3. Define GPU Functions

```python
@app.function(gpu="T4", timeout=3600)
def train_on_gpu(model_class_name, model_kwargs, data_bytes, config_dict, src_code):
    return _do_train(model_class_name, model_kwargs, data_bytes, config_dict, src_code)

@app.function(gpu="A10G", timeout=3600)
def train_on_a10g(model_class_name, model_kwargs, data_bytes, config_dict, src_code):
    return _do_train(model_class_name, model_kwargs, data_bytes, config_dict, src_code)

@app.function(gpu="A100", timeout=7200)
def train_on_a100(model_class_name, model_kwargs, data_bytes, config_dict, src_code):
    return _do_train(model_class_name, model_kwargs, data_bytes, config_dict, src_code)
```

Each function is identical except for the GPU type and timeout. The `_do_train` function handles all the work: deserialize the data, instantiate the model, move everything to GPU, train, evaluate, serialize the results back.

### 4. Serialize Everything

The graph data (PyTorch Geometric `Data` objects) gets pickled and sent as bytes. The trained model's state dict comes back the same way:

```python
# Client side: send data
data_bytes = pickle.dumps(pyg_data.cpu())

# Remote side: receive and use
pyg_data = pickle.loads(data_bytes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
pyg_data = pyg_data.to(device)

# After training: send results back
model = model.cpu()
return {
    "history": history,
    "metrics": metrics,
    "model_state_dict": pickle.dumps(model.state_dict()),
    "device_used": str(device),
}
```

Moving the model back to CPU before serializing is critical — the client machine likely doesn't have a CUDA GPU, so the tensors need to be CPU-compatible.

### 5. Wire It Into the UI

In the Streamlit training page, users choose between Local CPU, Mac GPU (MPS), and Remote GPU (Modal) with a simple radio button. If they pick Modal, they also choose the GPU tier (T4 at ~$0.60/hr, A10G at ~$1.10/hr, or A100 at ~$3.00/hr).

First run takes ~60 seconds for the cold start (building the container). Subsequent runs start in seconds.

## The Results

I trained all three models on the full human protein interaction network (20K+ nodes, 470K+ edges, 60 features per node). Same training config: 100 epochs, lr=0.001, 80/10/10 train/val/test split.

| Model | AUC-ROC | What it means |
|-------|---------|---------------|
| **GNN (GraphSAGE)** | **0.90** | Correctly ranks real connections above non-connections 90% of the time |
| **VAE** | **0.69** | Learns a compressed representation but sacrifices prediction accuracy |
| **Graph Transformer** | **0.63** | Attention mechanism hurts more than it helps on this graph |

**The simplest model won by a massive margin.**

### Why GNN Dominates

GraphSAGE with mean aggregation is the perfect inductive bias for protein interaction networks. The task is fundamentally: "do these two genes have similar neighborhoods?" Mean-aggregating neighbor features answers exactly that question. The 60-dimensional feature vectors (tissue expression profiles, AlphaFold structural data) carry strong biological signal — the GNN just needs to smooth these features across neighborhoods, not learn complex transformations. With only 2 layers and 64 hidden dimensions, it has very few parameters and generalizes cleanly.

### Why the Transformer Underperformed (0.63)

TransformerConv computes attention weights across all neighbors. In protein interaction networks, most neighbors carry roughly equal importance — unlike NLP where some tokens matter far more than others. The attention overhead adds parameters without adding useful inductive bias compared to simple mean aggregation.

The RWSE (Random Walk Structural Encoding) uses a Hutchinson estimator with random probes to approximate walk return probabilities. On a 20K-node graph, this approximation is noisy — the positional encodings aren't precise enough to give the attention mechanism clean structural information. More parameters + noisy structural encoding = overfitting the training edges while performing poorly on held-out edges.

### Why the VAE Landed in the Middle (0.69)

The VAE optimizes two competing objectives: reconstruction loss (predict real edges) and KL divergence (keep the latent space smooth and regular). The KL term with beta=1.0 pushes embeddings toward a standard normal distribution. This is great for generating novel samples and finding hidden clusters, but it directly penalizes the model for being too confident about specific edges — which is exactly what link prediction demands.

The VAE's real value isn't in raw AUC-ROC. It's in the latent space: genes that end up close together in the compressed 32-dimensional space likely share biological pathways, even if they aren't directly connected. It's a discovery tool more than a prediction tool.

## What I Learned

**1. Feature engineering beats model complexity.** The GNN with simple mean aggregation outperformed the Transformer because GTEx expression profiles (54-dimensional feature vector per gene) carry more signal than attention-weighted aggregation could add. Start simple.

**2. Serialization is the hard part of serverless ML.** Getting PyTorch Geometric data objects, model state dicts, and source code to cross the local→remote boundary cleanly took more thought than the actual training code. Everything must be CPU-compatible and pickle-safe.

**3. T4 is enough for most graph workloads.** Our largest graph trains in under a minute on a T4. Graph neural networks are more memory-bound than compute-bound. The A100 shines for larger graphs, but the T4 at ~$0.60/hr handles the common case.

**4. Keep the same trainer code locally and remotely.** The `Trainer` class doesn't know whether it's running on a laptop CPU or an A100. It just checks `torch.cuda.is_available()` and moves tensors accordingly. Local debugging works identically to remote execution.

**5. Ship your source code, don't bake it into the image.** Baking source code into the Modal image means rebuilding the container on every code change. Shipping it as a parameter means the image only rebuilds when dependencies change — which is rare.

**6. Cold starts are real but manageable.** First Modal invocation builds the container image — about 60 seconds. After that, containers stay warm for a few minutes. For iterative hyperparameter tuning, the experience is nearly instant.

## The Takeaway

A researcher without any GPU hardware or cloud infrastructure knowledge can now:
1. Open the Streamlit app
2. Select "Remote GPU (Modal)" from a dropdown
3. Pick a GPU tier
4. Click "Start Training"
5. Get results back in under 2 minutes

Total infra setup required: `pip install modal && modal setup`. That's it.

And the biggest lesson from the actual model comparison: **don't reach for a Transformer when a GNN will do.** On biological networks where the signal lives in local neighborhoods and rich node features, simple message passing beats attention. The 0.90 AUC-ROC came from the model with the fewest parameters — not the fanciest architecture.

---

Modal turned what would have been a week of DevOps into a single afternoon of Python. And running all three model comparisons on cloud GPUs took less time than writing this post.

---

#MachineLearning #Modal #ServerlessGPU #DeepLearning #GraphNeuralNetworks #PyTorch #CloudComputing #Bioinformatics #DrugDiscovery #Python
