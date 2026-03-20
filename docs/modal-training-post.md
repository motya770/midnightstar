# How I Trained Gene-Disease Discovery Models on Cloud GPUs Using Modal

**Serverless GPU training for graph neural networks — no infrastructure management required.**

---

Training deep learning models on biological networks with 20,000+ genes and 470,000+ protein interactions can be slow on a laptop CPU. I needed GPU acceleration without setting up AWS instances, managing Docker containers, or dealing with CUDA driver installs.

Enter **Modal** — a serverless GPU platform that lets you run Python functions on cloud GPUs with zero infrastructure.

## The Setup

MidnightStar trains three types of graph neural networks to predict hidden gene-disease connections. The training pipeline works locally on CPU or Apple Silicon (MPS), but the full human protein interaction network with 20K+ nodes and 470K+ edges benefits massively from GPU acceleration.

Here's how I integrated Modal into the training pipeline.

## Step 1: Define the Remote Environment

Modal uses container images. I defined one with all the dependencies needed for training:

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

That's it. No Dockerfile. No `apt-get`. Modal handles the CUDA runtime, cuDNN, and GPU drivers automatically.

## Step 2: Ship the Source Code

The biggest challenge with serverless functions: your remote environment doesn't have your source code. I solved this by collecting all Python source files and shipping them alongside the training data:

```python
def collect_source_code() -> dict:
    """Read all source files needed for remote training."""
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

## Step 3: Define GPU Functions

I created separate Modal functions for different GPU tiers:

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

## Step 4: Serialize Everything

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

## Step 5: Wire It Into the UI

In the Streamlit training page, users choose between Local CPU, Mac GPU (MPS), and Remote GPU (Modal) with a simple radio button. If they pick Modal, they also choose the GPU tier (T4, A10G, or A100).

The training call becomes:

```python
with modal_app.run():
    if gpu_type == "A10G":
        result = train_on_a10g.remote(model_class, model_kwargs, data_bytes, config_dict, src_code)
    elif gpu_type == "A100":
        result = train_on_a100.remote(model_class, model_kwargs, data_bytes, config_dict, src_code)
    else:
        result = train_on_gpu.remote(model_class, model_kwargs, data_bytes, config_dict, src_code)
```

First run takes ~60 seconds for the cold start (building the container image). Subsequent runs start in seconds.

## What I Learned

**1. Serialization is the hard part.** Getting PyTorch Geometric data objects, model state dicts, and source code to cross the local→remote boundary cleanly took more thought than the actual training code. Everything must be CPU-compatible and pickle-safe.

**2. Cold starts are real but manageable.** The first Modal invocation builds the container image — about 60 seconds. After that, containers stay warm for a few minutes. For iterative training (run, tweak hyperparameters, run again), the experience is nearly instant.

**3. T4 is enough for most graph workloads.** Our largest graph (20K nodes, 470K edges) trains in under a minute on a T4. Graph neural networks are more memory-bound than compute-bound. The A100 shines for larger graphs or longer training runs, but the T4 at ~$0.60/hr handles the common case.

**4. Keep the same trainer code locally and remotely.** The `Trainer` class doesn't know whether it's running on a laptop CPU or an A100. It just checks `torch.cuda.is_available()` and moves tensors accordingly. This means local debugging works identically to remote execution.

**5. Ship your source code, don't bake it into the image.** Baking source code into the Modal image means rebuilding the container on every code change. Shipping it as a parameter means the image only rebuilds when dependencies change — which is rare.

## The Result

A biologist without any GPU hardware or cloud infrastructure knowledge can now:
1. Open the Streamlit app
2. Select "Remote GPU (Modal)" from a dropdown
3. Pick a GPU tier
4. Click "Start Training"
5. Get results back in under 2 minutes

Total infra setup required: `pip install modal && modal setup`. That's it.

---

Modal turned what would have been a week of DevOps into a single afternoon of Python. For any ML project where you need occasional GPU access without permanent infrastructure, it's the fastest path from "works on my laptop" to "trains on an A100."

---

#MachineLearning #Modal #ServerlessGPU #DeepLearning #GraphNeuralNetworks #PyTorch #CloudComputing #Bioinformatics #Python
