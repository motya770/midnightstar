"""Modal app definition for remote GPU training."""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "torch-geometric>=2.5.0",
    "scikit-learn>=1.4.0",
    "networkx>=3.2",
    "scipy>=1.11",
    "tqdm>=4.66",
)

app = modal.App("midnightstar-training", image=image)


def _stage_source(src_code: dict) -> str:
    """Write the shipped source tree to a temp directory and put it on sys.path; returns the directory."""
    import os
    import sys
    import tempfile

    tmpdir = tempfile.mkdtemp()
    for rel_path, code in src_code.items():
        full_path = os.path.join(tmpdir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(code)
    sys.path.insert(0, tmpdir)
    return tmpdir


def _do_train(model_class_name: str, model_kwargs: dict,
              data_bytes: bytes, config_dict: dict, src_code: dict) -> dict:
    """Core training logic that runs on the remote GPU."""
    import pickle
    import torch

    _stage_source(src_code)

    # Deserialize data
    pyg_data = pickle.loads(data_bytes)

    # Create model
    if model_class_name == "GNNLinkPredictor":
        from src.models.gnn import GNNLinkPredictor
        model = GNNLinkPredictor(**model_kwargs)
    elif model_class_name == "GraphTransformerLinkPredictor":
        from src.models.graph_transformer import GraphTransformerLinkPredictor
        model = GraphTransformerLinkPredictor(**model_kwargs)
    elif model_class_name == "GraphVAE":
        from src.models.vae import GraphVAE
        model = GraphVAE(**model_kwargs)
    else:
        raise ValueError(f"Unknown model: {model_class_name}")

    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pyg_data = pyg_data.to(device)

    from dataclasses import fields
    from src.models.trainer import Trainer, TrainConfig
    # Forward every TrainConfig field the UI sent; the dataclass supplies defaults
    # for the rest, so new fields never need to be hand-copied here.
    allowed = {f.name for f in fields(TrainConfig)} - {"device"}
    config = TrainConfig(**{k: v for k, v in config_dict.items() if k in allowed}, device=str(device))
    trainer = Trainer(model, config)
    history = trainer.train(pyg_data)
    metrics = trainer.evaluate(pyg_data)

    model = model.cpu()
    return {
        "history": history,
        "metrics": metrics,
        "model_state_dict": pickle.dumps(model.state_dict()),
        "epochs_run": len(history["train_loss"]),
        "device_used": str(device),
    }


@app.function(gpu="T4", timeout=3600)
def train_on_gpu(model_class_name: str, model_kwargs: dict,
                 data_bytes: bytes, config_dict: dict, src_code: dict) -> dict:
    return _do_train(model_class_name, model_kwargs, data_bytes, config_dict, src_code)


@app.function(gpu="A10G", timeout=3600)
def train_on_a10g(model_class_name: str, model_kwargs: dict,
                  data_bytes: bytes, config_dict: dict, src_code: dict) -> dict:
    return _do_train(model_class_name, model_kwargs, data_bytes, config_dict, src_code)


@app.function(gpu="A100", timeout=7200)
def train_on_a100(model_class_name: str, model_kwargs: dict,
                  data_bytes: bytes, config_dict: dict, src_code: dict) -> dict:
    return _do_train(model_class_name, model_kwargs, data_bytes, config_dict, src_code)


def _do_train_hetero(payload_bytes: bytes, src_code: dict) -> dict:
    """Train the gene–disease hetero model from a pickled ``hetero_training_payload`` on the remote GPU.

    Returns the pickled CPU state dict, history and metrics; the caller benchmarks and saves locally.
    """
    import pickle
    import torch

    _stage_source(src_code)
    from src.models.hetero_pipeline import train_hetero_from_payload

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return train_hetero_from_payload(pickle.loads(payload_bytes), device=device)


@app.function(gpu="T4", timeout=3600)
def train_hetero_on_gpu(payload_bytes: bytes, src_code: dict) -> dict:
    return _do_train_hetero(payload_bytes, src_code)


@app.function(gpu="A10G", timeout=3600)
def train_hetero_on_a10g(payload_bytes: bytes, src_code: dict) -> dict:
    return _do_train_hetero(payload_bytes, src_code)


@app.function(gpu="A100", timeout=7200)
def train_hetero_on_a100(payload_bytes: bytes, src_code: dict) -> dict:
    return _do_train_hetero(payload_bytes, src_code)


# GPU label shown in the Training page's menu → Modal function for the hetero model.
HETERO_TRAIN_FUNCTIONS = {"T4": train_hetero_on_gpu, "A10G": train_hetero_on_a10g, "A100": train_hetero_on_a100}


def collect_source_code() -> dict:
    """Read all source files needed for remote training."""
    import os
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
