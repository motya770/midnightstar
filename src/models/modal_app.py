"""Modal app definition for remote GPU training."""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "torch-geometric>=2.5.0",
    "scikit-learn>=1.4.0",
    "networkx>=3.2",
)

app = modal.App("midnightstar-training", image=image)


def _do_train(model_class_name: str, model_kwargs: dict,
              data_bytes: bytes, config_dict: dict, src_code: dict) -> dict:
    """Core training logic that runs on the remote GPU."""
    import importlib.util
    import os
    import pickle
    import sys
    import tempfile
    import torch

    # Write source files to a temp directory so imports work
    tmpdir = tempfile.mkdtemp()
    src_root = os.path.join(tmpdir, "src")
    for rel_path, code in src_code.items():
        full_path = os.path.join(tmpdir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(code)

    sys.path.insert(0, tmpdir)

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

    from src.models.trainer import Trainer, TrainConfig
    config = TrainConfig(
        epochs=config_dict["epochs"],
        lr=config_dict["lr"],
        train_ratio=config_dict["train_ratio"],
        val_ratio=config_dict["val_ratio"],
        early_stopping=config_dict.get("early_stopping", False),
        patience=config_dict.get("patience", 10),
        device=str(device),
    )
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
