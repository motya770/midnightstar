"""Remote GPU training via Modal."""
import json
import logging
import os
import pickle
import tempfile

logger = logging.getLogger(__name__)

# Modal app definition — imported lazily to avoid requiring modal for local-only users
_app = None
_image = None


def _get_modal_app():
    global _app, _image
    if _app is not None:
        return _app, _image

    import modal

    _image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch>=2.2.0",
        "torch-geometric>=2.5.0",
        "scikit-learn>=1.4.0",
        "networkx>=3.2",
    )
    _app = modal.App("midnightstar-training", image=_image)
    return _app, _image


def is_modal_configured() -> bool:
    """Check if Modal is installed and authenticated."""
    try:
        import modal
        # Try to check if token exists
        token = modal.config._profile
        return True
    except Exception:
        return False


def get_modal_status() -> str:
    """Return a human-readable Modal status."""
    try:
        import modal
        return "Modal installed and ready"
    except ImportError:
        return "Modal not installed. Run: pip install modal && modal setup"


def train_remote(
    model_class_name: str,
    model_kwargs: dict,
    pyg_data_bytes: bytes,
    train_config: dict,
    gpu: str = "T4",
) -> dict:
    """
    Submit a training job to Modal and return results.

    Args:
        model_class_name: "GNNLinkPredictor", "GraphTransformerLinkPredictor", or "GraphVAE"
        model_kwargs: kwargs for model __init__
        pyg_data_bytes: pickle-serialized PyG Data object
        train_config: dict with epochs, lr, train_ratio, val_ratio, early_stopping, patience
        gpu: "T4", "A10G", "A100", or "H100"

    Returns:
        dict with: history, metrics, model_state_dict, epochs_run
    """
    import modal

    app, image = _get_modal_app()

    @app.function(gpu=gpu, timeout=3600)
    def _train_on_gpu(model_class_name, model_kwargs, data_bytes, config_dict):
        import pickle
        import torch
        from torch_geometric.data import Data
        from dataclasses import dataclass

        # Deserialize data
        pyg_data = pickle.loads(data_bytes)

        # Import and create model
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

        # Training setup
        from src.models.trainer import Trainer, TrainConfig
        config = TrainConfig(
            epochs=config_dict["epochs"],
            lr=config_dict["lr"],
            train_ratio=config_dict["train_ratio"],
            val_ratio=config_dict["val_ratio"],
            early_stopping=config_dict.get("early_stopping", False),
            patience=config_dict.get("patience", 10),
        )
        trainer = Trainer(model, config)

        # Train
        history = trainer.train(pyg_data)

        # Evaluate
        metrics = trainer.evaluate(pyg_data)

        # Serialize model state back to CPU
        model = model.cpu()
        state_dict_bytes = pickle.dumps(model.state_dict())

        return {
            "history": history,
            "metrics": metrics,
            "model_state_dict": state_dict_bytes,
            "epochs_run": len(history["train_loss"]),
            "device_used": str(device),
        }

    # Run on Modal
    with app.run():
        result = _train_on_gpu.remote(
            model_class_name, model_kwargs, pyg_data_bytes, train_config
        )

    # Deserialize model state dict
    result["model_state_dict"] = pickle.loads(result["model_state_dict"])
    return result


def serialize_pyg_data(pyg_data) -> bytes:
    """Serialize PyG Data for sending to Modal."""
    import torch
    # Move to CPU before serializing
    cpu_data = pyg_data.cpu()
    return pickle.dumps(cpu_data)
