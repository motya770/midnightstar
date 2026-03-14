# pages/3_Model_Training.py
import streamlit as st
import torch
import numpy as np
from torch_geometric.data import Data
from src.models.gnn import GNNLinkPredictor
from src.models.graph_transformer import GraphTransformerLinkPredictor
from src.models.vae import GraphVAE
from src.models.trainer import Trainer, TrainConfig
from src.utils import session

st.title("⚙️ Model Training")

graph = session.get_graph()
if graph is None:
    st.warning("No graph data loaded. Go to the **Search** page first.")
    st.stop()

# Convert NetworkX graph to PyG Data
def nx_to_pyg(G):
    node_list = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)

    # Build node features: expression (GTEx) + degree + node_type one-hot
    all_tissues = set()
    for n in node_list:
        expr = G.nodes[n].get("expression", {})
        all_tissues.update(expr.keys())
    all_tissues = sorted(all_tissues)

    # Feature vector: [expression_per_tissue..., degree, is_gene, is_disease]
    num_features = max(len(all_tissues), 1) + 3

    x = torch.zeros(num_nodes, num_features)
    for i, n in enumerate(node_list):
        # Expression features
        expr = G.nodes[n].get("expression", {})
        for j, tissue in enumerate(all_tissues):
            x[i, j] = expr.get(tissue, 0.0)
        # Degree feature
        x[i, -3] = float(G.degree(n))
        # Node type one-hot
        node_type = G.nodes[n].get("node_type", "")
        x[i, -2] = 1.0 if node_type == "gene" else 0.0
        x[i, -1] = 1.0 if node_type == "disease" else 0.0

    # Normalize expression columns only
    expr_cols = max(len(all_tissues), 1)
    max_vals = x[:, :expr_cols].max(dim=0).values.clamp(min=1.0)
    x[:, :expr_cols] = x[:, :expr_cols] / max_vals
    # Normalize degree
    max_deg = x[:, -3].max().clamp(min=1.0)
    x[:, -3] = x[:, -3] / max_deg

    # Build edges
    src, dst = [], []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            src.append(node_to_idx[u])
            dst.append(node_to_idx[v])
            src.append(node_to_idx[v])
            dst.append(node_to_idx[u])

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes), node_list, node_to_idx

pyg_data, node_list, node_to_idx = nx_to_pyg(graph)
in_channels = pyg_data.x.size(1)

# Layout: three columns
col_data, col_config, col_monitor = st.columns([1, 1, 1])

with col_data:
    st.subheader("📊 Data Selection")
    st.metric("Nodes", pyg_data.num_nodes)
    st.metric("Edges", pyg_data.edge_index.size(1) // 2)
    st.metric("Features", in_channels)
    train_ratio = st.slider("Train split", 0.5, 0.9, 0.8, 0.05)
    val_ratio = st.slider("Validation split", 0.05, 0.3, 0.1, 0.05)

with col_config:
    st.subheader("🧠 Model Configuration")
    model_type = st.radio("Model type", ["GNN", "Graph Transformer", "VAE"])

    if model_type == "GNN":
        num_layers = st.slider("Layers", 1, 5, 2)
        hidden_dim = st.select_slider("Hidden dimension", [16, 32, 64, 128, 256], value=64)
        aggr = st.selectbox("Aggregation", ["mean", "max", "sum"])
        with st.expander("What does this do?"):
            st.markdown("**GNN** learns by passing messages between connected genes. "
                       "It finds genes with similar neighborhoods that might share hidden connections.")
    elif model_type == "Graph Transformer":
        num_layers = st.slider("Layers", 1, 6, 2)
        hidden_dim = st.select_slider("Hidden dimension", [32, 64, 128, 256, 512], value=64)
        num_heads = st.select_slider("Attention heads", [1, 2, 4, 8], value=4)
        rwse_k = st.slider("RWSE walk length (k)", 8, 24, 16)
        with st.expander("What does this do?"):
            st.markdown("**Graph Transformer** uses attention to weigh which gene connections matter most. "
                       "It can detect longer-range patterns than GNN by looking at broader neighborhoods.")
    else:  # VAE
        hidden_dim = st.select_slider("Hidden dimension", [16, 32, 64, 128], value=64)
        latent_dim = st.select_slider("Latent dimension", [8, 16, 32, 64, 128], value=32)
        num_layers = st.slider("Encoder layers", 1, 4, 2)
        beta = st.slider("Beta (KL weight)", 0.1, 10.0, 1.0, 0.1)
        with st.expander("What does this do?"):
            st.markdown("**VAE** compresses gene data into a compact representation. "
                       "Genes that end up close together in this compressed space may share unknown pathways.")

    st.divider()
    epochs = st.slider("Epochs", 10, 500, 100)
    lr = st.select_slider("Learning rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    early_stop = st.checkbox("Early stopping", value=True)

with col_monitor:
    st.subheader("📈 Training Monitor")

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        # Build model
        if model_type == "GNN":
            model = GNNLinkPredictor(in_channels, hidden_dim, num_layers, aggr)
        elif model_type == "Graph Transformer":
            model = GraphTransformerLinkPredictor(
                in_channels, hidden_dim, num_layers, num_heads, rwse_dim=16, rwse_walk_length=rwse_k,
            )
        else:
            model = GraphVAE(in_channels, hidden_dim, latent_dim, num_layers, beta)

        config = TrainConfig(
            epochs=epochs, lr=lr, train_ratio=train_ratio,
            val_ratio=val_ratio, early_stopping=early_stop,
        )
        trainer = Trainer(model, config)

        progress_bar = st.progress(0)
        loss_chart = st.empty()
        status_text = st.empty()
        loss_history = []
        auc_history = []

        def on_epoch(epoch, loss, val_auc):
            loss_history.append(loss)
            auc_history.append(val_auc)
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch {epoch + 1}/{epochs} — Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")
            import pandas as pd
            df = pd.DataFrame({"Loss": loss_history, "Val AUC": auc_history})
            loss_chart.line_chart(df)

        with st.spinner("Training..."):
            history = trainer.train(pyg_data, on_epoch=on_epoch)

        metrics = trainer.evaluate(pyg_data)
        st.success("Training complete!")
        st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
        st.metric("Avg Precision", f"{metrics['avg_precision']:.4f}")

        # Store results
        session.set_training_results({
            "model": model,
            "trainer": trainer,
            "history": history,
            "metrics": metrics,
            "pyg_data": pyg_data,
            "node_list": node_list,
            "node_to_idx": node_to_idx,
            "model_type": model_type,
        })

        if st.button("📊 View Results"):
            st.switch_page("pages/4_Results.py")
