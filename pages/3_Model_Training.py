# pages/3_Model_Training.py
import streamlit as st
import torch
import numpy as np
from torch_geometric.data import Data
from src.models.gnn import GNNLinkPredictor
from src.models.graph_transformer import GraphTransformerLinkPredictor
from src.models.vae import GraphVAE
from src.models.trainer import Trainer, TrainConfig
from src.data.bulk_datasets import BulkDatasetManager
from src.utils import session

st.title("⚙️ Model Training")

graph = session.get_graph()
manager = BulkDatasetManager()
has_data = any(manager.is_downloaded(s) for s in ["gwas", "gtex", "hpa", "string"])

# If no graph in session but datasets are available, let user build or load one
if graph is None:
    if not has_data:
        st.warning("No data available. Go to **Download** to get datasets first.")
        if st.button("📥 Go to Download"):
            st.switch_page("pages/0_Download.py")
        st.stop()

    # Show saved graphs first
    saved_graphs = manager.list_saved_graphs()
    if saved_graphs:
        st.subheader("📂 Saved Graphs")
        for sg in saved_graphs:
            col_info, col_load, col_del = st.columns([5, 1, 1])
            with col_info:
                st.markdown(
                    f"**{sg['name']}** — {sg['nodes']:,} nodes, {sg['edges']:,} edges "
                    f"({sg['created_at'][:16]})"
                )
            with col_load:
                if st.button("Load", key=f"load_{sg['name']}"):
                    with st.spinner(f"Loading {sg['name']}..."):
                        graph = manager.load_graph(sg["name"])
                        if graph:
                            session.set_graph(graph)
                            st.rerun()
                        else:
                            st.error("Graph file not found.")
            with col_del:
                if st.button("🗑️", key=f"del_{sg['name']}"):
                    manager.delete_graph(sg["name"])
                    st.rerun()
        st.divider()

    # Build new graph
    st.subheader("🔨 Build New Graph")

    graph_mode = st.radio(
        "Data scope",
        ["Full dataset (all genes)", "Single gene subgraph"],
        help="Full dataset trains on all genes and interactions. Single gene focuses on a neighborhood.",
    )

    graph_name = st.text_input("Graph name (for saving)", placeholder="e.g., full_score700, tp53_depth2")

    if graph_mode == "Full dataset (all genes)":
        st.markdown("""
        Merges **all 5 sources** into one graph:
        - **Nodes:** All genes from HPA (~20K)
        - **Edges:** STRING protein-protein interactions
        - **Features:** GTEx expression (54 tissues) + AlphaFold structure (pLDDT, disorder, length) + HPA annotations + GWAS associations (diseases, traits, measurements)
        """)
        full_min_score = st.select_slider(
            "Min STRING score (higher = fewer but stronger edges)",
            options=[400, 500, 600, 700, 800, 900],
            value=700,
            help="700 = high confidence (~470K edges), 900 = highest (~200K edges). Lower values use more memory.",
        )
        include_gwas = st.checkbox("Include GWAS associations (as gene features)", value=True)
        max_pvalue = st.select_slider(
            "Max GWAS p-value",
            options=[5e-4, 5e-6, 5e-8, 5e-10, 5e-20, 5e-50],
            value=5e-8,
        ) if include_gwas else 5e-8

        # Auto-generate name if not provided
        if not graph_name:
            graph_name = f"full_s{full_min_score}{'_gwas' if include_gwas else ''}"

        if st.button("🔨 Build Full Graph", type="primary"):
            progress_text = st.empty()
            with st.spinner("Building full graph from all datasets..."):
                if not manager.is_downloaded("string_aliases"):
                    progress_text.text("Building STRING alias index...")
                    try:
                        manager.build_string_alias_table()
                    except Exception:
                        pass
                graph = manager.build_full_graph(
                    min_string_score=full_min_score,
                    include_diseases=include_gwas,
                    max_disease_pvalue=max_pvalue,
                    on_progress=lambda msg: progress_text.text(msg),
                )
                session.set_graph(graph)
                manager.save_graph(graph, graph_name)
                progress_text.text(f"Graph saved as '{graph_name}'")
            st.rerun()
    else:
        gene_input = st.text_input("Gene symbol", placeholder="e.g., TP53, BRCA1, SP4")
        depth = st.slider("Network depth (hops)", 1, 3, 2)
        min_score = st.slider("Min STRING score", 0, 1000, 400, 50)

        if not graph_name and gene_input:
            graph_name = f"{gene_input.upper()}_d{depth}_s{min_score}"

        if gene_input and st.button("🔨 Build Subgraph", type="primary"):
            with st.spinner(f"Building graph for {gene_input.upper()}..."):
                if not manager.is_downloaded("string_aliases"):
                    try:
                        manager.build_string_alias_table()
                    except Exception:
                        pass
                graph = manager.build_graph(gene_input.upper(), depth=depth, min_score=min_score)
                session.set_graph(graph)
                if graph_name:
                    manager.save_graph(graph, graph_name)
            st.rerun()
    st.stop()

# Convert NetworkX graph to PyG Data (cached to avoid recomputing on every widget change)
@st.cache_data(show_spinner="Converting graph to training format...")
def nx_to_pyg(_G, _graph_id):
    """_G prefixed with _ to tell Streamlit not to hash it. _graph_id is used for cache key."""
    G = _G
    node_list = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)

    # Build node features: expression (GTEx) + degree + node_type one-hot
    all_tissues = set()
    for n in node_list:
        expr = G.nodes[n].get("expression", {})
        all_tissues.update(expr.keys())
    all_tissues = sorted(all_tissues)

    # Feature vector: [expression_per_tissue..., plddt, disorder, seq_len, degree]
    # AlphaFold features: mean_plddt, disordered_fraction, sequence_length (3 features)
    has_alphafold = any(G.nodes[n].get("mean_plddt") is not None for n in node_list)
    af_features = 3 if has_alphafold else 0

    num_features = max(len(all_tissues), 1) + af_features + 1  # +1 for degree
    x = torch.zeros(num_nodes, num_features)
    expr_cols = max(len(all_tissues), 1)

    # GWAS features: build trait vocabulary and encode as token sequences
    gwas_categories = ["disease", "trait", "phenotype", "measurement", "biological_process", "other"]
    trait_vocab = {}  # trait_name -> token_id (0 = padding)
    next_id = 1
    for n in node_list:
        for entries in G.nodes[n].get("gwas", {}).values():
            for entry in entries:
                if entry["trait"] not in trait_vocab:
                    trait_vocab[entry["trait"]] = next_id
                    next_id += 1

    # Build category ID mapping for each trait
    cat_to_id = {cat: i for i, cat in enumerate(gwas_categories)}
    trait_to_cat_id = {}
    for n in node_list:
        gwas = G.nodes[n].get("gwas", {})
        for cat, entries in gwas.items():
            for entry in entries:
                trait_to_cat_id[entry["trait"]] = cat_to_id.get(cat, len(gwas_categories))

    # Find max tokens per node for padding
    max_gwas_tokens = 0
    for n in node_list:
        count = sum(len(entries) for entries in G.nodes[n].get("gwas", {}).values())
        max_gwas_tokens = max(max_gwas_tokens, count)
    max_gwas_tokens = max(max_gwas_tokens, 1)  # at least 1 for padding

    gwas_token_ids = torch.zeros(num_nodes, max_gwas_tokens, dtype=torch.long)   # 0 = pad
    gwas_scores = torch.zeros(num_nodes, max_gwas_tokens)
    gwas_cat_ids = torch.zeros(num_nodes, max_gwas_tokens, dtype=torch.long)

    for i, n in enumerate(node_list):
        # Numeric features
        expr = G.nodes[n].get("expression", {})
        for j, tissue in enumerate(all_tissues):
            x[i, j] = expr.get(tissue, 0.0)

        offset = expr_cols
        if has_alphafold:
            x[i, offset] = G.nodes[n].get("mean_plddt", 0.0) / 100.0
            x[i, offset + 1] = G.nodes[n].get("disordered_fraction", 0.0)
            x[i, offset + 2] = G.nodes[n].get("sequence_length", 0) / 5000.0
            offset += 3

        x[i, offset] = float(G.degree(n))

        # GWAS tokens
        tok_idx = 0
        for entries in G.nodes[n].get("gwas", {}).values():
            for entry in entries:
                if tok_idx < max_gwas_tokens:
                    gwas_token_ids[i, tok_idx] = trait_vocab[entry["trait"]]
                    gwas_scores[i, tok_idx] = entry["score"]
                    gwas_cat_ids[i, tok_idx] = trait_to_cat_id.get(entry["trait"], 0)
                    tok_idx += 1

    # Normalize expression columns
    max_vals = x[:, :expr_cols].max(dim=0).values.clamp(min=1.0)
    x[:, :expr_cols] = x[:, :expr_cols] / max_vals
    # Normalize degree
    deg_col = num_features - 1
    max_deg = x[:, deg_col].max().clamp(min=1.0)
    x[:, deg_col] = x[:, deg_col] / max_deg

    # Build edges
    src, dst = [], []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            src.append(node_to_idx[u])
            dst.append(node_to_idx[v])
            src.append(node_to_idx[v])
            dst.append(node_to_idx[u])

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    # Attach GWAS token data for embedding layer in models
    data.gwas_token_ids = gwas_token_ids      # [num_nodes, max_tokens] - trait IDs
    data.gwas_scores = gwas_scores             # [num_nodes, max_tokens] - association scores
    data.gwas_cat_ids = gwas_cat_ids           # [num_nodes, max_tokens] - category IDs
    data.gwas_vocab_size = next_id             # total traits + 1 (for padding)
    data.gwas_num_categories = len(gwas_categories) + 1
    return data, node_list, node_to_idx

graph_id = f"{graph.number_of_nodes()}_{graph.number_of_edges()}"
pyg_data, node_list, node_to_idx = nx_to_pyg(graph, graph_id)
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
        rwse_probes = st.select_slider("RWSE probes (accuracy)", [32, 64, 128, 256, 512], value=256,
                                        help="More probes = more accurate positional encodings. 256 is recommended for large graphs.")
        with st.expander("What does this do?"):
            st.markdown("**Graph Transformer** uses attention to weigh which gene connections matter most. "
                       "It can detect longer-range patterns than GNN by looking at broader neighborhoods.")
    else:  # VAE
        hidden_dim = st.select_slider("Hidden dimension", [16, 32, 64, 128, 256, 512], value=64)
        latent_dim = st.select_slider("Latent dimension", [8, 16, 32, 64, 128, 256], value=32)
        num_layers = st.slider("Encoder layers", 1, 4, 2)
        beta = st.slider("Beta (KL weight)", 0.1, 10.0, 1.0, 0.1)
        with st.expander("What does this do?"):
            st.markdown("**VAE** compresses gene data into a compact representation. "
                       "Genes that end up close together in this compressed space may share unknown pathways.")

    st.divider()
    epochs = st.slider("Epochs", 10, 500, 100)
    lr = st.select_slider("Learning rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    early_stop = st.checkbox("Early stopping", value=True)
    mini_batch = st.checkbox("Mini-batch training", value=False,
                             help="Samples subgraphs per batch instead of full graph. Reduces memory, required for large graphs with Transformer.")
    batch_size = 512
    num_neighbors = None
    if mini_batch:
        batch_size = st.select_slider("Batch size (edges)", [16, 32, 64, 128, 256, 512, 1024, 2048], value=512)
        edge_dropout = st.slider("Edge dropout", 0.0, 0.7, 0.3, 0.05,
                                  help="Fraction of edges randomly dropped during encode each epoch. "
                                       "Prevents overfitting. 0.3 = drop 30% of edges.")
        neighbor_opt = st.selectbox("Neighbors per layer",
                                     ["10, 5", "15, 10", "20, 10", "25, 15", "30, 20"],
                                     index=2)
        num_neighbors = [int(x) for x in neighbor_opt.split(", ")]

    st.divider()
    st.subheader("Compute")
    import torch as _torch
    compute_options = ["Local (CPU)"]
    if _torch.backends.mps.is_available():
        compute_options.append("Mac GPU (MPS)")
    compute_options.append("Remote GPU (Modal)")
    compute_mode = st.radio("Train on", compute_options)
    gpu_type = None
    if compute_mode == "Remote GPU (Modal)":
        gpu_type = st.selectbox("GPU", ["T4 (~$0.60/hr)", "A10G (~$1.10/hr)", "A100 (~$3.00/hr)"])
        gpu_type = gpu_type.split(" ")[0]
        try:
            import modal
            st.caption("Modal is installed")
        except ImportError:
            st.error("Modal not installed. Run: `pip install modal && modal setup`")
    elif compute_mode == "Mac GPU (MPS)":
        st.caption("Apple Silicon GPU detected")

with col_monitor:
    st.subheader("📈 Training Monitor")

    if st.button("🚀 Start Training", type="primary", width="stretch"):
        # GWAS token params for models
        gwas_params = dict(
            gwas_vocab_size=getattr(pyg_data, "gwas_vocab_size", 0),
            gwas_num_categories=getattr(pyg_data, "gwas_num_categories", 0),
            gwas_embed_dim=32,
        )

        # Build model kwargs
        if model_type == "GNN":
            model_class = "GNNLinkPredictor"
            model_kwargs = dict(in_channels=in_channels, hidden_channels=hidden_dim,
                                num_layers=num_layers, aggr=aggr, **gwas_params)
            params_str = f"layers={num_layers}, hidden={hidden_dim}, aggr={aggr}, lr={lr}, epochs={epochs}"
        elif model_type == "Graph Transformer":
            model_class = "GraphTransformerLinkPredictor"
            model_kwargs = dict(in_channels=in_channels, hidden_channels=hidden_dim,
                                num_layers=num_layers, num_heads=num_heads,
                                rwse_dim=16, rwse_walk_length=rwse_k,
                                rwse_probes=rwse_probes, **gwas_params)
            params_str = f"layers={num_layers}, hidden={hidden_dim}, heads={num_heads}, rwse_k={rwse_k}, lr={lr}, epochs={epochs}"
        else:
            model_class = "GraphVAE"
            model_kwargs = dict(in_channels=in_channels, hidden_channels=hidden_dim,
                                latent_dim=latent_dim, num_layers=num_layers, beta=beta, **gwas_params)
            params_str = f"layers={num_layers}, hidden={hidden_dim}, latent={latent_dim}, beta={beta}, lr={lr}, epochs={epochs}"

        # Create local model (needed for param count and local training)
        if model_type == "GNN":
            model = GNNLinkPredictor(**model_kwargs)
        elif model_type == "Graph Transformer":
            model = GraphTransformerLinkPredictor(**model_kwargs)
        else:
            model = GraphVAE(**model_kwargs)

        # Print training params
        compute_label = f"Remote GPU ({gpu_type})" if compute_mode == "Remote GPU (Modal)" else "Local CPU"
        st.markdown(f"**Training started** — {compute_label}")
        param_cols = st.columns(4)
        param_cols[0].metric("Model", model_type)
        param_cols[1].metric("Nodes", f"{pyg_data.num_nodes:,}")
        param_cols[2].metric("Edges", f"{pyg_data.edge_index.size(1) // 2:,}")
        param_cols[3].metric("Features", in_channels)
        st.code(f"{model_class}({params_str})", language=None)
        total_params = sum(p.numel() for p in model.parameters())
        st.caption(f"Model parameters: {total_params:,}")

        # Determine device
        if compute_mode == "Mac GPU (MPS)":
            device = "mps"
        elif compute_mode == "Remote GPU (Modal)":
            device = "cpu"  # remote handles its own device
        else:
            device = "cpu"

        config_dict = dict(epochs=epochs, lr=lr, train_ratio=train_ratio,
                           val_ratio=val_ratio, early_stopping=early_stop, patience=10,
                           mini_batch=mini_batch, batch_size=batch_size,
                           edge_dropout=edge_dropout if mini_batch else 0.0,
                           num_neighbors=num_neighbors, device=device)

        if compute_mode == "Remote GPU (Modal)":
            # Remote training via Modal
            import pickle
            st.info(f"Submitting to Modal ({gpu_type})... First run may take ~60s for cold start.")

            with st.spinner(f"Training on {gpu_type} GPU..."):
                try:
                    from src.models.modal_app import (
                        app as modal_app, train_on_gpu, train_on_a10g, train_on_a100,
                        collect_source_code,
                    )

                    data_bytes = pickle.dumps(pyg_data.cpu())
                    src_code = collect_source_code()

                    with modal_app.run():
                        if gpu_type == "A10G":
                            result = train_on_a10g.remote(model_class, model_kwargs, data_bytes, config_dict, src_code)
                        elif gpu_type == "A100":
                            result = train_on_a100.remote(model_class, model_kwargs, data_bytes, config_dict, src_code)
                        else:
                            result = train_on_gpu.remote(model_class, model_kwargs, data_bytes, config_dict, src_code)

                    history = result["history"]
                    metrics = result["metrics"]
                    epochs_run = result["epochs_run"]

                    # Load model state back
                    state_dict = pickle.loads(result["model_state_dict"])
                    model.load_state_dict(state_dict)

                    st.success(f"Training complete on {result['device_used']}!")
                    params_str += f", gpu={gpu_type}"

                except Exception as e:
                    st.error(f"Remote training failed: {e}")
                    st.stop()

        if compute_mode in ("Local (CPU)", "Mac GPU (MPS)"):
            # Local training (CPU or MPS)
            config = TrainConfig(**config_dict)
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

            device_label = "MPS GPU" if device == "mps" else "CPU"
            with st.spinner(f"Training on {device_label}..."):
                history = trainer.train(pyg_data, on_epoch=on_epoch)

            metrics = trainer.evaluate(pyg_data)
            st.success(f"Training complete on {device_label}!")

        # Show results
        st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
        st.metric("Avg Precision", f"{metrics['avg_precision']:.4f}")

        if "train_loss" in history:
            import pandas as pd
            df = pd.DataFrame({"Loss": history["train_loss"], "Val AUC": history["val_auc"]})
            st.line_chart(df)

        # Save training run
        import datetime
        model_name = f"{model_type.lower().replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_id = manager.save_training_run(
            model_type=model_type,
            parameters=params_str,
            graph_name=graph_id,
            nodes=pyg_data.num_nodes,
            edges=pyg_data.edge_index.size(1) // 2,
            features=in_channels,
            epochs_run=len(history["train_loss"]),
            auc_roc=metrics["auc_roc"],
            avg_precision=metrics["avg_precision"],
            model=model,
            model_name=model_name,
        )

        # Save full checkpoint for Results page reload
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".datasets", "trained_models")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "model_class": model_class,
            "model_kwargs": model_kwargs,
            "pyg_data": pyg_data,
            "node_list": node_list,
            "node_to_idx": node_to_idx,
            "model_type": model_type,
            "metrics": metrics,
        }, os.path.join(checkpoint_dir, f"{model_name}_checkpoint.pt"))
        st.caption(f"Run saved (#{run_id})")

        # Store results in session
        session.set_training_results({
            "model": model,
            "history": history,
            "metrics": metrics,
            "pyg_data": pyg_data,
            "node_list": node_list,
            "node_to_idx": node_to_idx,
            "model_type": model_type,
        })

        if st.button("📊 View Results"):
            st.switch_page("pages/4_Results.py")

# Training History
st.divider()
st.subheader("📋 Training History")

runs = manager.list_training_runs()
if runs:
    import pandas as pd
    df = pd.DataFrame(runs)
    df["created_at"] = df["created_at"].str[:19].str.replace("T", " ")
    display_df = df[["id", "created_at", "model_type", "parameters", "nodes", "edges",
                      "features", "epochs_run", "auc_roc", "avg_precision"]].copy()
    display_df.columns = ["#", "Date", "Model", "Parameters", "Nodes", "Edges",
                          "Features", "Epochs", "AUC-ROC", "Avg Precision"]
    display_df["AUC-ROC"] = display_df["AUC-ROC"].apply(lambda x: f"{x:.4f}")
    display_df["Avg Precision"] = display_df["Avg Precision"].apply(lambda x: f"{x:.4f}")

    st.dataframe(display_df, width="stretch", hide_index=True)

    # Delete run
    col_del1, col_del2 = st.columns([3, 1])
    with col_del1:
        del_id = st.number_input("Run # to delete", min_value=1, step=1, key="del_run_id")
    with col_del2:
        st.write("")
        if st.button("🗑️ Delete Run"):
            manager.delete_training_run(del_id)
            st.rerun()
else:
    st.info("No training runs yet. Train a model above to see results here.")
