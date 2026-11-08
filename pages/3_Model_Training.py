# pages/3_Model_Training.py — Train Model
import streamlit as st
import torch
import numpy as np
import os
from torch_geometric.data import Data
from src.models.gnn import GNNLinkPredictor
from src.models.graph_transformer import GraphTransformerLinkPredictor
from src.models.vae import GraphVAE
from src.models.trainer import Trainer, TrainConfig
from src.models.hetero_gnn import DECODERS, HeteroGNNLinkPredictor
from src.models.hetero_pipeline import (
    benchmark_hetero,
    describe_hetero_params,
    hetero_model_kwargs,
    hetero_training_payload,
    prepare_hetero_training,
    save_hetero_run,
    train_hetero,
)
from src.models.hetero_trainer import NEG_SAMPLING, HeteroTrainConfig
from src.utils.hetero_scorer import HETERO_MODEL_TYPE
from src.data.bulk_datasets import BulkDatasetManager
from src.utils import session

st.title("⚙️ Train a Prediction Model")
st.markdown(
    "Train an AI model on the gene interaction network to predict **new gene-disease connections** "
    "that haven't been discovered yet. No ML expertise required — sensible defaults are pre-selected."
)

graph = session.get_graph()
manager = BulkDatasetManager()
has_data = any(manager.is_downloaded(s) for s in ["gwas", "gtex", "hpa", "string"])

# ---------------------------------------------------------------------------
# No data → point to Download
# ---------------------------------------------------------------------------
if graph is None:
    if not has_data:
        st.warning("You need to download datasets before training a model.")
        st.markdown(
            "Head to **Download Datasets** to get the data (~5–10 min, one-time only), "
            "then come back here."
        )
        if st.button("📥 Go to Download Datasets"):
            st.switch_page("pages/0_Download.py")
        st.stop()

    # Show saved graphs
    saved_graphs = manager.list_saved_graphs()
    if saved_graphs:
        st.subheader("Load a previously built network")
        st.markdown("Pick a network you've already built, or create a new one below.")
        for sg in saved_graphs:
            col_info, col_load, col_del = st.columns([5, 1, 1])
            with col_info:
                st.markdown(
                    f"**{sg['name']}** — {sg['nodes']:,} genes, {sg['edges']:,} connections "
                    f"(built {sg['created_at'][:16].replace('T', ' ')})"
                )
            with col_load:
                if st.button("Load", key=f"load_{sg['name']}"):
                    with st.spinner(f"Loading {sg['name']}..."):
                        graph = manager.load_graph(sg["name"])
                        if graph:
                            session.set_graph(graph, name=sg["name"])
                            st.rerun()
                        else:
                            st.error("Graph file not found.")
            with col_del:
                if st.button("🗑️", key=f"del_{sg['name']}"):
                    manager.delete_graph(sg["name"])
                    st.rerun()
        st.divider()

    # Build new graph
    st.subheader("Build a new network")

    graph_mode = st.radio(
        "What do you want to analyze?",
        ["All genes (full dataset)", "One gene's neighborhood"],
        help="**All genes** creates a comprehensive network from every gene in the database (~20K genes). "
             "**One gene** focuses on a smaller subnetwork around a specific gene of interest.",
    )

    graph_name = st.text_input(
        "Name this network (optional)",
        placeholder="e.g., full_high_confidence, tp53_deep",
        help="Give it a name so you can reload it later without rebuilding.",
    )

    if graph_mode == "All genes (full dataset)":
        st.markdown(
            "This merges **all five data sources** into one large network: "
            "~20K genes as nodes, protein interactions as edges, with expression, structure, and disease data as features."
        )
        full_min_score = st.select_slider(
            "Connection confidence threshold",
            options=[400, 500, 600, 700, 800, 900],
            value=700,
            help="Higher = fewer but more reliable connections. "
                 "700 (recommended) gives ~470K edges. 900 gives ~200K edges.",
        )
        include_gwas = st.checkbox(
            "Include disease associations (from GWAS)",
            value=True,
            help="Adds known gene-disease links as features. Required for disease gene prediction.",
        )
        max_pvalue = st.select_slider(
            "GWAS significance threshold (p-value)",
            options=[5e-4, 5e-6, 5e-8, 5e-10, 5e-20, 5e-50],
            value=5e-8,
            help="Lower = stricter. 5e-8 is the standard genome-wide significance threshold.",
        ) if include_gwas else 5e-8

        # Auto-generate name if not provided
        if not graph_name:
            graph_name = f"full_s{full_min_score}{'_gwas' if include_gwas else ''}"

        if st.button("🔨 Build Network", type="primary"):
            progress_text = st.empty()
            with st.spinner("Building full network from all datasets (this may take a minute)..."):
                if not manager.is_downloaded("string_aliases"):
                    progress_text.text("Building gene name index...")
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
                session.set_graph(graph, name=graph_name)
                manager.save_graph(graph, graph_name)
                progress_text.text(f"Network saved as '{graph_name}'")
            st.rerun()
    else:
        gene_input = st.text_input("Gene symbol", placeholder="e.g., TP53, BRCA1, SP4")
        depth = st.slider(
            "Neighborhood depth",
            1, 3, 2,
            help="How many hops from the gene to include. "
                 "1 = direct partners, 2 = partners of partners (recommended).",
        )
        min_score = st.slider(
            "Min confidence score",
            0, 1000, 400, 50,
            help="STRING confidence (0–1000). 400 = medium, 700 = high.",
        )

        if not graph_name and gene_input:
            graph_name = f"{gene_input.upper()}_d{depth}_s{min_score}"

        if gene_input and st.button("🔨 Build Network", type="primary"):
            with st.spinner(f"Building network for {gene_input.upper()}..."):
                if not manager.is_downloaded("string_aliases"):
                    try:
                        manager.build_string_alias_table()
                    except Exception:
                        pass
                graph = manager.build_graph(gene_input.upper(), depth=depth, min_score=min_score)
                session.set_graph(graph, name=graph_name or None)
                if graph_name:
                    manager.save_graph(graph, graph_name)
            st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Convert graph to training format
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Preparing training data...", max_entries=3)
def nx_to_pyg(_G, graph_token):
    """_G is excluded from hashing (underscore prefix). graph_token IS hashed and
    changes whenever a different graph is stored in the session, so a rebuilt or
    reloaded network can never be served the previous network's tensors."""
    from src.utils.graph_features import nx_to_pyg_data
    return nx_to_pyg_data(_G)

graph_id = f"{graph.number_of_nodes()}_{graph.number_of_edges()}"
graph_token = session.get_graph_token() or f"{graph_id}_{id(graph)}"
pyg_data, node_list, node_to_idx = nx_to_pyg(graph, graph_token)
# Retrieve the real graph name for checkpoint persistence
_saved_graph_name = session.get_graph_name() or graph_id
in_channels = pyg_data.x.size(1)
edge_attr = getattr(pyg_data, "edge_attr", None)
edge_dim = edge_attr.size(1) if edge_attr is not None else 0

# ---------------------------------------------------------------------------
# Hetero GNN (gene–disease) training flow — the CLI's pipeline, driven from the page
# ---------------------------------------------------------------------------
def run_hetero_training(graph, graph_name, manager, *, model_options, min_trait_genes, holdout, config_kwargs,
                        compute_mode, gpu_type):
    """Prepare, train (here or on a Modal GPU), benchmark against enrichment, save the checkpoint and
    record the run. Nothing is stored in ``training_results``: the Predictions page would treat the
    model as a protein–protein model; the checkpoint appears there under *Ranking score* instead."""
    import pickle
    import pandas as pd

    try:
        with st.spinner("Building the gene–trait graph..."):
            prepared = prepare_hetero_training(graph, holdout=holdout, min_trait_genes=min_trait_genes)
    except ValueError as e:
        st.error(f"Cannot train on this network: {e}")
        return
    summary = (f"**{len(prepared.bundle.node_list):,} genes · {len(prepared.bundle.trait_list):,} traits · "
               f"{prepared.n_assoc:,} gene–trait associations**")
    if prepared.split is not None:
        summary += (f" ({prepared.n_hidden:,} hidden pairs of {len(prepared.split.diseases)} benchmark "
                    f"diseases removed before training)")
    st.markdown(summary)

    model_kwargs = hetero_model_kwargs(prepared.bundle, **model_options)
    device = "mps" if compute_mode == "This Mac's GPU" else "cpu"
    try:
        config = HeteroTrainConfig(**config_kwargs, device=device)
    except ValueError as e:
        st.error(f"Invalid training settings: {e}")
        return
    remote = compute_mode == "Cloud GPU (Modal)"
    st.markdown(f"**Training on:** {f'Cloud GPU ({gpu_type})' if remote else compute_mode}")

    if remote:
        st.info(f"Submitting to cloud ({gpu_type})... First run may take ~60s to start.")
        try:
            from src.models.modal_app import HETERO_TRAIN_FUNCTIONS, app as modal_app, collect_source_code
            payload_bytes = pickle.dumps(hetero_training_payload(prepared, model_kwargs, config))
            with st.spinner(f"Training on {gpu_type} GPU..."):
                with modal_app.run():
                    result = HETERO_TRAIN_FUNCTIONS[gpu_type].remote(payload_bytes, collect_source_code())
        except Exception as e:
            st.error(f"Cloud training failed: {e}")
            return
        model = HeteroGNNLinkPredictor(**model_kwargs)
        model.load_state_dict(pickle.loads(result["model_state_dict"]))
        history, metrics = result["history"], result["metrics"]
        st.success(f"Training complete on {result['device_used']}!")
    else:
        progress_bar = st.progress(0)
        chart = st.empty()
        status = st.empty()
        losses, mrrs = [], []
        redraw_every = max(1, config.epochs // 100)

        def on_epoch(epoch, loss, val_mrr):
            losses.append(loss)
            mrrs.append(val_mrr)
            progress_bar.progress((epoch + 1) / config.epochs)
            status.text(f"Round {epoch + 1}/{config.epochs} — Loss: {loss:.4f}, Validation MRR: {val_mrr:.4f}")
            if (epoch + 1) % redraw_every == 0:
                chart.line_chart(pd.DataFrame({"Loss (lower=better)": losses, "Validation MRR": mrrs}))

        try:
            with st.spinner(f"Training on {'Mac GPU' if device == 'mps' else 'CPU'}..."):
                trained = train_hetero(prepared.bundle, prepared.index, prepared.split, model_kwargs, config,
                                       on_epoch=on_epoch)
        except ValueError as e:
            st.error(f"Training could not start: {e}")
            return
        progress_bar.progress(1.0)
        model, history, metrics = trained.model, trained.history, trained.metrics
        st.success("Training complete!")

    with st.spinner("Benchmarking against network enrichment..." if holdout else "Saving the model..."):
        benchmark = benchmark_hetero(model, prepared, graph)
        params = describe_hetero_params(model_kwargs, config, holdout=holdout, gpu=gpu_type if remote else None)
        saved = save_hetero_run(manager, model=model, model_kwargs=model_kwargs, prepared=prepared, config=config,
                                metrics=metrics, benchmark=benchmark, graph_name=graph_name,
                                min_trait_genes=min_trait_genes, params=params)

    st.divider()
    st.markdown("**Results**")
    result_cols = st.columns(3)
    result_cols[0].metric("Validation MRR", f"{metrics['val_mrr']:.4f}",
                          help="Reciprocal rank of the best hidden validation gene, averaged over diseases. "
                               "1.0 = a hidden gene ranked first for every disease.")
    result_cols[1].metric("Validation recall@50", f"{metrics['val_recall50']:.4f}",
                          help="Share of hidden validation genes ranked in the top 50.")
    result_cols[2].metric("Validation AUC", f"{metrics['val_auc']:.4f}")
    st.caption(f"{metrics['epochs_run']} rounds run; best weights from round {metrics['best_epoch'] + 1}.")
    st.line_chart(pd.DataFrame({"Loss": history["train_loss"], "Validation MRR": history["val_mrr"]}))

    if benchmark:
        model_result, enrichment_result = benchmark["hetero"], benchmark["enrichment"]
        st.markdown(f"**Benchmark on the hidden pairs** ({model_result.n_hidden:,} hidden genes, "
                    f"{model_result.n_diseases} diseases)")
        rows = [{"Scorer": name, "recall@50": f"{r.recall[50]:.4f}", "recall@200": f"{r.recall[200]:.4f}",
                 "Median rank": f"{r.median_rank:.0f}", "MRR": f"{r.mrr:.4f}"} for name, r in benchmark.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if model_result.recall[50] > enrichment_result.recall[50] and model_result.mrr > enrichment_result.mrr:
            st.caption("The model beats network enrichment on recall@50 and MRR.")
        else:
            st.caption("The model does not beat network enrichment on recall@50 and MRR, so enrichment stays the "
                       "default ranking; the Predictions page labels this model experimental.")

    st.caption(f"Model saved (run #{saved.run_id}) — {saved.checkpoint_path}")
    st.page_link("pages/4_Results.py", label="📊 Use it under Ranking score on the Predictions page")


# ---------------------------------------------------------------------------
# Three-column layout: Data | Model | Training
# ---------------------------------------------------------------------------
col_data, col_config, col_monitor = st.columns([1, 1, 1])

with col_data:
    st.subheader("Network Summary")
    st.metric("Genes", f"{pyg_data.num_nodes:,}")
    st.metric("Connections", f"{pyg_data.edge_index.size(1) // 2:,}")
    st.metric("Features per gene", in_channels)

    with st.expander("Advanced: data split"):
        train_pct = st.slider("Training data %", 50, 90, 80, 5,
                              help="Percentage of edges used for training. The rest is split between validation and testing.")
        max_val_pct = min(30, 95 - train_pct)  # always leave at least 5% of edges for testing
        if max_val_pct > 5:
            val_pct = st.slider("Validation data %", 5, max_val_pct, min(10, max_val_pct), 5)
        else:
            val_pct = 5
            st.caption("Validation data: 5%")
        st.caption(f"Test data: {100 - train_pct - val_pct}% of edges")
        train_ratio = train_pct / 100
        val_ratio = val_pct / 100

with col_config:
    st.subheader("Model Selection")

    model_type = st.radio(
        "Choose a model",
        ["GNN (recommended)", "Graph Transformer", "VAE", "Hetero GNN (gene–disease)"],
        help="**GNN** — Fast, works well for most cases. Good default.\n\n"
             "**Graph Transformer** — More powerful for large networks, uses attention to weigh connections.\n\n"
             "**VAE** — Finds genes in similar 'latent space'. Good for discovering unexpected similarities.\n\n"
             "**Hetero GNN** — Learns gene–disease links directly: genes and GWAS traits are nodes of one "
             "graph. Its scores appear on the Predictions page under *Ranking score*.",
    )
    # Clean model type for internal use
    model_type_clean = model_type.split(" (")[0]  # Remove "(recommended)"

    if model_type_clean == "GNN":
        with st.expander("Model settings", expanded=False):
            num_layers = st.slider("Layers", 1, 5, 2)
            hidden_dim = st.select_slider("Hidden dimension", [16, 32, 64, 128, 256], value=64)
            aggr = st.selectbox("Aggregation", ["mean", "max", "sum"])
        st.caption("GNN learns by passing messages between connected genes to find similar neighborhoods.")
    elif model_type_clean == "Graph Transformer":
        with st.expander("Model settings", expanded=False):
            num_layers = st.slider("Layers", 1, 6, 2)
            hidden_dim = st.select_slider("Hidden dimension", [32, 64, 128, 256, 512], value=64)
            num_heads = st.select_slider("Attention heads", [1, 2, 4, 8], value=4)
            rwse_k = st.slider("Position encoding depth", 0, 16, 8,
                               help="0 disables RWSE entirely. Lower values are cheaper and often more stable.")
            rwse_probes = st.select_slider("Position encoding accuracy", [16, 32, 64, 128, 256], value=64)
            transformer_dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
            transformer_ff_mult = st.select_slider("FFN expansion", [1, 2, 4], value=2)
        st.caption("Transformer uses attention to weigh which gene connections matter most.")
    elif model_type_clean == HETERO_MODEL_TYPE:
        with st.expander("Model settings", expanded=False):
            hidden_dim = st.select_slider("Hidden dimension", [32, 64, 128, 256], value=128)
            num_layers = st.slider("Layers", 1, 3, 2)
            hetero_decoder = st.selectbox("Decoder", list(DECODERS),
                                          help="How a gene embedding and a trait embedding become a score.")
            hetero_dropout = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
            min_trait_genes = st.slider("Ignore traits with fewer known genes", 1, 10, 1,
                                        help="Traits with fewer genes than this get no node in the graph.")
            neg_gene = st.slider("Gene-corrupted negatives per known pair", 1, 8, 4)
            neg_trait = st.slider("Trait-corrupted negatives per known pair", 0, 4, 1)
            neg_sampling = st.selectbox("Negative sampling", list(NEG_SAMPLING),
                                        help="degree: draw negative genes in proportion to how many traits they have.")
        st.caption("Learns gene–disease links directly: genes and GWAS traits are nodes of one graph. "
                   "Benchmarked against network enrichment when the hold-out is on.")
    else:  # VAE
        with st.expander("Model settings", expanded=False):
            hidden_dim = st.select_slider("Hidden dimension", [16, 32, 64, 128, 256, 512], value=64)
            latent_dim = st.select_slider("Latent dimension", [8, 16, 32, 64, 128, 256], value=32)
            num_layers = st.slider("Encoder layers", 1, 4, 2)
            beta = st.slider("Regularization strength", 0.1, 10.0, 1.0, 0.1)
        st.caption("VAE compresses gene profiles — genes close in compressed space may share hidden connections.")

    is_hetero = model_type_clean == HETERO_MODEL_TYPE
    with st.expander("Training settings", expanded=False):
        max_epochs, default_epochs = (2000, 300) if is_hetero else (500, 100)
        epochs = st.slider("Training rounds (epochs)", 10, max_epochs, default_epochs,
                           help="More epochs = longer training but potentially better results. "
                                "Early stopping will halt if the model stops improving.")
        default_lr = 0.0005 if model_type_clean == "Graph Transformer" else 0.001
        lr = st.select_slider("Learning rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=default_lr)
        early_stop = st.checkbox("Stop early if no improvement", value=True,
                                 help="Keeps the weights from the best validation epoch.")
        seed = int(st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
                                   help="Same seed = same data split, negatives and initialization, "
                                        "so different models can be compared on equal footing."))
        holdout = True
        val_fraction, eval_every, patience = 0.1, 1, 20
        if is_hetero:
            holdout = st.checkbox("Hold out benchmark pairs and benchmark against enrichment", value=True,
                                  help="Removes 20 % of each disease's known genes (seeded) before training and "
                                       "scores the model on them next to network enrichment. Untick for a "
                                       "production model once the honest one has been validated.")
            val_fraction = st.slider("Validation share of each disease's genes", 0.05, 0.30, 0.10, 0.05,
                                     help="A second seeded hold-out that drives early stopping.")
            eval_every = st.slider("Validate every N rounds", 1, 20, 1)
            patience = st.slider("Patience (validations without improvement)", 5, 50, 20)
            st.caption("The edge split under Network Summary applies to protein–protein models; "
                       "this model holds out disease genes instead.")
        weight_decay = 0.0
        warmup_epochs = 0
        if model_type_clean == "Graph Transformer":
            weight_decay = st.select_slider("Weight decay", [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-4)
            warmup_epochs = st.slider("Warmup epochs", 0, min(20, epochs // 2), min(5, epochs // 2))
        mini_batch = False
        if not is_hetero:
            mini_batch = st.checkbox(
                "Use mini-batches (for large networks)",
                value=False,
                help="Processes the network in chunks instead of all at once. "
                     "Reduces memory usage for networks with >10K genes.",
            )
        batch_size = 512
        edge_dropout = 0.0
        if mini_batch:
            batch_size = st.select_slider("Batch size", [16, 32, 64, 128, 256, 512, 1024, 2048], value=512)
            edge_dropout = st.slider("Edge dropout", 0.0, 0.7, 0.3, 0.05,
                                     help="Randomly removes edges during training to prevent overfitting.")

    st.divider()
    st.subheader("Where to train")
    import torch as _torch
    compute_options = ["This computer (CPU)"]
    if _torch.backends.mps.is_available():
        compute_options.append("This Mac's GPU")
    compute_options.append("Cloud GPU (Modal)")
    compute_mode = st.radio("Run on", compute_options)
    gpu_type = None
    if compute_mode == "Cloud GPU (Modal)":
        gpu_type = st.selectbox("GPU type", ["T4 (~$0.60/hr)", "A10G (~$1.10/hr)", "A100 (~$3.00/hr)"])
        gpu_type = gpu_type.split(" ")[0]
        try:
            import modal
            st.caption("Modal is installed and ready")
        except ImportError:
            st.error("Modal not installed. Run: `pip install modal && modal setup`")
    elif compute_mode == "This Mac's GPU":
        if is_hetero:
            st.caption("Apple Silicon GPU detected. For this model the CPU was faster than the Mac GPU in our runs.")
        else:
            st.caption("Apple Silicon GPU detected — training will be faster")

# ---------------------------------------------------------------------------
# Training monitor
# ---------------------------------------------------------------------------
with col_monitor:
    st.subheader("Training")

    start_training = st.button("🚀 Start Training", type="primary", use_container_width=True)
    if start_training and is_hetero:
        run_hetero_training(
            graph, _saved_graph_name, manager,
            model_options=dict(hidden=hidden_dim, layers=num_layers, dropout=hetero_dropout, decoder=hetero_decoder),
            min_trait_genes=min_trait_genes, holdout=holdout,
            config_kwargs=dict(epochs=epochs, lr=lr, neg_gene=neg_gene, neg_trait=neg_trait,
                               neg_sampling=neg_sampling, val_fraction=val_fraction, early_stopping=early_stop,
                               patience=patience, eval_every=eval_every, seed=seed),
            compute_mode=compute_mode, gpu_type=gpu_type,
        )
    if start_training and not is_hetero:
        # GWAS token params for models
        gwas_params = dict(
            gwas_vocab_size=getattr(pyg_data, "gwas_vocab_size", 0),
            gwas_num_categories=getattr(pyg_data, "gwas_num_categories", 0),
            gwas_embed_dim=32,
        )

        # Build model kwargs
        if model_type_clean == "GNN":
            model_class = "GNNLinkPredictor"
            model_kwargs = dict(in_channels=in_channels, hidden_channels=hidden_dim,
                                num_layers=num_layers, aggr=aggr, **gwas_params)
            params_str = f"layers={num_layers}, hidden={hidden_dim}, aggr={aggr}, lr={lr}, epochs={epochs}"
        elif model_type_clean == "Graph Transformer":
            model_class = "GraphTransformerLinkPredictor"
            model_kwargs = dict(in_channels=in_channels, hidden_channels=hidden_dim,
                                num_layers=num_layers, num_heads=num_heads,
                                rwse_dim=8 if rwse_k > 0 else 0, rwse_walk_length=rwse_k,
                                rwse_probes=rwse_probes, edge_dim=edge_dim,
                                dropout=transformer_dropout, ff_mult=transformer_ff_mult,
                                **gwas_params)
            params_str = (
                f"layers={num_layers}, hidden={hidden_dim}, heads={num_heads}, rwse_k={rwse_k}, "
                f"dropout={transformer_dropout}, ff_mult={transformer_ff_mult}, "
                f"lr={lr}, wd={weight_decay}, warmup={warmup_epochs}, epochs={epochs}"
            )
        else:
            model_class = "GraphVAE"
            model_kwargs = dict(in_channels=in_channels, hidden_channels=hidden_dim,
                                latent_dim=latent_dim, num_layers=num_layers, beta=beta, **gwas_params)
            params_str = f"layers={num_layers}, hidden={hidden_dim}, latent={latent_dim}, beta={beta}, lr={lr}, epochs={epochs}"

        params_str += f", seed={seed}"

        # Create local model (seeded so initialization is reproducible)
        torch.manual_seed(seed)
        if model_type_clean == "GNN":
            model = GNNLinkPredictor(**model_kwargs)
        elif model_type_clean == "Graph Transformer":
            model = GraphTransformerLinkPredictor(**model_kwargs)
        else:
            model = GraphVAE(**model_kwargs)

        # Show training info
        compute_label = f"Cloud GPU ({gpu_type})" if compute_mode == "Cloud GPU (Modal)" else compute_mode
        st.markdown(f"**Training on:** {compute_label}")

        total_params = sum(p.numel() for p in model.parameters())
        param_cols = st.columns(2)
        param_cols[0].metric("Model", model_type_clean)
        param_cols[1].metric("Parameters", f"{total_params:,}")

        # Determine device
        if compute_mode == "This Mac's GPU":
            device = "mps"
        elif compute_mode == "Cloud GPU (Modal)":
            device = "cpu"  # remote handles its own device
        else:
            device = "cpu"

        config_dict = dict(epochs=epochs, lr=lr, train_ratio=train_ratio,
                           val_ratio=val_ratio, early_stopping=early_stop, patience=10,
                           mini_batch=mini_batch, batch_size=batch_size,
                           edge_dropout=edge_dropout if mini_batch else 0.0,
                           weight_decay=weight_decay, warmup_epochs=warmup_epochs,
                           seed=seed, device=device)

        if compute_mode == "Cloud GPU (Modal)":
            # Remote training via Modal
            import pickle
            st.info(f"Submitting to cloud ({gpu_type})... First run may take ~60s to start.")

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
                    st.error(f"Cloud training failed: {e}")
                    st.stop()

        if compute_mode in ("This computer (CPU)", "This Mac's GPU"):
            # Local training
            try:
                config = TrainConfig(**config_dict)
            except ValueError as e:
                st.error(f"Invalid training settings: {e}")
                st.stop()
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
                status_text.text(f"Round {epoch + 1}/{epochs} — Loss: {loss:.4f}, Accuracy: {val_auc:.4f}")
                import pandas as pd
                df = pd.DataFrame({"Loss (lower=better)": loss_history, "Accuracy (AUC)": auc_history})
                loss_chart.line_chart(df)

            device_label = "Mac GPU" if device == "mps" else "CPU"
            try:
                with st.spinner(f"Training on {device_label}..."):
                    history = trainer.train(pyg_data, on_epoch=on_epoch)
                metrics = trainer.evaluate(pyg_data)
            except ValueError as e:
                st.error(f"Training could not start: {e}")
                st.stop()
            # The trainer moved the model to the training device in place; bring it
            # back to CPU so the Results page can encode the CPU feature tensors.
            model = model.cpu()
            st.success("Training complete!")

        # Show results
        st.divider()
        st.markdown("**Results**")
        result_cols = st.columns(2)
        result_cols[0].metric("Accuracy (AUC-ROC)", f"{metrics['auc_roc']:.4f}",
                              help="Area under the ROC curve. 1.0 = perfect, 0.5 = random guessing.")
        result_cols[1].metric("Precision", f"{metrics['avg_precision']:.4f}",
                              help="Average precision — how many of the model's top predictions are correct.")

        if "train_loss" in history:
            import pandas as pd
            df = pd.DataFrame({"Loss": history["train_loss"], "Accuracy (AUC)": history["val_auc"]})
            st.line_chart(df)

        # Save training run
        import datetime
        model_name = f"{model_type_clean.lower().replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_id = manager.save_training_run(
            model_type=model_type_clean,
            parameters=params_str,
            graph_name=_saved_graph_name,
            nodes=pyg_data.num_nodes,
            edges=pyg_data.edge_index.size(1) // 2,
            features=in_channels,
            epochs_run=len(history["train_loss"]),
            auc_roc=metrics["auc_roc"],
            avg_precision=metrics["avg_precision"],
            model=model,
            model_name=model_name,
        )

        # Save checkpoint
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".datasets", "trained_models")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.pt")
        torch.save({
            "model_state": model.state_dict(),
            "model_class": model_class,
            "model_kwargs": model_kwargs,
            "model_type": model_type_clean,
            "metrics": metrics,
            "graph_name": _saved_graph_name,
            "train_config": config_dict,
        }, checkpoint_path)
        st.caption(f"Model saved (run #{run_id})")

        # Store results in session
        session.set_training_results({
            "model": model,
            "history": history,
            "metrics": metrics,
            "pyg_data": pyg_data,
            "node_list": node_list,
            "node_to_idx": node_to_idx,
            "model_type": model_type_clean,
            "checkpoint_path": checkpoint_path,
        })

        # A plain button here could never fire: clicking it reruns the page with
        # "Start Training" back to False, so this whole block would be skipped.
        st.page_link("pages/4_Results.py", label="📊 See Predictions")

# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Past training runs")

runs = manager.list_training_runs()
if runs:
    import pandas as pd
    df = pd.DataFrame(runs)
    df["created_at"] = df["created_at"].str[:19].str.replace("T", " ")
    display_df = df[["id", "created_at", "model_type", "parameters", "nodes", "edges",
                      "features", "epochs_run", "auc_roc", "avg_precision"]].copy()
    display_df.columns = ["#", "Date", "Model", "Settings", "Genes", "Connections",
                          "Features", "Rounds", "Accuracy (AUC)", "Precision"]
    display_df["Accuracy (AUC)"] = display_df["Accuracy (AUC)"].apply(lambda x: f"{x:.4f}")
    display_df["Precision"] = display_df["Precision"].apply(lambda x: f"{x:.4f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Delete run
    with st.expander("Delete a run"):
        col_del1, col_del2 = st.columns([3, 1])
        with col_del1:
            del_id = st.number_input("Run # to delete", min_value=1, step=1, key="del_run_id")
        with col_del2:
            st.write("")
            if st.button("🗑️ Delete"):
                manager.delete_training_run(del_id)
                st.rerun()
else:
    st.info("No training runs yet. Configure a model above and click **Start Training** to begin.")
