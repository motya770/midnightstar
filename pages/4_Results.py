# pages/4_Results.py — Predictions / Discovery Dashboard
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pyvis.network import Network
import tempfile
import os
import datetime
import glob as glob_mod
from src.utils import session
from src.utils.discovery import (
    index_disease_genes,
    known_links_rows,
    predict_gene_disease_links,
    select_diseases,
)
from src.utils.discovery_export import (
    ENRICHMENT_SCORER,
    HETERO_SCORER,
    PREDICTED_FILE,
    build_manifest,
    write_discovery_bundle,
    zip_bundle,
)
from src.utils.hetero_scorer import (
    HETERO_MODEL_CLASS,
    HETERO_MODEL_TYPE,
    find_hetero_checkpoints,
    hetero_encode,
    is_hetero_checkpoint_path,
    load_hetero_checkpoint,
    score_from_encoding,
)
from src.models.gnn import GNNLinkPredictor
from src.models.graph_transformer import GraphTransformerLinkPredictor
from src.models.vae import GraphVAE

st.title("📊 Predictions & Discovery")
st.markdown(
    "Use your trained model to predict **new gene-disease connections**, "
    "find genes with similar profiles, or check if two genes are likely to interact."
)

from src.data.bulk_datasets import BulkDatasetManager
_mgr = BulkDatasetManager()

results = session.get_training_results()
graph = session.get_graph()

# Load graph from disk if not in session
if graph is None:
    saved_graphs = _mgr.list_saved_graphs()
    if saved_graphs:
        graph = _mgr.load_graph(saved_graphs[0]["name"])
        if graph:
            session.set_graph(graph, name=saved_graphs[0]["name"])

# ---------------------------------------------------------------------------
# No model loaded — let user pick one
# ---------------------------------------------------------------------------
if results is None and not session.get_no_protein_model():
    from src.utils.graph_features import nx_to_pyg_data

    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".datasets", "trained_models")
    # Gene–disease (hetero) checkpoints rank diseases under "Ranking score" below; they are not
    # protein–protein models and cannot be loaded here.
    checkpoints = sorted(p for p in glob_mod.glob(os.path.join(checkpoint_dir, "*_checkpoint.pt"))
                         if not is_hetero_checkpoint_path(p))
    training_runs = [r for r in _mgr.list_training_runs() if r.get("model_type") != HETERO_MODEL_TYPE]

    # Build options list
    options = []
    option_meta = []
    for cp in reversed(checkpoints):
        name = os.path.basename(cp).replace("_checkpoint.pt", "")
        options.append(f"[checkpoint] {name}")
        option_meta.append(("checkpoint", cp))
    for run in training_runs:
        if run.get("model_path") and os.path.exists(run["model_path"]):
            label = (f"[run #{run['id']}] {run['model_type']} — "
                     f"AUC={run['auc_roc']:.4f}, {run['nodes']} genes, "
                     f"{run['parameters']}")
            options.append(label)
            option_meta.append(("run", run))

    if not options:
        st.info("No loadable protein–protein model found yet. Go to **Train Model** to build a network and train "
                "one (this typically takes a few minutes), or predict without one below.")
    else:
        st.subheader("Select a trained model")
        st.markdown("Choose which protein–protein model to use for predictions.")
        selected_idx = st.selectbox("Model", range(len(options)),
                                    format_func=lambda i: options[i])

    if options and st.button("Load Model", type="primary"):
        kind, meta = option_meta[selected_idx]

        if kind == "checkpoint":
            checkpoint = torch.load(meta, map_location="cpu", weights_only=False)
            model_class = checkpoint["model_class"]
            model_kwargs = checkpoint["model_kwargs"]
            if model_class == HETERO_MODEL_CLASS:
                st.error("This is a gene–disease model checkpoint, not a protein–protein model. Load a "
                         "protein–protein model here and choose the gene–disease model under **Ranking score** "
                         "in the Predict Disease Genes tab (it is listed there when the file is named "
                         "`hetero_gnn_<stamp>_checkpoint.pt`).")
                st.stop()

            # Load graph for this checkpoint
            graph_name = checkpoint.get("graph_name", "")
            if graph is None and graph_name:
                graph = _mgr.load_graph(graph_name)
                if graph:
                    session.set_graph(graph, name=graph_name)

            if graph is None:
                st.error("No network data available. Go to **Train Model** to build or load a network first.")
                st.stop()

            with st.spinner("Preparing model and data..."):
                pyg_data, node_list, node_to_idx = nx_to_pyg_data(graph)

            if model_class == "GNNLinkPredictor":
                mdl = GNNLinkPredictor(**model_kwargs)
            elif model_class == "GraphTransformerLinkPredictor":
                mdl = GraphTransformerLinkPredictor(**model_kwargs)
            else:
                mdl = GraphVAE(**model_kwargs)
            try:
                mdl.load_state_dict(checkpoint["model_state"])
            except RuntimeError as e:
                st.error("This checkpoint was saved with an older model layout and cannot be loaded. "
                         "Retrain the model to create a compatible checkpoint.")
                st.caption(str(e)[:300])
                st.stop()

            results = {
                "model": mdl,
                "pyg_data": pyg_data,
                "node_list": node_list,
                "node_to_idx": node_to_idx,
                "model_type": checkpoint["model_type"],
                "metrics": checkpoint["metrics"],
                "checkpoint_path": meta,
            }
        else:
            st.error("Old training runs require checkpoint format. Please retrain to create a loadable checkpoint.")
            st.stop()

        session.set_training_results(results)
        st.rerun()

    # Without a protein–protein model: enrichment and the gene–disease model need only the network.
    st.markdown("**Or predict without a protein–protein model**")
    hetero_count = len(find_hetero_checkpoints(checkpoint_dir))
    st.caption(
        "The Predict Disease Genes tab and the export rank candidates by network enrichment"
        + (f", or by one of the {hetero_count} gene–disease model checkpoint{'s' if hetero_count > 1 else ''} found,"
           if hetero_count else "")
        + " using only the network. Find Similar Genes and Check Gene Pair need a protein–protein model."
    )
    if graph is None:
        st.caption("No saved network found. Go to **Train Model** to build or load one first.")
    elif st.button("🧬 Predict without a protein model"):
        session.set_no_protein_model(True)
        st.rerun()

    st.stop()

# ---------------------------------------------------------------------------
# Protein model loaded — compute embeddings; or no protein model — the network alone
# (enrichment and the gene–disease model need no embeddings; z and z_norm stay None)
# ---------------------------------------------------------------------------
if results is not None:
    model = results["model"].cpu()  # training may have left it on MPS/CUDA; features live on CPU
    pyg_data = results["pyg_data"].cpu()
    node_list = results["node_list"]
    node_to_idx = results["node_to_idx"]
    metrics = results["metrics"]
    model_type = results["model_type"]

    model.eval()
    with torch.no_grad():
        z = model.encode(pyg_data)
        z_norm = F.normalize(z, dim=-1)

    # Model info bar
    st.markdown(
        f"**Active model:** {model_type} · "
        f"**Accuracy (AUC):** {metrics['auc_roc']:.4f} · "
        f"**Genes loaded:** {z.size(0):,}"
    )
else:
    if graph is None:
        st.error("No network data available. Go to **Train Model** to build or load a network first.")
        st.stop()
    model = pyg_data = z = z_norm = None
    node_list = sorted(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    metrics = {}
    model_type = None
    col_bar, col_switch = st.columns([3, 1])
    col_bar.markdown(
        "**Active model:** none — ranking by network enrichment or a gene–disease model · "
        f"**Genes loaded:** {len(node_list):,}"
    )
    if col_switch.button("Load a protein–protein model instead"):
        session.set_no_protein_model(False)
        st.rerun()

# ---------------------------------------------------------------------------
# Helper data
# ---------------------------------------------------------------------------
# Known GWAS associations per trait (exact trait names), shared by the disease tab and the export.
disease_index = index_disease_genes(graph, node_list) if graph else {}

# ---------------------------------------------------------------------------
# Ranking scorer: network enrichment by default, or a trained heterogeneous
# gene–disease model when a checkpoint from scripts/train_hetero_disease_model.py exists.
# ---------------------------------------------------------------------------
_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".datasets", "trained_models")
SCORER_ENRICHMENT_LABEL = "Network enrichment (default)"
# The hetero model has not beaten enrichment on the held-out benchmark (README, "Held-out
# disease-gene benchmark"), so it is offered as an experimental alternative, never as the default.
SCORER_HETERO_NOTE = "(experimental, below enrichment on the benchmark)"
_hetero_paths = list(reversed(find_hetero_checkpoints(_CHECKPOINT_DIR)))
scorer_options = [SCORER_ENRICHMENT_LABEL] + [
    f"Gene–disease model: {os.path.basename(p).replace('_checkpoint.pt', '')} {SCORER_HETERO_NOTE}"
    for p in _hetero_paths
]
_scorer_paths = dict(zip(scorer_options[1:], _hetero_paths))
SCORER_HELP = (
    "**Network enrichment** ranks by how strongly a gene's STRING partners are enriched for the "
    "disease's known genes (the validated default). **Gene–disease model** ranks by a heterogeneous graph "
    "neural network trained on gene–disease links. On the held-out benchmark the current checkpoint "
    "recovers fewer hidden disease genes than enrichment on its own (it only helps when combined with "
    "enrichment, see the README), so treat its ranking as experimental and compare checkpoints with "
    "scripts/benchmark_disease_genes.py before trusting one."
)


@st.cache_resource(show_spinner="Encoding the network with the gene–disease model...")
def _hetero_encoding(checkpoint_path, graph_token, _graph):
    """One graph conversion and forward pass per (checkpoint, network); every disease list is scored from it."""
    return hetero_encode(load_hetero_checkpoint(checkpoint_path), _graph)


def scorer_for(label, diseases):
    """(score matrix or None, scorer name for the manifest, checkpoint path or None).

    When the checkpoint cannot score this network (different traits or gene features, unreadable
    file), the ranking falls back to network enrichment with a visible warning, and the manifest
    records enrichment."""
    if label == SCORER_ENRICHMENT_LABEL or label not in _scorer_paths:
        return None, ENRICHMENT_SCORER, None
    path = _scorer_paths[label]
    try:
        encoding = _hetero_encoding(path, session.get_graph_token(), graph)
        matrix = score_from_encoding(encoding, list(diseases))
    except (ValueError, RuntimeError, KeyError) as e:
        st.warning("The gene–disease checkpoint cannot score this network, so this ranking uses network "
                   f"enrichment instead. {str(e)[:300]}")
        return None, ENRICHMENT_SCORER, None
    return matrix, HETERO_SCORER, path


def pick_scorer(key):
    if len(scorer_options) == 1:
        return SCORER_ENRICHMENT_LABEL
    return st.selectbox("Ranking score", scorer_options, key=key, help=SCORER_HELP)


def find_similar_genes(gene_name, top_k=20):
    if gene_name not in node_to_idx:
        return []
    idx = node_to_idx[gene_name]
    sims = (z_norm @ z_norm[idx]).squeeze()
    results_list = []
    for i in sims.argsort(descending=True):
        i = i.item()
        node = node_list[i]
        if node == gene_name:
            continue
        if graph and graph.nodes[node].get("node_type") != "gene":
            continue
        results_list.append({
            "gene": node,
            "similarity": round(sims[i].item(), 4),
            "description": graph.nodes[node].get("name", "") if graph else "",
        })
        if len(results_list) >= top_k:
            break
    return results_list


# ---------------------------------------------------------------------------
# Three discovery tabs
# ---------------------------------------------------------------------------
NEEDS_PROTEIN_MODEL = (
    "**{tab}** compares the learned profiles of a protein–protein model, and none is loaded. "
    "Use **Load a protein–protein model instead** above."
)

tab_disease, tab_gene, tab_link = st.tabs([
    "🧬 Predict Disease Genes",
    "🔎 Find Similar Genes",
    "🔗 Check Gene Pair",
])

# === Tab 1: Disease Gene Discovery ===
with tab_disease:
    st.markdown(
        "**Pick a disease** to rank candidate genes by how strongly their STRING interaction "
        "partners are enriched for the disease's known genes (guilt-by-association). "
        + ("Ties, mostly genes with no known partner, are broken by similarity of the model's learned "
           "profiles. " if z_norm is not None else "")
        + "When a trained gene–disease model is available you can rank by its score "
        "instead. Candidates are hypotheses for follow-up, not a diagnosis."
    )

    if disease_index:
        col_cat, col_min = st.columns(2)
        with col_cat:
            categories = sorted({v["category"] for v in disease_index.values()})
            sel_cat = st.multiselect(
                "Category",
                categories,
                default=["disease"] if "disease" in categories else categories[:1],
            )
        with col_min:
            min_genes = st.slider(
                "Min known genes",
                2, 50, 5,
                help="Only show diseases that have at least this many known gene associations. "
                     "More known genes = better predictions.",
            )

        filtered = {k: v for k, v in disease_index.items()
                    if v["category"] in sel_cat and len(v["genes"]) >= min_genes}
        disease_options = sorted(filtered.keys(), key=lambda x: len(filtered[x]["genes"]), reverse=True)

        if disease_options:
            selected_disease = st.selectbox(
                f"Disease or trait ({len(disease_options)} available)",
                disease_options,
                format_func=lambda x: f"{x} ({len(filtered[x]['genes'])} known genes)",
            )
            top_k = st.slider("How many predictions?", 10, 200, 50)
            disease_scorer = pick_scorer("disease_scorer")

            if st.button("🔮 Predict New Genes", type="primary"):
                known_genes = sorted(disease_index[selected_disease]["genes"])
                scores, scorer_name, _ = scorer_for(disease_scorer, [selected_disease])
                candidates = predict_gene_disease_links(
                    z_norm, node_list, node_to_idx, graph, disease_index, [selected_disease], top_k,
                    scores=scores,
                )

                if candidates:
                    col_known, col_pred = st.columns(2)

                    with col_known:
                        st.markdown(f"**Already known ({len(known_genes)} genes):**")
                        known_df = pd.DataFrame([
                            {"Gene": g, "Description": graph.nodes[g].get("name", "")}
                            for g in known_genes
                        ])
                        st.dataframe(known_df, height=300)

                    with col_pred:
                        st.markdown(f"**Candidates ({len(candidates)} genes):**")
                        if scorer_name == HETERO_SCORER:
                            st.caption("Ranked by the gene–disease model's score; enrichment shown for reference")
                        else:
                            st.caption("Enrichment = -log10 p that this many of the gene's partners are known disease genes")
                        pred_df = pd.DataFrame([{
                            "Gene": c["gene"],
                            **({"Model score": c["model_score"]} if scorer_name == HETERO_SCORER else {}),
                            "Enrichment": c["enrichment_score"],
                            "Known interactors": c["known_interactors"],
                            **({"Similarity": c["similarity"]} if z_norm is not None else {}),
                            "Description": c["gene_description"],
                        } for c in candidates])
                        st.dataframe(pred_df, height=300)

                    # Visualization: known genes that touch the shown candidates first
                    st.subheader("Discovery network")
                    st.caption("Blue = known genes, Orange = predicted candidates. Lines = protein interactions.")
                    shown_candidates = [c["gene"] for c in candidates[:20]]
                    touching = [g for g in known_genes
                                if any(graph.has_edge(g, c) for c in shown_candidates)]
                    known_shown = (touching + [g for g in known_genes if g not in touching])[:30]

                    net = Network(height="500px", width="100%", bgcolor="#0e1117", font_color="white")
                    net.barnes_hut(gravity=-3000)
                    for g in known_shown:
                        net.add_node(g, label=g, color="#2a5a8c", size=15, title=f"Known: {g}")
                    for c in candidates[:20]:
                        net.add_node(c["gene"], label=c["gene"], color="#ff8844", size=12,
                                     title=f"Predicted: {c['gene']} (enrichment={c['enrichment_score']:.2f}, "
                                           f"{c['known_interactors']} known interactors)")

                    shown = set(known_shown) | set(shown_candidates)
                    known_set = set(known_shown)
                    for u in shown:
                        for v in shown:
                            if u < v and graph.has_edge(u, v):
                                d = graph.edges[u, v]
                                color = "#2a5a8c" if u in known_set and v in known_set else "#ff8844"
                                net.add_edge(u, v, width=2, color=color,
                                             title=f"Confidence: {d.get('score', 'N/A')}")

                    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                        net.save_graph(f.name)
                        html = open(f.name).read()
                        components.html(html, height=520, scrolling=True)
                    os.unlink(f.name)

                    # Export this disease (same columns as the full bundle)
                    csv = pd.DataFrame(candidates).to_csv(index=False)
                    st.download_button(
                        "📥 Export predictions as CSV",
                        csv,
                        f"predictions_{selected_disease.replace(' ', '_')}.csv",
                        "text/csv",
                    )
                else:
                    st.warning("No candidates found. The disease may have too few known genes for reliable predictions.")
        else:
            st.info("No diseases match the current filters. Try lowering the minimum known genes or adding more categories.")
    else:
        st.warning("No disease/GWAS data found in the network. Rebuild with GWAS associations enabled.")

# === Tab 2: Gene Similarity ===
with tab_gene:
    gene_input = ""
    if z_norm is None:
        st.info(NEEDS_PROTEIN_MODEL.format(tab="Find Similar Genes"))
    else:
        st.markdown(
            "Enter a gene to find others with the most **similar AI-learned profiles**. "
            "These genes share expression patterns, network neighborhoods, and disease associations — "
            "they may be involved in the same biological pathways."
        )
        gene_input = st.text_input("Gene symbol", placeholder="e.g., TP53, BRCA1, APOE", key="sim_gene").strip().upper()

    if gene_input:
        if gene_input in node_to_idx:
            similar = find_similar_genes(gene_input, top_k=30)

            # Show query gene info
            if graph and gene_input in graph.nodes:
                gdata = graph.nodes[gene_input]
                st.markdown(f"**{gene_input}** — {gdata.get('name', 'Unknown')}")
                gwas = gdata.get("gwas", {})
                if gwas:
                    st.markdown("Known associations:")
                    for cat, entries in gwas.items():
                        traits = [e["trait"] for e in sorted(entries, key=lambda x: -x["score"])[:5]]
                        st.markdown(f"- *{cat}*: {', '.join(traits)}")

            st.divider()
            st.markdown(f"**Most similar genes to {gene_input}:**")
            st.caption("Higher similarity = more similar AI-learned profile")
            sim_df = pd.DataFrame(similar)
            sim_df.columns = ["Gene", "Similarity", "Description"]
            st.dataframe(sim_df, height=400)

            # Show shared diseases
            if graph:
                query_gwas = graph.nodes.get(gene_input, {}).get("gwas", {})
                query_traits = set()
                for entries in query_gwas.values():
                    for e in entries:
                        query_traits.add(e["trait"])

                if query_traits:
                    st.subheader("Shared disease/trait associations")
                    st.caption(f"Diseases linked to both {gene_input} and the similar gene")
                    for s in similar[:10]:
                        other_gwas = graph.nodes.get(s["gene"], {}).get("gwas", {})
                        other_traits = set()
                        for entries in other_gwas.values():
                            for e in entries:
                                other_traits.add(e["trait"])
                        shared = query_traits & other_traits
                        if shared:
                            st.markdown(f"**{s['gene']}** (similarity={s['similarity']:.3f}): "
                                        f"{', '.join(list(shared)[:5])}")
        else:
            st.warning(f"Gene '{gene_input}' not found in the network. Check the spelling or try a different gene.")

# === Tab 3: Link Prediction ===
with tab_link:
    gene_a = gene_b = ""
    if z_norm is None:
        st.info(NEEDS_PROTEIN_MODEL.format(tab="Check Gene Pair"))
    else:
        st.markdown(
            "Check whether two specific genes are predicted to interact. "
            "The model scores the likelihood based on their learned profiles."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            gene_a = st.text_input("First gene", placeholder="e.g., TP53", key="link_a").strip().upper()
        with col_b:
            gene_b = st.text_input("Second gene", placeholder="e.g., BRCA1", key="link_b").strip().upper()

    if gene_a and gene_b:
        if gene_a not in node_to_idx:
            st.warning(f"Gene '{gene_a}' not found in the network.")
        elif gene_b not in node_to_idx:
            st.warning(f"Gene '{gene_b}' not found in the network.")
        elif gene_a == gene_b:
            st.warning("Please enter two different genes.")
        else:
            idx_a = node_to_idx[gene_a]
            idx_b = node_to_idx[gene_b]

            sim = F.cosine_similarity(z[idx_a].unsqueeze(0), z[idx_b].unsqueeze(0)).item()

            with torch.no_grad():
                src_t = torch.tensor([idx_a])
                dst_t = torch.tensor([idx_b])
                link_score = model.decode(z, src_t, dst_t).item()

            has_edge = graph.has_edge(gene_a, gene_b) if graph else False

            st.markdown(f"### {gene_a} ↔ {gene_b}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted link score", f"{link_score:.4f}",
                        help="Model's confidence that these genes interact. Higher = more likely.")
            col2.metric("Profile similarity", f"{sim:.4f}",
                        help="How similar their AI-learned profiles are (cosine similarity).")
            col3.metric("Already known?", "Yes ✅" if has_edge else "No",
                        help="Whether this interaction already exists in the STRING database.")

            if has_edge and graph:
                edge_data = graph.edges[gene_a, gene_b]
                st.markdown(f"**Known interaction:** STRING confidence = {edge_data.get('score', 'N/A')}")

            if graph:
                neighbors_a = set(graph.neighbors(gene_a)) if gene_a in graph else set()
                neighbors_b = set(graph.neighbors(gene_b)) if gene_b in graph else set()
                shared = neighbors_a & neighbors_b
                if shared:
                    st.markdown(f"**Shared interaction partners ({len(shared)}):** "
                                f"{', '.join(list(shared)[:15])}")

# ---------------------------------------------------------------------------
# Save / Export
# ---------------------------------------------------------------------------
st.divider()
if model is not None and st.button("💾 Save Model to Disk"):
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".models")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_type.lower().replace(' ', '_')}_model.pt")
    torch.save(model.state_dict(), path)
    st.success(f"Model saved to `{path}`")

st.subheader("📦 Export discoveries")
st.markdown(
    "Write **every predicted gene–disease connection** for the diseases selected below to a "
    "folder of CSV files, with a manifest describing the model and method. Grant that folder "
    "to Claude Science (or open it in any other tool) to continue the analysis there."
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model_token = id(model) if model is not None else "no-protein-model"

if not disease_index:
    st.info("No disease/GWAS data in the network, so there is nothing to export.")
else:
    export_categories = sorted({rec["category"] for rec in disease_index.values()})
    with st.form("export_discoveries"):
        col_cat, col_min, col_k = st.columns(3)
        with col_cat:
            exp_categories = st.multiselect(
                "Categories",
                export_categories,
                default=["disease"] if "disease" in export_categories else export_categories[:1],
            )
        with col_min:
            exp_min_genes = st.slider(
                "Min known genes per disease", 2, 50, 5,
                help="Diseases with fewer known genes are skipped; more known genes = better signatures.",
            )
        with col_k:
            exp_top_k = st.slider("Candidates per disease", 10, 200, 50)
        exp_scorer = pick_scorer("export_scorer")
        exp_dir = st.text_input(
            "Export folder",
            value=os.path.join(_PROJECT_ROOT, ".datasets", "exports"),
            help="A time-stamped sub-folder is created here. Grant it to Claude Science afterwards.",
        )
        exp_submit = st.form_submit_button("📦 Export all predicted connections", type="primary")

    if exp_submit:
        chosen = select_diseases(disease_index, exp_categories, exp_min_genes)
        if not chosen:
            st.warning("No diseases match these filters. Lower the minimum known genes or add categories.")
        else:
            try:
                with st.spinner(f"Scoring candidates for {len(chosen)} diseases..."):
                    exp_scores, exp_scorer_name, exp_scorer_path = scorer_for(exp_scorer, chosen)
                    links = predict_gene_disease_links(
                        z_norm, node_list, node_to_idx, graph, disease_index, chosen, exp_top_k,
                        scores=exp_scores,
                    )
                    known = known_links_rows(disease_index, chosen)
                    manifest = build_manifest(
                        model_type=model_type,
                        metrics=metrics,
                        checkpoint_path=results.get("checkpoint_path") if results else None,
                        graph_name=session.get_graph_name(),
                        n_nodes=pyg_data.num_nodes if pyg_data is not None else graph.number_of_nodes(),
                        n_edges=pyg_data.edge_index.size(1) // 2 if pyg_data is not None else graph.number_of_edges(),
                        embedding_dim=z.size(1) if z is not None else None,
                        categories=exp_categories,
                        min_known_genes=exp_min_genes,
                        top_k=exp_top_k,
                        links=links,
                        known=known,
                        scorer=exp_scorer_name,
                        scorer_checkpoint=exp_scorer_path,
                    )
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    bundle = write_discovery_bundle(
                        os.path.join(exp_dir, f"discovery_{stamp}"), links, known, manifest,
                    )
            except OSError as e:
                st.error(f"Could not write to `{exp_dir}`: {e}")
            else:
                st.session_state["discovery_export"] = {
                    "model_token": _model_token,
                    "dir": str(bundle),
                    "zip": zip_bundle(bundle),
                    "links": links,
                    "manifest": manifest,
                }

    export = st.session_state.get("discovery_export")
    if export and export["model_token"] == _model_token:
        counts = export["manifest"]["counts"]
        st.success(
            f"Exported {counts['predicted_links']:,} predicted connections for "
            f"{counts['diseases']} diseases to `{export['dir']}`"
        )
        st.caption(
            "In Claude Science: paste this folder's path into the composer, allow read access, "
            f"and ask Claude to load `{PREDICTED_FILE}` and save it as an artifact."
        )
        if export["links"]:
            st.markdown("**Preview (first 20 rows):**")
            st.dataframe(pd.DataFrame(export["links"]).head(20), height=300)
        st.download_button(
            "📥 Download bundle as zip",
            export["zip"],
            f"{os.path.basename(export['dir'])}.zip",
            "application/zip",
        )
