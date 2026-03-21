# pages/4_Results.py
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pyvis.network import Network
import tempfile
import os
import glob as glob_mod
from src.utils import session
from src.models.gnn import GNNLinkPredictor
from src.models.graph_transformer import GraphTransformerLinkPredictor
from src.models.vae import GraphVAE

st.title("Discovery Dashboard")

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
            session.set_graph(graph)

# If no session results, let user pick from saved checkpoints or training runs
if results is None:
    from src.utils.graph_features import nx_to_pyg_data

    # Collect all loadable models
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".datasets", "trained_models")
    checkpoints = sorted(glob_mod.glob(os.path.join(checkpoint_dir, "*_checkpoint.pt")))
    training_runs = _mgr.list_training_runs()

    if not checkpoints and not training_runs:
        st.warning("No trained models found. Go to **Model Training** first.")
        st.stop()

    st.subheader("Load a trained model")

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
                     f"AUC={run['auc_roc']:.4f}, {run['nodes']} nodes, "
                     f"{run['parameters']}")
            options.append(label)
            option_meta.append(("run", run))

    if not options:
        st.warning("No loadable models found. Train a model first.")
        st.stop()

    selected_idx = st.selectbox("Select model", range(len(options)),
                                format_func=lambda i: options[i])

    if st.button("Load model", type="primary"):
        kind, meta = option_meta[selected_idx]

        if kind == "checkpoint":
            checkpoint = torch.load(meta, map_location="cpu", weights_only=False)
            model_class = checkpoint["model_class"]
            model_kwargs = checkpoint["model_kwargs"]

            # Load graph for this checkpoint
            graph_name = checkpoint.get("graph_name", "")
            if graph is None and graph_name:
                graph = _mgr.load_graph(graph_name)
                if graph:
                    session.set_graph(graph)

            if graph is None:
                st.error("No graph available. Build/load a graph from Model Training first.")
                st.stop()

            # Reconstruct pyg_data from graph
            with st.spinner("Reconstructing features from graph..."):
                pyg_data, node_list, node_to_idx = nx_to_pyg_data(graph)

            if model_class == "GNNLinkPredictor":
                mdl = GNNLinkPredictor(**model_kwargs)
            elif model_class == "GraphTransformerLinkPredictor":
                mdl = GraphTransformerLinkPredictor(**model_kwargs)
            else:
                mdl = GraphVAE(**model_kwargs)
            mdl.load_state_dict(checkpoint["model_state"])

            results = {
                "model": mdl,
                "pyg_data": pyg_data,
                "node_list": node_list,
                "node_to_idx": node_to_idx,
                "model_type": checkpoint["model_type"],
                "metrics": checkpoint["metrics"],
            }
        else:
            st.error("Old training runs require checkpoint format. Please retrain to create a loadable checkpoint.")
            st.stop()

        session.set_training_results(results)
        st.rerun()

    st.stop()

model = results["model"]
pyg_data = results["pyg_data"]
node_list = results["node_list"]
node_to_idx = results["node_to_idx"]
metrics = results["metrics"]
model_type = results["model_type"]

# --- Compute embeddings once ---
model.eval()
with torch.no_grad():
    z = model.encode(pyg_data)  # [num_nodes, hidden_dim]
    z_norm = F.normalize(z, dim=-1)  # for cosine similarity


# --- Helper functions ---
def get_disease_genes(disease_name):
    """Find genes associated with a disease in the graph's GWAS data."""
    genes = []
    for node in node_list:
        gwas = graph.nodes[node].get("gwas", {}) if graph else {}
        for cat, entries in gwas.items():
            for entry in entries:
                if disease_name.lower() in entry["trait"].lower():
                    genes.append((node, entry["trait"], entry["score"], cat))
    return genes


def get_all_diseases():
    """Collect all unique GWAS traits from the graph."""
    diseases = {}
    if not graph:
        return diseases
    for node in node_list:
        gwas = graph.nodes[node].get("gwas", {})
        for cat, entries in gwas.items():
            for entry in entries:
                trait = entry["trait"]
                if trait not in diseases:
                    diseases[trait] = {"category": cat, "gene_count": 0}
                diseases[trait]["gene_count"] += 1
    return diseases


def predict_disease_genes(disease_name, top_k=50):
    """Predict new genes for a disease using embedding similarity."""
    known = get_disease_genes(disease_name)
    known_genes = list({g[0] for g in known})
    if not known_genes:
        return [], known_genes

    # Compute disease signature (mean embedding of known genes)
    known_indices = [node_to_idx[g] for g in known_genes if g in node_to_idx]
    if not known_indices:
        return [], known_genes

    disease_sig = z_norm[known_indices].mean(dim=0, keepdim=True)
    disease_sig = F.normalize(disease_sig, dim=-1)

    # Cosine similarity to all genes
    similarities = (z_norm @ disease_sig.T).squeeze()

    # Rank and exclude known genes
    known_set = set(known_genes)
    candidates = []
    for idx in similarities.argsort(descending=True):
        idx = idx.item()
        gene = node_list[idx]
        if gene not in known_set:
            candidates.append({
                "gene": gene,
                "similarity": round(similarities[idx].item(), 4),
                "description": graph.nodes[gene].get("name", "") if graph else "",
            })
            if len(candidates) >= top_k:
                break

    return candidates, known_genes


def find_similar_genes(gene_name, top_k=20):
    """Find genes with most similar embeddings."""
    if gene_name not in node_to_idx:
        return []
    idx = node_to_idx[gene_name]
    sims = (z_norm @ z_norm[idx]).squeeze()
    results = []
    for i in sims.argsort(descending=True):
        i = i.item()
        if node_list[i] != gene_name:
            results.append({
                "gene": node_list[i],
                "similarity": round(sims[i].item(), 4),
                "description": graph.nodes[node_list[i]].get("name", "") if graph else "",
            })
            if len(results) >= top_k:
                break
    return results


# --- UI Layout ---
st.markdown(f"**Model:** {model_type} | **AUC-ROC:** {metrics['auc_roc']:.4f} | "
            f"**Avg Precision:** {metrics['avg_precision']:.4f} | "
            f"**Embedding dim:** {z.size(1)} | **Genes:** {z.size(0)}")

tab_disease, tab_gene, tab_link = st.tabs([
    "Disease Gene Discovery",
    "Gene Similarity",
    "Link Prediction",
])

# === Tab 1: Disease Gene Discovery ===
with tab_disease:
    st.subheader("Predict new genes for a disease")
    st.markdown("Select a disease/trait to find genes that the model predicts are involved "
                "but aren't yet in GWAS data.")

    all_diseases = get_all_diseases()
    if all_diseases:
        # Filter options
        col_cat, col_min = st.columns(2)
        with col_cat:
            categories = sorted({v["category"] for v in all_diseases.values()})
            sel_cat = st.multiselect("Filter by category", categories, default=["disease"])
        with col_min:
            min_genes = st.slider("Min known genes", 2, 50, 5)

        # Build filtered disease list
        filtered = {k: v for k, v in all_diseases.items()
                    if v["category"] in sel_cat and v["gene_count"] >= min_genes}
        disease_options = sorted(filtered.keys(), key=lambda x: filtered[x]["gene_count"], reverse=True)

        if disease_options:
            selected_disease = st.selectbox(
                f"Disease/trait ({len(disease_options)} available)",
                disease_options,
                format_func=lambda x: f"{x} ({filtered[x]['gene_count']} known genes, {filtered[x]['category']})",
            )
            top_k = st.slider("Number of predictions", 10, 200, 50)

            if st.button("Predict", type="primary"):
                candidates, known_genes = predict_disease_genes(selected_disease, top_k)

                if candidates:
                    col_known, col_pred = st.columns(2)

                    with col_known:
                        st.markdown(f"**Known genes ({len(known_genes)}):**")
                        known_df = pd.DataFrame([
                            {"Gene": g, "Description": graph.nodes[g].get("name", "") if graph else ""}
                            for g in sorted(known_genes)
                        ])
                        st.dataframe(known_df, height=300)

                    with col_pred:
                        st.markdown(f"**Predicted candidates ({len(candidates)}):**")
                        pred_df = pd.DataFrame(candidates)
                        pred_df.columns = ["Gene", "Similarity", "Description"]
                        st.dataframe(pred_df, height=300)

                    # Visualization
                    st.subheader("Discovery network")
                    net = Network(height="500px", width="100%", bgcolor="#0e1117", font_color="white")
                    net.barnes_hut(gravity=-3000)

                    # Add known genes
                    for g in known_genes[:30]:
                        net.add_node(g, label=g, color="#2a5a8c", size=15,
                                     title=f"Known: {g}")

                    # Add predicted genes
                    for c in candidates[:20]:
                        net.add_node(c["gene"], label=c["gene"], color="#ff8844", size=12,
                                     title=f"Predicted: {c['gene']} (sim={c['similarity']:.3f})")

                    # Add edges between genes that are connected in STRING
                    shown = set(known_genes[:30]) | {c["gene"] for c in candidates[:20]}
                    if graph:
                        for u in shown:
                            for v in shown:
                                if u < v and graph.has_edge(u, v):
                                    d = graph.edges[u, v]
                                    color = "#2a5a8c" if u in known_genes and v in known_genes else "#ff8844"
                                    net.add_edge(u, v, width=2, color=color,
                                                 title=f"STRING: {d.get('score', 'N/A')}")

                    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                        net.save_graph(f.name)
                        html = open(f.name).read()
                        components.html(html, height=520, scrolling=True)
                    os.unlink(f.name)

                    # Export
                    csv = pred_df.to_csv(index=False)
                    st.download_button("Export predictions CSV", csv,
                                       f"predictions_{selected_disease.replace(' ', '_')}.csv", "text/csv")
                else:
                    st.warning("No candidates found. The disease may have too few known genes.")
        else:
            st.info("No diseases match the current filters. Try lowering min known genes or adding categories.")
    else:
        st.warning("No GWAS data found in the graph. Rebuild with GWAS enabled.")

# === Tab 2: Gene Similarity ===
with tab_gene:
    st.subheader("Find genes with similar profiles")
    st.markdown("Enter a gene to find others with the most similar learned embeddings. "
                "These share expression patterns, network neighborhoods, and disease associations.")

    gene_input = st.text_input("Gene symbol", placeholder="e.g., TP53, BRCA1, APOE").strip().upper()

    if gene_input:
        if gene_input in node_to_idx:
            similar = find_similar_genes(gene_input, top_k=30)

            # Show query gene info
            if graph and gene_input in graph.nodes:
                gdata = graph.nodes[gene_input]
                st.markdown(f"**{gene_input}** — {gdata.get('name', 'N/A')}")
                gwas = gdata.get("gwas", {})
                if gwas:
                    for cat, entries in gwas.items():
                        traits = [e["trait"] for e in sorted(entries, key=lambda x: -x["score"])[:5]]
                        st.markdown(f"- *{cat}*: {', '.join(traits)}")

            st.divider()
            st.markdown(f"**Top similar genes to {gene_input}:**")
            sim_df = pd.DataFrame(similar)
            sim_df.columns = ["Gene", "Similarity", "Description"]
            st.dataframe(sim_df, height=400)

            # Show shared diseases between query and top similar genes
            if graph:
                query_gwas = graph.nodes.get(gene_input, {}).get("gwas", {})
                query_traits = set()
                for entries in query_gwas.values():
                    for e in entries:
                        query_traits.add(e["trait"])

                if query_traits:
                    st.subheader("Shared disease/trait associations")
                    for s in similar[:10]:
                        other_gwas = graph.nodes.get(s["gene"], {}).get("gwas", {})
                        other_traits = set()
                        for entries in other_gwas.values():
                            for e in entries:
                                other_traits.add(e["trait"])
                        shared = query_traits & other_traits
                        if shared:
                            st.markdown(f"**{s['gene']}** (sim={s['similarity']:.3f}): "
                                        f"{', '.join(list(shared)[:5])}")
        else:
            st.warning(f"Gene '{gene_input}' not found in the graph.")

# === Tab 3: Link Prediction ===
with tab_link:
    st.subheader("Predict interactions between genes")
    st.markdown("Check if two genes are predicted to interact.")

    col_a, col_b = st.columns(2)
    with col_a:
        gene_a = st.text_input("Gene A", placeholder="e.g., TP53").strip().upper()
    with col_b:
        gene_b = st.text_input("Gene B", placeholder="e.g., BRCA1").strip().upper()

    if gene_a and gene_b:
        if gene_a not in node_to_idx:
            st.warning(f"Gene '{gene_a}' not found.")
        elif gene_b not in node_to_idx:
            st.warning(f"Gene '{gene_b}' not found.")
        elif gene_a == gene_b:
            st.warning("Enter two different genes.")
        else:
            idx_a = node_to_idx[gene_a]
            idx_b = node_to_idx[gene_b]

            # Embedding similarity
            sim = F.cosine_similarity(z[idx_a].unsqueeze(0), z[idx_b].unsqueeze(0)).item()

            # Link prediction score
            with torch.no_grad():
                src_t = torch.tensor([idx_a])
                dst_t = torch.tensor([idx_b])
                link_score = model.decode(z, src_t, dst_t).item()

            # Known edge?
            has_edge = graph.has_edge(gene_a, gene_b) if graph else False

            st.markdown(f"**{gene_a} -- {gene_b}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Link score", f"{link_score:.4f}")
            col2.metric("Embedding similarity", f"{sim:.4f}")
            col3.metric("Known interaction", "Yes" if has_edge else "No")

            if has_edge and graph:
                edge_data = graph.edges[gene_a, gene_b]
                st.markdown(f"**Known edge:** STRING score={edge_data.get('score', 'N/A')}, "
                            f"evidence={edge_data.get('evidence', 'N/A')}")

            # Show top shared neighbors
            if graph:
                neighbors_a = set(graph.neighbors(gene_a)) if gene_a in graph else set()
                neighbors_b = set(graph.neighbors(gene_b)) if gene_b in graph else set()
                shared = neighbors_a & neighbors_b
                if shared:
                    st.markdown(f"**Shared interaction partners ({len(shared)}):** "
                                f"{', '.join(list(shared)[:15])}")

# --- Save/Export ---
st.divider()
col_save, col_export = st.columns(2)
with col_save:
    if st.button("Save Model"):
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".models")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_type.lower().replace(' ', '_')}_model.pt")
        torch.save(model.state_dict(), path)
        st.success(f"Model saved to `{path}`")
