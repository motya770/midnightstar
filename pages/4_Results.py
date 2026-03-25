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
import glob as glob_mod
from src.utils import session
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
            session.set_graph(graph)

# ---------------------------------------------------------------------------
# No model loaded — let user pick one
# ---------------------------------------------------------------------------
if results is None:
    from src.utils.graph_features import nx_to_pyg_data

    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".datasets", "trained_models")
    checkpoints = sorted(glob_mod.glob(os.path.join(checkpoint_dir, "*_checkpoint.pt")))
    training_runs = _mgr.list_training_runs()

    if not checkpoints and not training_runs:
        st.info(
            "No trained models found yet. You need to train a model first before you can see predictions."
        )
        st.markdown(
            "Go to **Train Model** to build a network and train an AI model on it. "
            "This typically takes a few minutes."
        )
        st.stop()

    st.subheader("Select a trained model")
    st.markdown("Choose which model to use for predictions.")

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
        st.warning("No loadable models found. Please retrain a model.")
        st.stop()

    selected_idx = st.selectbox("Model", range(len(options)),
                                format_func=lambda i: options[i])

    if st.button("Load Model", type="primary"):
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

# ---------------------------------------------------------------------------
# Model is loaded — compute embeddings
# ---------------------------------------------------------------------------
model = results["model"]
pyg_data = results["pyg_data"]
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

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_disease_genes(disease_name):
    genes = []
    for node in node_list:
        gwas = graph.nodes[node].get("gwas", {}) if graph else {}
        for cat, entries in gwas.items():
            for entry in entries:
                if disease_name.lower() in entry["trait"].lower():
                    genes.append((node, entry["trait"], entry["score"], cat))
    return genes


def get_all_diseases():
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
    known = get_disease_genes(disease_name)
    known_genes = list({g[0] for g in known})
    if not known_genes:
        return [], known_genes

    known_indices = [node_to_idx[g] for g in known_genes if g in node_to_idx]
    if not known_indices:
        return [], known_genes

    disease_sig = z_norm[known_indices].mean(dim=0, keepdim=True)
    disease_sig = F.normalize(disease_sig, dim=-1)

    similarities = (z_norm @ disease_sig.T).squeeze()

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
    if gene_name not in node_to_idx:
        return []
    idx = node_to_idx[gene_name]
    sims = (z_norm @ z_norm[idx]).squeeze()
    results_list = []
    for i in sims.argsort(descending=True):
        i = i.item()
        if node_list[i] != gene_name:
            results_list.append({
                "gene": node_list[i],
                "similarity": round(sims[i].item(), 4),
                "description": graph.nodes[node_list[i]].get("name", "") if graph else "",
            })
            if len(results_list) >= top_k:
                break
    return results_list


# ---------------------------------------------------------------------------
# Three discovery tabs
# ---------------------------------------------------------------------------
tab_disease, tab_gene, tab_link = st.tabs([
    "🧬 Predict Disease Genes",
    "🔎 Find Similar Genes",
    "🔗 Check Gene Pair",
])

# === Tab 1: Disease Gene Discovery ===
with tab_disease:
    st.markdown(
        "**Pick a disease** to find genes the model predicts are involved — "
        "even though they're not yet in the GWAS database. "
        "These are candidates for further research."
    )

    all_diseases = get_all_diseases()
    if all_diseases:
        col_cat, col_min = st.columns(2)
        with col_cat:
            categories = sorted({v["category"] for v in all_diseases.values()})
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

        filtered = {k: v for k, v in all_diseases.items()
                    if v["category"] in sel_cat and v["gene_count"] >= min_genes}
        disease_options = sorted(filtered.keys(), key=lambda x: filtered[x]["gene_count"], reverse=True)

        if disease_options:
            selected_disease = st.selectbox(
                f"Disease or trait ({len(disease_options)} available)",
                disease_options,
                format_func=lambda x: f"{x} ({filtered[x]['gene_count']} known genes)",
            )
            top_k = st.slider("How many predictions?", 10, 200, 50)

            if st.button("🔮 Predict New Genes", type="primary"):
                candidates, known_genes = predict_disease_genes(selected_disease, top_k)

                if candidates:
                    col_known, col_pred = st.columns(2)

                    with col_known:
                        st.markdown(f"**Already known ({len(known_genes)} genes):**")
                        known_df = pd.DataFrame([
                            {"Gene": g, "Description": graph.nodes[g].get("name", "") if graph else ""}
                            for g in sorted(known_genes)
                        ])
                        st.dataframe(known_df, height=300)

                    with col_pred:
                        st.markdown(f"**AI predictions ({len(candidates)} candidates):**")
                        st.caption("Higher similarity = stronger prediction")
                        pred_df = pd.DataFrame(candidates)
                        pred_df.columns = ["Gene", "Similarity Score", "Description"]
                        st.dataframe(pred_df, height=300)

                    # Visualization
                    st.subheader("Discovery network")
                    st.caption("Blue = known genes, Orange = predicted candidates. Lines = protein interactions.")
                    net = Network(height="500px", width="100%", bgcolor="#0e1117", font_color="white")
                    net.barnes_hut(gravity=-3000)

                    for g in known_genes[:30]:
                        net.add_node(g, label=g, color="#2a5a8c", size=15,
                                     title=f"Known: {g}")
                    for c in candidates[:20]:
                        net.add_node(c["gene"], label=c["gene"], color="#ff8844", size=12,
                                     title=f"Predicted: {c['gene']} (score={c['similarity']:.3f})")

                    shown = set(known_genes[:30]) | {c["gene"] for c in candidates[:20]}
                    if graph:
                        for u in shown:
                            for v in shown:
                                if u < v and graph.has_edge(u, v):
                                    d = graph.edges[u, v]
                                    color = "#2a5a8c" if u in known_genes and v in known_genes else "#ff8844"
                                    net.add_edge(u, v, width=2, color=color,
                                                 title=f"Confidence: {d.get('score', 'N/A')}")

                    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                        net.save_graph(f.name)
                        html = open(f.name).read()
                        components.html(html, height=520, scrolling=True)
                    os.unlink(f.name)

                    # Export
                    csv = pred_df.to_csv(index=False)
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
# Save/Export
# ---------------------------------------------------------------------------
st.divider()
col_save, col_export = st.columns(2)
with col_save:
    if st.button("💾 Save Model to Disk"):
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".models")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_type.lower().replace(' ', '_')}_model.pt")
        torch.save(model.state_dict(), path)
        st.success(f"Model saved to `{path}`")
