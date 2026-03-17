# pages/4_Results.py
import streamlit as st
import streamlit.components.v1 as components
import torch
import pandas as pd
import numpy as np
from pyvis.network import Network
import tempfile
import os
from src.utils.text_explainer import TextExplainer
from src.utils import session

st.title("📊 Discovery Dashboard")

results = session.get_training_results()
graph = session.get_graph()

if results is None:
    st.warning("No training results available. Go to **Model Training** first.")
    st.stop()

model = results["model"]
pyg_data = results["pyg_data"]
node_list = results["node_list"]
node_to_idx = results["node_to_idx"]
metrics = results["metrics"]
model_type = results["model_type"]
explainer = TextExplainer()

# Generate predictions for all non-existing edges
model.eval()
with torch.no_grad():
    existing_edges = set()
    for i in range(pyg_data.edge_index.size(1)):
        src_idx = pyg_data.edge_index[0, i].item()
        dst_idx = pyg_data.edge_index[1, i].item()
        existing_edges.add((min(src_idx, dst_idx), max(src_idx, dst_idx)))

    predictions = []
    num_nodes = pyg_data.num_nodes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if (i, j) not in existing_edges:
                src_t = torch.tensor([i])
                dst_t = torch.tensor([j])
                if hasattr(model, "predict_links"):
                    score = model.predict_links(pyg_data.x, pyg_data.edge_index, src_t, dst_t).item()
                else:
                    score = model(pyg_data, src_t, dst_t).item()
                if score > 0.3:
                    predictions.append({
                        "Gene A": node_list[i],
                        "Gene B": node_list[j],
                        "Predicted Score": round(score, 4),
                        "src_idx": i,
                        "dst_idx": j,
                    })

    predictions.sort(key=lambda x: x["Predicted Score"], reverse=True)

# 2x2 Grid
top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)

# Top-left: Discovery Network
with top_left:
    st.subheader("Discovery Network")
    show_only_new = st.checkbox("Show only discoveries", value=False)

    net = Network(height="400px", width="100%", bgcolor="#0e1117", font_color="white")
    net.barnes_hut(gravity=-2000)

    shown_nodes = set()
    # Add predicted edges
    for pred in predictions[:20]:
        for gene in [pred["Gene A"], pred["Gene B"]]:
            if gene not in shown_nodes:
                net.add_node(gene, label=gene, color="#ff8844", size=15, title=f"Discovery: {gene}")
                shown_nodes.add(gene)
        net.add_edge(pred["Gene A"], pred["Gene B"],
                    width=pred["Predicted Score"] * 4,
                    color="#ff8844", dashes=True,
                    title=f"Predicted: {pred['Predicted Score']:.2f}")

    if not show_only_new and graph:
        for u, v, d in graph.edges(data=True):
            for node in [u, v]:
                if node not in shown_nodes:
                    node_data = graph.nodes.get(node, {})
                    color = "#2a5a8c" if node_data.get("node_type") == "gene" else "#8c2a5c"
                    net.add_node(node, label=node, color=color, size=12)
                    shown_nodes.add(node)
            net.add_edge(u, v, width=d.get("score", 0.5) * 3, color="#2a5a8c",
                        title=f"Known: {d.get('score', 'N/A')}")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html = open(f.name).read()
        components.html(html, height=420, scrolling=True)
    os.unlink(f.name)

# Top-right: Summary
with top_right:
    st.subheader("Summary")
    st.markdown(f"**Model:** {model_type} | **AUC-ROC:** {metrics['auc_roc']:.4f} | "
                f"**Avg Precision:** {metrics['avg_precision']:.4f}")

    num_discoveries = len(predictions)
    if num_discoveries > 0:
        top = predictions[0]
        st.markdown(
            f"The {model_type} model found **{num_discoveries} potential new connections**. "
            f"The strongest predicted link is between **{top['Gene A']}** and "
            f"**{top['Gene B']}** (confidence: {top['Predicted Score']:.2f})."
        )

        st.markdown("**Top Discoveries:**")
        for i, pred in enumerate(predictions[:10], 1):
            score = pred["Predicted Score"]
            color = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🟠"
            st.markdown(f"{i}. {color} **{pred['Gene A']}** ↔ **{pred['Gene B']}** — {score:.2f}")

            with st.expander(f"Why {pred['Gene A']} ↔ {pred['Gene B']}?"):
                # Find shared info from graph
                shared_partners, shared_tissues = [], []
                if graph:
                    neighbors_a = set(graph.neighbors(pred["Gene A"])) if pred["Gene A"] in graph else set()
                    neighbors_b = set(graph.neighbors(pred["Gene B"])) if pred["Gene B"] in graph else set()
                    shared_partners = list(neighbors_a & neighbors_b)
                    expr_a = graph.nodes.get(pred["Gene A"], {}).get("expression", {})
                    expr_b = graph.nodes.get(pred["Gene B"], {}).get("expression", {})
                    shared_tissues = [t for t in expr_a if t in expr_b]

                explanation = explainer.explain_prediction(
                    pred["Gene A"], pred["Gene B"], score, shared_partners, shared_tissues,
                )
                st.markdown(explanation)
    else:
        st.info("No strong predictions found. Try training with more data or different parameters.")

# Bottom-left: Data Table
with bottom_left:
    st.subheader("Predictions Table")
    if predictions:
        df = pd.DataFrame(predictions)[["Gene A", "Gene B", "Predicted Score"]]
        st.dataframe(df, width="stretch", height=300)

        csv = df.to_csv(index=False)
        st.download_button("📥 Export CSV", csv, "predictions.csv", "text/csv")
    else:
        st.info("No predictions to display.")

# Bottom-right: Evidence
with bottom_right:
    st.subheader("Evidence Panel")
    if predictions:
        selected_pred = st.selectbox(
            "Select prediction",
            [f"{p['Gene A']} ↔ {p['Gene B']} ({p['Predicted Score']:.2f})" for p in predictions[:20]],
        )
        if selected_pred:
            idx = next(
                i for i, p in enumerate(predictions[:20])
                if f"{p['Gene A']} ↔ {p['Gene B']}" in selected_pred
            )
            pred = predictions[idx]

            if graph:
                for gene in [pred["Gene A"], pred["Gene B"]]:
                    expr = graph.nodes.get(gene, {}).get("expression", {})
                    if expr:
                        st.markdown(f"**{gene} expression:**")
                        top_tissues = sorted(expr.items(), key=lambda x: x[1], reverse=True)[:5]
                        tissue_df = pd.DataFrame(top_tissues, columns=["Tissue", "Level"])
                        st.bar_chart(tissue_df.set_index("Tissue"))

                st.markdown("**Supporting evidence:**")
                for gene in [pred["Gene A"], pred["Gene B"]]:
                    neighbors = list(graph.neighbors(gene)) if gene in graph else []
                    if neighbors:
                        st.markdown(f"- {gene} connects to: {', '.join(neighbors[:5])}")
    else:
        st.info("Select a prediction to see evidence.")

# Save/Export
st.divider()
col_save, col_export = st.columns(2)
with col_save:
    if st.button("💾 Save Model"):
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".models")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_type.lower().replace(' ', '_')}_model.pt")
        torch.save(model.state_dict(), path)
        st.success(f"Model saved to `{path}`")

with col_export:
    if predictions:
        full_df = pd.DataFrame(predictions)[["Gene A", "Gene B", "Predicted Score"]]
        html_report = full_df.to_html(index=False)
        st.download_button("📄 Export HTML Report", html_report, "report.html", "text/html")
