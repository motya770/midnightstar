# pages/2_Explorer.py
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import os
from src.data.bulk_datasets import BulkDatasetManager
from src.utils import session

st.title("🕸️ Network Explorer")

manager = BulkDatasetManager()
has_any_data = any(manager.is_downloaded(s) for s in ["gwas", "gtex", "hpa", "string"])

# Check if string aliases exist — if not but string is downloaded, build them automatically
if manager.is_downloaded("string") and not manager.is_downloaded("string_aliases"):
    with st.spinner("Building STRING gene alias index (one-time, ~25 MB download)..."):
        try:
            manager.build_string_alias_table()
        except Exception as e:
            st.warning(f"Could not build alias table: {e}. Gene name resolution may be limited.")

# Data source: bulk datasets or session graph
graph = session.get_graph()

# Sidebar: gene search + controls
with st.sidebar:
    st.subheader("Gene Search")

    if has_any_data:
        gene_query = st.text_input("Enter gene symbol", placeholder="e.g., TP53, BRCA1, SP4")
        depth = st.slider("Network depth (hops)", 1, 3, 1)
        min_score_int = st.slider("Min STRING score", 0, 1000, 400, 50,
                                  help="STRING combined score (0-1000). 400 = medium, 700 = high, 900 = highest confidence.")

        if gene_query:
            with st.spinner(f"Building network for {gene_query.upper()}..."):
                graph = manager.build_graph(gene_query.upper(), depth=depth, min_score=min_score_int)
                session.set_graph(graph)
    else:
        st.warning("No datasets found. Download them first.")
        if st.button("Go to Download"):
            st.switch_page("pages/0_Download.py")

    st.divider()
    st.subheader("Display Controls")
    min_display_score = st.slider("Min display score", 0.0, 1.0, 0.0, 0.05)

    layout_options = {
        "Force-directed": "forceAtlas2Based",
        "Hierarchical": "hierarchicalRepulsion",
        "Repulsion": "repulsion",
    }
    layout_name = st.selectbox("Layout", list(layout_options.keys()))

if graph is None or graph.number_of_nodes() == 0:
    if has_any_data:
        st.info("Enter a gene symbol in the sidebar to explore its interaction network.")
    else:
        st.warning("No datasets found in `.datasets/datasets.db`. Go to **Download** to get them.")
        if st.button("📥 Go to Download"):
            st.switch_page("pages/0_Download.py")
    st.stop()

# Collect available sources from edges
available_sources = set()
for _, _, data in graph.edges(data=True):
    for s in data.get("sources", [data.get("data_source", "unknown")]):
        available_sources.add(s)

with st.sidebar:
    selected_sources = st.multiselect(
        "Filter by source",
        sorted(available_sources),
        default=sorted(available_sources),
    )

# Filter graph
filtered_edges = [
    (u, v, d) for u, v, d in graph.edges(data=True)
    if d.get("score", 0) >= min_display_score
    and any(s in selected_sources for s in d.get("sources", [d.get("data_source", "")]))
]
filtered_nodes = set()
for u, v, _ in filtered_edges:
    filtered_nodes.add(u)
    filtered_nodes.add(v)

# Also include nodes with no edges (isolated gene)
for node in graph.nodes():
    if graph.nodes[node].get("node_type") == "gene" and graph.degree(node) == 0:
        filtered_nodes.add(node)

if not filtered_nodes:
    st.info("No nodes match current filters. Try lowering the minimum score.")
    st.stop()

# Build Pyvis network
net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white")
solver = layout_options[layout_name]
if solver == "hierarchicalRepulsion":
    net.set_options('{"layout": {"hierarchical": {"enabled": true}}}')
else:
    net.barnes_hut(gravity=-3000)

color_map = {"gene": "#2a5a8c", "disease": "#8c2a5c", "default": "#5c5c2a"}

for node_id in filtered_nodes:
    node_data = graph.nodes.get(node_id, {})
    node_type = node_data.get("node_type", "default")
    color = color_map.get(node_type, color_map["default"])
    label = node_data.get("symbol", node_data.get("name", node_id))
    title = f"{label}\nType: {node_type}"
    if "expression" in node_data:
        expr = node_data["expression"]
        if expr:
            top_tissue = max(expr, key=expr.get)
            title += f"\nTop tissue: {top_tissue} ({expr[top_tissue]:.1f} TPM)"
    if node_data.get("subcellular_location"):
        title += f"\nLocation: {node_data['subcellular_location']}"
    net.add_node(node_id, label=label, color=color, title=title, size=20)

for u, v, d in filtered_edges:
    width = max(1, d.get("score", 0.5) * 5)
    title = f"Score: {d.get('score', 'N/A')}\nSource: {d.get('data_source', 'unknown')}"
    if d.get("evidence"):
        title += f"\nEvidence: {d['evidence']}"
    net.add_edge(u, v, width=width, title=title)

# Render
col_graph, col_details = st.columns([7, 3])

with col_graph:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        with open(f.name) as rf:
            html_content = rf.read()
        components.html(html_content, height=620, scrolling=True)
    os.unlink(f.name)

with col_details:
    st.subheader("Node Details")
    node_list = sorted(filtered_nodes)
    selected = st.selectbox("Select a node", node_list)
    if selected:
        node_data = graph.nodes.get(selected, {})
        node_type = node_data.get("node_type", "unknown")
        st.markdown(f"**{selected}** ({node_type})")

        if node_data.get("description"):
            st.markdown(f"_{node_data['description']}_")

        if node_data.get("subcellular_location"):
            st.markdown(f"**Location:** {node_data['subcellular_location']}")

        if node_data.get("tissue_specificity"):
            st.markdown(f"**Tissue specificity:** {node_data['tissue_specificity']}")

        if "expression" in node_data and node_data["expression"]:
            st.markdown("**Top tissues (TPM):**")
            sorted_tissues = sorted(
                node_data["expression"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            for tissue, level in sorted_tissues:
                st.markdown(f"- {tissue.replace('_', ' ')}: {level:.1f}")

        neighbors = list(graph.neighbors(selected))
        gene_neighbors = [n for n in neighbors if graph.nodes.get(n, {}).get("node_type") == "gene"]
        disease_neighbors = [n for n in neighbors if graph.nodes.get(n, {}).get("node_type") == "disease"]

        st.markdown(f"**Connections:** {len(neighbors)} ({len(gene_neighbors)} genes, {len(disease_neighbors)} diseases)")

        if gene_neighbors:
            with st.expander(f"Gene partners ({len(gene_neighbors)})"):
                for n in sorted(gene_neighbors):
                    edge_data = graph.edges.get((selected, n), {})
                    score = edge_data.get("score", "?")
                    st.markdown(f"- **{n}** (score: {score})")

        if disease_neighbors:
            with st.expander(f"Disease associations ({len(disease_neighbors)})"):
                for n in sorted(disease_neighbors):
                    disease_name = graph.nodes.get(n, {}).get("name", n)
                    st.markdown(f"- {disease_name}")

        if st.button("⚙️ Train model on this subgraph"):
            session.set_selected_node(selected)
            st.switch_page("pages/3_Model_Training.py")

st.caption(f"Showing {len(filtered_nodes)} nodes, {len(filtered_edges)} edges")
