# pages/2_Explorer.py — Network Explorer
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import os
from src.data.bulk_datasets import BulkDatasetManager
from src.utils import session

st.title("🕸️ Network Explorer")
st.markdown(
    "Visualize how genes connect to each other and to diseases through protein interactions. "
    "Click any node to see its details."
)

manager = BulkDatasetManager()
has_any_data = any(manager.is_downloaded(s) for s in ["gwas", "gtex", "hpa", "string"])

# Check if string aliases exist — if not but string is downloaded, build them automatically
if manager.is_downloaded("string") and not manager.is_downloaded("string_aliases"):
    with st.spinner("Building gene name index (one-time setup, ~25 MB)..."):
        try:
            manager.build_string_alias_table()
        except Exception as e:
            st.warning(f"Could not build alias table: {e}. Gene name resolution may be limited.")

# Data source: bulk datasets or session graph
graph = session.get_graph()

# ---------------------------------------------------------------------------
# Sidebar — search and display controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Find a Gene")

    if has_any_data:
        gene_query = st.text_input(
            "Gene symbol",
            placeholder="e.g., TP53, BRCA1, SP4",
            help="Type a gene symbol to build its interaction network from downloaded data.",
        )
        depth = st.slider(
            "Network depth",
            1, 3, 1,
            help="How many 'hops' away from the gene to include. "
                 "1 = direct partners only, 2 = partners of partners, etc.",
        )
        min_score_int = st.slider(
            "Min confidence score",
            0, 1000, 400, 50,
            help="STRING confidence score (0–1000). "
                 "400 = medium confidence, 700 = high, 900 = highest. "
                 "Higher values show fewer but more reliable connections.",
        )

        if gene_query:
            with st.spinner(f"Building network for {gene_query.upper()}..."):
                graph = manager.build_graph(gene_query.upper(), depth=depth, min_score=min_score_int)
                session.set_graph(graph)
    else:
        st.warning("No datasets downloaded yet.")
        if st.button("📥 Go to Download Datasets"):
            st.switch_page("pages/0_Download.py")

    st.divider()
    st.subheader("Display Settings")
    min_display_score = st.slider(
        "Min edge score to show",
        0.0, 1.0, 0.0, 0.05,
        help="Hide weaker connections to reduce visual clutter.",
    )

    layout_options = {
        "Force-directed (default)": "forceAtlas2Based",
        "Hierarchical (tree)": "hierarchicalRepulsion",
        "Spread out": "repulsion",
    }
    layout_name = st.selectbox("Layout style", list(layout_options.keys()))

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if graph is None or graph.number_of_nodes() == 0:
    if has_any_data:
        st.info(
            "Enter a gene symbol in the sidebar to explore its interaction network. "
            "Or, search a gene first on the **Gene Search** page — the network will carry over."
        )
    else:
        st.warning(
            "You need to download datasets before you can explore networks. "
            "This only takes about 5–10 minutes."
        )
        if st.button("📥 Download Datasets"):
            st.switch_page("pages/0_Download.py")
    st.stop()

# ---------------------------------------------------------------------------
# Source filters
# ---------------------------------------------------------------------------
available_sources = set()
for _, _, data in graph.edges(data=True):
    for s in data.get("sources", [data.get("data_source", "unknown")]):
        available_sources.add(s)

with st.sidebar:
    selected_sources = st.multiselect(
        "Show data from",
        sorted(available_sources),
        default=sorted(available_sources),
        help="Filter which data sources are shown in the network.",
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
    st.info("No nodes match the current filters. Try lowering the minimum score in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Build Pyvis network visualization
# ---------------------------------------------------------------------------
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
            title += f"\nMost active in: {top_tissue} ({expr[top_tissue]:.1f} TPM)"
    if node_data.get("subcellular_location"):
        title += f"\nCell location: {node_data['subcellular_location']}"
    net.add_node(node_id, label=label, color=color, title=title, size=20)

for u, v, d in filtered_edges:
    width = max(1, d.get("score", 0.5) * 5)
    title = f"Confidence: {d.get('score', 'N/A')}\nSource: {d.get('data_source', 'unknown')}"
    if d.get("evidence"):
        title += f"\nEvidence: {d['evidence']}"
    net.add_edge(u, v, width=width, title=title)

# ---------------------------------------------------------------------------
# Render graph + details panel
# ---------------------------------------------------------------------------
col_graph, col_details = st.columns([7, 3])

with col_graph:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        with open(f.name) as rf:
            html_content = rf.read()
        components.html(html_content, height=620, scrolling=True)
    os.unlink(f.name)

# Legend
st.markdown(
    "🔵 **Gene** · 🟣 **Disease** · "
    "Line thickness = interaction confidence. Hover over nodes and edges for details."
)

with col_details:
    st.subheader("Node Details")
    node_list = sorted(filtered_nodes)
    selected = st.selectbox("Select a node", node_list,
                            help="Pick a node to see its details, or click one in the graph.")
    if selected:
        node_data = graph.nodes.get(selected, {})
        node_type = node_data.get("node_type", "unknown")
        type_label = "Gene" if node_type == "gene" else "Disease/Trait" if node_type == "disease" else node_type
        st.markdown(f"**{selected}** ({type_label})")

        if node_data.get("description"):
            st.markdown(f"_{node_data['description']}_")

        if node_data.get("subcellular_location"):
            st.markdown(f"**Found in cell:** {node_data['subcellular_location']}")

        if node_data.get("tissue_specificity"):
            st.markdown(f"**Tissue specificity:** {node_data['tissue_specificity']}")

        if "expression" in node_data and node_data["expression"]:
            st.markdown("**Most active tissues (TPM):**")
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
                    st.markdown(f"- **{n}** (confidence: {score})")

        if disease_neighbors:
            with st.expander(f"Disease associations ({len(disease_neighbors)})"):
                for n in sorted(disease_neighbors):
                    disease_name = graph.nodes.get(n, {}).get("name", n)
                    st.markdown(f"- {disease_name}")

        st.divider()
        if st.button("⚙️ Train a model on this network"):
            session.set_selected_node(selected)
            st.switch_page("pages/3_Model_Training.py")

st.caption(f"Showing {len(filtered_nodes)} nodes, {len(filtered_edges)} edges")
