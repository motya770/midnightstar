# pages/2_Explorer.py
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import os
from src.data.gene_resolver import GeneResolver
from src.data.string_client import STRINGClient
from src.utils.graph_builder import GraphBuilder
from src.utils.text_explainer import TextExplainer
from src.utils import session

st.title("🕸️ Network Explorer")

graph = session.get_graph()

if graph is None:
    st.warning("No graph data loaded. Go to the **Search** page first to download data.")
    if st.button("Go to Search"):
        st.switch_page("pages/1_Search.py")
    st.stop()

# Controls
with st.sidebar:
    st.subheader("Graph Controls")
    min_score = st.slider("Minimum confidence score", 0.0, 1.0, 0.4, 0.05)

    available_sources = set()
    for _, _, data in graph.edges(data=True):
        for s in data.get("sources", [data.get("data_source", "unknown")]):
            available_sources.add(s)

    selected_sources = st.multiselect(
        "Data sources",
        sorted(available_sources),
        default=sorted(available_sources),
    )

    layout_options = {
        "Force-directed": "forceAtlas2Based",
        "Hierarchical": "hierarchicalRepulsion",
        "Repulsion": "repulsion",
    }
    layout_name = st.selectbox("Layout", list(layout_options.keys()))

    depth = st.slider("Expansion depth (hops)", 1, 3, 1)

# Filter graph
filtered_edges = [
    (u, v, d) for u, v, d in graph.edges(data=True)
    if d.get("score", 0) >= min_score
    and any(s in selected_sources for s in d.get("sources", [d.get("data_source", "")]))
]
filtered_nodes = set()
for u, v, _ in filtered_edges:
    filtered_nodes.add(u)
    filtered_nodes.add(v)

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
        top_tissue = max(node_data["expression"], key=node_data["expression"].get)
        title += f"\nTop tissue: {top_tissue}"
    net.add_node(node_id, label=label, color=color, title=title, size=20)

for u, v, d in filtered_edges:
    width = max(1, d.get("score", 0.5) * 5)
    title = f"Score: {d.get('score', 'N/A')}\nSource: {d.get('data_source', 'unknown')}"
    net.add_edge(u, v, width=width, title=title)

# Render
col_graph, col_details = st.columns([7, 3])

with col_graph:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        f.seek(0)
        html_content = open(f.name).read()
        components.html(html_content, height=620, scrolling=True)
    os.unlink(f.name)

with col_details:
    st.subheader("Node Details")
    node_list = sorted(filtered_nodes)
    selected = st.selectbox("Select a node", node_list)
    if selected:
        node_data = graph.nodes.get(selected, {})
        st.markdown(f"**{selected}**")
        st.markdown(f"Type: {node_data.get('node_type', 'unknown')}")

        if "description" in node_data and node_data["description"]:
            st.markdown(f"_{node_data['description']}_")

        if "expression" in node_data:
            st.markdown("**Top tissues:**")
            sorted_tissues = sorted(
                node_data["expression"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            for tissue, level in sorted_tissues:
                st.markdown(f"- {tissue.replace('_', ' ')}: {level:.1f}")

        neighbors = list(graph.neighbors(selected))
        st.markdown(f"**Connections:** {len(neighbors)}")

        if st.button("⚙️ Train model on this subgraph"):
            session.set_selected_node(selected)
            st.switch_page("pages/3_Model_Training.py")

st.caption(f"Showing {len(filtered_nodes)} nodes, {len(filtered_edges)} edges")
