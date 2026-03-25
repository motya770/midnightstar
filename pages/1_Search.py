# pages/1_Search.py — Gene Search
import streamlit as st
import os
from src.data.cache import Cache
from src.data.gene_resolver import GeneResolver
from src.data.gwas_client import GWASClient
from src.data.gtex_client import GTExClient
from src.data.hpa_client import HPAClient
from src.data.string_client import STRINGClient
from src.data.bulk_downloader import BulkDownloader
from src.utils.graph_builder import GraphBuilder
from src.utils.text_explainer import TextExplainer
from src.utils import session

st.title("🔍 Gene Search")
st.markdown(
    "Look up any gene or disease name to see what's known about it — "
    "disease associations, tissue expression, and protein interaction partners."
)

# Initialize clients
cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "midnightstar.db")
os.makedirs(os.path.dirname(cache_path), exist_ok=True)
cache = Cache(cache_path)
resolver = GeneResolver()
gwas = GWASClient()
gtex = GTExClient()
hpa = HPAClient()
string = STRINGClient()
downloader = BulkDownloader(gwas, gtex, hpa, string)
explainer = TextExplainer()

# ---------------------------------------------------------------------------
# Search input
# ---------------------------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Gene or disease name",
        placeholder="e.g., SP4, BRCA1, TP53, Alzheimer's",
        help="Enter a gene symbol (like TP53) or a disease name. "
             "We'll resolve it to a specific gene and pull data from all sources.",
    )
with col2:
    st.write("")  # spacing
    download_all = st.button("📥 Fetch All Data", type="primary",
                             help="Download data for this gene from GWAS, GTEx, HPA, and STRING")

if not query:
    # Empty state — show helpful guidance
    st.divider()
    st.markdown("**Not sure where to start?** Try one of these popular genes:")
    example_cols = st.columns(4)
    examples = [("TP53", "Tumor suppressor"), ("BRCA1", "Breast cancer"),
                ("APOE", "Alzheimer's risk"), ("SP4", "Transcription factor")]
    for col, (gene, desc) in zip(example_cols, examples):
        with col:
            if st.button(f"**{gene}**\n{desc}", use_container_width=True, key=f"ex_{gene}"):
                st.session_state["gene_search_query"] = gene
                st.rerun()

    with st.expander("What kind of names can I search?"):
        st.markdown("""
        You can search by:
        - **Gene symbols** — TP53, BRCA1, APOE, SP4, EGFR
        - **Gene names** — "tumor protein p53", "apolipoprotein E"
        - **Disease names** — "Alzheimer's", "breast cancer" (will find associated genes)
        - **Ensembl IDs** — ENSG00000141510

        The search uses the HGNC gene naming database to resolve your query.
        """)
    st.stop()

# ---------------------------------------------------------------------------
# Resolve gene
# ---------------------------------------------------------------------------
with st.spinner("Looking up gene..."):
    gene = resolver.resolve(query)

if gene is None:
    st.error(
        f"Could not find a gene matching **{query}**. "
        "Check the spelling or try a different name — for example, "
        "use the official gene symbol (e.g., \"TP53\" instead of \"p53\")."
    )
    st.stop()

st.success(f"Found: **{gene.symbol}** ({gene.name}) — {gene.ensembl_id}")

if not download_all:
    st.info(
        "Click **Fetch All Data** above to pull this gene's associations, "
        "expression patterns, and interaction partners from all databases."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Download data from all sources
# ---------------------------------------------------------------------------
progress = st.progress(0, text="Fetching data from all sources...")
sources_done = []

def on_progress(source, status):
    sources_done.append(source)
    progress.progress(
        len(sources_done) / 4,
        text=f"{'✅' if status == 'done' else '⚠️'} {source} ({len(sources_done)}/4)"
    )

result = downloader.download_all(gene, on_progress=on_progress)
progress.progress(1.0, text="All data fetched!")

if result["errors"]:
    st.warning(f"Some sources had issues: {', '.join(result['errors'])}")

# Cache results
for assoc in result["associations"]:
    cache.set(assoc.data_source, gene.symbol, {"type": "association"}, assoc.to_dict(), pinned=True)

# Store in session for other pages
builder = GraphBuilder()
builder.add_genes([gene])
builder.add_diseases(result["diseases"])
builder.add_associations(result["associations"])
builder.add_expression(result["expression"])
graph = builder.build()
session.set_graph(graph)
session.set_search_query(query)

# ---------------------------------------------------------------------------
# Results in tabs
# ---------------------------------------------------------------------------
tab_summary, tab_assoc, tab_expr, tab_interact = st.tabs([
    "📝 Summary", "🔗 Disease Associations", "🧫 Tissue Expression", "🤝 Protein Partners"
])

with tab_summary:
    hpa_info = result["hpa_info"]
    explanation = explainer.explain_gene(gene.symbol, gene.name, hpa_info)
    st.markdown(explanation)

with tab_assoc:
    if result["associations"]:
        import pandas as pd
        assoc_data = [
            {
                "Gene": a.source_id,
                "Associated with": a.target_id,
                "Type": a.type,
                "Confidence": f"{a.score:.2f}",
                "Evidence": a.evidence,
                "Source": a.data_source,
            }
            for a in result["associations"]
        ]
        df = pd.DataFrame(assoc_data)
        st.dataframe(df.sort_values("Confidence", ascending=False), use_container_width=True)
    else:
        st.info("No disease associations found for this gene in GWAS data.")

with tab_expr:
    if result["expression"]:
        import pandas as pd
        st.markdown(
            "Shows where this gene is most active in the body. "
            "Higher TPM (transcripts per million) = more expression."
        )
        expr_data = [
            {"Tissue": e.tissue.replace("_", " "), "Expression (TPM)": e.expression_level}
            for e in sorted(result["expression"], key=lambda e: e.expression_level, reverse=True)
        ]
        df = pd.DataFrame(expr_data)
        st.bar_chart(df.set_index("Tissue")["Expression (TPM)"])
    else:
        st.info("No expression data found for this gene in GTEx.")

with tab_interact:
    string_assocs = [a for a in result["associations"] if a.data_source == "STRING"]
    if string_assocs:
        st.markdown(
            f"**{len(string_assocs)} protein interaction partners found.** "
            "These are proteins that physically bind to or functionally cooperate with "
            f"{gene.symbol}."
        )
        for a in sorted(string_assocs, key=lambda x: x.score, reverse=True)[:10]:
            partner = a.target_id if a.source_id == gene.symbol else a.source_id
            st.markdown(f"- **{partner}** (confidence: {a.score:.2f}) — {a.evidence}")

        st.divider()
        st.markdown("**Want to see the full network?**")
        if st.button("🕸️ Open in Network Explorer"):
            st.switch_page("pages/2_Explorer.py")
    else:
        st.info("No protein interactions found for this gene in STRING.")
