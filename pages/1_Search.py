# pages/1_Search.py
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

st.title("🔍 Search")
st.markdown("Search for a gene or disease to explore its connections.")

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

# Search input
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Gene or disease name", placeholder="e.g., SP4, BRCA1, Alzheimer's")
with col2:
    st.write("")  # spacing
    download_all = st.button("📥 Download All Data")

if query:
    # Resolve gene
    with st.spinner("Resolving gene..."):
        gene = resolver.resolve(query)

    if gene is None:
        st.error(f"Could not find gene or disease: **{query}**. Try a different name.")
        st.stop()

    st.success(f"Found: **{gene.symbol}** ({gene.name}) — {gene.ensembl_id}")

    if download_all:
        progress = st.progress(0, text="Downloading from all sources...")
        sources_done = []

        def on_progress(source, status):
            sources_done.append(source)
            progress.progress(
                len(sources_done) / 4,
                text=f"✅ {source} {'done' if status == 'done' else 'error'} ({len(sources_done)}/4)"
            )

        result = downloader.download_all(gene, on_progress=on_progress)
        progress.progress(1.0, text="Download complete!")

        if result["errors"]:
            st.warning(f"Some sources had errors: {', '.join(result['errors'])}")

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

        # Display results
        tab_summary, tab_assoc, tab_expr, tab_interact = st.tabs(
            ["Summary", "Associations", "Expression", "Interactions"]
        )

        with tab_summary:
            hpa_info = result["hpa_info"]
            explanation = explainer.explain_gene(gene.symbol, gene.name, hpa_info)
            st.markdown(explanation)

        with tab_assoc:
            if result["associations"]:
                import pandas as pd
                assoc_data = [
                    {
                        "Source Gene": a.source_id,
                        "Target": a.target_id,
                        "Type": a.type,
                        "Score": a.score,
                        "Evidence": a.evidence,
                        "Database": a.data_source,
                    }
                    for a in result["associations"]
                ]
                df = pd.DataFrame(assoc_data)
                st.dataframe(df.sort_values("Score", ascending=False), width="stretch")
            else:
                st.info("No associations found.")

        with tab_expr:
            if result["expression"]:
                import pandas as pd
                expr_data = [
                    {"Tissue": e.tissue.replace("_", " "), "Expression Level": e.expression_level}
                    for e in sorted(result["expression"], key=lambda e: e.expression_level, reverse=True)
                ]
                df = pd.DataFrame(expr_data)
                st.bar_chart(df.set_index("Tissue")["Expression Level"])
            else:
                st.info("No expression data found.")

        with tab_interact:
            string_assocs = [a for a in result["associations"] if a.data_source == "STRING"]
            if string_assocs:
                st.markdown(f"**{len(string_assocs)} protein interaction partners found.**")
                for a in sorted(string_assocs, key=lambda x: x.score, reverse=True)[:10]:
                    partner = a.target_id if a.source_id == gene.symbol else a.source_id
                    st.markdown(f"- **{partner}** (score: {a.score:.2f}) — {a.evidence}")
                if st.button("🕸️ Explore in Network Graph"):
                    st.switch_page("pages/2_Explorer.py")
            else:
                st.info("No protein interactions found.")
    else:
        st.info("Click **Download All Data** to fetch from GWAS, GTEx, HPA, and STRING.")
