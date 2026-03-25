# pages/5_Data_GWAS.py — GWAS Catalog Browser
import streamlit as st
import pandas as pd
import sqlite3
from src.data.bulk_datasets import BulkDatasetManager

st.title("📋 GWAS Catalog")
st.markdown(
    "Browse the Genome-Wide Association Studies catalog — "
    "the largest collection of known gene-disease associations from published research."
)

manager = BulkDatasetManager()
if not manager.is_downloaded("gwas"):
    st.warning("GWAS data hasn't been downloaded yet.")
    st.markdown("Go to **Download Datasets** to get it (~58 MB, takes about a minute).")
    st.stop()

status = manager.get_status()
st.caption(f"{status['gwas']['row_count']:,} associations · Downloaded {status['gwas']['downloaded_at'][:10]}")

# Filters
with st.sidebar:
    st.subheader("Filter Results")
    gene_filter = st.text_input("Gene symbol", placeholder="e.g., TP53",
                                help="Filter by gene name (partial match supported)")
    disease_filter = st.text_input("Disease or trait", placeholder="e.g., Alzheimer",
                                   help="Filter by disease/trait name (partial match)")
    pvalue_max = st.select_slider(
        "Max p-value (significance)",
        [1e-5, 1e-8, 1e-10, 1e-15, 1e-20, 1e-50],
        value=1e-5,
        help="Lower p-value = stronger statistical evidence. "
             "5e-8 is the standard genome-wide significance threshold.",
    )
    page_size = st.selectbox("Rows per page", [25, 50, 100, 500], index=1)

# Build query
conditions = []
params = []

if gene_filter:
    conditions.append("(mapped_gene LIKE ? OR reported_genes LIKE ?)")
    params.extend([f"%{gene_filter}%", f"%{gene_filter}%"])

if disease_filter:
    conditions.append("disease_trait LIKE ?")
    params.append(f"%{disease_filter}%")

conditions.append("pvalue IS NOT NULL AND pvalue <= ?")
params.append(pvalue_max)

where = " AND ".join(conditions)

with sqlite3.connect(manager.db_path) as conn:
    total = conn.execute(f"SELECT COUNT(*) FROM gwas WHERE {where}", params).fetchone()[0]

st.metric("Matching associations", f"{total:,}")

# Pagination
total_pages = max(1, (total + page_size - 1) // page_size)
page = st.number_input("Page", 1, total_pages, 1)
offset = (page - 1) * page_size

with sqlite3.connect(manager.db_path) as conn:
    df = pd.read_sql_query(
        f"SELECT snp, mapped_gene, disease_trait, pvalue, pvalue_mlog, risk_allele_freq, reported_genes, study, pubmedid "
        f"FROM gwas WHERE {where} ORDER BY pvalue ASC LIMIT ? OFFSET ?",
        conn, params=params + [page_size, offset]
    )

st.dataframe(df, use_container_width=True, height=600)
st.caption(f"Page {page} of {total_pages} · Showing {len(df)} of {total:,} results")

csv = df.to_csv(index=False)
st.download_button("📥 Export this page as CSV", csv, "gwas_data.csv", "text/csv")
