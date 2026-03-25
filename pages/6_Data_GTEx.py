# pages/6_Data_GTEx.py — GTEx Expression Browser
import streamlit as st
import pandas as pd
import sqlite3
from src.data.bulk_datasets import BulkDatasetManager

st.title("📋 GTEx Expression")
st.markdown(
    "Browse gene expression data from the GTEx project — "
    "showing **where in the body** each gene is most active across 54 human tissues."
)

manager = BulkDatasetManager()
if not manager.is_downloaded("gtex"):
    st.warning("GTEx data hasn't been downloaded yet.")
    st.markdown("Go to **Download Datasets** to get it (~7 MB, takes about 30 seconds).")
    st.stop()

status = manager.get_status()
st.caption(f"{status['gtex']['row_count']:,} expression records · Downloaded {status['gtex']['downloaded_at'][:10]}")

# Filters
with st.sidebar:
    st.subheader("Filter Results")
    gene_filter = st.text_input("Gene symbol", placeholder="e.g., TP53",
                                help="Search by gene name")
    tissue_filter = st.text_input("Tissue name", placeholder="e.g., Brain",
                                  help="Search by tissue (partial match)")
    min_tpm = st.number_input("Min expression (TPM)", 0.0, 10000.0, 1.0,
                              help="TPM = Transcripts Per Million. Higher = more expression. "
                                   "1.0 filters out very low/no expression.")
    page_size = st.selectbox("Rows per page", [25, 50, 100, 500], index=1)

    with sqlite3.connect(manager.db_path) as conn:
        tissues = conn.execute("SELECT DISTINCT tissue FROM gtex ORDER BY tissue").fetchall()
    tissue_list = [t[0] for t in tissues]
    selected_tissue = st.selectbox("Or pick a tissue", ["All"] + tissue_list)

# Build query
conditions = ["median_tpm >= ?"]
params = [min_tpm]

if gene_filter:
    conditions.append("gene_symbol LIKE ?")
    params.append(f"%{gene_filter}%")

if tissue_filter:
    conditions.append("tissue LIKE ?")
    params.append(f"%{tissue_filter}%")
elif selected_tissue != "All":
    conditions.append("tissue = ?")
    params.append(selected_tissue)

where = " AND ".join(conditions)

with sqlite3.connect(manager.db_path) as conn:
    total = conn.execute(f"SELECT COUNT(*) FROM gtex WHERE {where}", params).fetchone()[0]

st.metric("Matching records", f"{total:,}")

# Pagination
total_pages = max(1, (total + page_size - 1) // page_size)
page = st.number_input("Page", 1, total_pages, 1)
offset = (page - 1) * page_size

with sqlite3.connect(manager.db_path) as conn:
    df = pd.read_sql_query(
        f"SELECT gene_symbol, ensembl_id, tissue, median_tpm "
        f"FROM gtex WHERE {where} ORDER BY median_tpm DESC LIMIT ? OFFSET ?",
        conn, params=params + [page_size, offset]
    )

st.dataframe(df, use_container_width=True, height=600)
st.caption(f"Page {page} of {total_pages} · Showing {len(df)} of {total:,} results")

# Expression profile chart
if gene_filter and not df.empty:
    st.subheader(f"Expression profile: {gene_filter.upper()}")
    st.caption("Shows which tissues express this gene most (top 20)")
    with sqlite3.connect(manager.db_path) as conn:
        chart_df = pd.read_sql_query(
            "SELECT tissue, median_tpm FROM gtex WHERE gene_symbol = ? AND median_tpm > 0 ORDER BY median_tpm DESC LIMIT 20",
            conn, params=[gene_filter.upper()]
        )
    if not chart_df.empty:
        st.bar_chart(chart_df.set_index("tissue")["median_tpm"])

csv = df.to_csv(index=False)
st.download_button("📥 Export this page as CSV", csv, "gtex_data.csv", "text/csv")
