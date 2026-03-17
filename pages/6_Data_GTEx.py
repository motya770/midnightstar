# pages/6_Data_GTEx.py
import streamlit as st
import pandas as pd
import sqlite3
from src.data.bulk_datasets import BulkDatasetManager

st.title("📋 GTEx Expression Data")

manager = BulkDatasetManager()
if not manager.is_downloaded("gtex"):
    st.warning("GTEx data not downloaded. Go to **Download** first.")
    st.stop()

status = manager.get_status()
st.caption(f"{status['gtex']['row_count']:,} expression records — downloaded {status['gtex']['downloaded_at'][:10]}")

# Filters
with st.sidebar:
    st.subheader("Filters")
    gene_filter = st.text_input("Gene symbol", placeholder="e.g., TP53")
    tissue_filter = st.text_input("Tissue", placeholder="e.g., Brain")
    min_tpm = st.number_input("Min median TPM", 0.0, 10000.0, 1.0)
    page_size = st.selectbox("Rows per page", [25, 50, 100, 500], index=1)

    # Get available tissues for reference
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

st.metric("Matching rows", f"{total:,}")

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

st.dataframe(df, width="stretch", height=600)
st.caption(f"Page {page}/{total_pages} — showing {len(df)} of {total:,} results")

# Gene expression chart
if gene_filter and not df.empty:
    st.subheader(f"Expression profile: {gene_filter.upper()}")
    with sqlite3.connect(manager.db_path) as conn:
        chart_df = pd.read_sql_query(
            "SELECT tissue, median_tpm FROM gtex WHERE gene_symbol = ? AND median_tpm > 0 ORDER BY median_tpm DESC LIMIT 20",
            conn, params=[gene_filter.upper()]
        )
    if not chart_df.empty:
        st.bar_chart(chart_df.set_index("tissue")["median_tpm"])

csv = df.to_csv(index=False)
st.download_button("📥 Export current page as CSV", csv, "gtex_data.csv", "text/csv")
