# pages/7_Data_HPA.py
import streamlit as st
import pandas as pd
import sqlite3
from src.data.bulk_datasets import BulkDatasetManager

st.title("📋 Human Protein Atlas Data")

manager = BulkDatasetManager()
if not manager.is_downloaded("hpa"):
    st.warning("HPA data not downloaded. Go to **Download** first.")
    st.stop()

status = manager.get_status()
st.caption(f"{status['hpa']['row_count']:,} genes — downloaded {status['hpa']['downloaded_at'][:10]}")

# Filters
with st.sidebar:
    st.subheader("Filters")
    gene_filter = st.text_input("Gene symbol", placeholder="e.g., TP53")
    location_filter = st.text_input("Subcellular location", placeholder="e.g., Nucleus")
    tissue_filter = st.text_input("Tissue expression", placeholder="e.g., brain")
    protein_class_filter = st.text_input("Protein class", placeholder="e.g., kinase")
    page_size = st.selectbox("Rows per page", [25, 50, 100, 500], index=1)

# Build query
conditions = []
params = []

if gene_filter:
    conditions.append("gene LIKE ?")
    params.append(f"%{gene_filter}%")

if location_filter:
    conditions.append("subcellular_location LIKE ?")
    params.append(f"%{location_filter}%")

if tissue_filter:
    conditions.append("(rna_tissue_specificity LIKE ? OR tissue_expression_cluster LIKE ?)")
    params.extend([f"%{tissue_filter}%", f"%{tissue_filter}%"])

if protein_class_filter:
    conditions.append("protein_class LIKE ?")
    params.append(f"%{protein_class_filter}%")

where = " AND ".join(conditions) if conditions else "1=1"

with sqlite3.connect(manager.db_path) as conn:
    total = conn.execute(f"SELECT COUNT(*) FROM hpa WHERE {where}", params).fetchone()[0]

st.metric("Matching genes", f"{total:,}")

# Pagination
total_pages = max(1, (total + page_size - 1) // page_size)
page = st.number_input("Page", 1, total_pages, 1)
offset = (page - 1) * page_size

with sqlite3.connect(manager.db_path) as conn:
    df = pd.read_sql_query(
        f"SELECT gene, ensembl, gene_description, subcellular_location, "
        f"rna_tissue_specificity, tissue_expression_cluster, protein_class "
        f"FROM hpa WHERE {where} ORDER BY gene LIMIT ? OFFSET ?",
        conn, params=params + [page_size, offset]
    )

st.dataframe(df, width="stretch", height=600)
st.caption(f"Page {page}/{total_pages} — showing {len(df)} of {total:,} results")

csv = df.to_csv(index=False)
st.download_button("📥 Export current page as CSV", csv, "hpa_data.csv", "text/csv")
