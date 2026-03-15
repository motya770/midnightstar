# pages/5_Data_GWAS.py
import streamlit as st
import pandas as pd
import sqlite3
from src.data.bulk_datasets import BulkDatasetManager

st.title("📋 GWAS Catalog Data")

manager = BulkDatasetManager()
if not manager.is_downloaded("gwas"):
    st.warning("GWAS data not downloaded. Go to **Download** first.")
    st.stop()

status = manager.get_status()
st.caption(f"{status['gwas']['row_count']:,} associations — downloaded {status['gwas']['downloaded_at'][:10]}")

# Filters
with st.sidebar:
    st.subheader("Filters")
    gene_filter = st.text_input("Gene symbol", placeholder="e.g., TP53")
    disease_filter = st.text_input("Disease/Trait", placeholder="e.g., Alzheimer")
    pvalue_max = st.select_slider("Max p-value", [1e-5, 1e-8, 1e-10, 1e-15, 1e-20, 1e-50], value=1e-5)
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

# Get total count
with sqlite3.connect(manager.db_path) as conn:
    total = conn.execute(f"SELECT COUNT(*) FROM gwas WHERE {where}", params).fetchone()[0]

st.metric("Matching rows", f"{total:,}")

# Pagination
total_pages = max(1, (total + page_size - 1) // page_size)
page = st.number_input("Page", 1, total_pages, 1)
offset = (page - 1) * page_size

# Fetch data
with sqlite3.connect(manager.db_path) as conn:
    df = pd.read_sql_query(
        f"SELECT snp, mapped_gene, disease_trait, pvalue, pvalue_mlog, risk_allele_freq, reported_genes, study, pubmedid "
        f"FROM gwas WHERE {where} ORDER BY pvalue ASC LIMIT ? OFFSET ?",
        conn, params=params + [page_size, offset]
    )

st.dataframe(df, use_container_width=True, height=600)
st.caption(f"Page {page}/{total_pages} — showing {len(df)} of {total:,} results")

# Export
csv = df.to_csv(index=False)
st.download_button("📥 Export current page as CSV", csv, "gwas_data.csv", "text/csv")
