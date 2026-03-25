# pages/8_Data_STRING.py — STRING Interactions Browser
import streamlit as st
import pandas as pd
import sqlite3
from src.data.bulk_datasets import BulkDatasetManager

st.title("📋 STRING Interactions")
st.markdown(
    "Browse protein-protein interaction data from the STRING database — "
    "showing **which proteins work together**, with confidence scores from multiple evidence types."
)

manager = BulkDatasetManager()
if not manager.is_downloaded("string"):
    st.warning("STRING data hasn't been downloaded yet.")
    st.markdown("Go to **Download Datasets** to get it (~133 MB, takes a few minutes).")
    st.stop()

status = manager.get_status()
st.caption(f"{status['string']['row_count']:,} interactions · Downloaded {status['string']['downloaded_at'][:10]}")

# Filters
with st.sidebar:
    st.subheader("Filter Results")
    protein_filter = st.text_input(
        "Gene or protein ID",
        placeholder="e.g., TP53 or ENSP00000269305",
        help="Enter a gene symbol (like TP53) or an Ensembl protein ID. "
             "Gene symbols are automatically resolved if the alias index is built.",
    )
    min_combined = st.slider(
        "Min confidence score",
        0, 1000, 700, 50,
        help="STRING combined score (0–1000). "
             "400 = medium, 700 = high confidence, 900 = highest confidence.",
    )
    evidence_type = st.multiselect(
        "Require evidence from",
        ["experimental", "database_score", "textmining", "coexpression", "neighborhood", "fusion", "cooccurence"],
        help="Only show interactions that have evidence from these specific channels.",
    )
    page_size = st.selectbox("Rows per page", [25, 50, 100, 500], index=1)

# Resolve gene name to protein ID
resolved_protein = None
if protein_filter and manager.is_downloaded("string_aliases"):
    with sqlite3.connect(manager.db_path) as conn:
        row = conn.execute(
            "SELECT protein_id FROM string_aliases WHERE gene_symbol = ? LIMIT 1",
            (protein_filter.upper(),)
        ).fetchone()
        if row:
            resolved_protein = row[0]
            st.sidebar.caption(f"Resolved to protein: {resolved_protein}")

# Build query
conditions = ["combined_score >= ?"]
params = [min_combined]

if protein_filter:
    if resolved_protein:
        conditions.append("(protein1 = ? OR protein2 = ?)")
        params.extend([resolved_protein, resolved_protein])
    else:
        conditions.append("(protein1 LIKE ? OR protein2 LIKE ?)")
        params.extend([f"%{protein_filter}%", f"%{protein_filter}%"])

for ev in evidence_type:
    conditions.append(f"{ev} > 0")

where = " AND ".join(conditions)

with sqlite3.connect(manager.db_path) as conn:
    total = conn.execute(f"SELECT COUNT(*) FROM string WHERE {where}", params).fetchone()[0]

st.metric("Matching interactions", f"{total:,}")

# Pagination
total_pages = max(1, (total + page_size - 1) // page_size)
page = st.number_input("Page", 1, total_pages, 1)
offset = (page - 1) * page_size

with sqlite3.connect(manager.db_path) as conn:
    df = pd.read_sql_query(
        f"SELECT protein1, protein2, neighborhood, fusion, cooccurence, "
        f"coexpression, experimental, database_score, textmining, combined_score "
        f"FROM string WHERE {where} ORDER BY combined_score DESC LIMIT ? OFFSET ?",
        conn, params=params + [page_size, offset]
    )

# Add gene symbol columns if aliases are available
if manager.is_downloaded("string_aliases") and not df.empty:
    with sqlite3.connect(manager.db_path) as conn:
        all_proteins = set(df["protein1"].tolist() + df["protein2"].tolist())
        placeholders = ",".join(["?"] * len(all_proteins))
        alias_rows = conn.execute(
            f"SELECT protein_id, gene_symbol FROM string_aliases WHERE protein_id IN ({placeholders})",
            list(all_proteins)
        ).fetchall()
        alias_map = {r[0]: r[1] for r in alias_rows}

    df.insert(1, "gene1", df["protein1"].map(alias_map).fillna(""))
    df.insert(3, "gene2", df["protein2"].map(alias_map).fillna(""))

st.dataframe(df, use_container_width=True, height=600)
st.caption(f"Page {page} of {total_pages} · Showing {len(df)} of {total:,} results")

csv = df.to_csv(index=False)
st.download_button("📥 Export this page as CSV", csv, "string_data.csv", "text/csv")
