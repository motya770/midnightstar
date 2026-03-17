# pages/0_Download.py
import streamlit as st
from src.data.bulk_datasets import BulkDatasetManager

st.title("📥 Download Full Datasets")
st.markdown("""
Download complete databases from all 5 sources. Data is stored locally in SQLite — **never expires**.

| Source | What you get | Size |
|--------|-------------|------|
| **GWAS Catalog** | All gene-disease associations, SNPs, p-values | ~58 MB |
| **GTEx v8** | Median gene expression across 54 tissues (56K genes) | ~7 MB |
| **Human Protein Atlas** | Protein expression, subcellular location, tissue data | ~6 MB |
| **STRING v12** | All human protein-protein interactions with subscores | ~133 MB |
| **AlphaFold DB** | Protein structure confidence (pLDDT), disorder fraction | API (~20K queries) |
""")

manager = BulkDatasetManager()
status = manager.get_status()

# Show current status
st.subheader("Dataset Status")
col1, col2, col3, col4, col5 = st.columns(5)

sources = [
    ("gwas", "GWAS Catalog", col1),
    ("gtex", "GTEx v8", col2),
    ("hpa", "Human Protein Atlas", col3),
    ("string", "STRING v12", col4),
    ("alphafold", "AlphaFold DB", col5),
]

for source, label, col in sources:
    with col:
        if source in status and status[source]["status"] == "complete":
            rows = status[source]["row_count"]
            date = status[source]["downloaded_at"][:10]
            st.success(f"**{label}**")
            st.metric("Rows", f"{rows:,}")
            st.caption(f"Downloaded {date}")
        else:
            st.warning(f"**{label}**")
            st.caption("Not downloaded")

db_size = manager.db_size_mb()
if db_size > 0:
    st.metric("Total database size", f"{db_size:.1f} MB")

st.divider()

# Download controls
st.subheader("Download")

tab_all, tab_individual = st.tabs(["Download All", "Individual Sources"])

with tab_all:
    st.markdown("Download all 5 datasets + aliases in sequence. Core datasets take ~5-10 min, AlphaFold takes longer (~20K API queries).")

    if st.button("🚀 Download All Datasets", type="primary", width="stretch"):
        progress_text = st.empty()
        progress_bar = st.progress(0)
        log = st.container()

        step = 0
        total_steps = 6

        def on_progress(msg):
            progress_text.text(msg)

        for source_key, label, method in [
            ("gwas", "GWAS Catalog", manager.download_gwas),
            ("gtex", "GTEx", manager.download_gtex),
            ("hpa", "Human Protein Atlas", manager.download_hpa),
            ("string", "STRING", manager.download_string),
            ("string_aliases", "STRING Gene Aliases", manager.build_string_alias_table),
            ("alphafold", "AlphaFold DB", manager.download_alphafold),
        ]:
            if manager.is_downloaded(source_key):
                with log:
                    st.info(f"**{label}** already downloaded, skipping.")
                step += 1
                progress_bar.progress(step / total_steps)
                continue

            try:
                with log:
                    st.write(f"Downloading **{label}**...")
                count = method(on_progress=on_progress)
                with log:
                    st.success(f"**{label}**: {count:,} rows indexed")
            except Exception as e:
                with log:
                    st.error(f"**{label}** failed: {e}")

            step += 1
            progress_bar.progress(step / total_steps)

        progress_bar.progress(1.0)
        progress_text.text("All downloads complete!")
        st.balloons()
        st.rerun()

with tab_individual:
    for source_key, label, method in [
        ("gwas", "GWAS Catalog", manager.download_gwas),
        ("gtex", "GTEx v8", manager.download_gtex),
        ("hpa", "Human Protein Atlas", manager.download_hpa),
        ("string", "STRING v12", manager.download_string),
        ("string_aliases", "STRING Aliases", manager.build_string_alias_table),
        ("alphafold", "AlphaFold DB", manager.download_alphafold),
    ]:
        already = manager.is_downloaded(source_key)
        btn_label = f"Re-download {label}" if already else f"Download {label}"
        if st.button(btn_label, key=f"dl_{source_key}"):
            with st.spinner(f"Downloading {label}..."):
                progress_text = st.empty()
                try:
                    count = method(on_progress=lambda msg: progress_text.text(msg))
                    st.success(f"**{label}**: {count:,} rows indexed")
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.rerun()

# Quick test
st.divider()
st.subheader("Quick Test")
test_gene = st.text_input("Test a gene lookup", placeholder="e.g., TP53, BRCA1, SP4")
if test_gene:
    result = manager.query_gene(test_gene.upper())
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("GWAS associations", len(result["gwas"]))
        st.metric("GTEx tissues", len(result["gtex"]))
    with col_b:
        st.metric("STRING interactions", len(result["string"]))
        st.metric("HPA info", "Yes" if result["hpa"] else "No")
    with col_c:
        af = result.get("alphafold")
        if af:
            st.metric("AlphaFold pLDDT", f"{af['mean_plddt']:.1f}")
            st.metric("Disordered", f"{af['disordered_fraction']:.0%}")
        else:
            st.metric("AlphaFold", "No data")

    if result["gtex"]:
        import pandas as pd
        df = pd.DataFrame(result["gtex"]).sort_values("median_tpm", ascending=False).head(10)
        st.bar_chart(df.set_index("tissue")["median_tpm"])
