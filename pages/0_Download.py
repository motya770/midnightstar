# pages/0_Download.py — Download Datasets
import streamlit as st
from src.data.bulk_datasets import BulkDatasetManager

st.title("📥 Download Datasets")
st.markdown(
    "Download public genomics databases to your computer. "
    "This is a **one-time setup** — data is stored locally and never expires. "
    "No account or API key needed."
)

# ---------------------------------------------------------------------------
# What you'll get
# ---------------------------------------------------------------------------
with st.expander("What gets downloaded?", expanded=False):
    st.markdown("""
    | Database | What it contains | Size |
    |----------|-----------------|------|
    | **GWAS Catalog** | Gene-disease associations from published studies | ~58 MB |
    | **GTEx v8** | Gene expression across 54 human tissues | ~7 MB |
    | **Human Protein Atlas** | Protein location and expression | ~6 MB |
    | **STRING v12** | Protein-protein interaction scores | ~133 MB |
    | **AlphaFold DB** | 3D structure confidence scores | API (~20K queries) |

    Everything is stored in a local SQLite database. Nothing is sent to external servers.
    """)

manager = BulkDatasetManager()
status = manager.get_status()

# ---------------------------------------------------------------------------
# Dataset status cards
# ---------------------------------------------------------------------------
st.subheader("Dataset Status")
col1, col2, col3, col4, col5 = st.columns(5)

sources = [
    ("gwas", "GWAS Catalog", col1),
    ("gtex", "GTEx v8", col2),
    ("hpa", "Human Protein Atlas", col3),
    ("string", "STRING v12", col4),
    ("alphafold", "AlphaFold DB", col5),
]

all_downloaded = True
for source, label, col in sources:
    with col:
        if source in status and status[source]["status"] == "complete":
            rows = status[source]["row_count"]
            date = status[source]["downloaded_at"][:10]
            st.success(f"**{label}**")
            st.metric("Rows", f"{rows:,}")
            st.caption(f"Downloaded {date}")
        else:
            all_downloaded = False
            st.warning(f"**{label}**")
            st.caption("Not yet downloaded")

db_size = manager.db_size_mb()
if db_size > 0:
    st.metric("Total database size", f"{db_size:.1f} MB")

if all_downloaded:
    st.success("All datasets are downloaded and ready to use!")

st.divider()

# ---------------------------------------------------------------------------
# Download controls
# ---------------------------------------------------------------------------
st.subheader("Download")

tab_all, tab_individual = st.tabs(["Download All (recommended)", "Individual Sources"])

with tab_all:
    st.markdown(
        "Downloads all 5 datasets in sequence. "
        "The core datasets take **~5–10 minutes**; AlphaFold takes longer due to API rate limits (~20K queries)."
    )

    if st.button("🚀 Download All Datasets", type="primary", use_container_width=True):
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
                    st.info(f"**{label}** already downloaded — skipping.")
                step += 1
                progress_bar.progress(step / total_steps)
                continue

            try:
                with log:
                    st.write(f"Downloading **{label}**...")
                count = method(on_progress=on_progress)
                with log:
                    st.success(f"**{label}**: {count:,} rows downloaded")
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
    st.markdown("Download or re-download individual datasets.")
    for source_key, label, method in [
        ("gwas", "GWAS Catalog", manager.download_gwas),
        ("gtex", "GTEx v8", manager.download_gtex),
        ("hpa", "Human Protein Atlas", manager.download_hpa),
        ("string", "STRING v12", manager.download_string),
        ("string_aliases", "STRING Gene Name Index", manager.build_string_alias_table),
        ("alphafold", "AlphaFold DB", manager.download_alphafold),
    ]:
        already = manager.is_downloaded(source_key)
        btn_label = f"Re-download {label}" if already else f"Download {label}"
        if st.button(btn_label, key=f"dl_{source_key}"):
            with st.spinner(f"Downloading {label}..."):
                progress_text = st.empty()
                try:
                    count = method(on_progress=lambda msg: progress_text.text(msg))
                    st.success(f"**{label}**: {count:,} rows downloaded")
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.rerun()

# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Quick Test")
st.markdown("Verify the data is working by looking up a gene.")

test_gene = st.text_input("Test a gene lookup", placeholder="e.g., TP53, BRCA1, SP4")
if test_gene:
    result = manager.query_gene(test_gene.upper())
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("GWAS associations", len(result["gwas"]))
        st.metric("GTEx tissues", len(result["gtex"]))
    with col_b:
        st.metric("STRING interactions", len(result["string"]))
        st.metric("HPA protein info", "Yes" if result["hpa"] else "No")
    with col_c:
        af = result.get("alphafold")
        if af:
            st.metric("AlphaFold confidence", f"{af['mean_plddt']:.1f}")
            st.metric("Disordered regions", f"{af['disordered_fraction']:.0%}")
        else:
            st.metric("AlphaFold", "No data")

    if result["gtex"]:
        import pandas as pd
        st.markdown(f"**Top tissues for {test_gene.upper()}:**")
        df = pd.DataFrame(result["gtex"]).sort_values("median_tpm", ascending=False).head(10)
        st.bar_chart(df.set_index("tissue")["median_tpm"])
