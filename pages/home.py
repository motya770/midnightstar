# pages/home.py — Landing page for MidnightStar
import streamlit as st
import os
from src.data.bulk_datasets import BulkDatasetManager

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .hero-title { font-size: 2.8rem; font-weight: 700; margin-bottom: 0.2rem; }
    .hero-subtitle { font-size: 1.25rem; color: #aaa; margin-bottom: 2rem; }
    .step-card {
        background: #1a1f2e;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        border-left: 4px solid #7eb8ff;
    }
    .step-card h3 { margin-top: 0; color: #7eb8ff; }
    .step-number {
        display: inline-block;
        background: #7eb8ff;
        color: #0e1117;
        font-weight: 700;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        border-radius: 50%;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="hero-title">MidnightStar</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    'Find hidden gene-disease connections using AI. '
    'Search a gene, explore its network, and let machine learning predict new associations.'
    '</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# What does this app do? (plain-language explanation)
# ---------------------------------------------------------------------------
st.markdown("""
**What is this?**  MidnightStar combines public genomics databases with graph neural networks
to predict which genes may be involved in diseases — even when no direct evidence exists yet.

**Who is it for?**  Researchers, biologists, and anyone curious about gene-disease relationships.
No coding or machine-learning expertise required.
""")

st.divider()

# ---------------------------------------------------------------------------
# How it works — 4-step visual workflow
# ---------------------------------------------------------------------------
st.subheader("How it works")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="step-card">
        <p><span class="step-number">1</span> <strong>Get Data</strong></p>
        <p>Download five public datasets (GWAS, GTEx, HPA, STRING, AlphaFold) with one click. Everything is stored locally on your machine — no cloud needed.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="step-card">
        <p><span class="step-number">2</span> <strong>Search a Gene</strong></p>
        <p>Type a gene name (like TP53 or BRCA1) to see its known disease links, tissue expression, and protein interactions at a glance.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="step-card">
        <p><span class="step-number">3</span> <strong>Explore the Network</strong></p>
        <p>Visualize how genes connect to each other and to diseases in an interactive graph. Zoom, filter, and click to dig deeper.</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="step-card">
        <p><span class="step-number">4</span> <strong>Discover with AI</strong></p>
        <p>Train a model on the gene network and let it predict new gene-disease associations that haven't been found yet.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Current status + quick-start actions
# ---------------------------------------------------------------------------
st.subheader("Your workspace")

manager = BulkDatasetManager()
status = manager.get_status()
downloaded_sources = [s for s in ["gwas", "gtex", "hpa", "string", "alphafold"]
                      if s in status and status[s]["status"] == "complete"]
pending_sources = [s for s in ["gwas", "gtex", "hpa", "string", "alphafold"]
                   if s not in downloaded_sources]

col_status, col_action = st.columns([2, 1])

with col_status:
    if downloaded_sources:
        total_rows = sum(status[s]["row_count"] for s in downloaded_sources)
        st.metric("Datasets ready", f"{len(downloaded_sources)} / 5")
        st.metric("Total data rows", f"{total_rows:,}")
        st.metric("Database size", f"{manager.db_size_mb():.1f} MB")
    else:
        st.info(
            "You haven't downloaded any datasets yet. "
            "This takes about 5-10 minutes and only needs to happen once."
        )

with col_action:
    st.markdown("**Quick start**")

    if not downloaded_sources:
        if st.button("📥 Download all datasets", type="primary", use_container_width=True):
            st.switch_page("pages/0_Download.py")
    else:
        if st.button("🔍 Search a gene", type="primary", use_container_width=True):
            st.switch_page("pages/1_Search.py")
        if st.button("🕸️ Explore networks", use_container_width=True):
            st.switch_page("pages/2_Explorer.py")
        if st.button("⚙️ Train a model", use_container_width=True):
            st.switch_page("pages/3_Model_Training.py")

    if pending_sources:
        missing = ", ".join(s.upper() for s in pending_sources)
        st.caption(f"Missing: {missing}")

# ---------------------------------------------------------------------------
# Training runs summary (if any exist)
# ---------------------------------------------------------------------------
runs = manager.list_training_runs()
if runs:
    st.divider()
    st.subheader("Recent model runs")
    import pandas as pd
    df = pd.DataFrame(runs[-5:])  # show last 5
    display_cols = {
        "created_at": "Date",
        "model_type": "Model",
        "nodes": "Genes",
        "auc_roc": "AUC-ROC",
    }
    display_df = df[[c for c in display_cols if c in df.columns]].copy()
    display_df.columns = [display_cols[c] for c in display_df.columns]
    if "Date" in display_df.columns:
        display_df["Date"] = display_df["Date"].str[:16].str.replace("T", " ")
    if "AUC-ROC" in display_df.columns:
        display_df["AUC-ROC"] = display_df["AUC-ROC"].apply(lambda x: f"{x:.4f}")
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    if st.button("📊 View all predictions"):
        st.switch_page("pages/4_Results.py")

# ---------------------------------------------------------------------------
# Data sources explanation
# ---------------------------------------------------------------------------
st.divider()

with st.expander("About the data sources"):
    st.markdown("""
    MidnightStar pulls from five well-established public databases:

    | Database | What it contains | Why it matters |
    |----------|-----------------|----------------|
    | **GWAS Catalog** | Known gene-disease associations from thousands of studies | The "ground truth" for which genes are linked to which diseases |
    | **GTEx** | Gene expression levels across 54 human tissues | Shows *where* in the body each gene is active |
    | **Human Protein Atlas** | Protein location and expression data | Shows *where inside cells* a protein works |
    | **STRING** | Protein-protein interaction scores | Maps which proteins work together |
    | **AlphaFold** | 3D protein structure predictions | Adds structural context (folding confidence, disorder) |

    All data is downloaded once and stored locally in a SQLite database. Nothing is sent to external servers.
    """)

with st.expander("How does the AI prediction work?"):
    st.markdown("""
    MidnightStar uses **graph neural networks** — a type of AI that learns from connections between things.

    Here's the intuition: if Gene A is linked to a disease, and Gene B has a very similar "neighborhood"
    in the protein interaction network (similar expression patterns, similar protein partners, similar
    structural features), then Gene B might also be involved in that disease — even if nobody has tested it yet.

    The model learns these patterns from the known data and then scores every gene for its likely involvement
    in any disease in the dataset. High-scoring predictions are the most promising candidates for further research.

    **No prior ML knowledge is needed** — you can use the default settings and get meaningful results.
    """)
