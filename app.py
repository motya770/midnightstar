import streamlit as st

st.set_page_config(
    page_title="MidnightStar — Gene-Disease Discovery",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Navigation using st.navigation + st.Page for grouped sidebar sections
# ---------------------------------------------------------------------------

home = st.Page("pages/home.py", title="Home", icon="🏠", default=True)

# --- Discover section: the core exploratory tools ---
search = st.Page("pages/1_Search.py", title="Gene Search", icon="🔍")
explorer = st.Page("pages/2_Explorer.py", title="Network Explorer", icon="🕸️")
structure = st.Page("pages/9_Structure_Viewer.py", title="3D Structure", icon="🧬")

# --- Analyze section: ML-powered discovery ---
training = st.Page("pages/3_Model_Training.py", title="Train Model", icon="⚙️")
results = st.Page("pages/4_Results.py", title="Predictions", icon="📊")

# --- Data section: data management and browsing ---
download = st.Page("pages/0_Download.py", title="Download Datasets", icon="📥")
data_gwas = st.Page("pages/5_Data_GWAS.py", title="GWAS Catalog", icon="📋")
data_gtex = st.Page("pages/6_Data_GTEx.py", title="GTEx Expression", icon="📋")
data_hpa = st.Page("pages/7_Data_HPA.py", title="Human Protein Atlas", icon="📋")
data_string = st.Page("pages/8_Data_STRING.py", title="STRING Interactions", icon="📋")

nav = st.navigation({
    "": [home],
    "Discover": [search, explorer, structure],
    "Analyze": [training, results],
    "Data Management": [download, data_gwas, data_gtex, data_hpa, data_string],
})

# ---------------------------------------------------------------------------
# Sidebar — persistent across all pages
# ---------------------------------------------------------------------------
with st.sidebar:
    st.divider()

    from src.data.cache import Cache
    from src.data.bulk_datasets import BulkDatasetManager
    import os

    # Quick dataset status
    manager = BulkDatasetManager()
    status = manager.get_status()
    downloaded = sum(1 for s in ["gwas", "gtex", "hpa", "string", "alphafold"]
                     if s in status and status[s]["status"] == "complete")

    if downloaded == 5:
        st.success(f"All datasets ready ({manager.db_size_mb():.0f} MB)", icon="✅")
    elif downloaded > 0:
        st.info(f"{downloaded}/5 datasets downloaded", icon="📦")
    else:
        st.warning("No datasets yet — start with **Download Datasets**", icon="📦")

    # Cache info (collapsed)
    with st.expander("Cache & Storage"):
        cache_path = os.path.join(os.path.dirname(__file__), ".cache", "midnightstar.db")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache = Cache(cache_path)
        size_mb = cache.size_bytes() / (1024 * 1024)
        st.metric("Cache Size", f"{size_mb:.1f} MB")
        if st.button("Clear Cache"):
            cache.clear_all()
            st.rerun()

    st.divider()
    st.caption("v0.1.0 — Local-first, privacy-first")

nav.run()
