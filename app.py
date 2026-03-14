import streamlit as st

st.set_page_config(
    page_title="MidnightStar",
    page_icon="🧬",
    layout="wide",
)

with st.sidebar:
    st.title("🧬 MidnightStar")
    st.caption("Gene-disease correlation discovery")
    st.divider()

    st.subheader("Data Manager")
    from src.data.cache import Cache
    import os

    cache_path = os.path.join(os.path.dirname(__file__), ".cache", "midnightstar.db")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = Cache(cache_path)

    size_mb = cache.size_bytes() / (1024 * 1024)
    st.metric("Cache Size", f"{size_mb:.1f} MB")
    if st.button("Clear Cache"):
        cache.clear_all()
        st.rerun()

    st.divider()
    st.caption("v0.1.0 — Local-first compute")

st.title("Welcome to MidnightStar")
st.markdown("""
Discover gene-disease correlations using deep learning.

**Get started:**
1. **Search** — Look up a gene or disease
2. **Explorer** — Navigate gene interaction networks
3. **Model Training** — Train ML models on your data
4. **Results** — View discovered patterns

Use the sidebar to navigate between pages.
""")
