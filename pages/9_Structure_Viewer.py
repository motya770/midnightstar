# pages/9_Structure_Viewer.py
import streamlit as st
import streamlit.components.v1 as components
import sqlite3
from src.data.bulk_datasets import BulkDatasetManager
from src.data.alphafold_client import AlphaFoldClient

st.title("🧬 Protein Structure Viewer")
st.markdown("View AlphaFold predicted 3D structures colored by confidence (pLDDT).")

manager = BulkDatasetManager()
client = AlphaFoldClient()

# Gene search
gene = st.text_input("Gene symbol", placeholder="e.g., TP53, BRCA1, SP4")

if not gene:
    st.info("Enter a gene symbol to view its predicted protein structure.")
    st.stop()

gene = gene.upper()

# Look up AlphaFold data
af_data = None
if manager.is_downloaded("alphafold"):
    with sqlite3.connect(manager.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM alphafold WHERE gene = ?", (gene,)).fetchone()
        if row:
            af_data = dict(row)

# If not in local DB, try API directly
uniprot_id = None
if af_data:
    uniprot_id = af_data["uniprot_id"]
    pdb_url = af_data["pdb_url"]
else:
    st.warning(f"No local AlphaFold data for {gene}. Fetching from API...")
    # Resolve gene to UniProt ID
    import requests
    try:
        resp = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={"query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
                    "format": "json", "size": "1", "fields": "accession"},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            uniprot_id = results[0]["primaryAccession"]
    except Exception as e:
        st.error(f"UniProt lookup failed: {e}")

    if not uniprot_id:
        st.error(f"Could not find UniProt ID for **{gene}**. Check the gene symbol is correct.")
        st.stop()

    st.info(f"Resolved {gene} → UniProt **{uniprot_id}**")

    pdb_url = client.get_pdb_url(uniprot_id)

# Show structure info
if af_data:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean pLDDT", f"{af_data['mean_plddt']:.1f}")
    col2.metric("Sequence length", af_data["sequence_length"])
    col3.metric("Disordered", f"{af_data['disordered_fraction']:.0%}")
    col4.metric("High confidence", f"{af_data['frac_very_high']:.0%}")

    st.markdown(f"**UniProt:** {af_data['uniprot_id']}")
else:
    # Fetch features on the fly
    features = client.get_structural_features(uniprot_id)
    if features:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean pLDDT", f"{features['mean_plddt']:.1f}")
        col2.metric("Sequence length", features["sequence_length"])
        col3.metric("Disordered", f"{features['disordered_fraction']:.0%}")
        col4.metric("High confidence", f"{features['frac_very_high']:.0%}")

    st.markdown(f"**UniProt:** {uniprot_id}")

# pLDDT legend
st.markdown("""
**pLDDT color guide:**
🔵 Very high (>90) · 🟢 Confident (70-90) · 🟡 Low (50-70) · 🟠 Very low (<50)
""")

# 3D viewer using 3Dmol.js
st.subheader("3D Structure")

viewer_html = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #viewer {{ width: 100%; height: 550px; position: relative; }}
    </style>
</head>
<body>
    <div id="viewer"></div>
    <script>
        let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "0x0e1117"}});

        // Load PDB from AlphaFold
        jQuery.ajax("{pdb_url}", {{
            success: function(data) {{
                viewer.addModel(data, "pdb");

                // Color by pLDDT (stored as B-factor)
                // Very high (>90): blue, Confident (70-90): cyan, Low (50-70): yellow, Very low (<50): orange
                viewer.setStyle({{}}, {{
                    cartoon: {{
                        colorfunc: function(atom) {{
                            var b = atom.b;
                            if (b > 90) return '0x0077ff';       // blue - very high
                            else if (b > 70) return '0x00ccaa';  // cyan - confident
                            else if (b > 50) return '0xffdd00';  // yellow - low
                            else return '0xff7700';               // orange - very low
                        }}
                    }}
                }});

                viewer.zoomTo();
                viewer.render();
                viewer.spin("y", 0.5);
            }},
            error: function() {{
                document.getElementById("viewer").innerHTML =
                    '<p style="color:white;text-align:center;padding-top:200px;">Could not load structure</p>';
            }}
        }});
    </script>
</body>
</html>
"""

components.html(viewer_html, height=570, scrolling=False)

# Controls
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"[📥 Download PDB]({pdb_url})")
with col_b:
    cif_url = client.get_cif_url(uniprot_id) if uniprot_id else ""
    if cif_url:
        st.markdown(f"[📥 Download CIF]({cif_url})")
with col_c:
    if uniprot_id:
        st.markdown(f"[🔗 View on AlphaFold DB](https://alphafold.ebi.ac.uk/entry/{uniprot_id})")

# Per-residue pLDDT chart
if uniprot_id:
    with st.expander("Per-residue pLDDT scores"):
        plddt = client.get_plddt_scores(uniprot_id)
        if plddt and "confidenceScore" in plddt:
            import pandas as pd
            df = pd.DataFrame({
                "Residue": plddt["residueNumber"],
                "pLDDT": plddt["confidenceScore"],
            })
            st.line_chart(df.set_index("Residue")["pLDDT"])
        else:
            st.info("Could not fetch per-residue scores.")
