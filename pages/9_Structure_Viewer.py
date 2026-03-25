# pages/9_Structure_Viewer.py — 3D Protein Structure
import streamlit as st
import streamlit.components.v1 as components
import sqlite3
from src.data.bulk_datasets import BulkDatasetManager
from src.data.alphafold_client import AlphaFoldClient

st.title("🧬 3D Protein Structure")
st.markdown(
    "View the predicted 3D shape of any protein, colored by how confident AlphaFold is "
    "in each part of the structure. Blue = high confidence, orange = uncertain."
)

manager = BulkDatasetManager()
client = AlphaFoldClient()

# Gene search
gene = st.text_input(
    "Gene symbol",
    placeholder="e.g., TP53, BRCA1, SP4",
    help="Enter a gene name to view its predicted protein structure from AlphaFold.",
)

if not gene:
    st.info("Enter a gene symbol above to view its predicted 3D protein structure.")
    with st.expander("What is this?"):
        st.markdown("""
        **AlphaFold** is an AI system that predicts the 3D shape of proteins from their amino acid sequence.

        The **pLDDT score** (predicted Local Distance Difference Test) tells you how confident
        AlphaFold is about each part of the structure:
        - **Blue (>90):** Very high confidence — the structure is well-defined
        - **Cyan (70–90):** Confident — reliable backbone prediction
        - **Yellow (50–70):** Low confidence — may be flexible or disordered
        - **Orange (<50):** Very low confidence — likely a disordered/flexible region

        Disordered regions are biologically important — they're often involved in protein-protein
        interactions and signaling.
        """)
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
    st.warning(f"No local data for {gene}. Looking it up via API...")
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
        st.error(f"Could not look up this gene: {e}")

    if not uniprot_id:
        st.error(
            f"Could not find a protein for **{gene}**. "
            "Check that the gene symbol is correct (e.g., TP53, not p53)."
        )
        st.stop()

    st.info(f"Found: {gene} → UniProt **{uniprot_id}**")
    pdb_url = client.get_pdb_url(uniprot_id)

# ---------------------------------------------------------------------------
# Structure quality metrics
# ---------------------------------------------------------------------------
if af_data:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confidence (pLDDT)", f"{af_data['mean_plddt']:.1f}",
                help="Average predicted confidence (0–100). Higher = better.")
    col2.metric("Protein length", f"{af_data['sequence_length']} residues")
    col3.metric("Disordered", f"{af_data['disordered_fraction']:.0%}",
                help="Fraction of the protein that is predicted to be disordered (pLDDT < 50)")
    col4.metric("High confidence", f"{af_data['frac_very_high']:.0%}",
                help="Fraction with pLDDT > 90 (very reliable structure)")
    st.markdown(f"**UniProt ID:** {af_data['uniprot_id']}")
else:
    features = client.get_structural_features(uniprot_id)
    if features:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Confidence (pLDDT)", f"{features['mean_plddt']:.1f}")
        col2.metric("Protein length", f"{features['sequence_length']} residues")
        col3.metric("Disordered", f"{features['disordered_fraction']:.0%}")
        col4.metric("High confidence", f"{features['frac_very_high']:.0%}")
    st.markdown(f"**UniProt ID:** {uniprot_id}")

# Color legend
st.markdown(
    "**Structure coloring:**  "
    "🔵 Very high confidence (>90) · 🟢 Confident (70–90) · 🟡 Low (50–70) · 🟠 Very low (<50)"
)

# ---------------------------------------------------------------------------
# 3D viewer
# ---------------------------------------------------------------------------
st.subheader("Interactive 3D View")
st.caption("Drag to rotate, scroll to zoom, right-click to move")

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

        jQuery.ajax("{pdb_url}", {{
            success: function(data) {{
                viewer.addModel(data, "pdb");

                viewer.setStyle({{}}, {{
                    cartoon: {{
                        colorfunc: function(atom) {{
                            var b = atom.b;
                            if (b > 90) return '0x0077ff';
                            else if (b > 70) return '0x00ccaa';
                            else if (b > 50) return '0xffdd00';
                            else return '0xff7700';
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

# Download links
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"[📥 Download PDB file]({pdb_url})")
with col_b:
    cif_url = client.get_cif_url(uniprot_id) if uniprot_id else ""
    if cif_url:
        st.markdown(f"[📥 Download CIF file]({cif_url})")
with col_c:
    if uniprot_id:
        st.markdown(f"[🔗 View on AlphaFold DB](https://alphafold.ebi.ac.uk/entry/{uniprot_id})")

# Per-residue pLDDT chart
if uniprot_id:
    with st.expander("Per-residue confidence scores"):
        st.caption("Shows the confidence score for each amino acid position along the protein chain")
        plddt = client.get_plddt_scores(uniprot_id)
        if plddt and "confidenceScore" in plddt:
            import pandas as pd
            df = pd.DataFrame({
                "Position": plddt["residueNumber"],
                "Confidence (pLDDT)": plddt["confidenceScore"],
            })
            st.line_chart(df.set_index("Position")["Confidence (pLDDT)"])
        else:
            st.info("Could not fetch per-residue scores.")
