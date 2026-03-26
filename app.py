"""
AlphaFold Explorer — Protein Structure Viewer
Streamlit app for looking up protein structures from the AlphaFold DB
and folding novel sequences via ESMFold (Meta).

Usage:
    pip install streamlit requests plotly biopython py3Dmol stmol --break-system-packages
    streamlit run app.py
"""

import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from io import StringIO
import re

# ── Config ──────────────────────────────────────────────────────────────
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api"
ESMFOLD_API = "https://api.esmatlas.com/foldSequence/v1/pdb/"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"

st.set_page_config(
    page_title="AlphaFold Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ─────────────────────────────────────────────────────────────

def clean_sequence(seq: str) -> str:
    """Strip whitespace, numbers, FASTA headers. Return uppercase amino acids only."""
    lines = seq.strip().split("\n")
    cleaned = []
    for line in lines:
        if line.startswith(">"):
            continue
        cleaned.append(re.sub(r"[^A-Za-z]", "", line))
    return "".join(cleaned).upper()


def validate_sequence(seq: str) -> tuple[bool, str]:
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not seq:
        return False, "Sequence is empty."
    if len(seq) < 10:
        return False, "Sequence too short (min 10 residues)."
    if len(seq) > 400:
        return False, f"Sequence too long for ESMFold ({len(seq)} residues, max 400). Use AlphaFold DB lookup for longer proteins."
    invalid = set(seq) - valid_aa
    if invalid:
        return False, f"Invalid amino acid characters: {invalid}"
    return True, "OK"


def fetch_alphafold_prediction(uniprot_id: str) -> dict | None:
    """Fetch AlphaFold prediction metadata from EBI."""
    url = f"{ALPHAFOLD_API}/prediction/{uniprot_id}"
    resp = requests.get(url, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        return data[0] if isinstance(data, list) else data
    return None


def fetch_alphafold_pdb(prediction: dict) -> str | None:
    """Fetch PDB file from AlphaFold DB using URL from prediction metadata."""
    url = prediction.get("pdbUrl")
    if not url:
        return None
    resp = requests.get(url, timeout=15)
    if resp.status_code == 200:
        return resp.text
    return None


def fetch_alphafold_pae(prediction: dict) -> dict | None:
    """Fetch PAE (Predicted Aligned Error) JSON using URL from prediction metadata."""
    url = prediction.get("paeDocUrl")
    if not url:
        return None
    resp = requests.get(url, timeout=15)
    if resp.status_code == 200:
        return resp.json()
    return None


def fold_with_esmfold(sequence: str) -> str | None:
    """Fold a sequence using ESMFold API. Returns PDB string."""
    resp = requests.post(
        ESMFOLD_API,
        data=sequence,
        headers={"Content-Type": "text/plain"},
        timeout=120,
    )
    if resp.status_code == 200:
        return resp.text
    return None


def search_uniprot(query: str, limit: int = 10) -> list[dict]:
    """Search UniProt for protein entries."""
    url = f"{UNIPROT_API}/search"
    params = {"query": query, "format": "json", "size": limit}
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        results = []
        for entry in data.get("results", []):
            acc = entry.get("primaryAccession", "")
            name = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "Unknown")
            organism = entry.get("organism", {}).get("scientificName", "Unknown")
            length = entry.get("sequence", {}).get("length", 0)
            results.append({
                "accession": acc,
                "name": name,
                "organism": organism,
                "length": length,
            })
        return results
    return []


def parse_plddt_from_pdb(pdb_text: str) -> list[dict]:
    """Extract per-residue pLDDT from B-factor column of PDB (AlphaFold convention)."""
    residues = {}
    for line in pdb_text.split("\n"):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            res_num = int(line[22:26].strip())
            bfactor = float(line[60:66].strip())
            chain = line[21:22].strip()
            res_name = line[17:20].strip()
            residues[res_num] = {
                "residue_num": res_num,
                "residue_name": res_name,
                "chain": chain,
                "plddt": bfactor,
            }
    return [residues[k] for k in sorted(residues.keys())]


def plddt_color(val: float) -> str:
    """AlphaFold confidence color scheme."""
    if val > 90:
        return "#0053D6"  # Very high (blue)
    elif val > 70:
        return "#65CBF3"  # Confident (light blue)
    elif val > 50:
        return "#FFDB13"  # Low (yellow)
    else:
        return "#FF7D45"  # Very low (orange)


def make_plddt_chart(residues: list[dict]) -> go.Figure:
    """Create per-residue pLDDT confidence plot."""
    nums = [r["residue_num"] for r in residues]
    scores = [r["plddt"] for r in residues]
    colors = [plddt_color(s) for s in scores]

    fig = go.Figure()

    # Background bands
    fig.add_hrect(y0=90, y1=100, fillcolor="#0053D6", opacity=0.08, line_width=0)
    fig.add_hrect(y0=70, y1=90, fillcolor="#65CBF3", opacity=0.08, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor="#FFDB13", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=50, fillcolor="#FF7D45", opacity=0.08, line_width=0)

    fig.add_trace(go.Bar(
        x=nums, y=scores,
        marker_color=colors,
        hovertemplate="Residue %{x}<br>pLDDT: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title="Per-Residue pLDDT Confidence",
        xaxis_title="Residue Number",
        yaxis_title="pLDDT Score",
        yaxis=dict(range=[0, 100]),
        height=400,
        margin=dict(t=50, b=50, l=60, r=20),
        plot_bgcolor="white",
        annotations=[
            dict(x=1.02, y=0.95, xref="paper", yref="paper", text="Very high (>90)", font=dict(color="#0053D6", size=10), showarrow=False, xanchor="left"),
            dict(x=1.02, y=0.80, xref="paper", yref="paper", text="Confident (70-90)", font=dict(color="#65CBF3", size=10), showarrow=False, xanchor="left"),
            dict(x=1.02, y=0.60, xref="paper", yref="paper", text="Low (50-70)", font=dict(color="#FFDB13", size=10), showarrow=False, xanchor="left"),
            dict(x=1.02, y=0.40, xref="paper", yref="paper", text="Very low (<50)", font=dict(color="#FF7D45", size=10), showarrow=False, xanchor="left"),
        ],
    )
    return fig


def make_pae_heatmap(pae_data: dict) -> go.Figure:
    """Create PAE (Predicted Aligned Error) heatmap."""
    if isinstance(pae_data, list):
        pae_data = pae_data[0]
    pae_matrix = np.array(pae_data.get("predicted_aligned_error", pae_data.get("pae", [])))

    fig = go.Figure(data=go.Heatmap(
        z=pae_matrix,
        colorscale="Greens_r",
        zmin=0, zmax=30,
        colorbar=dict(title="PAE (Å)"),
        hovertemplate="Residue %{x} vs %{y}<br>PAE: %{z:.1f} Å<extra></extra>",
    ))
    fig.update_layout(
        title="Predicted Aligned Error (PAE)",
        xaxis_title="Scored Residue",
        yaxis_title="Aligned Residue",
        height=500,
        width=500,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def render_3d_structure(pdb_text: str):
    """Render 3D protein structure colored by pLDDT."""
    try:
        import py3Dmol
        from stmol import showmol

        view = py3Dmol.view(width=700, height=500)
        view.addModel(pdb_text, "pdb")
        view.setStyle({"cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb", "min": 0, "max": 100}}})
        view.zoomTo()
        view.setBackgroundColor("white")
        showmol(view, height=500, width=700)
    except ImportError:
        st.warning("Install py3Dmol and stmol for 3D visualization: `pip install py3Dmol stmol`")
        st.code(pdb_text[:2000] + "\n... (truncated)", language="text")


def compute_structure_stats(residues: list[dict]) -> dict:
    """Compute summary statistics from pLDDT scores."""
    scores = [r["plddt"] for r in residues]
    return {
        "num_residues": len(scores),
        "mean_plddt": np.mean(scores),
        "median_plddt": np.median(scores),
        "min_plddt": np.min(scores),
        "max_plddt": np.max(scores),
        "pct_very_high": sum(1 for s in scores if s > 90) / len(scores) * 100,
        "pct_confident": sum(1 for s in scores if s > 70) / len(scores) * 100,
        "pct_low": sum(1 for s in scores if s <= 50) / len(scores) * 100,
    }


# ── Sidebar ─────────────────────────────────────────────────────────────
st.sidebar.title("🧬 AlphaFold Explorer")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Mode", ["🔍 Lookup (AlphaFold DB)", "🧪 Fold (ESMFold)"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About**

**Lookup mode**: Search the AlphaFold Protein Structure Database (200M+ structures) by UniProt ID or protein name.

**Fold mode**: Fold a novel amino acid sequence using Meta's ESMFold (sequences up to 400 residues).

Both modes show per-residue pLDDT confidence scores and 3D structure visualization.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Example proteins:**
- `P69905` — Human hemoglobin subunit alpha
- `P0DTC2` — SARS-CoV-2 spike protein
- `P04637` — Human p53 tumor suppressor
- `P68871` — Human hemoglobin subunit beta
- `Q9BYF1` — Human ACE2 receptor
""")

# ── Main ────────────────────────────────────────────────────────────────

if "🔍" in mode:
    # ── LOOKUP MODE ──
    st.title("AlphaFold Structure Lookup")
    st.markdown("Search the AlphaFold DB by UniProt ID or protein name.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("UniProt ID or protein name", placeholder="e.g. P69905 or hemoglobin human")
    with col2:
        search_btn = st.button("Search", use_container_width=True, type="primary")

    if search_btn and query:
        # Check if it looks like a UniProt accession
        is_accession = bool(re.match(r"^[A-Z][0-9][A-Z0-9]{3}[0-9]$", query.strip().upper()))

        if is_accession:
            uniprot_id = query.strip().upper()
            with st.spinner(f"Fetching AlphaFold prediction for {uniprot_id}..."):
                prediction = fetch_alphafold_prediction(uniprot_id)

            if prediction:
                st.success(f"Found AlphaFold prediction for **{uniprot_id}**")

                # Fetch PDB + PAE using URLs from prediction metadata
                pdb_text = fetch_alphafold_pdb(prediction)
                pae_data = fetch_alphafold_pae(prediction)

                if pdb_text:
                    residues = parse_plddt_from_pdb(pdb_text)
                    stats = compute_structure_stats(residues)

                    # Stats row
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Residues", stats["num_residues"])
                    c2.metric("Mean pLDDT", f"{stats['mean_plddt']:.1f}")
                    c3.metric("% Very High (>90)", f"{stats['pct_very_high']:.0f}%")
                    c4.metric("% Low (<50)", f"{stats['pct_low']:.0f}%")

                    # Tabs for visualization
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 pLDDT Chart", "🔬 3D Structure", "🗺️ PAE Heatmap", "📥 Download"])

                    with tab1:
                        fig = make_plddt_chart(residues)
                        st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        render_3d_structure(pdb_text)

                    with tab3:
                        if pae_data:
                            fig = make_pae_heatmap(pae_data)
                            st.plotly_chart(fig)
                        else:
                            st.info("PAE data not available for this prediction.")

                    with tab4:
                        st.download_button("Download PDB", pdb_text, f"AF-{uniprot_id}.pdb", "chemical/x-pdb")
                        if pae_data:
                            st.download_button("Download PAE JSON", json.dumps(pae_data), f"AF-{uniprot_id}-pae.json", "application/json")

                else:
                    st.error(f"PDB file not found for {uniprot_id}.")
            else:
                st.warning(f"No AlphaFold prediction found for {uniprot_id}. Try searching by name instead.")

        else:
            # Search by name
            with st.spinner(f"Searching UniProt for '{query}'..."):
                results = search_uniprot(query)

            if results:
                st.markdown(f"**{len(results)} results found.** Click a UniProt ID to load the structure.")

                for r in results:
                    col_a, col_b, col_c, col_d = st.columns([1, 3, 2, 1])
                    col_a.code(r["accession"])
                    col_b.write(r["name"])
                    col_c.write(f"*{r['organism']}*")
                    col_d.write(f"{r['length']} aa")

                st.info("Copy a UniProt accession from above and search again to load the full structure.")
            else:
                st.warning(f"No results found for '{query}'.")

else:
    # ── FOLD MODE ──
    st.title("ESMFold — Fold a Sequence")
    st.markdown("Fold a novel amino acid sequence using Meta's ESMFold. Max 400 residues.")

    sequence_input = st.text_area(
        "Amino acid sequence (FASTA or raw)",
        height=150,
        placeholder=">my_protein\nMKTAYIAKQRQISFVKSHFSRQDLDALK...",
    )

    fold_btn = st.button("Fold Sequence", type="primary", use_container_width=True)

    if fold_btn and sequence_input:
        sequence = clean_sequence(sequence_input)
        valid, msg = validate_sequence(sequence)

        if not valid:
            st.error(msg)
        else:
            st.info(f"Folding {len(sequence)} residues with ESMFold...")

            with st.spinner("This may take 30-60 seconds for longer sequences..."):
                pdb_text = fold_with_esmfold(sequence)

            if pdb_text:
                st.success("Folding complete!")

                residues = parse_plddt_from_pdb(pdb_text)
                stats = compute_structure_stats(residues)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Residues", stats["num_residues"])
                c2.metric("Mean pLDDT", f"{stats['mean_plddt']:.1f}")
                c3.metric("% Very High (>90)", f"{stats['pct_very_high']:.0f}%")
                c4.metric("% Low (<50)", f"{stats['pct_low']:.0f}%")

                tab1, tab2, tab3 = st.tabs(["📊 pLDDT Chart", "🔬 3D Structure", "📥 Download"])

                with tab1:
                    fig = make_plddt_chart(residues)
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    render_3d_structure(pdb_text)

                with tab3:
                    st.download_button("Download PDB", pdb_text, "esmfold_prediction.pdb", "chemical/x-pdb")

            else:
                st.error("ESMFold failed. The server may be busy — try a shorter sequence or try again in a few minutes.")

# Footer
st.markdown("---")
st.caption("Data: [AlphaFold DB](https://alphafold.ebi.ac.uk/) (DeepMind/EBI) | [ESMFold](https://esmatlas.com/) (Meta AI) | [UniProt](https://www.uniprot.org/)")
