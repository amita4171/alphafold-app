"""
AlphaFold Explorer — Protein Structure Viewer
Streamlit app for looking up protein structures from the AlphaFold DB
and folding novel sequences via ESMFold (Meta).

Phase 2 features: batch fold, compare, domain annotation, PDF export, local ESM-2.

Usage:
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

import csv
import datetime
import json
import re
from io import BytesIO, StringIO

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config ──────────────────────────────────────────────────────────────
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api"
ESMFOLD_API = "https://api.esmatlas.com/foldSequence/v1/pdb/"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"
DOMAIN_FEATURE_TYPES = {"Domain", "Region", "Motif", "Zinc finger", "DNA binding"}

st.set_page_config(
    page_title="AlphaFold Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Optional Local ESM-2 ───────────────────────────────────────────────
try:
    import torch
    import esm as esm_module
    LOCAL_ESM_AVAILABLE = True
except ImportError:
    LOCAL_ESM_AVAILABLE = False


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


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into list of (name, sequence) tuples."""
    sequences = []
    current_name = None
    current_seq_lines: list[str] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if current_name is not None:
                seq = re.sub(r"[^A-Za-z]", "", "".join(current_seq_lines)).upper()
                if seq:
                    sequences.append((current_name, seq))
            current_name = line[1:].strip().split()[0] if line[1:].strip() else "unnamed"
            current_seq_lines = []
        else:
            current_seq_lines.append(line)
    if current_name is not None:
        seq = re.sub(r"[^A-Za-z]", "", "".join(current_seq_lines)).upper()
        if seq:
            sequences.append((current_name, seq))
    return sequences


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


def fetch_uniprot_domains(uniprot_id: str) -> list[dict]:
    """Fetch domain annotations from UniProt REST API."""
    url = f"{UNIPROT_API}/{uniprot_id}.json"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        domains = []
        for feat in data.get("features", []):
            if feat.get("type") in DOMAIN_FEATURE_TYPES:
                loc = feat.get("location", {})
                start = loc.get("start", {}).get("value")
                end = loc.get("end", {}).get("value")
                desc = feat.get("description", feat.get("type", "Unknown"))
                if start is not None and end is not None:
                    domains.append({
                        "name": desc,
                        "start": int(start),
                        "end": int(end),
                        "type": feat["type"],
                    })
        return domains
    except Exception:
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


def make_plddt_chart(residues: list[dict], domains: list[dict] | None = None) -> go.Figure:
    """Create per-residue pLDDT confidence plot with optional domain overlays."""
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

    # Domain annotation overlays
    if domains:
        palette = px.colors.qualitative.Set2
        for i, dom in enumerate(domains):
            color = palette[i % len(palette)]
            fig.add_vrect(
                x0=dom["start"] - 0.5, x1=dom["end"] + 0.5,
                fillcolor=color, opacity=0.15, line_width=1,
                line_color=color,
                annotation_text=dom["name"],
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color=color,
            )

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


def generate_pdf_report(
    title: str,
    stats: dict,
    plddt_fig: go.Figure,
    uniprot_id: str | None = None,
    domains: list[dict] | None = None,
) -> bytes:
    """Generate PDF report as bytes for st.download_button."""
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image as RLImage,
        SimpleDocTemplate,
        Spacer,
        Paragraph,
        Table,
        TableStyle,
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("AlphaFold Explorer Report", styles["Title"]))
    elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
    if uniprot_id:
        elements.append(Paragraph(f"UniProt: {uniprot_id}", styles["Normal"]))
    elements.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Stats table
    table_data = [
        ["Metric", "Value"],
        ["Residues", str(stats["num_residues"])],
        ["Mean pLDDT", f"{stats['mean_plddt']:.1f}"],
        ["Median pLDDT", f"{stats['median_plddt']:.1f}"],
        ["Min pLDDT", f"{stats['min_plddt']:.1f}"],
        ["Max pLDDT", f"{stats['max_plddt']:.1f}"],
        ["% Very High (>90)", f"{stats['pct_very_high']:.1f}%"],
        ["% Confident (>70)", f"{stats['pct_confident']:.1f}%"],
        ["% Low (<=50)", f"{stats['pct_low']:.1f}%"],
    ]
    if domains:
        for dom in domains:
            table_data.append([f"Domain: {dom['name']}", f"Residues {dom['start']}-{dom['end']}"])

    t = Table(table_data, colWidths=[3 * inch, 3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#0053D6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#F0F4FF")]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3 * inch))

    # pLDDT chart as image
    try:
        img_bytes = plddt_fig.to_image(format="png", width=900, height=400, scale=2)
        img_buf = BytesIO(img_bytes)
        elements.append(Paragraph("Per-Residue pLDDT Confidence", styles["Heading3"]))
        elements.append(RLImage(img_buf, width=6.5 * inch, height=2.9 * inch))
    except Exception:
        elements.append(Paragraph(
            "<i>Chart image unavailable (install kaleido for chart export)</i>",
            styles["Normal"],
        ))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("<i>3D structure visualization available in the app.</i>", styles["Normal"]))

    # Footer
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(
        "Data: AlphaFold DB (DeepMind/EBI) | ESMFold (Meta AI) | UniProt",
        styles["Normal"],
    ))

    doc.build(elements)
    return buf.getvalue()


# ── Local ESM-2 Helpers ────────────────────────────────────────────────

def check_local_esm_capabilities() -> str:
    """Returns 'full_fold', 'analysis_only', or 'unavailable'."""
    if not LOCAL_ESM_AVAILABLE:
        return "unavailable"
    try:
        from esm.pretrained import esmfold_v1  # noqa: F401
        return "full_fold"
    except (ImportError, AttributeError):
        return "analysis_only"


def fold_local_esm(sequence: str) -> str | None:
    """Attempt local folding with ESMFold. Returns PDB string or None."""
    try:
        from esm.pretrained import esmfold_v1
        model = esmfold_v1()
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        with torch.no_grad():
            output = model.infer_pdb(sequence)
        return output
    except Exception:
        return None


def analyze_local_esm(sequence: str) -> dict | None:
    """Run ESM-2 analysis: contact map prediction. Returns dict or None."""
    try:
        model, alphabet = esm_module.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model = model.eval()

        data = [("protein", sequence)]
        _, _, batch_tokens = batch_converter(data)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=True)

        contact_map = results["contacts"][0].numpy()
        return {"contact_map": contact_map}
    except Exception:
        return None


# ── Sidebar ─────────────────────────────────────────────────────────────
st.sidebar.title("🧬 AlphaFold Explorer")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Mode",
    ["🔍 Lookup (AlphaFold DB)", "🧪 Fold (ESMFold)", "📦 Batch Fold", "⚖️ Compare"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About**

**Lookup mode**: Search the AlphaFold DB (200M+ structures) by UniProt ID. Includes domain annotations.

**Fold mode**: Fold a novel sequence using ESMFold (max 400 residues). Optional local ESM-2 analysis.

**Batch Fold**: Upload a FASTA file and fold multiple sequences. Export stats as CSV.

**Compare**: Compare two proteins side by side — pLDDT charts and 3D structures.

All modes include pLDDT confidence charts and PDF export.
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

                # Fetch PDB + PAE + domains
                pdb_text = fetch_alphafold_pdb(prediction)
                pae_data = fetch_alphafold_pae(prediction)
                domains = fetch_uniprot_domains(uniprot_id)

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
                        show_domains = st.checkbox("Show domain annotations", value=True) if domains else False
                        fig = make_plddt_chart(residues, domains=domains if show_domains else None)
                        st.plotly_chart(fig, use_container_width=True)
                        if domains and show_domains:
                            st.caption(f"Showing {len(domains)} domain/region annotations from UniProt")

                    with tab2:
                        render_3d_structure(pdb_text)

                    with tab3:
                        if pae_data:
                            pae_fig = make_pae_heatmap(pae_data)
                            st.plotly_chart(pae_fig)
                        else:
                            st.info("PAE data not available for this prediction.")

                    with tab4:
                        st.download_button("Download PDB", pdb_text, f"AF-{uniprot_id}.pdb", "chemical/x-pdb")
                        if pae_data:
                            st.download_button("Download PAE JSON", json.dumps(pae_data), f"AF-{uniprot_id}-pae.json", "application/json")
                        # PDF export
                        pdf_fig = make_plddt_chart(residues, domains=domains if domains else None)
                        pdf_bytes = generate_pdf_report(
                            title=f"AlphaFold Prediction — {uniprot_id}",
                            stats=stats,
                            plddt_fig=pdf_fig,
                            uniprot_id=uniprot_id,
                            domains=domains,
                        )
                        st.download_button("📄 Export PDF Report", pdf_bytes, f"AF-{uniprot_id}-report.pdf", "application/pdf")

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

elif "🧪" in mode:
    # ── FOLD MODE ──
    st.title("ESMFold — Fold a Sequence")
    st.markdown("Fold a novel amino acid sequence using Meta's ESMFold. Max 400 residues.")

    # Local ESM toggle
    esm_capability = check_local_esm_capabilities()
    use_local = False
    if esm_capability != "unavailable":
        use_local = st.checkbox("Use local ESM-2 model (offline)", value=False)
    else:
        st.caption("💡 Install `torch` and `fair-esm` for optional offline ESM-2 analysis")

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
            st.info(f"Folding {len(sequence)} residues...")
            analysis = None

            if use_local and esm_capability == "full_fold":
                with st.spinner("Folding locally with ESMFold..."):
                    pdb_text = fold_local_esm(sequence)
                if not pdb_text:
                    st.warning("Local folding failed, falling back to API...")
                    with st.spinner("Folding with ESMFold API..."):
                        pdb_text = fold_with_esmfold(sequence)
            else:
                with st.spinner("This may take 30-60 seconds for longer sequences..."):
                    pdb_text = fold_with_esmfold(sequence)
                if use_local and esm_capability == "analysis_only":
                    with st.spinner("Running local ESM-2 analysis..."):
                        analysis = analyze_local_esm(sequence)

            if pdb_text:
                st.success("Folding complete!")

                residues = parse_plddt_from_pdb(pdb_text)
                stats = compute_structure_stats(residues)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Residues", stats["num_residues"])
                c2.metric("Mean pLDDT", f"{stats['mean_plddt']:.1f}")
                c3.metric("% Very High (>90)", f"{stats['pct_very_high']:.0f}%")
                c4.metric("% Low (<50)", f"{stats['pct_low']:.0f}%")

                tab_names = ["📊 pLDDT Chart", "🔬 3D Structure", "📥 Download"]
                if analysis and "contact_map" in analysis:
                    tab_names.insert(2, "🧠 Contact Map (ESM-2)")

                tabs = st.tabs(tab_names)

                with tabs[0]:
                    fig = make_plddt_chart(residues)
                    st.plotly_chart(fig, use_container_width=True)

                with tabs[1]:
                    render_3d_structure(pdb_text)

                if analysis and "contact_map" in analysis:
                    with tabs[2]:
                        st.markdown("**Predicted Contact Map** (ESM-2 local model)")
                        cm = analysis["contact_map"]
                        cm_fig = go.Figure(data=go.Heatmap(
                            z=cm, colorscale="Blues",
                            hovertemplate="Residue %{x} vs %{y}<br>Contact: %{z:.3f}<extra></extra>",
                        ))
                        cm_fig.update_layout(
                            xaxis_title="Residue", yaxis_title="Residue",
                            height=500, width=500,
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(cm_fig)

                with tabs[-1]:
                    st.download_button("Download PDB", pdb_text, "esmfold_prediction.pdb", "chemical/x-pdb")
                    pdf_fig = make_plddt_chart(residues)
                    pdf_bytes = generate_pdf_report(title="ESMFold Prediction", stats=stats, plddt_fig=pdf_fig)
                    st.download_button("📄 Export PDF Report", pdf_bytes, "esmfold-report.pdf", "application/pdf")

            else:
                st.error("ESMFold failed. The server may be busy — try a shorter sequence or try again in a few minutes.")

elif "📦" in mode:
    # ── BATCH FOLD MODE ──
    st.title("Batch Fold — Multiple Sequences")
    st.markdown("Upload a FASTA file to fold multiple sequences with ESMFold.")

    uploaded = st.file_uploader("Upload FASTA file", type=["fasta", "fa", "faa", "txt"])

    if uploaded:
        fasta_text = uploaded.read().decode("utf-8")
        sequences = parse_fasta(fasta_text)

        if len(sequences) == 0:
            st.error("No valid sequences found in the uploaded file.")
        elif len(sequences) > 20:
            st.error(f"Too many sequences ({len(sequences)}). Maximum is 20 per batch.")
        else:
            # Validate all sequences
            valid_seqs = []
            skipped = []
            for name, seq in sequences:
                ok, msg = validate_sequence(seq)
                if ok:
                    valid_seqs.append((name, seq))
                else:
                    skipped.append((name, msg))

            st.info(f"**{len(valid_seqs)}** valid sequences, **{len(skipped)}** skipped")
            if skipped:
                with st.expander("Skipped sequences"):
                    for name, reason in skipped:
                        st.write(f"- **{name}**: {reason}")

            if valid_seqs and st.button("Fold All", type="primary", use_container_width=True):
                results = []
                progress = st.progress(0, text="Folding sequences...")

                for i, (name, seq) in enumerate(valid_seqs):
                    progress.progress(i / len(valid_seqs), text=f"Folding {name} ({i + 1}/{len(valid_seqs)})...")
                    pdb_text = fold_with_esmfold(seq)
                    if pdb_text:
                        residues = parse_plddt_from_pdb(pdb_text)
                        batch_stats = compute_structure_stats(residues)
                        results.append({"name": name, "sequence": seq, "pdb": pdb_text, "residues": residues, "stats": batch_stats})
                    else:
                        results.append({"name": name, "sequence": seq, "pdb": None, "residues": None, "stats": None})

                progress.progress(1.0, text="Done!")
                st.session_state["batch_results"] = results

            # Display results from session state
            if "batch_results" in st.session_state and st.session_state["batch_results"]:
                results = st.session_state["batch_results"]
                st.markdown("### Results")

                # Summary table
                table_rows = []
                for r in results:
                    if r["stats"]:
                        s = r["stats"]
                        table_rows.append({
                            "Sequence": r["name"],
                            "Length": s["num_residues"],
                            "Mean pLDDT": f"{s['mean_plddt']:.1f}",
                            "Median pLDDT": f"{s['median_plddt']:.1f}",
                            "% Very High": f"{s['pct_very_high']:.0f}%",
                            "% Confident": f"{s['pct_confident']:.0f}%",
                            "% Low": f"{s['pct_low']:.0f}%",
                        })
                    else:
                        table_rows.append({
                            "Sequence": r["name"],
                            "Length": len(r["sequence"]),
                            "Mean pLDDT": "FAILED",
                            "Median pLDDT": "-",
                            "% Very High": "-",
                            "% Confident": "-",
                            "% Low": "-",
                        })

                st.dataframe(table_rows, use_container_width=True)

                # Expandable details per sequence
                for r in results:
                    if r["stats"]:
                        with st.expander(f"{r['name']} — Mean pLDDT: {r['stats']['mean_plddt']:.1f}"):
                            batch_fig = make_plddt_chart(r["residues"])
                            st.plotly_chart(batch_fig, use_container_width=True)

                # CSV export
                csv_buf = StringIO()
                writer = csv.DictWriter(csv_buf, fieldnames=[
                    "sequence_name", "length", "mean_plddt", "median_plddt",
                    "pct_very_high", "pct_confident", "pct_low",
                ])
                writer.writeheader()
                for r in results:
                    if r["stats"]:
                        s = r["stats"]
                        writer.writerow({
                            "sequence_name": r["name"],
                            "length": s["num_residues"],
                            "mean_plddt": f"{s['mean_plddt']:.2f}",
                            "median_plddt": f"{s['median_plddt']:.2f}",
                            "pct_very_high": f"{s['pct_very_high']:.1f}",
                            "pct_confident": f"{s['pct_confident']:.1f}",
                            "pct_low": f"{s['pct_low']:.1f}",
                        })
                st.download_button("📥 Download CSV", csv_buf.getvalue(), "batch_results.csv", "text/csv")

elif "⚖️" in mode:
    # ── COMPARE MODE ──
    st.title("Compare Proteins")
    st.markdown("Compare pLDDT confidence and structure for two proteins side by side.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Protein A")
        input_type_a = st.selectbox("Input type", ["UniProt ID", "Paste Sequence"], key="type_a")
        if input_type_a == "UniProt ID":
            val_a = st.text_input("UniProt ID", key="id_a", placeholder="e.g. P69905")
        else:
            val_a = st.text_area("Sequence", key="seq_a", height=100, placeholder="MKTAYIAK...")

    with col_b:
        st.subheader("Protein B")
        input_type_b = st.selectbox("Input type", ["UniProt ID", "Paste Sequence"], key="type_b")
        if input_type_b == "UniProt ID":
            val_b = st.text_input("UniProt ID", key="id_b", placeholder="e.g. P68871")
        else:
            val_b = st.text_area("Sequence", key="seq_b", height=100, placeholder="MVHLTPE...")

    if st.button("Compare", type="primary", use_container_width=True):

        def _fetch_protein(input_type: str, value: str, label: str):
            """Returns (label, pdb_text, residues, stats) or None."""
            if input_type == "UniProt ID":
                uid = value.strip().upper()
                if not uid:
                    st.error(f"{label}: Please enter a UniProt ID")
                    return None
                pred = fetch_alphafold_prediction(uid)
                if not pred:
                    st.error(f"No AlphaFold prediction for {uid}")
                    return None
                pdb = fetch_alphafold_pdb(pred)
                if not pdb:
                    st.error(f"PDB not found for {uid}")
                    return None
                residues = parse_plddt_from_pdb(pdb)
                return uid, pdb, residues, compute_structure_stats(residues)
            else:
                seq = clean_sequence(value)
                ok, msg = validate_sequence(seq)
                if not ok:
                    st.error(f"{label}: {msg}")
                    return None
                pdb = fold_with_esmfold(seq)
                if not pdb:
                    st.error(f"ESMFold failed for {label}")
                    return None
                residues = parse_plddt_from_pdb(pdb)
                return label, pdb, residues, compute_structure_stats(residues)

        with st.spinner("Fetching/folding proteins..."):
            data_a = _fetch_protein(input_type_a, val_a, "Protein A")
            data_b = _fetch_protein(input_type_b, val_b, "Protein B")

        if data_a and data_b:
            label_a, pdb_a, res_a, stats_a = data_a
            label_b, pdb_b, res_b, stats_b = data_b

            # Overlaid pLDDT chart
            st.markdown("### pLDDT Comparison")
            cmp_fig = go.Figure()
            cmp_fig.add_hrect(y0=90, y1=100, fillcolor="#0053D6", opacity=0.08, line_width=0)
            cmp_fig.add_hrect(y0=70, y1=90, fillcolor="#65CBF3", opacity=0.08, line_width=0)
            cmp_fig.add_hrect(y0=50, y1=70, fillcolor="#FFDB13", opacity=0.08, line_width=0)
            cmp_fig.add_hrect(y0=0, y1=50, fillcolor="#FF7D45", opacity=0.08, line_width=0)
            cmp_fig.add_trace(go.Scatter(
                x=[r["residue_num"] for r in res_a],
                y=[r["plddt"] for r in res_a],
                mode="lines", name=label_a,
                line=dict(color="#0053D6", width=2),
            ))
            cmp_fig.add_trace(go.Scatter(
                x=[r["residue_num"] for r in res_b],
                y=[r["plddt"] for r in res_b],
                mode="lines", name=label_b,
                line=dict(color="#FF7D45", width=2),
            ))
            cmp_fig.update_layout(
                title="Per-Residue pLDDT Comparison",
                xaxis_title="Residue Number",
                yaxis_title="pLDDT Score",
                yaxis=dict(range=[0, 100]),
                height=400,
                plot_bgcolor="white",
            )
            st.plotly_chart(cmp_fig, use_container_width=True)

            # Stats comparison
            st.markdown("### Statistics")
            metrics = ["num_residues", "mean_plddt", "median_plddt", "min_plddt", "max_plddt", "pct_very_high", "pct_confident", "pct_low"]
            labels = ["Residues", "Mean pLDDT", "Median pLDDT", "Min pLDDT", "Max pLDDT", "% Very High (>90)", "% Confident (>70)", "% Low (<=50)"]
            comp_rows = []
            for metric, lbl in zip(metrics, labels):
                va = stats_a[metric]
                vb = stats_b[metric]
                fmt = ".1f" if isinstance(va, float) else "d"
                comp_rows.append({"Metric": lbl, label_a: f"{va:{fmt}}", label_b: f"{vb:{fmt}}"})
            st.dataframe(comp_rows, use_container_width=True)

            # 3D structures side by side
            st.markdown("### 3D Structures")
            s_a, s_b = st.columns(2)
            with s_a:
                st.caption(label_a)
                render_3d_structure(pdb_a)
            with s_b:
                st.caption(label_b)
                render_3d_structure(pdb_b)

# Footer
st.markdown("---")
st.caption("Data: [AlphaFold DB](https://alphafold.ebi.ac.uk/) (DeepMind/EBI) | [ESMFold](https://esmatlas.com/) (Meta AI) | [UniProt](https://www.uniprot.org/)")
