"""
AlphaFold Explorer — Protein Structure Viewer & Analysis Suite
Streamlit app for looking up protein structures from the AlphaFold DB,
folding novel sequences via ESMFold, and comprehensive protein analysis.

Features: lookup, fold, batch fold, compare, PDB upload, sequence properties,
domain annotation, Ramachandran plot, hydrophobicity, PDF export, local ESM-2.

Usage:
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import re
import tempfile
from io import BytesIO, StringIO
from collections import Counter

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

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

# Kyte-Doolittle hydrophobicity scale
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

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


# ── Helpers: Sequence ───────────────────────────────────────────────────

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


def extract_sequence_from_pdb(pdb_text: str) -> str:
    """Extract amino acid sequence from PDB CA atoms."""
    residues = parse_plddt_from_pdb(pdb_text)
    return "".join(THREE_TO_ONE.get(r["residue_name"], "X") for r in residues)


# ── Helpers: API (cached) ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alphafold_prediction(uniprot_id: str) -> dict | None:
    """Fetch AlphaFold prediction metadata from EBI."""
    url = f"{ALPHAFOLD_API}/prediction/{uniprot_id}"
    resp = requests.get(url, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        return data[0] if isinstance(data, list) else data
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alphafold_pdb(pdb_url: str) -> str | None:
    """Fetch PDB file from AlphaFold DB."""
    resp = requests.get(pdb_url, timeout=15)
    if resp.status_code == 200:
        return resp.text
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alphafold_pae(pae_url: str) -> dict | None:
    """Fetch PAE JSON from AlphaFold DB."""
    resp = requests.get(pae_url, timeout=15)
    if resp.status_code == 200:
        return resp.json()
    return None


@st.cache_data(ttl=600, show_spinner=False)
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


@st.cache_data(ttl=3600, show_spinner=False)
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


@st.cache_data(ttl=3600, show_spinner=False)
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


# ── Helpers: PDB Parsing ───────────────────────────────────────────────

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


def parse_backbone_atoms(pdb_text: str) -> dict[int, dict[str, tuple[float, float, float]]]:
    """Parse N, CA, C backbone atom coordinates per residue."""
    backbone = {}
    for line in pdb_text.split("\n"):
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name not in ("N", "CA", "C"):
            continue
        res_num = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        if res_num not in backbone:
            backbone[res_num] = {}
        backbone[res_num][atom_name] = (x, y, z)
    return backbone


def _dihedral(p0, p1, p2, p3):
    """Calculate dihedral angle between four points in degrees."""
    b0 = np.array(p0) - np.array(p1)
    b1 = np.array(p2) - np.array(p1)
    b2 = np.array(p3) - np.array(p2)
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.degrees(math.atan2(y, x))


def calculate_phi_psi(pdb_text: str) -> list[tuple[float, float]]:
    """Calculate phi/psi dihedral angles from PDB backbone atoms."""
    backbone = parse_backbone_atoms(pdb_text)
    res_nums = sorted(backbone.keys())
    angles = []
    for i in range(1, len(res_nums) - 1):
        prev_num = res_nums[i - 1]
        curr_num = res_nums[i]
        next_num = res_nums[i + 1]
        prev = backbone.get(prev_num, {})
        curr = backbone.get(curr_num, {})
        nxt = backbone.get(next_num, {})
        if all(k in prev for k in ("C",)) and all(k in curr for k in ("N", "CA", "C")) and all(k in nxt for k in ("N",)):
            phi = _dihedral(prev["C"], curr["N"], curr["CA"], curr["C"])
            psi = _dihedral(curr["N"], curr["CA"], curr["C"], nxt["N"])
            angles.append((phi, psi))
    return angles


def parse_secondary_structure(pdb_text: str) -> dict:
    """Parse HELIX and SHEET records from PDB to get secondary structure summary."""
    helix_residues = 0
    sheet_residues = 0
    for line in pdb_text.split("\n"):
        if line.startswith("HELIX"):
            try:
                start = int(line[21:25].strip())
                end = int(line[33:37].strip())
                helix_residues += max(0, end - start + 1)
            except (ValueError, IndexError):
                pass
        elif line.startswith("SHEET"):
            try:
                start = int(line[22:26].strip())
                end = int(line[33:37].strip())
                sheet_residues += max(0, end - start + 1)
            except (ValueError, IndexError):
                pass
    return {"helix": helix_residues, "sheet": sheet_residues}


# ── Helpers: Sequence Analysis ─────────────────────────────────────────

def compute_sequence_properties(sequence: str) -> dict:
    """Compute biophysical properties using BioPython ProteinAnalysis."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    analysis = ProteinAnalysis(sequence)
    aa_counts = analysis.count_amino_acids()
    aa_percent = analysis.amino_acids_percent
    return {
        "length": len(sequence),
        "molecular_weight": analysis.molecular_weight(),
        "isoelectric_point": analysis.isoelectric_point(),
        "gravy": analysis.gravy(),
        "instability_index": analysis.instability_index(),
        "aromaticity": analysis.aromaticity(),
        "helix_fraction": analysis.secondary_structure_fraction()[0],
        "turn_fraction": analysis.secondary_structure_fraction()[1],
        "sheet_fraction": analysis.secondary_structure_fraction()[2],
        "aa_counts": aa_counts,
        "aa_percent": aa_percent,
    }


def compute_hydrophobicity(sequence: str, window: int = 9) -> list[tuple[int, float]]:
    """Kyte-Doolittle hydrophobicity with sliding window."""
    if len(sequence) < window:
        return []
    half = window // 2
    values = []
    for i in range(half, len(sequence) - half):
        segment = sequence[i - half:i + half + 1]
        score = sum(KD_SCALE.get(aa, 0) for aa in segment) / window
        values.append((i + 1, score))
    return values


# ── Helpers: Visualization ─────────────────────────────────────────────

def plddt_color(val: float) -> str:
    """AlphaFold confidence color scheme."""
    if val > 90:
        return "#0053D6"
    elif val > 70:
        return "#65CBF3"
    elif val > 50:
        return "#FFDB13"
    else:
        return "#FF7D45"


def make_plddt_chart(residues: list[dict], domains: list[dict] | None = None) -> go.Figure:
    """Create per-residue pLDDT confidence plot with optional domain overlays."""
    nums = [r["residue_num"] for r in residues]
    scores = [r["plddt"] for r in residues]
    colors = [plddt_color(s) for s in scores]

    fig = go.Figure()
    fig.add_hrect(y0=90, y1=100, fillcolor="#0053D6", opacity=0.08, line_width=0)
    fig.add_hrect(y0=70, y1=90, fillcolor="#65CBF3", opacity=0.08, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor="#FFDB13", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=50, fillcolor="#FF7D45", opacity=0.08, line_width=0)

    fig.add_trace(go.Bar(
        x=nums, y=scores,
        marker_color=colors,
        hovertemplate="Residue %{x}<br>pLDDT: %{y:.1f}<extra></extra>",
    ))

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
        z=pae_matrix, colorscale="Greens_r", zmin=0, zmax=30,
        colorbar=dict(title="PAE (Å)"),
        hovertemplate="Residue %{x} vs %{y}<br>PAE: %{z:.1f} Å<extra></extra>",
    ))
    fig.update_layout(
        title="Predicted Aligned Error (PAE)",
        xaxis_title="Scored Residue", yaxis_title="Aligned Residue",
        height=500, width=500, yaxis=dict(autorange="reversed"),
    )
    return fig


def make_ramachandran_plot(phi_psi: list[tuple[float, float]]) -> go.Figure:
    """Create Ramachandran plot from phi/psi angles."""
    if not phi_psi:
        fig = go.Figure()
        fig.update_layout(title="Ramachandran Plot — No backbone angles available")
        return fig

    phis = [p[0] for p in phi_psi]
    psis = [p[1] for p in phi_psi]

    fig = go.Figure()

    # Favored regions (approximate)
    # Alpha-helix region
    fig.add_shape(type="rect", x0=-160, y0=-80, x1=-20, y1=40,
                  fillcolor="rgba(0,83,214,0.08)", line=dict(width=0))
    # Beta-sheet region
    fig.add_shape(type="rect", x0=-180, y0=80, x1=-40, y1=180,
                  fillcolor="rgba(101,203,243,0.08)", line=dict(width=0))
    # Left-handed helix
    fig.add_shape(type="rect", x0=20, y0=-60, x1=120, y1=80,
                  fillcolor="rgba(255,219,19,0.08)", line=dict(width=0))

    fig.add_trace(go.Scatter(
        x=phis, y=psis, mode="markers",
        marker=dict(size=4, color="#0053D6", opacity=0.6),
        hovertemplate="Phi: %{x:.1f}°<br>Psi: %{y:.1f}°<extra></extra>",
    ))

    fig.update_layout(
        title="Ramachandran Plot",
        xaxis_title="Phi (°)", yaxis_title="Psi (°)",
        xaxis=dict(range=[-180, 180], dtick=60),
        yaxis=dict(range=[-180, 180], dtick=60),
        height=500, width=500,
        plot_bgcolor="white",
        annotations=[
            dict(x=-90, y=-30, text="α-helix", showarrow=False, font=dict(size=10, color="#0053D6")),
            dict(x=-120, y=140, text="β-sheet", showarrow=False, font=dict(size=10, color="#65CBF3")),
            dict(x=60, y=20, text="L-helix", showarrow=False, font=dict(size=10, color="#FFDB13")),
        ],
    )
    return fig


def make_aa_composition_chart(sequence: str) -> go.Figure:
    """Bar chart of amino acid composition."""
    counts = Counter(sequence)
    aa_order = "ARNDCQEGHILKMFPSTWYV"
    aa_names = {
        "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
        "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
        "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
        "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
    }
    aa_list = [aa for aa in aa_order if aa in set(sequence)]
    vals = [counts.get(aa, 0) for aa in aa_list]
    pcts = [100 * v / len(sequence) for v in vals]

    # Color by property
    hydrophobic = set("AILMFWV")
    charged = set("DEKRH")
    polar = set("STNQCY")
    colors = []
    for aa in aa_list:
        if aa in hydrophobic:
            colors.append("#FF7D45")
        elif aa in charged:
            colors.append("#0053D6")
        elif aa in polar:
            colors.append("#65CBF3")
        else:
            colors.append("#FFDB13")

    fig = go.Figure(go.Bar(
        x=[f"{aa} ({aa_names.get(aa, aa)})" for aa in aa_list],
        y=pcts,
        marker_color=colors,
        hovertemplate="%{x}<br>Count: " + str(vals).replace("[", "").replace("]", "").split(", ")[0] + "<br>%{y:.1f}%<extra></extra>",
        customdata=list(zip(vals, pcts)),
    ))
    # Fix hover to show actual count per bar
    fig.data[0].hovertemplate = None
    fig.data[0].hoverinfo = "text"
    fig.data[0].text = [f"{aa_names.get(aa, aa)}: {c} ({p:.1f}%)" for aa, c, p in zip(aa_list, vals, pcts)]

    fig.update_layout(
        title="Amino Acid Composition",
        xaxis_title="Amino Acid", yaxis_title="Frequency (%)",
        height=350, plot_bgcolor="white",
        margin=dict(t=50, b=80),
    )
    return fig


def make_hydrophobicity_plot(sequence: str, window: int = 9) -> go.Figure:
    """Kyte-Doolittle hydrophobicity sliding window plot."""
    values = compute_hydrophobicity(sequence, window)
    if not values:
        fig = go.Figure()
        fig.update_layout(title="Hydrophobicity — Sequence too short for window")
        return fig

    positions = [v[0] for v in values]
    scores = [v[1] for v in values]

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)

    # Color positive (hydrophobic) and negative (hydrophilic) differently
    colors = ["#FF7D45" if s > 0 else "#0053D6" for s in scores]

    fig.add_trace(go.Scatter(
        x=positions, y=scores,
        mode="lines",
        line=dict(color="#555", width=1),
        fill="tozeroy",
        fillcolor="rgba(0,83,214,0.1)",
        hovertemplate="Residue %{x}<br>Hydrophobicity: %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Kyte-Doolittle Hydrophobicity (window={window})",
        xaxis_title="Residue Number",
        yaxis_title="Hydrophobicity Score",
        height=350,
        plot_bgcolor="white",
        annotations=[
            dict(x=1.02, y=0.9, xref="paper", yref="paper", text="Hydrophobic", font=dict(color="#FF7D45", size=10), showarrow=False, xanchor="left"),
            dict(x=1.02, y=0.1, xref="paper", yref="paper", text="Hydrophilic", font=dict(color="#0053D6", size=10), showarrow=False, xanchor="left"),
        ],
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


def render_overlay_3d(pdb_a: str, pdb_b: str, label_a: str = "A", label_b: str = "B"):
    """Render two structures overlaid in one 3D viewer."""
    try:
        import py3Dmol
        from stmol import showmol

        view = py3Dmol.view(width=700, height=500)
        view.addModel(pdb_a, "pdb")
        view.setStyle({"model": 0}, {"cartoon": {"color": "#0053D6", "opacity": 0.8}})
        view.addModel(pdb_b, "pdb")
        view.setStyle({"model": 1}, {"cartoon": {"color": "#FF7D45", "opacity": 0.8}})
        view.zoomTo()
        view.setBackgroundColor("white")
        showmol(view, height=500, width=700)
        st.caption(f"Blue: {label_a} | Orange: {label_b}")
    except ImportError:
        st.warning("Install py3Dmol and stmol for 3D visualization.")


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


# ── Helpers: PDF Report ────────────────────────────────────────────────

def generate_pdf_report(
    title: str,
    stats: dict,
    plddt_fig: go.Figure,
    uniprot_id: str | None = None,
    domains: list[dict] | None = None,
    seq_props: dict | None = None,
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
    if seq_props:
        table_data.append(["Molecular Weight", f"{seq_props['molecular_weight']:.1f} Da"])
        table_data.append(["Isoelectric Point", f"{seq_props['isoelectric_point']:.2f}"])
        table_data.append(["GRAVY", f"{seq_props['gravy']:.3f}"])
        table_data.append(["Instability Index", f"{seq_props['instability_index']:.1f}"])
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
    elements.append(Paragraph("<i>3D structure and Ramachandran plot available in the app.</i>", styles["Normal"]))

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


# ── Shared UI Components ───────────────────────────────────────────────

def show_stats_row(stats: dict):
    """Display the 4-column stats metrics row."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Residues", stats["num_residues"])
    c2.metric("Mean pLDDT", f"{stats['mean_plddt']:.1f}")
    c3.metric("% Very High (>90)", f"{stats['pct_very_high']:.0f}%")
    c4.metric("% Low (<50)", f"{stats['pct_low']:.0f}%")


def show_properties_tab(sequence: str):
    """Display sequence properties panel."""
    props = compute_sequence_properties(sequence)

    col1, col2, col3 = st.columns(3)
    col1.metric("Molecular Weight", f"{props['molecular_weight']:.0f} Da")
    col2.metric("Isoelectric Point (pI)", f"{props['isoelectric_point']:.2f}")
    col3.metric("GRAVY", f"{props['gravy']:.3f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Instability Index", f"{props['instability_index']:.1f}")
    col5.metric("Aromaticity", f"{props['aromaticity']:.3f}")
    stability = "Stable" if props["instability_index"] < 40 else "Unstable"
    col6.metric("Predicted Stability", stability)

    # Secondary structure fractions (from sequence-based prediction)
    st.markdown("**Predicted Secondary Structure (sequence-based)**")
    ss_cols = st.columns(3)
    ss_cols[0].metric("Helix", f"{props['helix_fraction'] * 100:.0f}%")
    ss_cols[1].metric("Turn", f"{props['turn_fraction'] * 100:.0f}%")
    ss_cols[2].metric("Sheet", f"{props['sheet_fraction'] * 100:.0f}%")

    # AA composition chart
    aa_fig = make_aa_composition_chart(sequence)
    st.plotly_chart(aa_fig, use_container_width=True)

    # Hydrophobicity plot
    hydro_fig = make_hydrophobicity_plot(sequence)
    st.plotly_chart(hydro_fig, use_container_width=True)

    return props


def show_ramachandran_tab(pdb_text: str):
    """Display Ramachandran plot tab."""
    phi_psi = calculate_phi_psi(pdb_text)
    if phi_psi:
        rama_fig = make_ramachandran_plot(phi_psi)
        st.plotly_chart(rama_fig)
        st.caption(f"{len(phi_psi)} residues with calculable backbone angles")

        # Region distribution
        alpha_count = sum(1 for p, s in phi_psi if -160 < p < -20 and -80 < s < 40)
        beta_count = sum(1 for p, s in phi_psi if -180 < p < -40 and 80 < s < 180)
        other_count = len(phi_psi) - alpha_count - beta_count
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Alpha-helix region", f"{100 * alpha_count / len(phi_psi):.0f}%")
        rc2.metric("Beta-sheet region", f"{100 * beta_count / len(phi_psi):.0f}%")
        rc3.metric("Other regions", f"{100 * other_count / len(phi_psi):.0f}%")
    else:
        st.info("Cannot calculate Ramachandran angles — insufficient backbone atoms in PDB.")


# ── Sidebar ─────────────────────────────────────────────────────────────
st.sidebar.title("🧬 AlphaFold Explorer")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Mode",
    [
        "🔍 Lookup (AlphaFold DB)",
        "🧪 Fold (ESMFold)",
        "📦 Batch Fold",
        "⚖️ Compare",
        "📂 Upload PDB",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About**

**Lookup**: Search AlphaFold DB (200M+ structures) by UniProt ID. Domain annotations, Ramachandran plot, sequence properties.

**Fold**: Fold a sequence with ESMFold (max 400 residues). Optional local ESM-2 analysis.

**Batch Fold**: Upload FASTA, fold multiple sequences, export CSV.

**Compare**: Side-by-side pLDDT, stats, 3D structures, and structure overlay.

**Upload PDB**: Analyze your own PDB file — pLDDT, Ramachandran, properties.

All modes include PDF export.
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
        is_accession = bool(re.match(r"^[A-Z][0-9][A-Z0-9]{3}[0-9]$", query.strip().upper()))

        if is_accession:
            uniprot_id = query.strip().upper()
            with st.spinner(f"Fetching AlphaFold prediction for {uniprot_id}..."):
                prediction = fetch_alphafold_prediction(uniprot_id)

            if prediction:
                st.success(f"Found AlphaFold prediction for **{uniprot_id}**")

                pdb_url = prediction.get("pdbUrl")
                pae_url = prediction.get("paeDocUrl")
                pdb_text = fetch_alphafold_pdb(pdb_url) if pdb_url else None
                pae_data = fetch_alphafold_pae(pae_url) if pae_url else None
                domains = fetch_uniprot_domains(uniprot_id)

                if pdb_text:
                    residues = parse_plddt_from_pdb(pdb_text)
                    stats = compute_structure_stats(residues)
                    sequence = extract_sequence_from_pdb(pdb_text)
                    show_stats_row(stats)

                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                        "📊 pLDDT Chart", "🔬 3D Structure", "🗺️ PAE Heatmap",
                        "🧬 Properties", "📐 Ramachandran", "📥 Download",
                    ])

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
                        seq_props = show_properties_tab(sequence)

                    with tab5:
                        show_ramachandran_tab(pdb_text)

                    with tab6:
                        st.download_button("Download PDB", pdb_text, f"AF-{uniprot_id}.pdb", "chemical/x-pdb")
                        if pae_data:
                            st.download_button("Download PAE JSON", json.dumps(pae_data), f"AF-{uniprot_id}-pae.json", "application/json")
                        pdf_fig = make_plddt_chart(residues, domains=domains if domains else None)
                        pdf_bytes = generate_pdf_report(
                            title=f"AlphaFold Prediction — {uniprot_id}",
                            stats=stats, plddt_fig=pdf_fig,
                            uniprot_id=uniprot_id, domains=domains,
                            seq_props=compute_sequence_properties(sequence),
                        )
                        st.download_button("📄 Export PDF Report", pdf_bytes, f"AF-{uniprot_id}-report.pdf", "application/pdf")
                else:
                    st.error(f"PDB file not found for {uniprot_id}.")
            else:
                st.warning(f"No AlphaFold prediction found for {uniprot_id}. Try searching by name instead.")

        else:
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

    esm_capability = check_local_esm_capabilities()
    use_local = False
    if esm_capability != "unavailable":
        use_local = st.checkbox("Use local ESM-2 model (offline)", value=False)
    else:
        st.caption("💡 Install `torch` and `fair-esm` for optional offline ESM-2 analysis")

    sequence_input = st.text_area(
        "Amino acid sequence (FASTA or raw)", height=150,
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
                show_stats_row(stats)

                tab_names = ["📊 pLDDT Chart", "🔬 3D Structure", "🧬 Properties", "📐 Ramachandran", "📥 Download"]
                if analysis and "contact_map" in analysis:
                    tab_names.insert(4, "🧠 Contact Map (ESM-2)")

                tabs = st.tabs(tab_names)

                with tabs[0]:
                    fig = make_plddt_chart(residues)
                    st.plotly_chart(fig, use_container_width=True)

                with tabs[1]:
                    render_3d_structure(pdb_text)

                with tabs[2]:
                    seq_props = show_properties_tab(sequence)

                with tabs[3]:
                    show_ramachandran_tab(pdb_text)

                if analysis and "contact_map" in analysis:
                    with tabs[4]:
                        st.markdown("**Predicted Contact Map** (ESM-2 local model)")
                        cm = analysis["contact_map"]
                        cm_fig = go.Figure(data=go.Heatmap(
                            z=cm, colorscale="Blues",
                            hovertemplate="Residue %{x} vs %{y}<br>Contact: %{z:.3f}<extra></extra>",
                        ))
                        cm_fig.update_layout(
                            xaxis_title="Residue", yaxis_title="Residue",
                            height=500, width=500, yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(cm_fig)

                with tabs[-1]:
                    st.download_button("Download PDB", pdb_text, "esmfold_prediction.pdb", "chemical/x-pdb")
                    pdf_fig = make_plddt_chart(residues)
                    pdf_bytes = generate_pdf_report(
                        title="ESMFold Prediction", stats=stats, plddt_fig=pdf_fig,
                        seq_props=compute_sequence_properties(sequence),
                    )
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

            if "batch_results" in st.session_state and st.session_state["batch_results"]:
                results = st.session_state["batch_results"]
                st.markdown("### Results")

                table_rows = []
                for r in results:
                    if r["stats"]:
                        s = r["stats"]
                        table_rows.append({
                            "Sequence": r["name"], "Length": s["num_residues"],
                            "Mean pLDDT": f"{s['mean_plddt']:.1f}",
                            "Median pLDDT": f"{s['median_plddt']:.1f}",
                            "% Very High": f"{s['pct_very_high']:.0f}%",
                            "% Confident": f"{s['pct_confident']:.0f}%",
                            "% Low": f"{s['pct_low']:.0f}%",
                        })
                    else:
                        table_rows.append({
                            "Sequence": r["name"], "Length": len(r["sequence"]),
                            "Mean pLDDT": "FAILED", "Median pLDDT": "-",
                            "% Very High": "-", "% Confident": "-", "% Low": "-",
                        })

                st.dataframe(table_rows, use_container_width=True)

                for r in results:
                    if r["stats"]:
                        with st.expander(f"{r['name']} — Mean pLDDT: {r['stats']['mean_plddt']:.1f}"):
                            batch_fig = make_plddt_chart(r["residues"])
                            st.plotly_chart(batch_fig, use_container_width=True)

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
                            "sequence_name": r["name"], "length": s["num_residues"],
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
            if input_type == "UniProt ID":
                uid = value.strip().upper()
                if not uid:
                    st.error(f"{label}: Please enter a UniProt ID")
                    return None
                pred = fetch_alphafold_prediction(uid)
                if not pred:
                    st.error(f"No AlphaFold prediction for {uid}")
                    return None
                pdb_url = pred.get("pdbUrl")
                pdb = fetch_alphafold_pdb(pdb_url) if pdb_url else None
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
                x=[r["residue_num"] for r in res_a], y=[r["plddt"] for r in res_a],
                mode="lines", name=label_a, line=dict(color="#0053D6", width=2),
            ))
            cmp_fig.add_trace(go.Scatter(
                x=[r["residue_num"] for r in res_b], y=[r["plddt"] for r in res_b],
                mode="lines", name=label_b, line=dict(color="#FF7D45", width=2),
            ))
            cmp_fig.update_layout(
                title="Per-Residue pLDDT Comparison",
                xaxis_title="Residue Number", yaxis_title="pLDDT Score",
                yaxis=dict(range=[0, 100]), height=400, plot_bgcolor="white",
            )
            st.plotly_chart(cmp_fig, use_container_width=True)

            # Stats comparison
            st.markdown("### Statistics")
            metrics = ["num_residues", "mean_plddt", "median_plddt", "min_plddt", "max_plddt", "pct_very_high", "pct_confident", "pct_low"]
            metric_labels = ["Residues", "Mean pLDDT", "Median pLDDT", "Min pLDDT", "Max pLDDT", "% Very High (>90)", "% Confident (>70)", "% Low (<=50)"]
            comp_rows = []
            for metric, lbl in zip(metrics, metric_labels):
                va = stats_a[metric]
                vb = stats_b[metric]
                fmt = ".1f" if isinstance(va, float) else "d"
                comp_rows.append({"Metric": lbl, label_a: f"{va:{fmt}}", label_b: f"{vb:{fmt}}"})
            st.dataframe(comp_rows, use_container_width=True)

            # 3D structures: side by side + overlay
            st.markdown("### 3D Structures")
            view_mode = st.radio("View", ["Side by Side", "Overlay"], horizontal=True, key="3d_view")

            if view_mode == "Side by Side":
                s_a, s_b = st.columns(2)
                with s_a:
                    st.caption(label_a)
                    render_3d_structure(pdb_a)
                with s_b:
                    st.caption(label_b)
                    render_3d_structure(pdb_b)
            else:
                render_overlay_3d(pdb_a, pdb_b, label_a, label_b)

elif "📂" in mode:
    # ── UPLOAD PDB MODE ──
    st.title("Upload PDB — Analyze Your Structure")
    st.markdown("Upload a PDB file to view pLDDT/B-factors, Ramachandran plot, and sequence properties.")

    pdb_upload = st.file_uploader("Upload PDB file", type=["pdb", "ent"])

    if pdb_upload:
        pdb_text = pdb_upload.read().decode("utf-8")
        residues = parse_plddt_from_pdb(pdb_text)

        if not residues:
            st.error("No CA atoms found in the uploaded PDB file.")
        else:
            stats = compute_structure_stats(residues)
            sequence = extract_sequence_from_pdb(pdb_text)
            st.success(f"Loaded structure: **{len(residues)} residues**")
            show_stats_row(stats)

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 B-factor / pLDDT Chart", "🔬 3D Structure",
                "🧬 Properties", "📐 Ramachandran", "📥 Download",
            ])

            with tab1:
                fig = make_plddt_chart(residues)
                fig.update_layout(title="Per-Residue B-factor / pLDDT")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                render_3d_structure(pdb_text)

            with tab3:
                if sequence and "X" not in sequence:
                    seq_props = show_properties_tab(sequence)
                else:
                    st.warning("Could not extract clean amino acid sequence from PDB for property analysis.")

            with tab4:
                show_ramachandran_tab(pdb_text)

            with tab5:
                st.download_button("Download PDB", pdb_text, pdb_upload.name, "chemical/x-pdb")
                pdf_fig = make_plddt_chart(residues)
                props_for_pdf = compute_sequence_properties(sequence) if sequence and "X" not in sequence else None
                pdf_bytes = generate_pdf_report(
                    title=f"Uploaded Structure — {pdb_upload.name}",
                    stats=stats, plddt_fig=pdf_fig,
                    seq_props=props_for_pdf,
                )
                st.download_button("📄 Export PDF Report", pdf_bytes, f"{pdb_upload.name}-report.pdf", "application/pdf")

# Footer
st.markdown("---")
st.caption("Data: [AlphaFold DB](https://alphafold.ebi.ac.uk/) (DeepMind/EBI) | [ESMFold](https://esmatlas.com/) (Meta AI) | [UniProt](https://www.uniprot.org/)")
