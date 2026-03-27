"""
AlphaFold Explorer — Comprehensive Protein Analysis Suite
Streamlit app for protein structure lookup, folding, comparison,
and in-depth biophysical/structural analysis.

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
from collections import Counter
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

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

POSITIVE_RESIDUES = {"ARG", "LYS", "HIS"}
NEGATIVE_RESIDUES = {"ASP", "GLU"}

st.set_page_config(page_title="AlphaFold Explorer", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

# ── Optional Local ESM-2 ───────────────────────────────────────────────
try:
    import torch
    import esm as esm_module
    LOCAL_ESM_AVAILABLE = True
except ImportError:
    LOCAL_ESM_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# HELPERS: SEQUENCE
# ═══════════════════════════════════════════════════════════════════════

def clean_sequence(seq: str) -> str:
    lines = seq.strip().split("\n")
    cleaned = []
    for line in lines:
        if line.startswith(">"):
            continue
        cleaned.append(re.sub(r"[^A-Za-z]", "", line))
    return "".join(cleaned).upper()


def parse_fasta(text: str) -> list[tuple[str, str]]:
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
        return False, f"Sequence too long for ESMFold ({len(seq)} residues, max 400)."
    invalid = set(seq) - valid_aa
    if invalid:
        return False, f"Invalid amino acid characters: {invalid}"
    return True, "OK"


def extract_sequence_from_pdb(pdb_text: str) -> str:
    residues = parse_plddt_from_pdb(pdb_text)
    return "".join(THREE_TO_ONE.get(r["residue_name"], "X") for r in residues)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS: API (CACHED)
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alphafold_prediction(uniprot_id: str) -> dict | None:
    url = f"{ALPHAFOLD_API}/prediction/{uniprot_id}"
    resp = requests.get(url, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        return data[0] if isinstance(data, list) else data
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alphafold_pdb(pdb_url: str) -> str | None:
    resp = requests.get(pdb_url, timeout=15)
    return resp.text if resp.status_code == 200 else None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alphafold_pae(pae_url: str) -> dict | None:
    resp = requests.get(pae_url, timeout=15)
    return resp.json() if resp.status_code == 200 else None


@st.cache_data(ttl=600, show_spinner=False)
def fold_with_esmfold(sequence: str) -> str | None:
    resp = requests.post(ESMFOLD_API, data=sequence, headers={"Content-Type": "text/plain"}, timeout=120)
    return resp.text if resp.status_code == 200 else None


@st.cache_data(ttl=3600, show_spinner=False)
def search_uniprot(query: str, limit: int = 10) -> list[dict]:
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
            results.append({"accession": acc, "name": name, "organism": organism, "length": length})
        return results
    return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_uniprot_full(uniprot_id: str) -> dict | None:
    """Fetch full UniProt entry JSON."""
    url = f"{UNIPROT_API}/{uniprot_id}.json"
    try:
        resp = requests.get(url, timeout=15)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def extract_uniprot_domains(data: dict) -> list[dict]:
    """Extract domain annotations from full UniProt data."""
    domains = []
    for feat in data.get("features", []):
        if feat.get("type") in DOMAIN_FEATURE_TYPES:
            loc = feat.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            desc = feat.get("description", feat.get("type", "Unknown"))
            if start is not None and end is not None:
                domains.append({"name": desc, "start": int(start), "end": int(end), "type": feat["type"]})
    return domains


def extract_uniprot_annotations(data: dict) -> dict:
    """Extract comprehensive annotations from full UniProt JSON."""
    ann = {
        "gene_name": "", "synonyms": [], "function": "", "subcellular_location": "",
        "disease": "", "keywords": [], "go_terms": [], "pdb_refs": [],
        "protein_existence": data.get("proteinExistence", ""),
    }
    # Gene names
    genes = data.get("genes", [])
    if genes:
        ann["gene_name"] = genes[0].get("geneName", {}).get("value", "")
        ann["synonyms"] = [s.get("value", "") for s in genes[0].get("synonyms", [])]
    # Comments
    for c in data.get("comments", []):
        ct = c.get("commentType", "")
        if ct == "FUNCTION":
            texts = c.get("texts", [])
            if texts:
                ann["function"] = texts[0].get("value", "")
        elif ct == "SUBCELLULAR LOCATION":
            # Try subcellularLocations first, then note
            locs = c.get("subcellularLocations", [])
            if locs:
                loc_strs = []
                for loc in locs:
                    loc_val = loc.get("location", {}).get("value", "")
                    if loc_val:
                        loc_strs.append(loc_val)
                ann["subcellular_location"] = "; ".join(loc_strs)
            else:
                note = c.get("note", {})
                texts = note.get("texts", [])
                if texts:
                    ann["subcellular_location"] = texts[0].get("value", "")
        elif ct == "DISEASE":
            disease_obj = c.get("disease", {})
            if disease_obj:
                ann["disease"] = disease_obj.get("diseaseId", "")
            else:
                note = c.get("note", {})
                texts = note.get("texts", [])
                if texts:
                    ann["disease"] = texts[0].get("value", "")[:500]
    # Keywords
    ann["keywords"] = [{"name": kw.get("name", ""), "category": kw.get("category", "")} for kw in data.get("keywords", [])]
    # Cross-references
    for xref in data.get("uniProtKBCrossReferences", []):
        db = xref.get("database", "")
        if db == "GO":
            props = {p["key"]: p["value"] for p in xref.get("properties", [])}
            term = props.get("GoTerm", "")
            ann["go_terms"].append({"id": xref["id"], "term": term})
        elif db == "PDB":
            props = {p["key"]: p["value"] for p in xref.get("properties", [])}
            ann["pdb_refs"].append({
                "id": xref["id"], "method": props.get("Method", ""),
                "resolution": props.get("Resolution", ""), "chains": props.get("Chains", ""),
            })
    return ann


# ═══════════════════════════════════════════════════════════════════════
# HELPERS: PDB PARSING & STRUCTURAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def parse_plddt_from_pdb(pdb_text: str) -> list[dict]:
    residues = {}
    for line in pdb_text.split("\n"):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            res_num = int(line[22:26].strip())
            bfactor = float(line[60:66].strip())
            chain = line[21:22].strip()
            res_name = line[17:20].strip()
            residues[res_num] = {"residue_num": res_num, "residue_name": res_name, "chain": chain, "plddt": bfactor}
    return [residues[k] for k in sorted(residues.keys())]


def parse_backbone_atoms(pdb_text: str) -> dict[int, dict[str, tuple[float, float, float]]]:
    backbone = {}
    for line in pdb_text.split("\n"):
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name not in ("N", "CA", "C"):
            continue
        res_num = int(line[22:26].strip())
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        if res_num not in backbone:
            backbone[res_num] = {}
        backbone[res_num][atom_name] = (x, y, z)
    return backbone


def parse_all_atoms(pdb_text: str) -> list[dict]:
    """Parse all ATOM records for structural analysis."""
    atoms = []
    for line in pdb_text.split("\n"):
        if not line.startswith("ATOM"):
            continue
        atoms.append({
            "name": line[12:16].strip(), "res_name": line[17:20].strip(),
            "chain": line[21:22].strip(), "res_num": int(line[22:26].strip()),
            "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
            "bfactor": float(line[60:66].strip()),
        })
    return atoms


def _dihedral(p0, p1, p2, p3):
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
    backbone = parse_backbone_atoms(pdb_text)
    res_nums = sorted(backbone.keys())
    angles = []
    for i in range(1, len(res_nums) - 1):
        prev, curr, nxt = backbone.get(res_nums[i - 1], {}), backbone.get(res_nums[i], {}), backbone.get(res_nums[i + 1], {})
        if "C" in prev and all(k in curr for k in ("N", "CA", "C")) and "N" in nxt:
            phi = _dihedral(prev["C"], curr["N"], curr["CA"], curr["C"])
            psi = _dihedral(curr["N"], curr["CA"], curr["C"], nxt["N"])
            angles.append((phi, psi))
    return angles


def calculate_distance_map(pdb_text: str) -> tuple[np.ndarray, list[int]]:
    """Calculate CA-CA distance matrix."""
    backbone = parse_backbone_atoms(pdb_text)
    res_nums = sorted(r for r in backbone if "CA" in backbone[r])
    n = len(res_nums)
    coords = np.array([backbone[r]["CA"] for r in res_nums])
    # Vectorized distance calculation
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist_matrix, res_nums


def calculate_radius_of_gyration(pdb_text: str) -> float:
    backbone = parse_backbone_atoms(pdb_text)
    ca_coords = np.array([backbone[r]["CA"] for r in sorted(backbone) if "CA" in backbone[r]])
    center = np.mean(ca_coords, axis=0)
    return float(np.sqrt(np.mean(np.sum((ca_coords - center) ** 2, axis=1))))


def detect_disulfide_bonds(pdb_text: str) -> list[dict]:
    """Detect Cys-Cys disulfide bonds (SG-SG distance < 3.0 Å)."""
    sg_atoms = {}
    for line in pdb_text.split("\n"):
        if line.startswith("ATOM") and line[12:16].strip() == "SG" and line[17:20].strip() == "CYS":
            res_num = int(line[22:26].strip())
            sg_atoms[res_num] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    bonds = []
    res_nums = sorted(sg_atoms.keys())
    for i in range(len(res_nums)):
        for j in range(i + 1, len(res_nums)):
            dist = float(np.linalg.norm(sg_atoms[res_nums[i]] - sg_atoms[res_nums[j]]))
            if dist < 3.0:
                bonds.append({"cys1": res_nums[i], "cys2": res_nums[j], "distance": dist})
    return bonds


def detect_salt_bridges(pdb_text: str, threshold: float = 4.0) -> list[dict]:
    """Detect salt bridges between oppositely charged residues."""
    charged_atoms = {}  # res_num -> (res_name, centroid_of_charged_atoms)
    atoms = parse_all_atoms(pdb_text)
    # Group charged sidechain atoms by residue
    charged_sc = {"ARG": {"NH1", "NH2", "NE"}, "LYS": {"NZ"}, "HIS": {"ND1", "NE2"},
                  "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
    res_charged_coords: dict[int, list] = {}
    res_names: dict[int, str] = {}
    for a in atoms:
        if a["res_name"] in charged_sc and a["name"] in charged_sc[a["res_name"]]:
            res_num = a["res_num"]
            res_names[res_num] = a["res_name"]
            if res_num not in res_charged_coords:
                res_charged_coords[res_num] = []
            res_charged_coords[res_num].append([a["x"], a["y"], a["z"]])

    centroids = {r: np.mean(res_charged_coords[r], axis=0) for r in res_charged_coords}
    bridges = []
    pos_res = sorted(r for r in centroids if res_names[r] in POSITIVE_RESIDUES)
    neg_res = sorted(r for r in centroids if res_names[r] in NEGATIVE_RESIDUES)
    for pr in pos_res:
        for nr in neg_res:
            dist = float(np.linalg.norm(centroids[pr] - centroids[nr]))
            if dist < threshold:
                bridges.append({"pos_res": pr, "pos_name": res_names[pr],
                                "neg_res": nr, "neg_name": res_names[nr], "distance": dist})
    return bridges


# ═══════════════════════════════════════════════════════════════════════
# HELPERS: SEQUENCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def compute_sequence_properties(sequence: str) -> dict:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    analysis = ProteinAnalysis(sequence)
    ss = analysis.secondary_structure_fraction()
    return {
        "length": len(sequence),
        "molecular_weight": analysis.molecular_weight(),
        "isoelectric_point": analysis.isoelectric_point(),
        "gravy": analysis.gravy(),
        "instability_index": analysis.instability_index(),
        "aromaticity": analysis.aromaticity(),
        "helix_fraction": ss[0], "turn_fraction": ss[1], "sheet_fraction": ss[2],
        "aa_counts": analysis.count_amino_acids(),
        "aa_percent": analysis.amino_acids_percent,
        "extinction_coeff": analysis.molar_extinction_coefficient(),  # (reduced, oxidized)
    }


def compute_charge_at_ph(sequence: str) -> list[tuple[float, float]]:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    analysis = ProteinAnalysis(sequence)
    return [(ph / 10, analysis.charge_at_pH(ph / 10)) for ph in range(0, 141)]


def compute_hydrophobicity(sequence: str, window: int = 9) -> list[tuple[int, float]]:
    if len(sequence) < window:
        return []
    half = window // 2
    return [(i + 1, sum(KD_SCALE.get(sequence[j], 0) for j in range(i - half, i + half + 1)) / window)
            for i in range(half, len(sequence) - half)]


def find_glycosylation_sites(sequence: str) -> list[dict]:
    """Find N-linked glycosylation consensus: N-X-S/T where X != P."""
    sites = []
    for i in range(len(sequence) - 2):
        if sequence[i] == "N" and sequence[i + 1] != "P" and sequence[i + 2] in "ST":
            sites.append({"position": i + 1, "motif": sequence[i:i + 3]})
    return sites


def find_phosphorylation_sites(sequence: str) -> list[dict]:
    """Find potential phosphorylation sites (S, T, Y)."""
    return [{"position": i + 1, "residue": aa} for i, aa in enumerate(sequence) if aa in "STY"]


def predict_transmembrane(sequence: str, window: int = 21, threshold: float = 1.6) -> list[dict]:
    """Simple TM prediction: hydrophobic stretches >= window with avg KD > threshold."""
    if len(sequence) < window:
        return []
    regions = []
    in_tm = False
    start = 0
    for i in range(len(sequence) - window + 1):
        segment = sequence[i:i + window]
        avg = sum(KD_SCALE.get(aa, 0) for aa in segment) / window
        if avg > threshold and not in_tm:
            in_tm = True
            start = i
        elif avg <= threshold and in_tm:
            in_tm = False
            regions.append({"start": start + 1, "end": i + window, "length": i + window - start})
    if in_tm:
        regions.append({"start": start + 1, "end": len(sequence), "length": len(sequence) - start})
    return regions


# ═══════════════════════════════════════════════════════════════════════
# HELPERS: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def plddt_color(val: float) -> str:
    if val > 90: return "#0053D6"
    elif val > 70: return "#65CBF3"
    elif val > 50: return "#FFDB13"
    else: return "#FF7D45"


def make_plddt_chart(residues: list[dict], domains: list[dict] | None = None) -> go.Figure:
    nums = [r["residue_num"] for r in residues]
    scores = [r["plddt"] for r in residues]
    colors = [plddt_color(s) for s in scores]
    fig = go.Figure()
    fig.add_hrect(y0=90, y1=100, fillcolor="#0053D6", opacity=0.08, line_width=0)
    fig.add_hrect(y0=70, y1=90, fillcolor="#65CBF3", opacity=0.08, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor="#FFDB13", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=50, fillcolor="#FF7D45", opacity=0.08, line_width=0)
    fig.add_trace(go.Bar(x=nums, y=scores, marker_color=colors, hovertemplate="Residue %{x}<br>pLDDT: %{y:.1f}<extra></extra>"))
    if domains:
        palette = px.colors.qualitative.Set2
        for i, dom in enumerate(domains):
            color = palette[i % len(palette)]
            fig.add_vrect(x0=dom["start"] - 0.5, x1=dom["end"] + 0.5, fillcolor=color, opacity=0.15,
                          line_width=1, line_color=color, annotation_text=dom["name"],
                          annotation_position="top left", annotation_font_size=9, annotation_font_color=color)
    fig.update_layout(
        title="Per-Residue pLDDT Confidence", xaxis_title="Residue Number", yaxis_title="pLDDT Score",
        yaxis=dict(range=[0, 100]), height=400, margin=dict(t=50, b=50, l=60, r=20), plot_bgcolor="white",
        annotations=[
            dict(x=1.02, y=0.95, xref="paper", yref="paper", text="Very high (>90)", font=dict(color="#0053D6", size=10), showarrow=False, xanchor="left"),
            dict(x=1.02, y=0.80, xref="paper", yref="paper", text="Confident (70-90)", font=dict(color="#65CBF3", size=10), showarrow=False, xanchor="left"),
            dict(x=1.02, y=0.60, xref="paper", yref="paper", text="Low (50-70)", font=dict(color="#FFDB13", size=10), showarrow=False, xanchor="left"),
            dict(x=1.02, y=0.40, xref="paper", yref="paper", text="Very low (<50)", font=dict(color="#FF7D45", size=10), showarrow=False, xanchor="left"),
        ],
    )
    return fig


def make_pae_heatmap(pae_data: dict) -> go.Figure:
    if isinstance(pae_data, list):
        pae_data = pae_data[0]
    pae_matrix = np.array(pae_data.get("predicted_aligned_error", pae_data.get("pae", [])))
    fig = go.Figure(data=go.Heatmap(z=pae_matrix, colorscale="Greens_r", zmin=0, zmax=30,
                                     colorbar=dict(title="PAE (Å)"),
                                     hovertemplate="Residue %{x} vs %{y}<br>PAE: %{z:.1f} Å<extra></extra>"))
    fig.update_layout(title="Predicted Aligned Error (PAE)", xaxis_title="Scored Residue",
                      yaxis_title="Aligned Residue", height=500, width=500, yaxis=dict(autorange="reversed"))
    return fig


def make_ramachandran_plot(phi_psi: list[tuple[float, float]]) -> go.Figure:
    if not phi_psi:
        fig = go.Figure()
        fig.update_layout(title="Ramachandran Plot — No backbone angles available")
        return fig
    phis, psis = zip(*phi_psi)
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-160, y0=-80, x1=-20, y1=40, fillcolor="rgba(0,83,214,0.08)", line=dict(width=0))
    fig.add_shape(type="rect", x0=-180, y0=80, x1=-40, y1=180, fillcolor="rgba(101,203,243,0.08)", line=dict(width=0))
    fig.add_shape(type="rect", x0=20, y0=-60, x1=120, y1=80, fillcolor="rgba(255,219,19,0.08)", line=dict(width=0))
    fig.add_trace(go.Scatter(x=list(phis), y=list(psis), mode="markers",
                              marker=dict(size=4, color="#0053D6", opacity=0.6),
                              hovertemplate="Phi: %{x:.1f}°<br>Psi: %{y:.1f}°<extra></extra>"))
    fig.update_layout(title="Ramachandran Plot", xaxis_title="Phi (°)", yaxis_title="Psi (°)",
                      xaxis=dict(range=[-180, 180], dtick=60), yaxis=dict(range=[-180, 180], dtick=60),
                      height=500, width=500, plot_bgcolor="white",
                      annotations=[
                          dict(x=-90, y=-30, text="α-helix", showarrow=False, font=dict(size=10, color="#0053D6")),
                          dict(x=-120, y=140, text="β-sheet", showarrow=False, font=dict(size=10, color="#65CBF3")),
                          dict(x=60, y=20, text="L-helix", showarrow=False, font=dict(size=10, color="#FFDB13")),
                      ])
    return fig


def make_aa_composition_chart(sequence: str) -> go.Figure:
    counts = Counter(sequence)
    aa_order = "ARNDCQEGHILKMFPSTWYV"
    aa_names = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu",
                "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe",
                "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val"}
    hydrophobic, charged, polar = set("AILMFWV"), set("DEKRH"), set("STNQCY")
    aa_list = [aa for aa in aa_order if counts.get(aa, 0) > 0]
    vals = [counts[aa] for aa in aa_list]
    pcts = [100 * v / len(sequence) for v in vals]
    colors = ["#FF7D45" if aa in hydrophobic else "#0053D6" if aa in charged else "#65CBF3" if aa in polar else "#FFDB13" for aa in aa_list]
    fig = go.Figure(go.Bar(x=[f"{aa} ({aa_names[aa]})" for aa in aa_list], y=pcts, marker_color=colors,
                            text=[f"{c} ({p:.1f}%)" for c, p in zip(vals, pcts)], hoverinfo="text"))
    fig.update_layout(title="Amino Acid Composition", xaxis_title="Amino Acid", yaxis_title="Frequency (%)",
                      height=350, plot_bgcolor="white", margin=dict(t=50, b=80))
    return fig


def make_hydrophobicity_plot(sequence: str, window: int = 9) -> go.Figure:
    values = compute_hydrophobicity(sequence, window)
    if not values:
        fig = go.Figure()
        fig.update_layout(title="Hydrophobicity — Sequence too short")
        return fig
    positions, scores = zip(*values)
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    fig.add_trace(go.Scatter(x=list(positions), y=list(scores), mode="lines", line=dict(color="#555", width=1),
                              fill="tozeroy", fillcolor="rgba(0,83,214,0.1)",
                              hovertemplate="Residue %{x}<br>Hydrophobicity: %{y:.2f}<extra></extra>"))
    fig.update_layout(title=f"Kyte-Doolittle Hydrophobicity (window={window})",
                      xaxis_title="Residue Number", yaxis_title="Score", height=350, plot_bgcolor="white")
    return fig


def make_charge_at_ph_plot(sequence: str) -> go.Figure:
    data = compute_charge_at_ph(sequence)
    phs, charges = zip(*data)
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    fig.add_trace(go.Scatter(x=list(phs), y=list(charges), mode="lines", line=dict(color="#0053D6", width=2),
                              hovertemplate="pH %{x:.1f}<br>Charge: %{y:.1f}<extra></extra>"))
    fig.update_layout(title="Net Charge vs pH", xaxis_title="pH", yaxis_title="Net Charge",
                      height=350, plot_bgcolor="white")
    return fig


def make_distance_map(pdb_text: str) -> go.Figure:
    dist_matrix, res_nums = calculate_distance_map(pdb_text)
    fig = go.Figure(data=go.Heatmap(z=dist_matrix, x=res_nums, y=res_nums, colorscale="Viridis_r",
                                     colorbar=dict(title="Distance (Å)"),
                                     hovertemplate="Res %{x} vs %{y}<br>Distance: %{z:.1f} Å<extra></extra>"))
    fig.update_layout(title="CA-CA Distance Map", xaxis_title="Residue", yaxis_title="Residue",
                      height=550, width=550, yaxis=dict(autorange="reversed"))
    return fig


def make_bfactor_histogram(residues: list[dict]) -> go.Figure:
    scores = [r["plddt"] for r in residues]
    fig = go.Figure(go.Histogram(x=scores, nbinsx=30,
                                  marker_color="#0053D6", opacity=0.7,
                                  hovertemplate="pLDDT: %{x:.0f}<br>Count: %{y}<extra></extra>"))
    fig.add_vline(x=np.mean(scores), line_dash="dash", line_color="red", annotation_text=f"Mean: {np.mean(scores):.1f}")
    fig.update_layout(title="pLDDT / B-factor Distribution", xaxis_title="pLDDT Score", yaxis_title="Count",
                      height=350, plot_bgcolor="white")
    return fig


def render_3d_structure(pdb_text: str, style: str = "cartoon", color_scheme: str = "pLDDT"):
    """Render 3D protein structure with configurable style and color."""
    try:
        import py3Dmol
        from stmol import showmol
        view = py3Dmol.view(width=700, height=500)
        view.addModel(pdb_text, "pdb")
        if style == "cartoon":
            if color_scheme == "pLDDT":
                view.setStyle({"cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb", "min": 0, "max": 100}}})
            elif color_scheme == "Chain":
                view.setStyle({"cartoon": {"color": "spectrum"}})
            elif color_scheme == "Hydrophobicity":
                view.setStyle({"cartoon": {"colorscheme": "hydrophobicity"}})
            elif color_scheme == "Secondary Structure":
                view.setStyle({"cartoon": {"colorscheme": "ssJmol"}})
            else:
                view.setStyle({"cartoon": {"color": "#0053D6"}})
        elif style == "stick":
            view.setStyle({"stick": {"colorscheme": "Jmol"}})
        elif style == "sphere":
            view.setStyle({"sphere": {"colorscheme": "Jmol", "scale": 0.3}})
        elif style == "surface":
            view.addSurface("VDW", {"opacity": 0.8, "colorscheme": {"prop": "b", "gradient": "roygb", "min": 0, "max": 100}})
        elif style == "line":
            view.setStyle({"line": {"colorscheme": "Jmol"}})
        view.zoomTo()
        view.setBackgroundColor("white")
        showmol(view, height=500, width=700)
    except ImportError:
        st.warning("Install py3Dmol and stmol for 3D visualization: `pip install py3Dmol stmol`")
        st.code(pdb_text[:2000] + "\n... (truncated)", language="text")


def render_overlay_3d(pdb_a: str, pdb_b: str, label_a: str = "A", label_b: str = "B"):
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
    scores = [r["plddt"] for r in residues]
    return {
        "num_residues": len(scores), "mean_plddt": np.mean(scores), "median_plddt": np.median(scores),
        "min_plddt": np.min(scores), "max_plddt": np.max(scores),
        "pct_very_high": sum(1 for s in scores if s > 90) / len(scores) * 100,
        "pct_confident": sum(1 for s in scores if s > 70) / len(scores) * 100,
        "pct_low": sum(1 for s in scores if s <= 50) / len(scores) * 100,
    }


# ═══════════════════════════════════════════════════════════════════════
# HELPERS: EXPORT (PDF & JSON)
# ═══════════════════════════════════════════════════════════════════════

def generate_pdf_report(title: str, stats: dict, plddt_fig: go.Figure,
                        uniprot_id: str | None = None, domains: list[dict] | None = None,
                        seq_props: dict | None = None) -> bytes:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image as RLImage, SimpleDocTemplate, Spacer, Paragraph, Table, TableStyle

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("AlphaFold Explorer Report", styles["Title"]))
    elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
    if uniprot_id:
        elements.append(Paragraph(f"UniProt: {uniprot_id}", styles["Normal"]))
    elements.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    table_data = [["Metric", "Value"],
                  ["Residues", str(stats["num_residues"])],
                  ["Mean pLDDT", f"{stats['mean_plddt']:.1f}"], ["Median pLDDT", f"{stats['median_plddt']:.1f}"],
                  ["Min pLDDT", f"{stats['min_plddt']:.1f}"], ["Max pLDDT", f"{stats['max_plddt']:.1f}"],
                  ["% Very High (>90)", f"{stats['pct_very_high']:.1f}%"],
                  ["% Confident (>70)", f"{stats['pct_confident']:.1f}%"],
                  ["% Low (<=50)", f"{stats['pct_low']:.1f}%"]]
    if seq_props:
        table_data += [["Molecular Weight", f"{seq_props['molecular_weight']:.1f} Da"],
                       ["Isoelectric Point", f"{seq_props['isoelectric_point']:.2f}"],
                       ["GRAVY", f"{seq_props['gravy']:.3f}"],
                       ["Instability Index", f"{seq_props['instability_index']:.1f}"],
                       ["Ext. Coefficient (reduced)", f"{seq_props['extinction_coeff'][0]} M⁻¹cm⁻¹"],
                       ["Ext. Coefficient (oxidized)", f"{seq_props['extinction_coeff'][1]} M⁻¹cm⁻¹"]]
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

    try:
        img_bytes = plddt_fig.to_image(format="png", width=900, height=400, scale=2)
        elements.append(Paragraph("Per-Residue pLDDT Confidence", styles["Heading3"]))
        elements.append(RLImage(BytesIO(img_bytes), width=6.5 * inch, height=2.9 * inch))
    except Exception:
        elements.append(Paragraph("<i>Chart image unavailable</i>", styles["Normal"]))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("<i>3D structure, Ramachandran plot, and full analysis available in the app.</i>", styles["Normal"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("Data: AlphaFold DB (DeepMind/EBI) | ESMFold (Meta AI) | UniProt", styles["Normal"]))
    doc.build(elements)
    return buf.getvalue()


def export_analysis_json(stats: dict, seq_props: dict | None = None, domains: list[dict] | None = None,
                         phi_psi: list | None = None, disulfide: list | None = None,
                         salt_bridges: list | None = None, rg: float | None = None,
                         annotations: dict | None = None) -> str:
    """Export comprehensive analysis as JSON string."""
    data = {"structure_stats": stats}
    if seq_props:
        sp = {k: v for k, v in seq_props.items() if k not in ("aa_counts", "aa_percent")}
        sp["extinction_coeff_reduced"] = seq_props["extinction_coeff"][0]
        sp["extinction_coeff_oxidized"] = seq_props["extinction_coeff"][1]
        data["sequence_properties"] = sp
    if domains:
        data["domains"] = domains
    if phi_psi:
        data["ramachandran_angles_count"] = len(phi_psi)
    if disulfide:
        data["disulfide_bonds"] = disulfide
    if salt_bridges:
        data["salt_bridges"] = salt_bridges
    if rg is not None:
        data["radius_of_gyration"] = rg
    if annotations:
        data["uniprot_annotations"] = annotations
    return json.dumps(data, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS: LOCAL ESM-2
# ═══════════════════════════════════════════════════════════════════════

def check_local_esm_capabilities() -> str:
    if not LOCAL_ESM_AVAILABLE:
        return "unavailable"
    try:
        from esm.pretrained import esmfold_v1  # noqa: F401
        return "full_fold"
    except (ImportError, AttributeError):
        return "analysis_only"


def fold_local_esm(sequence: str) -> str | None:
    try:
        from esm.pretrained import esmfold_v1
        model = esmfold_v1().eval()
        if torch.cuda.is_available():
            model = model.cuda()
        with torch.no_grad():
            return model.infer_pdb(sequence)
    except Exception:
        return None


def analyze_local_esm(sequence: str) -> dict | None:
    try:
        model, alphabet = esm_module.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model = model.eval()
        _, _, batch_tokens = batch_converter([("protein", sequence)])
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=True)
        return {"contact_map": results["contacts"][0].numpy()}
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# SHARED UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════

def show_stats_row(stats: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Residues", stats["num_residues"])
    c2.metric("Mean pLDDT", f"{stats['mean_plddt']:.1f}")
    c3.metric("% Very High (>90)", f"{stats['pct_very_high']:.0f}%")
    c4.metric("% Low (<50)", f"{stats['pct_low']:.0f}%")


def show_3d_tab(pdb_text: str):
    """3D viewer with style and color controls."""
    col_style, col_color = st.columns(2)
    with col_style:
        style = st.selectbox("Style", ["cartoon", "stick", "sphere", "surface", "line"], key=f"style_{id(pdb_text)}")
    with col_color:
        color = st.selectbox("Color by", ["pLDDT", "Chain", "Hydrophobicity", "Secondary Structure", "Uniform"],
                              key=f"color_{id(pdb_text)}")
    render_3d_structure(pdb_text, style=style, color_scheme=color)


def show_properties_tab(sequence: str) -> dict:
    props = compute_sequence_properties(sequence)
    # Row 1: Basic metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Molecular Weight", f"{props['molecular_weight']:.0f} Da")
    c2.metric("Isoelectric Point (pI)", f"{props['isoelectric_point']:.2f}")
    c3.metric("GRAVY", f"{props['gravy']:.3f}")
    c4.metric("Instability Index", f"{props['instability_index']:.1f}")
    # Row 2: More metrics
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Aromaticity", f"{props['aromaticity']:.3f}")
    c6.metric("Stability", "Stable" if props["instability_index"] < 40 else "Unstable")
    c7.metric("Ext. Coeff (reduced)", f"{props['extinction_coeff'][0]}")
    c8.metric("Ext. Coeff (oxidized)", f"{props['extinction_coeff'][1]}")
    # Secondary structure
    st.markdown("**Predicted Secondary Structure (sequence-based)**")
    ss1, ss2, ss3 = st.columns(3)
    ss1.metric("Helix", f"{props['helix_fraction'] * 100:.0f}%")
    ss2.metric("Turn", f"{props['turn_fraction'] * 100:.0f}%")
    ss3.metric("Sheet", f"{props['sheet_fraction'] * 100:.0f}%")
    # Plots
    st.plotly_chart(make_aa_composition_chart(sequence), use_container_width=True)
    st.plotly_chart(make_hydrophobicity_plot(sequence), use_container_width=True)
    st.plotly_chart(make_charge_at_ph_plot(sequence), use_container_width=True)
    # PTM sites
    glyco = find_glycosylation_sites(sequence)
    phospho = find_phosphorylation_sites(sequence)
    tm_regions = predict_transmembrane(sequence)
    st.markdown("**Post-Translational Modification Sites**")
    ptm1, ptm2, ptm3 = st.columns(3)
    ptm1.metric("N-Glycosylation sites", len(glyco))
    ptm2.metric("Potential phospho sites (S/T/Y)", len(phospho))
    ptm3.metric("Predicted TM regions", len(tm_regions))
    if glyco:
        with st.expander(f"N-Glycosylation sites ({len(glyco)})"):
            for g in glyco:
                st.write(f"Position **{g['position']}**: {g['motif']}")
    if tm_regions:
        with st.expander(f"Transmembrane regions ({len(tm_regions)})"):
            for t in tm_regions:
                st.write(f"Residues **{t['start']}-{t['end']}** ({t['length']} aa)")
    return props


def show_ramachandran_tab(pdb_text: str):
    phi_psi = calculate_phi_psi(pdb_text)
    if phi_psi:
        st.plotly_chart(make_ramachandran_plot(phi_psi))
        st.caption(f"{len(phi_psi)} residues with calculable backbone angles")
        alpha = sum(1 for p, s in phi_psi if -160 < p < -20 and -80 < s < 40)
        beta = sum(1 for p, s in phi_psi if -180 < p < -40 and 80 < s < 180)
        other = len(phi_psi) - alpha - beta
        r1, r2, r3 = st.columns(3)
        r1.metric("Alpha-helix region", f"{100 * alpha / len(phi_psi):.0f}%")
        r2.metric("Beta-sheet region", f"{100 * beta / len(phi_psi):.0f}%")
        r3.metric("Other regions", f"{100 * other / len(phi_psi):.0f}%")
    else:
        st.info("Cannot calculate Ramachandran angles — insufficient backbone atoms.")


def show_structural_analysis_tab(pdb_text: str, residues: list[dict]):
    """Structural analysis: distance map, Rg, disulfide bonds, salt bridges, B-factor histogram."""
    # Radius of gyration
    rg = calculate_radius_of_gyration(pdb_text)
    disulfide = detect_disulfide_bonds(pdb_text)
    salt = detect_salt_bridges(pdb_text)

    c1, c2, c3 = st.columns(3)
    c1.metric("Radius of Gyration", f"{rg:.1f} Å")
    c2.metric("Disulfide Bonds", len(disulfide))
    c3.metric("Salt Bridges", len(salt))

    # B-factor histogram
    st.plotly_chart(make_bfactor_histogram(residues), use_container_width=True)

    # Distance map
    st.plotly_chart(make_distance_map(pdb_text))

    # Details
    if disulfide:
        with st.expander(f"Disulfide Bonds ({len(disulfide)})"):
            for bond in disulfide:
                st.write(f"Cys **{bond['cys1']}** — Cys **{bond['cys2']}** ({bond['distance']:.2f} Å)")
    if salt:
        with st.expander(f"Salt Bridges ({len(salt)})"):
            for sb in salt:
                st.write(f"{sb['pos_name']} **{sb['pos_res']}** — {sb['neg_name']} **{sb['neg_res']}** ({sb['distance']:.1f} Å)")


def show_annotations_tab(uniprot_data: dict):
    """Display UniProt biological annotations."""
    ann = extract_uniprot_annotations(uniprot_data)

    if ann["gene_name"]:
        synonyms = f" ({', '.join(ann['synonyms'])})" if ann["synonyms"] else ""
        st.markdown(f"**Gene:** {ann['gene_name']}{synonyms}")
    if ann["protein_existence"]:
        st.markdown(f"**Evidence:** {ann['protein_existence']}")
    st.markdown("---")

    if ann["function"]:
        st.markdown("**Function**")
        st.write(ann["function"])
    if ann["subcellular_location"]:
        st.markdown("**Subcellular Location**")
        st.write(ann["subcellular_location"])
    if ann["disease"]:
        st.markdown("**Disease Association**")
        st.write(ann["disease"])

    # GO Terms
    if ann["go_terms"]:
        st.markdown("**Gene Ontology (GO) Terms**")
        go_c = [g for g in ann["go_terms"] if g["term"].startswith("C:")]
        go_f = [g for g in ann["go_terms"] if g["term"].startswith("F:")]
        go_p = [g for g in ann["go_terms"] if g["term"].startswith("P:")]
        if go_c:
            with st.expander(f"Cellular Component ({len(go_c)})"):
                for g in go_c:
                    st.write(f"- {g['term'][2:]} (`{g['id']}`)")
        if go_f:
            with st.expander(f"Molecular Function ({len(go_f)})"):
                for g in go_f:
                    st.write(f"- {g['term'][2:]} (`{g['id']}`)")
        if go_p:
            with st.expander(f"Biological Process ({len(go_p)})"):
                for g in go_p:
                    st.write(f"- {g['term'][2:]} (`{g['id']}`)")

    # PDB cross-references
    if ann["pdb_refs"]:
        st.markdown(f"**PDB Structures ({len(ann['pdb_refs'])})**")
        pdb_rows = [{"PDB ID": p["id"], "Method": p["method"], "Resolution": p["resolution"], "Chains": p["chains"]}
                    for p in ann["pdb_refs"]]
        st.dataframe(pdb_rows, use_container_width=True)

    # Keywords
    if ann["keywords"]:
        st.markdown("**Keywords**")
        kw_by_cat: dict[str, list[str]] = {}
        for kw in ann["keywords"]:
            cat = kw["category"] or "Other"
            kw_by_cat.setdefault(cat, []).append(kw["name"])
        for cat, names in sorted(kw_by_cat.items()):
            st.write(f"**{cat}:** {', '.join(names)}")


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

st.sidebar.title("🧬 AlphaFold Explorer")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode", [
    "🔍 Lookup (AlphaFold DB)", "🧪 Fold (ESMFold)", "📦 Batch Fold", "⚖️ Compare", "📂 Upload PDB",
], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**About**

**Lookup**: AlphaFold DB search with domain annotations, GO terms, disease associations, structural analysis.

**Fold**: ESMFold API or local ESM-2. Full biophysical analysis.

**Batch Fold**: FASTA upload, fold up to 20 sequences, CSV export.

**Compare**: Side-by-side or overlay. pLDDT, stats, 3D structures.

**Upload PDB**: Analyze your own PDB — full analysis suite.

All modes: pLDDT, Ramachandran, properties, structural analysis, PDF & JSON export.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Example proteins:**
- `P69905` — Hemoglobin alpha
- `P0DTC2` — SARS-CoV-2 spike
- `P04637` — p53 tumor suppressor
- `P68871` — Hemoglobin beta
- `Q9BYF1` — ACE2 receptor
""")

# ═══════════════════════════════════════════════════════════════════════
# MAIN: LOOKUP MODE
# ═══════════════════════════════════════════════════════════════════════

if "🔍" in mode:
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
            with st.spinner(f"Fetching data for {uniprot_id}..."):
                prediction = fetch_alphafold_prediction(uniprot_id)
                uniprot_data = fetch_uniprot_full(uniprot_id)

            if prediction:
                st.success(f"Found AlphaFold prediction for **{uniprot_id}**")
                pdb_url = prediction.get("pdbUrl")
                pae_url = prediction.get("paeDocUrl")
                pdb_text = fetch_alphafold_pdb(pdb_url) if pdb_url else None
                pae_data = fetch_alphafold_pae(pae_url) if pae_url else None
                domains = extract_uniprot_domains(uniprot_data) if uniprot_data else []

                if pdb_text:
                    residues = parse_plddt_from_pdb(pdb_text)
                    stats = compute_structure_stats(residues)
                    sequence = extract_sequence_from_pdb(pdb_text)
                    show_stats_row(stats)

                    tabs = st.tabs(["📊 pLDDT", "🔬 3D Structure", "🗺️ PAE", "🧬 Properties",
                                    "📐 Ramachandran", "🔩 Structural", "📚 Annotations", "📥 Export"])

                    with tabs[0]:
                        show_domains = st.checkbox("Show domain annotations", value=True) if domains else False
                        fig = make_plddt_chart(residues, domains=domains if show_domains else None)
                        st.plotly_chart(fig, use_container_width=True)
                        if domains and show_domains:
                            st.caption(f"Showing {len(domains)} domain/region annotations from UniProt")

                    with tabs[1]:
                        show_3d_tab(pdb_text)

                    with tabs[2]:
                        if pae_data:
                            st.plotly_chart(make_pae_heatmap(pae_data))
                        else:
                            st.info("PAE data not available.")

                    with tabs[3]:
                        seq_props = show_properties_tab(sequence)

                    with tabs[4]:
                        show_ramachandran_tab(pdb_text)

                    with tabs[5]:
                        show_structural_analysis_tab(pdb_text, residues)

                    with tabs[6]:
                        if uniprot_data:
                            show_annotations_tab(uniprot_data)
                        else:
                            st.info("UniProt annotations not available.")

                    with tabs[7]:
                        st.download_button("Download PDB", pdb_text, f"AF-{uniprot_id}.pdb", "chemical/x-pdb")
                        if pae_data:
                            st.download_button("Download PAE JSON", json.dumps(pae_data), f"AF-{uniprot_id}-pae.json", "application/json")
                        pdf_fig = make_plddt_chart(residues, domains=domains if domains else None)
                        pdf_bytes = generate_pdf_report(
                            title=f"AlphaFold Prediction — {uniprot_id}", stats=stats, plddt_fig=pdf_fig,
                            uniprot_id=uniprot_id, domains=domains,
                            seq_props=compute_sequence_properties(sequence))
                        st.download_button("📄 Export PDF Report", pdf_bytes, f"AF-{uniprot_id}-report.pdf", "application/pdf")
                        ann = extract_uniprot_annotations(uniprot_data) if uniprot_data else None
                        json_str = export_analysis_json(
                            stats=stats, seq_props=compute_sequence_properties(sequence), domains=domains,
                            phi_psi=calculate_phi_psi(pdb_text), disulfide=detect_disulfide_bonds(pdb_text),
                            salt_bridges=detect_salt_bridges(pdb_text), rg=calculate_radius_of_gyration(pdb_text),
                            annotations=ann)
                        st.download_button("📦 Export Full Analysis JSON", json_str, f"AF-{uniprot_id}-analysis.json", "application/json")
                else:
                    st.error(f"PDB file not found for {uniprot_id}.")
            else:
                st.warning(f"No AlphaFold prediction found for {uniprot_id}. Try searching by name.")
        else:
            with st.spinner(f"Searching UniProt for '{query}'..."):
                results = search_uniprot(query)
            if results:
                st.markdown(f"**{len(results)} results found.**")
                for r in results:
                    ca, cb, cc, cd = st.columns([1, 3, 2, 1])
                    ca.code(r["accession"])
                    cb.write(r["name"])
                    cc.write(f"*{r['organism']}*")
                    cd.write(f"{r['length']} aa")
                st.info("Copy a UniProt accession from above and search again.")
            else:
                st.warning(f"No results found for '{query}'.")

# ═══════════════════════════════════════════════════════════════════════
# MAIN: FOLD MODE
# ═══════════════════════════════════════════════════════════════════════

elif "🧪" in mode:
    st.title("ESMFold — Fold a Sequence")
    st.markdown("Fold a novel amino acid sequence using Meta's ESMFold. Max 400 residues.")

    esm_capability = check_local_esm_capabilities()
    use_local = False
    if esm_capability != "unavailable":
        use_local = st.checkbox("Use local ESM-2 model (offline)", value=False)
    else:
        st.caption("💡 Install `torch` and `fair-esm` for optional offline ESM-2 analysis")

    sequence_input = st.text_area("Amino acid sequence (FASTA or raw)", height=150,
                                   placeholder=">my_protein\nMKTAYIAKQRQISFVKSHFSRQDLDALK...")
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
                with st.spinner("Folding locally..."):
                    pdb_text = fold_local_esm(sequence)
                if not pdb_text:
                    st.warning("Local folding failed, falling back to API...")
                    with st.spinner("Folding with ESMFold API..."):
                        pdb_text = fold_with_esmfold(sequence)
            else:
                with st.spinner("This may take 30-60 seconds..."):
                    pdb_text = fold_with_esmfold(sequence)
                if use_local and esm_capability == "analysis_only":
                    with st.spinner("Running local ESM-2 analysis..."):
                        analysis = analyze_local_esm(sequence)

            if pdb_text:
                st.success("Folding complete!")
                residues = parse_plddt_from_pdb(pdb_text)
                stats = compute_structure_stats(residues)
                show_stats_row(stats)

                tab_names = ["📊 pLDDT", "🔬 3D Structure", "🧬 Properties", "📐 Ramachandran", "🔩 Structural", "📥 Export"]
                if analysis and "contact_map" in analysis:
                    tab_names.insert(5, "🧠 Contact Map")
                tabs = st.tabs(tab_names)

                with tabs[0]:
                    st.plotly_chart(make_plddt_chart(residues), use_container_width=True)
                with tabs[1]:
                    show_3d_tab(pdb_text)
                with tabs[2]:
                    seq_props = show_properties_tab(sequence)
                with tabs[3]:
                    show_ramachandran_tab(pdb_text)
                with tabs[4]:
                    show_structural_analysis_tab(pdb_text, residues)

                if analysis and "contact_map" in analysis:
                    with tabs[5]:
                        cm_fig = go.Figure(data=go.Heatmap(z=analysis["contact_map"], colorscale="Blues",
                                                            hovertemplate="Res %{x} vs %{y}<br>Contact: %{z:.3f}<extra></extra>"))
                        cm_fig.update_layout(xaxis_title="Residue", yaxis_title="Residue", height=500, width=500,
                                              yaxis=dict(autorange="reversed"))
                        st.plotly_chart(cm_fig)

                with tabs[-1]:
                    st.download_button("Download PDB", pdb_text, "esmfold_prediction.pdb", "chemical/x-pdb")
                    pdf_fig = make_plddt_chart(residues)
                    pdf_bytes = generate_pdf_report(title="ESMFold Prediction", stats=stats, plddt_fig=pdf_fig,
                                                     seq_props=compute_sequence_properties(sequence))
                    st.download_button("📄 Export PDF Report", pdf_bytes, "esmfold-report.pdf", "application/pdf")
                    json_str = export_analysis_json(
                        stats=stats, seq_props=compute_sequence_properties(sequence),
                        phi_psi=calculate_phi_psi(pdb_text), disulfide=detect_disulfide_bonds(pdb_text),
                        salt_bridges=detect_salt_bridges(pdb_text), rg=calculate_radius_of_gyration(pdb_text))
                    st.download_button("📦 Export Full Analysis JSON", json_str, "esmfold-analysis.json", "application/json")
            else:
                st.error("ESMFold failed. Try a shorter sequence or try again later.")

# ═══════════════════════════════════════════════════════════════════════
# MAIN: BATCH FOLD
# ═══════════════════════════════════════════════════════════════════════

elif "📦" in mode:
    st.title("Batch Fold — Multiple Sequences")
    st.markdown("Upload a FASTA file to fold multiple sequences with ESMFold.")
    uploaded = st.file_uploader("Upload FASTA file", type=["fasta", "fa", "faa", "txt"])

    if uploaded:
        fasta_text = uploaded.read().decode("utf-8")
        sequences = parse_fasta(fasta_text)
        if len(sequences) == 0:
            st.error("No valid sequences found.")
        elif len(sequences) > 20:
            st.error(f"Too many sequences ({len(sequences)}). Maximum is 20.")
        else:
            valid_seqs, skipped = [], []
            for name, seq in sequences:
                ok, msg = validate_sequence(seq)
                (valid_seqs if ok else skipped).append((name, seq) if ok else (name, msg))

            st.info(f"**{len(valid_seqs)}** valid, **{len(skipped)}** skipped")
            if skipped:
                with st.expander("Skipped"):
                    for name, reason in skipped:
                        st.write(f"- **{name}**: {reason}")

            if valid_seqs and st.button("Fold All", type="primary", use_container_width=True):
                results = []
                progress = st.progress(0, text="Folding...")
                for i, (name, seq) in enumerate(valid_seqs):
                    progress.progress(i / len(valid_seqs), text=f"Folding {name} ({i + 1}/{len(valid_seqs)})...")
                    pdb_text = fold_with_esmfold(seq)
                    if pdb_text:
                        res = parse_plddt_from_pdb(pdb_text)
                        results.append({"name": name, "sequence": seq, "pdb": pdb_text, "residues": res, "stats": compute_structure_stats(res)})
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
                        table_rows.append({"Sequence": r["name"], "Length": s["num_residues"],
                                            "Mean pLDDT": f"{s['mean_plddt']:.1f}", "Median": f"{s['median_plddt']:.1f}",
                                            "% High": f"{s['pct_very_high']:.0f}%", "% Low": f"{s['pct_low']:.0f}%"})
                    else:
                        table_rows.append({"Sequence": r["name"], "Length": len(r["sequence"]),
                                            "Mean pLDDT": "FAILED", "Median": "-", "% High": "-", "% Low": "-"})
                st.dataframe(table_rows, use_container_width=True)

                for r in results:
                    if r["stats"]:
                        with st.expander(f"{r['name']} — pLDDT: {r['stats']['mean_plddt']:.1f}"):
                            st.plotly_chart(make_plddt_chart(r["residues"]), use_container_width=True)

                csv_buf = StringIO()
                writer = csv.DictWriter(csv_buf, fieldnames=["sequence_name", "length", "mean_plddt", "median_plddt", "pct_very_high", "pct_confident", "pct_low"])
                writer.writeheader()
                for r in results:
                    if r["stats"]:
                        s = r["stats"]
                        writer.writerow({"sequence_name": r["name"], "length": s["num_residues"],
                                          "mean_plddt": f"{s['mean_plddt']:.2f}", "median_plddt": f"{s['median_plddt']:.2f}",
                                          "pct_very_high": f"{s['pct_very_high']:.1f}", "pct_confident": f"{s['pct_confident']:.1f}",
                                          "pct_low": f"{s['pct_low']:.1f}"})
                st.download_button("📥 Download CSV", csv_buf.getvalue(), "batch_results.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════
# MAIN: COMPARE MODE
# ═══════════════════════════════════════════════════════════════════════

elif "⚖️" in mode:
    st.title("Compare Proteins")
    st.markdown("Compare two proteins side by side.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Protein A")
        input_type_a = st.selectbox("Input type", ["UniProt ID", "Paste Sequence"], key="type_a")
        val_a = st.text_input("UniProt ID", key="id_a", placeholder="e.g. P69905") if input_type_a == "UniProt ID" else st.text_area("Sequence", key="seq_a", height=100)
    with col_b:
        st.subheader("Protein B")
        input_type_b = st.selectbox("Input type", ["UniProt ID", "Paste Sequence"], key="type_b")
        val_b = st.text_input("UniProt ID", key="id_b", placeholder="e.g. P68871") if input_type_b == "UniProt ID" else st.text_area("Sequence", key="seq_b", height=100)

    if st.button("Compare", type="primary", use_container_width=True):
        def _fetch_protein(input_type, value, label):
            if input_type == "UniProt ID":
                uid = value.strip().upper()
                if not uid:
                    st.error(f"{label}: Enter a UniProt ID"); return None
                pred = fetch_alphafold_prediction(uid)
                if not pred:
                    st.error(f"No prediction for {uid}"); return None
                pdb = fetch_alphafold_pdb(pred["pdbUrl"]) if pred.get("pdbUrl") else None
                if not pdb:
                    st.error(f"PDB not found for {uid}"); return None
                res = parse_plddt_from_pdb(pdb)
                return uid, pdb, res, compute_structure_stats(res)
            else:
                seq = clean_sequence(value)
                ok, msg = validate_sequence(seq)
                if not ok:
                    st.error(f"{label}: {msg}"); return None
                pdb = fold_with_esmfold(seq)
                if not pdb:
                    st.error(f"ESMFold failed for {label}"); return None
                res = parse_plddt_from_pdb(pdb)
                return label, pdb, res, compute_structure_stats(res)

        with st.spinner("Fetching/folding..."):
            data_a = _fetch_protein(input_type_a, val_a, "Protein A")
            data_b = _fetch_protein(input_type_b, val_b, "Protein B")

        if data_a and data_b:
            label_a, pdb_a, res_a, stats_a = data_a
            label_b, pdb_b, res_b, stats_b = data_b

            # pLDDT overlay
            st.markdown("### pLDDT Comparison")
            cmp_fig = go.Figure()
            for y0, y1, c in [(90, 100, "#0053D6"), (70, 90, "#65CBF3"), (50, 70, "#FFDB13"), (0, 50, "#FF7D45")]:
                cmp_fig.add_hrect(y0=y0, y1=y1, fillcolor=c, opacity=0.08, line_width=0)
            cmp_fig.add_trace(go.Scatter(x=[r["residue_num"] for r in res_a], y=[r["plddt"] for r in res_a],
                                          mode="lines", name=label_a, line=dict(color="#0053D6", width=2)))
            cmp_fig.add_trace(go.Scatter(x=[r["residue_num"] for r in res_b], y=[r["plddt"] for r in res_b],
                                          mode="lines", name=label_b, line=dict(color="#FF7D45", width=2)))
            cmp_fig.update_layout(title="Per-Residue pLDDT Comparison", xaxis_title="Residue Number",
                                   yaxis_title="pLDDT Score", yaxis=dict(range=[0, 100]), height=400, plot_bgcolor="white")
            st.plotly_chart(cmp_fig, use_container_width=True)

            # Stats
            st.markdown("### Statistics")
            metrics = ["num_residues", "mean_plddt", "median_plddt", "min_plddt", "max_plddt", "pct_very_high", "pct_confident", "pct_low"]
            mlabels = ["Residues", "Mean pLDDT", "Median pLDDT", "Min pLDDT", "Max pLDDT", "% Very High", "% Confident", "% Low"]
            comp_rows = []
            for m, l in zip(metrics, mlabels):
                fmt = ".1f" if isinstance(stats_a[m], float) else "d"
                comp_rows.append({"Metric": l, label_a: f"{stats_a[m]:{fmt}}", label_b: f"{stats_b[m]:{fmt}}"})
            st.dataframe(comp_rows, use_container_width=True)

            # 3D
            st.markdown("### 3D Structures")
            view_mode = st.radio("View", ["Side by Side", "Overlay"], horizontal=True, key="3d_view")
            if view_mode == "Side by Side":
                sa, sb = st.columns(2)
                with sa:
                    st.caption(label_a); render_3d_structure(pdb_a)
                with sb:
                    st.caption(label_b); render_3d_structure(pdb_b)
            else:
                render_overlay_3d(pdb_a, pdb_b, label_a, label_b)

            # Structural comparison
            st.markdown("### Structural Comparison")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric(f"Rg — {label_a}", f"{calculate_radius_of_gyration(pdb_a):.1f} Å")
                st.metric(f"Disulfide bonds — {label_a}", len(detect_disulfide_bonds(pdb_a)))
            with sc2:
                st.metric(f"Rg — {label_b}", f"{calculate_radius_of_gyration(pdb_b):.1f} Å")
                st.metric(f"Disulfide bonds — {label_b}", len(detect_disulfide_bonds(pdb_b)))

# ═══════════════════════════════════════════════════════════════════════
# MAIN: UPLOAD PDB
# ═══════════════════════════════════════════════════════════════════════

elif "📂" in mode:
    st.title("Upload PDB — Analyze Your Structure")
    st.markdown("Upload a PDB file for comprehensive analysis.")
    pdb_upload = st.file_uploader("Upload PDB file", type=["pdb", "ent"])

    if pdb_upload:
        pdb_text = pdb_upload.read().decode("utf-8")
        residues = parse_plddt_from_pdb(pdb_text)
        if not residues:
            st.error("No CA atoms found in the PDB file.")
        else:
            stats = compute_structure_stats(residues)
            sequence = extract_sequence_from_pdb(pdb_text)
            st.success(f"Loaded: **{len(residues)} residues**")
            show_stats_row(stats)

            tabs = st.tabs(["📊 B-factor", "🔬 3D Structure", "🧬 Properties", "📐 Ramachandran", "🔩 Structural", "📥 Export"])
            with tabs[0]:
                fig = make_plddt_chart(residues)
                fig.update_layout(title="Per-Residue B-factor / pLDDT")
                st.plotly_chart(fig, use_container_width=True)
            with tabs[1]:
                show_3d_tab(pdb_text)
            with tabs[2]:
                if "X" not in sequence:
                    seq_props = show_properties_tab(sequence)
                else:
                    st.warning("Could not extract clean sequence from PDB.")
            with tabs[3]:
                show_ramachandran_tab(pdb_text)
            with tabs[4]:
                show_structural_analysis_tab(pdb_text, residues)
            with tabs[5]:
                st.download_button("Download PDB", pdb_text, pdb_upload.name, "chemical/x-pdb")
                pdf_fig = make_plddt_chart(residues)
                props = compute_sequence_properties(sequence) if "X" not in sequence else None
                pdf_bytes = generate_pdf_report(title=f"Uploaded — {pdb_upload.name}", stats=stats, plddt_fig=pdf_fig, seq_props=props)
                st.download_button("📄 Export PDF", pdf_bytes, f"{pdb_upload.name}-report.pdf", "application/pdf")
                json_str = export_analysis_json(
                    stats=stats, seq_props=props, phi_psi=calculate_phi_psi(pdb_text),
                    disulfide=detect_disulfide_bonds(pdb_text), salt_bridges=detect_salt_bridges(pdb_text),
                    rg=calculate_radius_of_gyration(pdb_text))
                st.download_button("📦 Export JSON", json_str, f"{pdb_upload.name}-analysis.json", "application/json")

# Footer
st.markdown("---")
st.caption("Data: [AlphaFold DB](https://alphafold.ebi.ac.uk/) (DeepMind/EBI) | [ESMFold](https://esmatlas.com/) (Meta AI) | [UniProt](https://www.uniprot.org/)")
