from __future__ import annotations

from collections import Counter

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Constants ────────────────────────────────────────────────────────────
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


# ── 1. pLDDT color helper ───────────────────────────────────────────────
def plddt_color(val: float) -> str:
    if val > 90: return "#0053D6"
    elif val > 70: return "#65CBF3"
    elif val > 50: return "#FFDB13"
    else: return "#FF7D45"


# ── 2. pLDDT chart ──────────────────────────────────────────────────────
def make_plddt_chart(residues: list[dict], domains: list[dict] | None = None) -> go.Figure:
    nums = [r["residue_num"] for r in residues]
    scores = [r["plddt"] for r in residues]
    colors = [plddt_color(s) for s in scores]
    fig = go.Figure()
    fig.add_hrect(y0=90, y1=100, fillcolor="#0053D6", opacity=0.08, line_width=0)
    fig.add_hrect(y0=70, y1=90, fillcolor="#65CBF3", opacity=0.08, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor="#FFDB13", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=50, fillcolor="#FF7D45", opacity=0.08, line_width=0)
    fig.add_trace(go.Bar(x=nums, y=scores, marker_color=colors,
                         hovertemplate="Residue %{x}<br>pLDDT: %{y:.1f}<extra></extra>"))
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


# ── 3. PAE heatmap ──────────────────────────────────────────────────────
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


# ── 4. Ramachandran plot ─────────────────────────────────────────────────
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


# ── 5. Amino acid composition ───────────────────────────────────────────
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


# ── 6. Hydrophobicity plot ──────────────────────────────────────────────
def _compute_hydrophobicity(sequence: str, window: int = 9) -> list[tuple[int, float]]:
    if len(sequence) < window:
        return []
    half = window // 2
    return [(i + 1, sum(KD_SCALE.get(sequence[j], 0) for j in range(i - half, i + half + 1)) / window)
            for i in range(half, len(sequence) - half)]


def make_hydrophobicity_plot(sequence: str, window: int = 9) -> go.Figure:
    values = _compute_hydrophobicity(sequence, window)
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


# ── 7. Charge vs pH ─────────────────────────────────────────────────────
def make_charge_at_ph_plot(sequence: str) -> go.Figure:
    from analysis import compute_charge_at_ph
    data = compute_charge_at_ph(sequence)
    phs, charges = zip(*data)
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    fig.add_trace(go.Scatter(x=list(phs), y=list(charges), mode="lines", line=dict(color="#0053D6", width=2),
                              hovertemplate="pH %{x:.1f}<br>Charge: %{y:.1f}<extra></extra>"))
    fig.update_layout(title="Net Charge vs pH", xaxis_title="pH", yaxis_title="Net Charge",
                      height=350, plot_bgcolor="white")
    return fig


# ── 8. Distance map ─────────────────────────────────────────────────────
def make_distance_map(pdb_text: str) -> go.Figure:
    from analysis import calculate_distance_map
    dist_matrix, res_nums = calculate_distance_map(pdb_text)
    fig = go.Figure(data=go.Heatmap(z=dist_matrix, x=res_nums, y=res_nums, colorscale="Viridis_r",
                                     colorbar=dict(title="Distance (Å)"),
                                     hovertemplate="Res %{x} vs %{y}<br>Distance: %{z:.1f} Å<extra></extra>"))
    fig.update_layout(title="CA-CA Distance Map", xaxis_title="Residue", yaxis_title="Residue",
                      height=550, width=550, yaxis=dict(autorange="reversed"))
    return fig


# ── 9. B-factor / pLDDT histogram ───────────────────────────────────────
def make_bfactor_histogram(residues: list[dict]) -> go.Figure:
    scores = [r["plddt"] for r in residues]
    fig = go.Figure(go.Histogram(x=scores, nbinsx=30,
                                  marker_color="#0053D6", opacity=0.7,
                                  hovertemplate="pLDDT: %{x:.0f}<br>Count: %{y}<extra></extra>"))
    fig.add_vline(x=np.mean(scores), line_dash="dash", line_color="red", annotation_text=f"Mean: {np.mean(scores):.1f}")
    fig.update_layout(title="pLDDT / B-factor Distribution", xaxis_title="pLDDT Score", yaxis_title="Count",
                      height=350, plot_bgcolor="white")
    return fig


# ── 10. 3D structure viewer ─────────────────────────────────────────────
def render_3d_structure(pdb_text: str, style: str = "cartoon", color_scheme: str = "pLDDT"):
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


# ── 11. Overlay two structures ──────────────────────────────────────────
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


# ── 12. Flexibility plot ────────────────────────────────────────────────
def make_flexibility_plot(sequence: str) -> go.Figure:
    from analysis import compute_flexibility
    scores = compute_flexibility(sequence)
    if not scores:
        fig = go.Figure()
        fig.update_layout(title="Flexibility — No data available")
        return fig
    positions = list(range(1, len(scores) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=positions, y=scores, mode="lines", line=dict(color="#0053D6", width=1.5),
                              hovertemplate="Residue %{x}<br>Flexibility: %{y:.3f}<extra></extra>"))
    fig.update_layout(title="Per-Residue Flexibility", xaxis_title="Residue Position", yaxis_title="Flexibility Score",
                      height=350, plot_bgcolor="white")
    return fig


# ── 13. Sequence complexity (Shannon entropy) ───────────────────────────
def make_complexity_plot(sequence: str) -> go.Figure:
    from analysis import compute_sequence_complexity
    entropy_values = compute_sequence_complexity(sequence)
    if not entropy_values:
        fig = go.Figure()
        fig.update_layout(title="Sequence Complexity — No data available")
        return fig
    positions = list(range(1, len(entropy_values) + 1))
    fig = go.Figure()
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", opacity=0.6,
                  annotation_text="Low-complexity threshold", annotation_position="top left")
    fig.add_trace(go.Scatter(x=positions, y=entropy_values, mode="lines", line=dict(color="#0053D6", width=1.5),
                              hovertemplate="Residue %{x}<br>Entropy: %{y:.2f}<extra></extra>"))
    fig.update_layout(title="Sequence Complexity (Shannon Entropy)", xaxis_title="Residue Position",
                      yaxis_title="Entropy (bits)", height=350, plot_bgcolor="white")
    return fig


# ── 14. Burial plot ─────────────────────────────────────────────────────
def make_burial_plot(burial_data: list[dict]) -> go.Figure:
    if not burial_data:
        fig = go.Figure()
        fig.update_layout(title="Burial Analysis — No data available")
        return fig
    positions = [d["residue_num"] for d in burial_data]
    scores = [d["burial_score"] for d in burial_data]
    colors = ["#FF7D45" if s > 0.5 else "#0053D6" for s in scores]
    fig = go.Figure(go.Bar(x=positions, y=scores, marker_color=colors,
                            hovertemplate="Residue %{x}<br>Burial: %{y:.2f}<extra></extra>"))
    fig.update_layout(title="Per-Residue Burial Score", xaxis_title="Residue Number",
                      yaxis_title="Burial Score", height=400, plot_bgcolor="white",
                      annotations=[
                          dict(x=1.02, y=0.9, xref="paper", yref="paper", text="Buried", font=dict(color="#FF7D45", size=10), showarrow=False, xanchor="left"),
                          dict(x=1.02, y=0.7, xref="paper", yref="paper", text="Exposed", font=dict(color="#0053D6", size=10), showarrow=False, xanchor="left"),
                      ])
    return fig


# ── 15. Disorder plot ───────────────────────────────────────────────────
def make_disorder_plot(residues: list[dict], disordered_regions: list[dict]) -> go.Figure:
    nums = [r["residue_num"] for r in residues]
    scores = [r["plddt"] for r in residues]
    fig = go.Figure()
    for region in disordered_regions:
        fig.add_vrect(x0=region["start"] - 0.5, x1=region["end"] + 0.5,
                      fillcolor="red", opacity=0.12, line_width=0,
                      annotation_text="disordered", annotation_position="top left",
                      annotation_font_size=9, annotation_font_color="red")
    fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.6,
                  annotation_text="Disorder threshold", annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=nums, y=scores, mode="lines", line=dict(color="#0053D6", width=1.5),
                              hovertemplate="Residue %{x}<br>pLDDT: %{y:.1f}<extra></extra>"))
    fig.update_layout(title="Disorder Prediction (pLDDT-based)", xaxis_title="Residue Number",
                      yaxis_title="pLDDT Score", yaxis=dict(range=[0, 100]),
                      height=400, plot_bgcolor="white")
    return fig


# ── 16. Hydrogen bond contact map ───────────────────────────────────────
def make_hydrogen_bond_plot(hbonds: list[dict], num_residues: int) -> go.Figure:
    if not hbonds:
        fig = go.Figure()
        fig.update_layout(title="Hydrogen Bonds — None detected")
        return fig
    donors = [hb["donor_res"] for hb in hbonds]
    acceptors = [hb["acceptor_res"] for hb in hbonds]
    distances = [hb.get("distance", 0) for hb in hbonds]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=donors, y=acceptors, mode="markers",
                              marker=dict(size=5, color=distances, colorscale="Blues_r",
                                          colorbar=dict(title="Dist (Å)"), opacity=0.7),
                              hovertemplate="Donor: %{x}<br>Acceptor: %{y}<br>Dist: %{marker.color:.2f} Å<extra></extra>"))
    # diagonal reference
    fig.add_trace(go.Scatter(x=[1, num_residues], y=[1, num_residues], mode="lines",
                              line=dict(color="grey", dash="dot", width=0.5), showlegend=False))
    fig.update_layout(title="Hydrogen Bond Contact Map", xaxis_title="Donor Residue",
                      yaxis_title="Acceptor Residue", height=500, width=500,
                      xaxis=dict(range=[1, num_residues]), yaxis=dict(range=[1, num_residues], autorange="reversed"),
                      plot_bgcolor="white")
    return fig


# ── 17. Alignment visualization ─────────────────────────────────────────
# BLOSUM62 positive-score pairs for similarity detection
_BLOSUM62_SIMILAR = {
    "A": "AS", "R": "RKQ", "N": "NDS", "D": "DNE", "C": "C",
    "Q": "QRK", "E": "EDZ", "G": "G", "H": "HNY", "I": "ILV",
    "L": "LIM", "K": "KRQ", "M": "MLI", "F": "FYW", "P": "P",
    "S": "STA", "T": "TS", "W": "WYF", "Y": "YFW", "V": "VIL",
}


def make_alignment_viz(alignment: dict) -> str:
    aligned_a = alignment.get("aligned_a", "")
    aligned_b = alignment.get("aligned_b", "")
    match_chars = []
    for a, b in zip(aligned_a, aligned_b):
        if a == "-" or b == "-":
            match_chars.append(" ")
        elif a == b:
            match_chars.append("|")
        elif b in _BLOSUM62_SIMILAR.get(a, ""):
            match_chars.append(":")
        else:
            match_chars.append(".")
    match_line = "".join(match_chars)
    # Format in blocks of 60
    lines: list[str] = []
    block = 60
    for i in range(0, len(aligned_a), block):
        lines.append(f"Seq A  {aligned_a[i:i+block]}")
        lines.append(f"       {match_line[i:i+block]}")
        lines.append(f"Seq B  {aligned_b[i:i+block]}")
        lines.append("")
    return "\n".join(lines)


# ── 18. Protein interaction network ─────────────────────────────────────
def make_interaction_network_plot(interactions: list[dict]) -> go.Figure:
    if not interactions:
        fig = go.Figure()
        fig.update_layout(title="Interaction Network — No interactions found")
        return fig
    # Collect unique proteins and build adjacency
    proteins: set[str] = set()
    for ix in interactions:
        proteins.add(ix["protein_a"])
        proteins.add(ix["protein_b"])
    protein_list = sorted(proteins)
    n = len(protein_list)
    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {p: (float(np.cos(a)), float(np.sin(a))) for p, a in zip(protein_list, angles)}
    # Edges
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for ix in interactions:
        x0, y0 = pos[ix["protein_a"]]
        x1, y1 = pos[ix["protein_b"]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                              line=dict(color="#CCCCCC", width=1), hoverinfo="none", showlegend=False))
    node_x = [pos[p][0] for p in protein_list]
    node_y = [pos[p][1] for p in protein_list]
    # Node degree for sizing
    degree = {p: 0 for p in protein_list}
    for ix in interactions:
        degree[ix["protein_a"]] += 1
        degree[ix["protein_b"]] += 1
    sizes = [8 + 3 * degree[p] for p in protein_list]
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=protein_list, textposition="top center",
                              marker=dict(size=sizes, color="#0053D6", line=dict(width=1, color="white")),
                              hovertemplate="%{text}<br>Connections: %{marker.size}<extra></extra>",
                              showlegend=False))
    fig.update_layout(title="Protein Interaction Network (STRING)", showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      height=550, width=550, plot_bgcolor="white")
    return fig
