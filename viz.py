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


# ── 19. Per-residue SASA bar chart ────────────────────────────────────
def make_sasa_plot(sasa_data: list[dict]) -> go.Figure:
    """Per-residue SASA bar chart colored by exposure level."""
    if not sasa_data:
        fig = go.Figure()
        fig.update_layout(title="SASA — No data available")
        return fig
    residue_nums = [d["residue_num"] for d in sasa_data]
    relative_sasa = [d["relative_sasa"] for d in sasa_data]
    residue_names = [d["residue_name"] for d in sasa_data]
    colors = []
    for val in relative_sasa:
        if val > 0.5:
            colors.append("#0053D6")   # exposed — blue
        elif val < 0.2:
            colors.append("#FF7D45")   # buried — orange
        else:
            colors.append("#999999")   # intermediate — grey
    fig = go.Figure(go.Bar(
        x=residue_nums, y=relative_sasa, marker_color=colors,
        hovertemplate="Residue %{x} (%{customdata})<br>Relative SASA: %{y:.3f}<extra></extra>",
        customdata=residue_names,
    ))
    fig.update_layout(
        title="Per-Residue Solvent Accessible Surface Area",
        xaxis_title="Residue Number", yaxis_title="Relative SASA (0-1)",
        yaxis=dict(range=[0, 1]),
        height=400, plot_bgcolor="white",
        margin=dict(t=50, b=50, l=60, r=20),
        annotations=[
            dict(x=1.02, y=0.9, xref="paper", yref="paper", text="Exposed (>0.5)",
                 font=dict(color="#0053D6", size=10), showarrow=False, xanchor="left"),
            dict(x=1.02, y=0.7, xref="paper", yref="paper", text="Intermediate",
                 font=dict(color="#999999", size=10), showarrow=False, xanchor="left"),
            dict(x=1.02, y=0.5, xref="paper", yref="paper", text="Buried (<0.2)",
                 font=dict(color="#FF7D45", size=10), showarrow=False, xanchor="left"),
        ],
    )
    return fig


# ── 20. NMA mean-square fluctuations ──────────────────────────────────
def make_nma_fluctuations_plot(sqflucts: list[float], residue_nums: list[int] | None = None) -> go.Figure:
    """Line plot of mean square fluctuations from Normal Mode Analysis.

    Peaks exceeding 2x the mean are highlighted in red.
    """
    if not sqflucts:
        fig = go.Figure()
        fig.update_layout(title="NMA Fluctuations — No data available")
        return fig
    if residue_nums is None:
        residue_nums = list(range(1, len(sqflucts) + 1))
    mean_val = float(np.mean(sqflucts))
    threshold = 2.0 * mean_val
    # Base line trace
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=residue_nums, y=sqflucts, mode="lines",
        line=dict(color="#0053D6", width=1.5),
        hovertemplate="Residue %{x}<br>Fluctuation: %{y:.4f}<extra></extra>",
        name="Fluctuation",
    ))
    # Highlight peaks above threshold
    peak_x = [r for r, f in zip(residue_nums, sqflucts) if f > threshold]
    peak_y = [f for f in sqflucts if f > threshold]
    if peak_x:
        fig.add_trace(go.Scatter(
            x=peak_x, y=peak_y, mode="markers",
            marker=dict(color="red", size=6, symbol="circle"),
            hovertemplate="Residue %{x}<br>Fluctuation: %{y:.4f} (peak)<extra></extra>",
            name="Peak (>2x mean)",
        ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text=f"2x mean ({threshold:.4f})", annotation_position="top right")
    fig.update_layout(
        title="NMA Mean Square Fluctuations",
        xaxis_title="Residue Number", yaxis_title="Mean Square Fluctuation",
        height=400, plot_bgcolor="white",
    )
    return fig


# ── 21. NMA cross-correlation heatmap ─────────────────────────────────
def make_nma_cross_correlation(cross_corr: "np.ndarray") -> go.Figure:
    """Heatmap of inter-residue cross-correlations from NMA."""
    fig = go.Figure(data=go.Heatmap(
        z=cross_corr,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation"),
        hovertemplate="Residue %{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="NMA Cross-Correlation Map",
        xaxis_title="Residue", yaxis_title="Residue",
        height=550, width=550,
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ── 22. GNM B-factor comparison ───────────────────────────────────────
def make_gnm_bfactor_comparison(
    experimental: list[dict], predicted: list[tuple[int, float]]
) -> go.Figure:
    """Overlay plot of experimental vs GNM-predicted B-factors.

    *experimental* — list of dicts with keys ``residue_num`` and ``bfactor``.
    *predicted* — list of (residue_num, predicted_bfactor) tuples.
    """
    fig = go.Figure()
    if experimental:
        exp_nums = [d["residue_num"] for d in experimental]
        exp_vals = [d.get("bfactor", d.get("plddt", 0)) for d in experimental]
        fig.add_trace(go.Scatter(
            x=exp_nums, y=exp_vals, mode="lines",
            line=dict(color="#0053D6", width=1.5),
            name="Experimental B-factor",
            hovertemplate="Residue %{x}<br>B-factor: %{y:.2f}<extra></extra>",
        ))
    if predicted:
        pred_nums = [p[0] for p in predicted]
        pred_vals = [p[1] for p in predicted]
        fig.add_trace(go.Scatter(
            x=pred_nums, y=pred_vals, mode="lines",
            line=dict(color="#FF7D45", width=1.5),
            name="GNM Predicted",
            hovertemplate="Residue %{x}<br>Predicted: %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        title="Experimental vs GNM-Predicted B-factors",
        xaxis_title="Residue Number", yaxis_title="B-factor",
        height=400, plot_bgcolor="white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


# ── 23. Enhanced protein-protein interaction network ──────────────────
def make_ppi_network(interactions: list[dict]) -> go.Figure:
    """Enhanced PPI network using networkx spring layout.

    Each interaction dict should have keys: ``protein_a``, ``protein_b``,
    ``combined_score``, and optionally ``score``.
    """
    import networkx as nx

    if not interactions:
        fig = go.Figure()
        fig.update_layout(title="PPI Network — No interactions found")
        return fig

    G = nx.Graph()
    for ix in interactions:
        G.add_edge(
            ix["protein_a"], ix["protein_b"],
            weight=ix.get("combined_score", ix.get("score", 0.5)),
        )

    pos = nx.spring_layout(G, seed=42)

    # Edges — width scaled by combined score
    edge_traces: list[go.Scatter] = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get("weight", 0.5)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(color="rgba(180,180,180,0.6)", width=1 + 3 * weight),
            hoverinfo="none", showlegend=False,
        ))

    # Node properties
    degrees = dict(G.degree())
    proteins = list(G.nodes())
    node_x = [pos[p][0] for p in proteins]
    node_y = [pos[p][1] for p in proteins]
    node_sizes = [10 + 4 * degrees[p] for p in proteins]

    # Color by average interaction score
    node_scores: list[float] = []
    for p in proteins:
        edges = G.edges(p, data=True)
        avg_score = float(np.mean([d.get("weight", 0.5) for _, _, d in edges])) if edges else 0.5
        node_scores.append(avg_score)

    fig = go.Figure()
    for trace in edge_traces:
        fig.add_trace(trace)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=proteins, textposition="top center",
        textfont=dict(size=10),
        marker=dict(
            size=node_sizes,
            color=node_scores,
            colorscale="Viridis",
            colorbar=dict(title="Avg Score"),
            line=dict(width=1, color="white"),
        ),
        hovertemplate="%{text}<br>Degree: %{customdata[0]}<br>Avg Score: %{customdata[1]:.3f}<extra></extra>",
        customdata=list(zip([degrees[p] for p in proteins], node_scores)),
        showlegend=False,
    ))

    fig.update_layout(
        title="Protein-Protein Interaction Network",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600, width=600, plot_bgcolor="white",
    )
    return fig


# ── 24. Sequence logo ─────────────────────────────────────────────────
def make_sequence_logo(logo_data: dict) -> "matplotlib.figure.Figure":
    """Sequence logo from per-position amino acid frequency matrix.

    *logo_data* must contain key ``matrix`` — a pandas DataFrame of
    per-position amino acid frequencies.
    """
    import logomaker
    import matplotlib
    import matplotlib.pyplot as plt

    matrix = logo_data["matrix"]
    fig, ax = plt.subplots(figsize=(max(10, len(matrix) * 0.3), 3))
    logomaker.Logo(matrix, ax=ax)
    ax.set_xlabel("Position")
    ax.set_ylabel("Frequency")
    ax.set_title("Sequence Logo")
    plt.tight_layout()
    return fig


# ── 25. Phylogenetic tree (UPGMA dendrogram) ─────────────────────────
def make_phylogenetic_tree(tree: dict) -> "matplotlib.figure.Figure":
    """Draw a UPGMA tree as a horizontal dendrogram.

    *tree* is a nested dict: ``{name, distance, children: [...]}``.
    Leaf nodes have ``name`` set and no ``children`` (or empty list).
    """
    import matplotlib
    import matplotlib.pyplot as plt

    def _count_leaves(node: dict) -> int:
        children = node.get("children", [])
        if not children:
            return 1
        return sum(_count_leaves(c) for c in children)

    def _draw(node: dict, ax, x: float, y_range: tuple[float, float]) -> float:
        """Recursively draw the dendrogram. Returns the y-center of this node."""
        children = node.get("children", [])
        dist = node.get("distance", 0.0)

        if not children:
            # Leaf
            y_center = (y_range[0] + y_range[1]) / 2.0
            ax.text(x + 0.002, y_center, f"  {node.get('name', '')}", va="center", ha="left", fontsize=9)
            return y_center

        # Partition y-range among children proportionally to leaf count
        total_leaves = _count_leaves(node)
        child_centers: list[float] = []
        y_start = y_range[0]
        for child in children:
            child_leaves = _count_leaves(child)
            y_end = y_start + (y_range[1] - y_range[0]) * child_leaves / total_leaves
            child_x = x + dist - child.get("distance", 0.0)
            center = _draw(child, ax, child_x, (y_start, y_end))
            child_centers.append(center)
            # Horizontal line from child to this node's x
            ax.plot([child_x, x + dist], [center, center], color="#0053D6", linewidth=1.2)
            y_start = y_end

        # Vertical connector line
        ax.plot([x + dist, x + dist], [min(child_centers), max(child_centers)],
                color="#0053D6", linewidth=1.2)

        return (min(child_centers) + max(child_centers)) / 2.0

    n_leaves = _count_leaves(tree)
    fig, ax = plt.subplots(figsize=(8, max(4, n_leaves * 0.4)))
    _draw(tree, ax, x=0.0, y_range=(0.0, float(n_leaves)))
    ax.set_xlabel("Distance")
    ax.set_title("UPGMA Phylogenetic Tree")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    return fig


# ── 26. Topology diagram ──────────────────────────────────────────────
def make_topology_diagram(topology: list[dict]) -> "matplotlib.figure.Figure":
    """2D topology diagram: helices (red rectangles), strands (blue arrows),
    coils (grey lines). Each element dict has keys: ``type``, ``start``, ``end``.
    """
    import matplotlib
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    if not topology:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No topology data", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        return fig

    fig, ax = plt.subplots(figsize=(max(10, len(topology) * 1.5), 4))
    y_center = 0.5
    x_cursor = 0.0
    gap = 0.3

    for elem in topology:
        elem_type = elem.get("type", "coil").lower()
        start = elem.get("start", 0)
        end = elem.get("end", 0)
        length = max(end - start + 1, 1)
        width = length * 0.1  # scale factor

        if elem_type == "helix":
            rect = patches.FancyBboxPatch(
                (x_cursor, y_center - 0.15), width, 0.3,
                boxstyle="round,pad=0.02",
                facecolor="#D9534F", edgecolor="#B52B27", linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(x_cursor + width / 2, y_center, f"H\n{start}-{end}",
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        elif elem_type == "strand":
            # Arrow shape
            arrow = patches.FancyArrow(
                x_cursor, y_center, width, 0,
                width=0.25, head_width=0.35, head_length=0.15,
                fc="#337AB7", ec="#2A6496", linewidth=1.5,
            )
            ax.add_patch(arrow)
            ax.text(x_cursor + width / 2, y_center, f"E\n{start}-{end}",
                    ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        else:
            # Coil — simple line
            ax.plot([x_cursor, x_cursor + width], [y_center, y_center],
                    color="#999999", linewidth=2, linestyle="-")
            ax.text(x_cursor + width / 2, y_center + 0.2, f"{start}-{end}",
                    ha="center", va="center", fontsize=6, color="#666666")

        x_cursor += width + gap

    ax.set_xlim(-0.5, x_cursor + 0.5)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.set_title("Protein Topology Diagram")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    return fig


# ── 27. RMSD / TM-align summary ──────────────────────────────────────
def make_rmsd_summary(rmsd_data: dict) -> str:
    """Format TM-align results as a readable text block for ``st.code`` display."""
    lines: list[str] = [
        "=" * 50,
        "  TM-align Structural Alignment Summary",
        "=" * 50,
        "",
    ]
    if "rmsd" in rmsd_data:
        lines.append(f"  RMSD:              {rmsd_data['rmsd']:.3f} A")
    if "tm_score" in rmsd_data:
        tm = rmsd_data["tm_score"]
        if isinstance(tm, (list, tuple)):
            lines.append(f"  TM-score (chain 1): {tm[0]:.4f}")
            if len(tm) > 1:
                lines.append(f"  TM-score (chain 2): {tm[1]:.4f}")
        else:
            lines.append(f"  TM-score:          {tm:.4f}")
    if "aligned_length" in rmsd_data:
        lines.append(f"  Aligned length:    {rmsd_data['aligned_length']}")
    if "seq_identity" in rmsd_data:
        lines.append(f"  Sequence identity: {rmsd_data['seq_identity']:.1%}")
    if "n_residues_a" in rmsd_data:
        lines.append(f"  Residues (struct A): {rmsd_data['n_residues_a']}")
    if "n_residues_b" in rmsd_data:
        lines.append(f"  Residues (struct B): {rmsd_data['n_residues_b']}")

    lines.append("")
    lines.append("=" * 50)

    # Interpretation
    tm_val = rmsd_data.get("tm_score")
    if tm_val is not None:
        if isinstance(tm_val, (list, tuple)):
            tm_val = max(tm_val)
        if tm_val >= 0.5:
            lines.append("  Interpretation: Same fold (TM-score >= 0.5)")
        elif tm_val >= 0.17:
            lines.append("  Interpretation: Possible structural similarity")
        else:
            lines.append("  Interpretation: Different folds (TM-score < 0.17)")

    return "\n".join(lines)
