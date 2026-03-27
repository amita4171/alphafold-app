"""Export utilities: PDF reports and JSON analysis export."""
from __future__ import annotations

import datetime
import json
from io import BytesIO

import plotly.graph_objects as go


def generate_pdf_report(title: str, stats: dict, plddt_fig: go.Figure,
                        uniprot_id: str | None = None, domains: list[dict] | None = None,
                        seq_props: dict | None = None) -> bytes:
    """Generate PDF report as bytes for st.download_button."""
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
                  ["Mean pLDDT", f"{stats['mean_plddt']:.1f}"],
                  ["Median pLDDT", f"{stats['median_plddt']:.1f}"],
                  ["Min pLDDT", f"{stats['min_plddt']:.1f}"],
                  ["Max pLDDT", f"{stats['max_plddt']:.1f}"],
                  ["% Very High (>90)", f"{stats['pct_very_high']:.1f}%"],
                  ["% Confident (>70)", f"{stats['pct_confident']:.1f}%"],
                  ["% Low (<=50)", f"{stats['pct_low']:.1f}%"]]
    if seq_props:
        table_data += [["Molecular Weight", f"{seq_props['molecular_weight']:.1f} Da"],
                       ["Isoelectric Point", f"{seq_props['isoelectric_point']:.2f}"],
                       ["GRAVY", f"{seq_props['gravy']:.3f}"],
                       ["Instability Index", f"{seq_props['instability_index']:.1f}"],
                       ["Ext. Coeff (reduced)", f"{seq_props['extinction_coeff'][0]} M-1cm-1"],
                       ["Ext. Coeff (oxidized)", f"{seq_props['extinction_coeff'][1]} M-1cm-1"]]
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
    elements.append(Paragraph("<i>Full analysis available in the app.</i>", styles["Normal"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("Data: AlphaFold DB | ESMFold | UniProt", styles["Normal"]))
    doc.build(elements)
    return buf.getvalue()


def export_analysis_json(stats: dict, seq_props: dict | None = None, domains: list[dict] | None = None,
                         phi_psi: list | None = None, disulfide: list | None = None,
                         salt_bridges: list | None = None, rg: float | None = None,
                         annotations: dict | None = None, hbonds: list | None = None,
                         contact_order: float | None = None, disordered: list | None = None) -> str:
    """Export comprehensive analysis as JSON string."""
    data: dict = {"structure_stats": stats}
    if seq_props:
        sp = {k: v for k, v in seq_props.items() if k not in ("aa_counts", "aa_percent")}
        sp["extinction_coeff_reduced"] = seq_props["extinction_coeff"][0]
        sp["extinction_coeff_oxidized"] = seq_props["extinction_coeff"][1]
        data["sequence_properties"] = sp
    if domains:
        data["domains"] = domains
    if phi_psi is not None:
        data["ramachandran_angles_count"] = len(phi_psi)
    if disulfide:
        data["disulfide_bonds"] = disulfide
    if salt_bridges:
        data["salt_bridges"] = salt_bridges
    if rg is not None:
        data["radius_of_gyration"] = rg
    if hbonds:
        data["hydrogen_bonds_count"] = len(hbonds)
    if contact_order is not None:
        data["contact_order"] = contact_order
    if disordered:
        data["disordered_regions"] = disordered
    if annotations:
        data["uniprot_annotations"] = annotations
    return json.dumps(data, indent=2, default=str)
