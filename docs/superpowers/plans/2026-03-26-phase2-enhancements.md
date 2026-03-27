# Phase 2 Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add batch fold, sequence comparison, domain annotation, PDF export, and optional local ESM-2 to the AlphaFold Explorer Streamlit app.

**Architecture:** All changes go into the existing single-file `app.py`. New helpers are added to the helpers section, sidebar gets 2 new modes, existing modes get additive features (domain overlay, PDF export, local ESM toggle).

**Tech Stack:** Streamlit, Plotly, reportlab, kaleido, requests, numpy, optional torch+esm

---

### Task 1: Update dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add reportlab and kaleido to requirements.txt**

```
reportlab>=4.0
kaleido>=0.2.1
```

- [ ] **Step 2: Install new dependencies**

Run: `pip install reportlab kaleido --break-system-packages`

- [ ] **Step 3: Verify imports**

Run: `python3 -c "import reportlab; import kaleido; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "feat: add reportlab and kaleido dependencies for PDF export"
```

---

### Task 2: Add new helper functions (parse_fasta, fetch_uniprot_domains)

**Files:**
- Modify: `app.py` (helpers section, lines 33-249)

- [ ] **Step 1: Add parse_fasta helper after clean_sequence**

```python
def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into list of (name, sequence) tuples."""
    sequences = []
    current_name = None
    current_seq_lines = []
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
```

- [ ] **Step 2: Add fetch_uniprot_domains helper after search_uniprot**

```python
DOMAIN_FEATURE_TYPES = {"Domain", "Region", "Motif", "Zinc finger", "DNA binding"}

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
```

- [ ] **Step 3: Verify helpers parse correctly**

Run: `python3 -c "from app import parse_fasta, fetch_uniprot_domains; print(parse_fasta('>seq1\nMKTAYI\n>seq2\nACDEFG')); print(fetch_uniprot_domains('P69905'))"`

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add parse_fasta and fetch_uniprot_domains helpers"
```

---

### Task 3: Add domain overlay to make_plddt_chart

**Files:**
- Modify: `app.py` — `make_plddt_chart` function (line 158)

- [ ] **Step 1: Add optional domains parameter to make_plddt_chart**

Update signature to `make_plddt_chart(residues: list[dict], domains: list[dict] | None = None)` and add domain overlay logic after the bar trace:

```python
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
```

- [ ] **Step 2: Verify chart still renders without domains**

Run: `python3 -c "from app import parse_plddt_from_pdb, make_plddt_chart, fetch_alphafold_prediction, fetch_alphafold_pdb; pred = fetch_alphafold_prediction('P69905'); pdb = fetch_alphafold_pdb(pred); res = parse_plddt_from_pdb(pdb); fig = make_plddt_chart(res); print('OK:', len(fig.data), 'traces')"`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add optional domain annotation overlay to pLDDT chart"
```

---

### Task 4: Add PDF report generation helper

**Files:**
- Modify: `app.py` — add `generate_pdf_report` after `compute_structure_stats`

- [ ] **Step 1: Add generate_pdf_report helper**

```python
def generate_pdf_report(
    title: str,
    stats: dict,
    plddt_fig: go.Figure,
    uniprot_id: str | None = None,
    domains: list[dict] | None = None,
) -> bytes:
    """Generate PDF report as bytes for st.download_button."""
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    import datetime

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph(f"AlphaFold Explorer Report", styles["Title"]))
    elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
    if uniprot_id:
        elements.append(Paragraph(f"UniProt: {uniprot_id}", styles["Normal"]))
    elements.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 0.3*inch))

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

    t = Table(table_data, colWidths=[3*inch, 3*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0053D6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4FF")]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3*inch))

    # pLDDT chart as image
    try:
        img_bytes = plddt_fig.to_image(format="png", width=900, height=400, scale=2)
        img_buf = BytesIO(img_bytes)
        elements.append(Paragraph("Per-Residue pLDDT Confidence", styles["Heading3"]))
        elements.append(RLImage(img_buf, width=6.5*inch, height=2.9*inch))
    except Exception:
        elements.append(Paragraph("<i>Chart image unavailable (install kaleido for chart export)</i>", styles["Normal"]))

    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("<i>3D structure visualization available in the app.</i>", styles["Normal"]))

    # Footer
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph(
        "Data: AlphaFold DB (DeepMind/EBI) | ESMFold (Meta AI) | UniProt",
        styles["Normal"],
    ))

    doc.build(elements)
    return buf.getvalue()
```

- [ ] **Step 2: Verify PDF generation**

Run: `python3 -c "from app import *; pred = fetch_alphafold_prediction('P69905'); pdb = fetch_alphafold_pdb(pred); res = parse_plddt_from_pdb(pdb); stats = compute_structure_stats(res); fig = make_plddt_chart(res); pdf = generate_pdf_report('Hemoglobin alpha', stats, fig, 'P69905'); print(f'PDF OK: {len(pdf)} bytes')"`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add PDF report generation helper"
```

---

### Task 5: Add local ESM-2 helpers

**Files:**
- Modify: `app.py` — add after `generate_pdf_report`

- [ ] **Step 1: Add ESM capability detection and analysis helpers**

```python
# ── Optional Local ESM-2 ───────────────────────────────────────────────
try:
    import torch
    import esm as esm_module
    LOCAL_ESM_AVAILABLE = True
except ImportError:
    LOCAL_ESM_AVAILABLE = False


def check_local_esm_capabilities() -> str:
    """Returns 'full_fold', 'analysis_only', or 'unavailable'."""
    if not LOCAL_ESM_AVAILABLE:
        return "unavailable"
    try:
        from esm.pretrained import esmfold_v1
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
```

- [ ] **Step 2: Verify detection works (should be 'unavailable' without torch)**

Run: `python3 -c "from app import check_local_esm_capabilities; print(check_local_esm_capabilities())"`
Expected: `unavailable`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add optional local ESM-2 analysis helpers"
```

---

### Task 6: Update sidebar with new modes

**Files:**
- Modify: `app.py` — sidebar section (lines 252-277)

- [ ] **Step 1: Update sidebar radio to include all 4 modes**

Replace the existing radio with:
```python
mode = st.sidebar.radio(
    "Mode",
    ["🔍 Lookup (AlphaFold DB)", "🧪 Fold (ESMFold)", "📦 Batch Fold", "⚖️ Compare"],
    index=0,
)
```

Update the About text to include new modes:
```python
st.sidebar.markdown("""
**About**

**Lookup mode**: Search the AlphaFold DB (200M+ structures) by UniProt ID. Includes domain annotations.

**Fold mode**: Fold a novel sequence using ESMFold (max 400 residues). Optional local ESM-2 analysis.

**Batch Fold**: Upload a FASTA file and fold multiple sequences. Export stats as CSV.

**Compare**: Compare two proteins side by side — pLDDT charts and 3D structures.

All modes include pLDDT confidence charts and PDF export.
""")
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: update sidebar with batch fold and compare modes"
```

---

### Task 7: Add domain annotation + PDF export to Lookup mode

**Files:**
- Modify: `app.py` — Lookup mode section (lines 281-363)

- [ ] **Step 1: Integrate domain annotation and PDF export into Lookup mode**

After fetching PDB (line 306), add domain fetch:
```python
domains = fetch_uniprot_domains(uniprot_id)
```

In the pLDDT chart tab, add domain toggle:
```python
with tab1:
    show_domains = st.checkbox("Show domain annotations", value=True) if domains else False
    fig = make_plddt_chart(residues, domains=domains if show_domains else None)
    st.plotly_chart(fig, use_container_width=True)
    if domains and show_domains:
        st.caption(f"Showing {len(domains)} domain/region annotations from UniProt")
```

In the download tab, add PDF export button after existing downloads:
```python
with tab4:
    st.download_button("Download PDB", pdb_text, f"AF-{uniprot_id}.pdb", "chemical/x-pdb")
    if pae_data:
        st.download_button("Download PAE JSON", json.dumps(pae_data), f"AF-{uniprot_id}-pae.json", "application/json")
    # PDF export
    plddt_fig = make_plddt_chart(residues, domains=domains if domains else None)
    pdf_bytes = generate_pdf_report(
        title=f"AlphaFold Prediction — {uniprot_id}",
        stats=stats, plddt_fig=plddt_fig,
        uniprot_id=uniprot_id, domains=domains,
    )
    st.download_button("📄 Export PDF Report", pdf_bytes, f"AF-{uniprot_id}-report.pdf", "application/pdf")
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add domain annotation and PDF export to lookup mode"
```

---

### Task 8: Add local ESM toggle + PDF export to Fold mode

**Files:**
- Modify: `app.py` — Fold mode section (lines 365-415)

- [ ] **Step 1: Add local ESM checkbox and PDF export to Fold mode**

Before fold_btn handling, add local ESM toggle:
```python
esm_capability = check_local_esm_capabilities()
use_local = False
if esm_capability != "unavailable":
    use_local = st.checkbox("Use local ESM-2 model (offline)", value=False)
else:
    st.caption("💡 Install `torch` and `fair-esm` for optional offline ESM-2 analysis")
```

In the fold logic, if `use_local` and `esm_capability == "full_fold"`:
```python
if use_local and esm_capability == "full_fold":
    pdb_text = fold_local_esm(sequence)
elif use_local and esm_capability == "analysis_only":
    pdb_text = fold_with_esmfold(sequence)  # still use API for structure
    analysis = analyze_local_esm(sequence)  # get contact map locally
else:
    pdb_text = fold_with_esmfold(sequence)
```

Add contact map tab if analysis available. Add PDF export to download tab:
```python
with download_tab:
    st.download_button("Download PDB", pdb_text, "esmfold_prediction.pdb", "chemical/x-pdb")
    plddt_fig = make_plddt_chart(residues)
    pdf_bytes = generate_pdf_report(title="ESMFold Prediction", stats=stats, plddt_fig=plddt_fig)
    st.download_button("📄 Export PDF Report", pdf_bytes, "esmfold-report.pdf", "application/pdf")
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add local ESM-2 toggle and PDF export to fold mode"
```

---

### Task 9: Implement Batch Fold mode

**Files:**
- Modify: `app.py` — add new elif block for Batch Fold

- [ ] **Step 1: Add Batch Fold mode UI**

```python
elif "📦" in mode:
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
            # Validate
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
                    progress.progress((i) / len(valid_seqs), text=f"Folding {name} ({i+1}/{len(valid_seqs)})...")
                    pdb_text = fold_with_esmfold(seq)
                    if pdb_text:
                        residues = parse_plddt_from_pdb(pdb_text)
                        stats = compute_structure_stats(residues)
                        results.append({"name": name, "sequence": seq, "pdb": pdb_text, "residues": residues, "stats": stats})
                    else:
                        results.append({"name": name, "sequence": seq, "pdb": None, "residues": None, "stats": None})

                progress.progress(1.0, text="Done!")
                st.session_state["batch_results"] = results

            # Display results
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

                # Expandable details
                for r in results:
                    if r["stats"]:
                        with st.expander(f"{r['name']} — Mean pLDDT: {r['stats']['mean_plddt']:.1f}"):
                            fig = make_plddt_chart(r["residues"])
                            st.plotly_chart(fig, use_container_width=True)

                # CSV export
                import csv
                csv_buf = StringIO()
                writer = csv.DictWriter(csv_buf, fieldnames=["sequence_name", "length", "mean_plddt", "median_plddt", "pct_very_high", "pct_confident", "pct_low"])
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
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: implement batch fold mode with FASTA upload and CSV export"
```

---

### Task 10: Implement Compare mode

**Files:**
- Modify: `app.py` — add new elif block for Compare

- [ ] **Step 1: Add Compare mode UI**

```python
elif "⚖️" in mode:
    st.title("Compare Proteins")
    st.markdown("Compare pLDDT confidence and structure for two proteins side by side.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Protein A")
        input_type_a = st.selectbox("Input type", ["UniProt ID", "Paste Sequence"], key="type_a")
        if input_type_a == "UniProt ID":
            id_a = st.text_input("UniProt ID", key="id_a", placeholder="e.g. P69905")
        else:
            seq_a = st.text_area("Sequence", key="seq_a", height=100, placeholder="MKTAYIAK...")

    with col_b:
        st.subheader("Protein B")
        input_type_b = st.selectbox("Input type", ["UniProt ID", "Paste Sequence"], key="type_b")
        if input_type_b == "UniProt ID":
            id_b = st.text_input("UniProt ID", key="id_b", placeholder="e.g. P68871")
        else:
            seq_b = st.text_area("Sequence", key="seq_b", height=100, placeholder="MVHLTPE...")

    if st.button("Compare", type="primary", use_container_width=True):
        # Fetch/fold both proteins
        def get_protein_data(input_type, label):
            """Returns (label, pdb_text, residues, stats) or shows error."""
            if input_type == "UniProt ID":
                uid = st.session_state.get(f"id_{label.lower()[-1]}", "").strip().upper()
                if not uid:
                    return None
                pred = fetch_alphafold_prediction(uid)
                if not pred:
                    st.error(f"No AlphaFold prediction for {uid}")
                    return None
                pdb = fetch_alphafold_pdb(pred)
                if not pdb:
                    st.error(f"PDB not found for {uid}")
                    return None
                return uid, pdb, parse_plddt_from_pdb(pdb), compute_structure_stats(parse_plddt_from_pdb(pdb))
            else:
                raw = st.session_state.get(f"seq_{label.lower()[-1]}", "")
                seq = clean_sequence(raw)
                ok, msg = validate_sequence(seq)
                if not ok:
                    st.error(f"Protein {label[-1]}: {msg}")
                    return None
                pdb = fold_with_esmfold(seq)
                if not pdb:
                    st.error(f"ESMFold failed for Protein {label[-1]}")
                    return None
                return f"Sequence {label[-1]}", pdb, parse_plddt_from_pdb(pdb), compute_structure_stats(parse_plddt_from_pdb(pdb))

        with st.spinner("Fetching/folding proteins..."):
            data_a = get_protein_data(input_type_a, "Protein A")
            data_b = get_protein_data(input_type_b, "Protein B")

        if data_a and data_b:
            label_a, pdb_a, res_a, stats_a = data_a
            label_b, pdb_b, res_b, stats_b = data_b

            # Overlaid pLDDT chart
            st.markdown("### pLDDT Comparison")
            fig = go.Figure()
            fig.add_hrect(y0=90, y1=100, fillcolor="#0053D6", opacity=0.08, line_width=0)
            fig.add_hrect(y0=70, y1=90, fillcolor="#65CBF3", opacity=0.08, line_width=0)
            fig.add_hrect(y0=50, y1=70, fillcolor="#FFDB13", opacity=0.08, line_width=0)
            fig.add_hrect(y0=0, y1=50, fillcolor="#FF7D45", opacity=0.08, line_width=0)
            fig.add_trace(go.Scatter(
                x=[r["residue_num"] for r in res_a], y=[r["plddt"] for r in res_a],
                mode="lines", name=label_a, line=dict(color="#0053D6", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=[r["residue_num"] for r in res_b], y=[r["plddt"] for r in res_b],
                mode="lines", name=label_b, line=dict(color="#FF7D45", width=2),
            ))
            fig.update_layout(
                title="Per-Residue pLDDT Comparison",
                xaxis_title="Residue Number", yaxis_title="pLDDT Score",
                yaxis=dict(range=[0, 100]), height=400,
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

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
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: implement compare mode with side-by-side pLDDT and 3D views"
```

---

### Task 11: Update mode routing and footer

**Files:**
- Modify: `app.py` — main routing logic

- [ ] **Step 1: Update the if/elif chain**

Ensure the main routing is:
```python
if "🔍" in mode:
    # Lookup
elif "🧪" in mode:
    # Fold
elif "📦" in mode:
    # Batch Fold
elif "⚖️" in mode:
    # Compare
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: wire up all mode routing"
```

---

### Task 12: Update README.md and requirements.txt

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt`

- [ ] **Step 1: Update README with new features and optional deps**

Add sections for new features, batch fold, compare, domain annotation, PDF export, optional local ESM-2.

- [ ] **Step 2: Verify final requirements.txt has all deps**

```
streamlit>=1.30.0
requests>=2.31.0
plotly>=5.18.0
numpy>=1.24.0
py3Dmol>=2.0.0
stmol>=0.0.9
biopython>=1.83
reportlab>=4.0
kaleido>=0.2.1
```

- [ ] **Step 3: Commit**

```bash
git add README.md requirements.txt
git commit -m "docs: update README and requirements for Phase 2 features"
```

---

### Task 13: Final verification

- [ ] **Step 1: Verify all imports work**

Run: `python3 -c "from app import *; print('All imports OK')"`

- [ ] **Step 2: Verify helper functions work end-to-end**

Run: `python3 -c "
from app import *
# Test parse_fasta
seqs = parse_fasta('>s1\nMKTAYI\n>s2\nACDEFG')
assert len(seqs) == 2
# Test domain fetch
doms = fetch_uniprot_domains('P04637')
print(f'Domains for P04637: {len(doms)}')
# Test PDF
pred = fetch_alphafold_prediction('P69905')
pdb = fetch_alphafold_pdb(pred)
res = parse_plddt_from_pdb(pdb)
stats = compute_structure_stats(res)
fig = make_plddt_chart(res, domains=fetch_uniprot_domains('P69905'))
pdf = generate_pdf_report('Test', stats, fig, 'P69905')
print(f'PDF: {len(pdf)} bytes')
print('ALL OK')
"`

- [ ] **Step 3: Commit all remaining changes**

```bash
git add -A
git commit -m "feat: Phase 2 complete — batch fold, compare, domains, PDF export, local ESM"
```
