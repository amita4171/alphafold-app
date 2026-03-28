"""
AlphaFold Explorer — Comprehensive Protein Analysis Suite
Streamlit app: structure lookup, folding, comparison, and deep biophysical analysis.

Modules: analysis.py, api_clients.py, viz.py, ui_components.py, export_utils.py
"""
from __future__ import annotations

import csv
import json
from io import StringIO

import streamlit as st

from analysis import (
    clean_sequence, parse_fasta, validate_sequence, extract_sequence_from_pdb,
    parse_plddt_from_pdb, compute_structure_stats, compute_sequence_properties,
    calculate_phi_psi, calculate_radius_of_gyration, detect_disulfide_bonds,
    detect_salt_bridges, detect_hydrogen_bonds, calculate_contact_order,
    find_disordered_regions, align_sequences, parse_pdb_chains, parse_plddt_by_chain,
)
from api_clients import (
    fetch_alphafold_prediction, fetch_alphafold_pdb, fetch_alphafold_pae,
    fold_with_esmfold, search_uniprot, fetch_uniprot_full,
    extract_uniprot_domains, extract_uniprot_annotations,
)
from viz import (
    make_plddt_chart, make_pae_heatmap, render_3d_structure, render_overlay_3d,
    make_distance_map, make_alignment_viz,
)
from ui_components import (
    show_stats_row, show_3d_tab, show_properties_tab, show_ramachandran_tab,
    show_structural_analysis_tab, show_annotations_tab, show_external_databases_tab,
    show_compare_alignment_tab, show_sasa_tab, show_nma_tab, show_topology_tab,
    show_compare_structural_tab,
)
from export_utils import generate_pdf_report, export_analysis_json

# ── Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AlphaFold Explorer", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

# Optional ESM-2
try:
    import torch
    import esm as esm_module
    LOCAL_ESM_AVAILABLE = True
except ImportError:
    LOCAL_ESM_AVAILABLE = False


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


# ── Sidebar ─────────────────────────────────────────────────────────────
st.sidebar.title("🧬 AlphaFold Explorer")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode", [
    "🔍 Lookup (AlphaFold DB)", "🧪 Fold (ESMFold)", "📦 Batch Fold", "⚖️ Compare", "📂 Upload PDB",
], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**About**

**Lookup**: AlphaFold DB + UniProt + InterPro + STRING + Reactome + KEGG

**Fold**: ESMFold API or local ESM-2. Full biophysical analysis.

**Batch**: FASTA upload, fold up to 20 sequences, CSV export.

**Compare**: pLDDT overlay, sequence alignment, structure overlay, structural metrics.

**Upload PDB**: Full analysis of your own structures.

All modes: pLDDT, Ramachandran, properties, structural analysis, PDF & JSON export.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Examples:** `P69905` (Hb alpha) | `P04637` (p53) | `P68871` (Hb beta) | `Q9BYF1` (ACE2)
""")

# ═══════════════════════════════════════════════════════════════════════
# LOOKUP MODE
# ═══════════════════════════════════════════════════════════════════════

if "🔍" in mode:
    st.title("AlphaFold Structure Lookup")
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("UniProt ID or protein name", placeholder="e.g. P69905 or hemoglobin human")
    with col2:
        search_btn = st.button("Search", use_container_width=True, type="primary")

    if search_btn and query:
        import re
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
                    chains = parse_pdb_chains(pdb_text)
                    show_stats_row(stats)
                    if len(chains) > 1:
                        st.info(f"Multi-chain structure: {', '.join(chains)}")

                    tabs = st.tabs(["📊 pLDDT", "🔬 3D", "🗺️ PAE", "🧬 Properties",
                                    "📐 Ramachandran", "🔩 Structural", "💧 SASA",
                                    "🔄 Dynamics", "🧩 Topology", "📚 Annotations",
                                    "🌐 Databases", "📥 Export"])

                    with tabs[0]:
                        show_domains = st.checkbox("Show domains", value=True) if domains else False
                        fig = make_plddt_chart(residues, domains=domains if show_domains else None)
                        st.plotly_chart(fig, use_container_width=True)

                    with tabs[1]:
                        show_3d_tab(pdb_text, key_suffix="lookup")

                    with tabs[2]:
                        if pae_data:
                            st.plotly_chart(make_pae_heatmap(pae_data))
                        else:
                            st.info("PAE not available.")

                    with tabs[3]:
                        seq_props = show_properties_tab(sequence)

                    with tabs[4]:
                        show_ramachandran_tab(pdb_text)

                    with tabs[5]:
                        show_structural_analysis_tab(pdb_text, residues)

                    with tabs[6]:
                        show_sasa_tab(pdb_text)

                    with tabs[7]:
                        show_nma_tab(pdb_text, residues)

                    with tabs[8]:
                        show_topology_tab(pdb_text)

                    with tabs[9]:
                        if uniprot_data:
                            show_annotations_tab(uniprot_data)

                    with tabs[10]:
                        show_external_databases_tab(uniprot_id, sequence)

                    with tabs[11]:
                        st.download_button("Download PDB", pdb_text, f"AF-{uniprot_id}.pdb", "chemical/x-pdb")
                        if pae_data:
                            st.download_button("Download PAE JSON", json.dumps(pae_data), f"AF-{uniprot_id}-pae.json")
                        st.download_button("Download FASTA", f">{uniprot_id}\n{sequence}\n", f"{uniprot_id}.fasta", "text/plain")
                        pdf_fig = make_plddt_chart(residues, domains=domains)
                        pdf_bytes = generate_pdf_report(
                            title=f"AlphaFold — {uniprot_id}", stats=stats, plddt_fig=pdf_fig,
                            uniprot_id=uniprot_id, domains=domains,
                            seq_props=compute_sequence_properties(sequence))
                        st.download_button("📄 PDF Report", pdf_bytes, f"AF-{uniprot_id}-report.pdf", "application/pdf")
                        ann = extract_uniprot_annotations(uniprot_data) if uniprot_data else None
                        json_str = export_analysis_json(
                            stats=stats, seq_props=compute_sequence_properties(sequence), domains=domains,
                            phi_psi=calculate_phi_psi(pdb_text), disulfide=detect_disulfide_bonds(pdb_text),
                            salt_bridges=detect_salt_bridges(pdb_text), rg=calculate_radius_of_gyration(pdb_text),
                            hbonds=detect_hydrogen_bonds(pdb_text), contact_order=calculate_contact_order(pdb_text),
                            disordered=find_disordered_regions(residues), annotations=ann)
                        st.download_button("📦 Full Analysis JSON", json_str, f"AF-{uniprot_id}-analysis.json")
                else:
                    st.error(f"PDB not found for {uniprot_id}.")
            else:
                st.warning(f"No AlphaFold prediction for {uniprot_id}.")
        else:
            with st.spinner(f"Searching UniProt for '{query}'..."):
                results = search_uniprot(query)
            if results:
                st.markdown(f"**{len(results)} results.**")
                for r in results:
                    ca, cb, cc, cd = st.columns([1, 3, 2, 1])
                    ca.code(r["accession"]); cb.write(r["name"]); cc.write(f"*{r['organism']}*"); cd.write(f"{r['length']} aa")
                st.info("Copy a UniProt accession and search again.")
            else:
                st.warning(f"No results for '{query}'.")

# ═══════════════════════════════════════════════════════════════════════
# FOLD MODE
# ═══════════════════════════════════════════════════════════════════════

elif "🧪" in mode:
    st.title("ESMFold — Fold a Sequence")
    esm_capability = check_local_esm_capabilities()
    use_local = st.checkbox("Use local ESM-2 (offline)", value=False) if esm_capability != "unavailable" else False
    if esm_capability == "unavailable":
        st.caption("💡 Install `torch` + `fair-esm` for offline ESM-2")

    sequence_input = st.text_area("Amino acid sequence (FASTA or raw)", height=150,
                                   placeholder=">my_protein\nMKTAYIAKQRQISFVKSHFSRQDLDALK...")
    if st.button("Fold Sequence", type="primary", use_container_width=True) and sequence_input:
        sequence = clean_sequence(sequence_input)
        valid, msg = validate_sequence(sequence)
        if not valid:
            st.error(msg)
        else:
            analysis = None
            if use_local and esm_capability == "full_fold":
                with st.spinner("Folding locally..."): pdb_text = fold_local_esm(sequence)
                if not pdb_text:
                    st.warning("Local fold failed, using API...")
                    with st.spinner("Folding via API..."): pdb_text = fold_with_esmfold(sequence)
            else:
                with st.spinner("Folding (30-60s)..."): pdb_text = fold_with_esmfold(sequence)
                if use_local and esm_capability == "analysis_only":
                    with st.spinner("ESM-2 analysis..."): analysis = analyze_local_esm(sequence)

            if pdb_text:
                st.success("Done!")
                residues = parse_plddt_from_pdb(pdb_text)
                stats = compute_structure_stats(residues)
                show_stats_row(stats)

                tab_names = ["📊 pLDDT", "🔬 3D", "🧬 Properties", "📐 Ramachandran",
                             "🔩 Structural", "💧 SASA", "🔄 Dynamics", "🧩 Topology", "📥 Export"]
                if analysis:
                    tab_names.insert(8, "🧠 Contact Map")
                tabs = st.tabs(tab_names)

                with tabs[0]: st.plotly_chart(make_plddt_chart(residues), use_container_width=True)
                with tabs[1]: show_3d_tab(pdb_text, key_suffix="fold")
                with tabs[2]: show_properties_tab(sequence)
                with tabs[3]: show_ramachandran_tab(pdb_text)
                with tabs[4]: show_structural_analysis_tab(pdb_text, residues)
                with tabs[5]: show_sasa_tab(pdb_text)
                with tabs[6]: show_nma_tab(pdb_text, residues)
                with tabs[7]: show_topology_tab(pdb_text)

                if analysis:
                    import plotly.graph_objects as go
                    with tabs[5]:
                        cm_fig = go.Figure(data=go.Heatmap(z=analysis["contact_map"], colorscale="Blues"))
                        cm_fig.update_layout(height=500, width=500, yaxis=dict(autorange="reversed"))
                        st.plotly_chart(cm_fig)

                with tabs[-1]:
                    st.download_button("Download PDB", pdb_text, "prediction.pdb", "chemical/x-pdb")
                    st.download_button("Download FASTA", f">prediction\n{sequence}\n", "prediction.fasta", "text/plain")
                    pdf_bytes = generate_pdf_report("ESMFold Prediction", stats, make_plddt_chart(residues),
                                                     seq_props=compute_sequence_properties(sequence))
                    st.download_button("📄 PDF Report", pdf_bytes, "prediction-report.pdf", "application/pdf")
                    json_str = export_analysis_json(
                        stats=stats, seq_props=compute_sequence_properties(sequence),
                        phi_psi=calculate_phi_psi(pdb_text), disulfide=detect_disulfide_bonds(pdb_text),
                        salt_bridges=detect_salt_bridges(pdb_text), rg=calculate_radius_of_gyration(pdb_text),
                        hbonds=detect_hydrogen_bonds(pdb_text), contact_order=calculate_contact_order(pdb_text),
                        disordered=find_disordered_regions(residues))
                    st.download_button("📦 Full Analysis JSON", json_str, "prediction-analysis.json")
            else:
                st.error("ESMFold failed. Try shorter sequence or retry later.")

# ═══════════════════════════════════════════════════════════════════════
# BATCH FOLD
# ═══════════════════════════════════════════════════════════════════════

elif "📦" in mode:
    st.title("Batch Fold — Multiple Sequences")
    uploaded = st.file_uploader("Upload FASTA", type=["fasta", "fa", "faa", "txt"])

    if uploaded:
        sequences = parse_fasta(uploaded.read().decode("utf-8"))
        if not sequences:
            st.error("No sequences found.")
        elif len(sequences) > 20:
            st.error(f"Too many ({len(sequences)}). Max 20.")
        else:
            valid_seqs, skipped = [], []
            for name, seq in sequences:
                ok, msg = validate_sequence(seq)
                (valid_seqs if ok else skipped).append((name, seq) if ok else (name, msg))

            st.info(f"**{len(valid_seqs)}** valid, **{len(skipped)}** skipped")
            if skipped:
                with st.expander("Skipped"):
                    for n, r in skipped: st.write(f"- **{n}**: {r}")

            if valid_seqs and st.button("Fold All", type="primary", use_container_width=True):
                results = []
                progress = st.progress(0)
                for i, (name, seq) in enumerate(valid_seqs):
                    progress.progress(i / len(valid_seqs), text=f"Folding {name} ({i+1}/{len(valid_seqs)})")
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
                rows = []
                for r in results:
                    if r["stats"]:
                        s = r["stats"]
                        rows.append({"Sequence": r["name"], "Length": s["num_residues"],
                                      "Mean pLDDT": f"{s['mean_plddt']:.1f}", "% High": f"{s['pct_very_high']:.0f}%"})
                    else:
                        rows.append({"Sequence": r["name"], "Length": len(r["sequence"]), "Mean pLDDT": "FAILED", "% High": "-"})
                st.dataframe(rows, use_container_width=True)

                for r in results:
                    if r["stats"]:
                        with st.expander(f"{r['name']} — pLDDT: {r['stats']['mean_plddt']:.1f}"):
                            st.plotly_chart(make_plddt_chart(r["residues"]), use_container_width=True)

                csv_buf = StringIO()
                writer = csv.DictWriter(csv_buf, fieldnames=["name", "length", "mean_plddt", "median_plddt", "pct_high", "pct_low"])
                writer.writeheader()
                for r in results:
                    if r["stats"]:
                        s = r["stats"]
                        writer.writerow({"name": r["name"], "length": s["num_residues"],
                                          "mean_plddt": f"{s['mean_plddt']:.2f}", "median_plddt": f"{s['median_plddt']:.2f}",
                                          "pct_high": f"{s['pct_very_high']:.1f}", "pct_low": f"{s['pct_low']:.1f}"})
                st.download_button("📥 CSV", csv_buf.getvalue(), "batch.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════
# COMPARE MODE
# ═══════════════════════════════════════════════════════════════════════

elif "⚖️" in mode:
    import plotly.graph_objects as go

    st.title("Compare Proteins")
    ca, cb = st.columns(2)
    with ca:
        st.subheader("Protein A")
        type_a = st.selectbox("Input", ["UniProt ID", "Sequence"], key="ta")
        val_a = st.text_input("ID", key="ia", placeholder="P69905") if type_a == "UniProt ID" else st.text_area("Seq", key="sa", height=100)
    with cb:
        st.subheader("Protein B")
        type_b = st.selectbox("Input", ["UniProt ID", "Sequence"], key="tb")
        val_b = st.text_input("ID", key="ib", placeholder="P68871") if type_b == "UniProt ID" else st.text_area("Seq", key="sb", height=100)

    if st.button("Compare", type="primary", use_container_width=True):
        def _fetch(itype, val, label):
            if itype == "UniProt ID":
                uid = val.strip().upper()
                if not uid: st.error(f"{label}: enter ID"); return None
                pred = fetch_alphafold_prediction(uid)
                if not pred: st.error(f"No prediction for {uid}"); return None
                pdb = fetch_alphafold_pdb(pred["pdbUrl"]) if pred.get("pdbUrl") else None
                if not pdb: st.error(f"No PDB for {uid}"); return None
                res = parse_plddt_from_pdb(pdb)
                return uid, pdb, res, compute_structure_stats(res), extract_sequence_from_pdb(pdb)
            else:
                seq = clean_sequence(val)
                ok, msg = validate_sequence(seq)
                if not ok: st.error(f"{label}: {msg}"); return None
                pdb = fold_with_esmfold(seq)
                if not pdb: st.error(f"Fold failed for {label}"); return None
                res = parse_plddt_from_pdb(pdb)
                return label, pdb, res, compute_structure_stats(res), seq

        with st.spinner("Processing..."):
            da = _fetch(type_a, val_a, "Protein A")
            db = _fetch(type_b, val_b, "Protein B")

        if da and db:
            la, pdb_a, res_a, stats_a, seq_a = da
            lb, pdb_b, res_b, stats_b, seq_b = db

            tabs = st.tabs(["📊 pLDDT", "📈 Stats", "🔬 3D", "🧬 Alignment", "🔩 Structure"])

            with tabs[0]:
                fig = go.Figure()
                for y0, y1, c in [(90,100,"#0053D6"),(70,90,"#65CBF3"),(50,70,"#FFDB13"),(0,50,"#FF7D45")]:
                    fig.add_hrect(y0=y0, y1=y1, fillcolor=c, opacity=0.08, line_width=0)
                fig.add_trace(go.Scatter(x=[r["residue_num"] for r in res_a], y=[r["plddt"] for r in res_a],
                                          mode="lines", name=la, line=dict(color="#0053D6", width=2)))
                fig.add_trace(go.Scatter(x=[r["residue_num"] for r in res_b], y=[r["plddt"] for r in res_b],
                                          mode="lines", name=lb, line=dict(color="#FF7D45", width=2)))
                fig.update_layout(title="pLDDT Comparison", xaxis_title="Residue", yaxis_title="pLDDT",
                                   yaxis=dict(range=[0,100]), height=400, plot_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                metrics = ["num_residues","mean_plddt","median_plddt","min_plddt","max_plddt","pct_very_high","pct_confident","pct_low"]
                mlabels = ["Residues","Mean","Median","Min","Max","% High","% Confident","% Low"]
                rows = []
                for m, l in zip(metrics, mlabels):
                    fmt = ".1f" if isinstance(stats_a[m], float) else "d"
                    rows.append({"Metric": l, la: f"{stats_a[m]:{fmt}}", lb: f"{stats_b[m]:{fmt}}"})
                st.dataframe(rows, use_container_width=True)

            with tabs[2]:
                view = st.radio("View", ["Side by Side", "Overlay"], horizontal=True)
                if view == "Side by Side":
                    s1, s2 = st.columns(2)
                    with s1: st.caption(la); render_3d_structure(pdb_a)
                    with s2: st.caption(lb); render_3d_structure(pdb_b)
                else:
                    render_overlay_3d(pdb_a, pdb_b, la, lb)

            with tabs[3]:
                show_compare_alignment_tab(seq_a, seq_b, la, lb)

            with tabs[4]:
                show_compare_structural_tab(pdb_a, pdb_b, la, lb, res_a, res_b)

# ═══════════════════════════════════════════════════════════════════════
# UPLOAD PDB
# ═══════════════════════════════════════════════════════════════════════

elif "📂" in mode:
    st.title("Upload PDB — Analyze Your Structure")
    pdb_upload = st.file_uploader("Upload PDB", type=["pdb", "ent"])

    if pdb_upload:
        pdb_text = pdb_upload.read().decode("utf-8")
        residues = parse_plddt_from_pdb(pdb_text)
        if not residues:
            st.error("No CA atoms found.")
        else:
            stats = compute_structure_stats(residues)
            sequence = extract_sequence_from_pdb(pdb_text)
            chains = parse_pdb_chains(pdb_text)
            st.success(f"Loaded: **{len(residues)} residues** | Chains: {', '.join(chains)}")
            show_stats_row(stats)

            tabs = st.tabs(["📊 B-factor", "🔬 3D", "🧬 Properties", "📐 Ramachandran",
                            "🔩 Structural", "💧 SASA", "🔄 Dynamics", "🧩 Topology", "📥 Export"])
            with tabs[0]:
                fig = make_plddt_chart(residues)
                fig.update_layout(title="Per-Residue B-factor / pLDDT")
                st.plotly_chart(fig, use_container_width=True)
            with tabs[1]:
                show_3d_tab(pdb_text, key_suffix="upload")
            with tabs[2]:
                if "X" not in sequence:
                    show_properties_tab(sequence)
                else:
                    st.warning("Could not extract clean sequence from PDB.")
            with tabs[3]:
                show_ramachandran_tab(pdb_text)
            with tabs[4]:
                show_structural_analysis_tab(pdb_text, residues)
            with tabs[5]:
                show_sasa_tab(pdb_text)
            with tabs[6]:
                show_nma_tab(pdb_text, residues)
            with tabs[7]:
                show_topology_tab(pdb_text)
            with tabs[8]:
                st.download_button("Download PDB", pdb_text, pdb_upload.name, "chemical/x-pdb")
                if "X" not in sequence:
                    st.download_button("Download FASTA", f">uploaded\n{sequence}\n", "uploaded.fasta", "text/plain")
                props = compute_sequence_properties(sequence) if "X" not in sequence else None
                pdf_bytes = generate_pdf_report(f"Uploaded — {pdb_upload.name}", stats, make_plddt_chart(residues), seq_props=props)
                st.download_button("📄 PDF Report", pdf_bytes, f"{pdb_upload.name}-report.pdf", "application/pdf")
                json_str = export_analysis_json(
                    stats=stats, seq_props=props, phi_psi=calculate_phi_psi(pdb_text),
                    disulfide=detect_disulfide_bonds(pdb_text), salt_bridges=detect_salt_bridges(pdb_text),
                    rg=calculate_radius_of_gyration(pdb_text), hbonds=detect_hydrogen_bonds(pdb_text),
                    contact_order=calculate_contact_order(pdb_text), disordered=find_disordered_regions(residues))
                st.download_button("📦 Full Analysis JSON", json_str, f"{pdb_upload.name}-analysis.json")

# Footer
st.markdown("---")
st.caption("Data: [AlphaFold DB](https://alphafold.ebi.ac.uk/) | [ESMFold](https://esmatlas.com/) | [UniProt](https://www.uniprot.org/) | [STRING](https://string-db.org/) | [Reactome](https://reactome.org/) | [InterPro](https://www.ebi.ac.uk/interpro/)")
