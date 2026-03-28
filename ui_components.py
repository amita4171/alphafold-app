"""Shared Streamlit UI components for all modes."""
from __future__ import annotations

import streamlit as st


def show_stats_row(stats: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Residues", stats["num_residues"])
    c2.metric("Mean pLDDT", f"{stats['mean_plddt']:.1f}")
    c3.metric("% Very High (>90)", f"{stats['pct_very_high']:.0f}%")
    c4.metric("% Low (<50)", f"{stats['pct_low']:.0f}%")


def show_3d_tab(pdb_text: str, key_suffix: str = ""):
    from viz import render_3d_structure
    c1, c2 = st.columns(2)
    with c1:
        style = st.selectbox("Style", ["cartoon", "stick", "sphere", "surface", "line"], key=f"style_{key_suffix}")
    with c2:
        color = st.selectbox("Color by", ["pLDDT", "Chain", "Hydrophobicity", "Secondary Structure", "Uniform"],
                              key=f"color_{key_suffix}")
    render_3d_structure(pdb_text, style=style, color_scheme=color)


def show_properties_tab(sequence: str) -> dict:
    from analysis import (compute_sequence_properties, find_glycosylation_sites,
                          find_phosphorylation_sites, predict_transmembrane,
                          compute_aliphatic_index, compute_half_life, detect_signal_peptide,
                          classify_protein)
    from viz import make_aa_composition_chart, make_hydrophobicity_plot, make_charge_at_ph_plot, make_flexibility_plot

    props = compute_sequence_properties(sequence)
    # Row 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Molecular Weight", f"{props['molecular_weight']:.0f} Da")
    c2.metric("Isoelectric Point", f"{props['isoelectric_point']:.2f}")
    c3.metric("GRAVY", f"{props['gravy']:.3f}")
    c4.metric("Instability Index", f"{props['instability_index']:.1f}")
    # Row 2
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Aromaticity", f"{props['aromaticity']:.3f}")
    c5_label = "Stable" if props["instability_index"] < 40 else "Unstable"
    c6.metric("Stability", c5_label)
    c7.metric("Ext. Coeff (red)", f"{props['extinction_coeff'][0]}")
    c8.metric("Ext. Coeff (ox)", f"{props['extinction_coeff'][1]}")
    # Row 3: Additional metrics
    aliphatic = compute_aliphatic_index(sequence)
    half_life = compute_half_life(sequence)
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Aliphatic Index", f"{aliphatic:.1f}")
    c10.metric("Half-life (mammalian)", half_life["mammalian"])
    c11.metric("Half-life (yeast)", half_life["yeast"])
    c12.metric("Half-life (E. coli)", half_life["ecoli"])
    # Classification tags
    tags = classify_protein(sequence, props)
    if tags:
        st.markdown("**Classification:** " + " | ".join(f"`{t}`" for t in tags))
    # Signal peptide
    signal = detect_signal_peptide(sequence)
    if signal:
        st.info(f"Potential signal peptide detected: residues {signal['start']}-{signal['end']} (hydrophobicity score: {signal['score']:.2f})")
    # Secondary structure
    st.markdown("**Predicted Secondary Structure (sequence-based)**")
    s1, s2, s3 = st.columns(3)
    s1.metric("Helix", f"{props['helix_fraction'] * 100:.0f}%")
    s2.metric("Turn", f"{props['turn_fraction'] * 100:.0f}%")
    s3.metric("Sheet", f"{props['sheet_fraction'] * 100:.0f}%")
    # Charts
    st.plotly_chart(make_aa_composition_chart(sequence), use_container_width=True)
    st.plotly_chart(make_hydrophobicity_plot(sequence), use_container_width=True)
    st.plotly_chart(make_charge_at_ph_plot(sequence), use_container_width=True)
    st.plotly_chart(make_flexibility_plot(sequence), use_container_width=True)
    # PTM sites
    glyco = find_glycosylation_sites(sequence)
    phospho = find_phosphorylation_sites(sequence)
    tm = predict_transmembrane(sequence)
    st.markdown("**Post-Translational Modification Sites**")
    p1, p2, p3 = st.columns(3)
    p1.metric("N-Glycosylation", len(glyco))
    p2.metric("Phospho sites (S/T/Y)", len(phospho))
    p3.metric("TM regions", len(tm))
    if glyco:
        with st.expander(f"N-Glycosylation sites ({len(glyco)})"):
            for g in glyco:
                st.write(f"Position **{g['position']}**: {g['motif']}")
    if tm:
        with st.expander(f"Transmembrane regions ({len(tm)})"):
            for t in tm:
                st.write(f"Residues **{t['start']}-{t['end']}** ({t['length']} aa)")
    return props


def show_ramachandran_tab(pdb_text: str):
    from analysis import calculate_phi_psi, count_ramachandran_outliers, assign_ss_from_phi_psi
    from viz import make_ramachandran_plot
    phi_psi = calculate_phi_psi(pdb_text)
    if phi_psi:
        st.plotly_chart(make_ramachandran_plot(phi_psi))
        outliers = count_ramachandran_outliers(phi_psi)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Favored", f"{100 * outliers['favored'] / outliers['total']:.0f}%")
        r2.metric("Allowed", f"{100 * outliers['allowed'] / outliers['total']:.0f}%")
        r3.metric("Outlier", f"{100 * outliers['outlier'] / outliers['total']:.0f}%")
        r4.metric("Total angles", outliers["total"])
        # SS assignment
        ss = assign_ss_from_phi_psi(phi_psi)
        h = ss.count("H")
        e = ss.count("E")
        c = ss.count("C")
        st.markdown(f"**Phi/Psi-based SS:** Helix {100*h/len(ss):.0f}% | Sheet {100*e/len(ss):.0f}% | Coil {100*c/len(ss):.0f}%")
    else:
        st.info("Cannot calculate Ramachandran angles.")


def show_structural_analysis_tab(pdb_text: str, residues: list[dict]):
    from analysis import (calculate_radius_of_gyration, detect_disulfide_bonds, detect_salt_bridges,
                          detect_hydrogen_bonds, calculate_contact_order, find_disordered_regions,
                          calculate_residue_burial)
    from viz import make_bfactor_histogram, make_distance_map, make_disorder_plot, make_burial_plot

    rg = calculate_radius_of_gyration(pdb_text)
    disulfide = detect_disulfide_bonds(pdb_text)
    salt = detect_salt_bridges(pdb_text)
    hbonds = detect_hydrogen_bonds(pdb_text)
    co = calculate_contact_order(pdb_text)
    disordered = find_disordered_regions(residues)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Radius of Gyration", f"{rg:.1f} Å")
    c2.metric("Disulfide Bonds", len(disulfide))
    c3.metric("Salt Bridges", len(salt))
    c4.metric("H-bonds (backbone)", len(hbonds))

    c5, c6, c7 = st.columns(3)
    c5.metric("Contact Order", f"{co:.4f}")
    c6.metric("Disordered Regions", len(disordered))
    c7.metric("Disordered Residues", sum(d["length"] for d in disordered))

    # Disorder plot
    if disordered:
        st.plotly_chart(make_disorder_plot(residues, disordered), use_container_width=True)

    # B-factor histogram
    st.plotly_chart(make_bfactor_histogram(residues), use_container_width=True)

    # Burial
    burial = calculate_residue_burial(pdb_text)
    if burial:
        st.plotly_chart(make_burial_plot(burial), use_container_width=True)

    # Distance map
    st.plotly_chart(make_distance_map(pdb_text))

    # Details
    if disulfide:
        with st.expander(f"Disulfide Bonds ({len(disulfide)})"):
            for b in disulfide:
                st.write(f"Cys **{b['cys1']}** — Cys **{b['cys2']}** ({b['distance']:.2f} Å)")
    if salt:
        with st.expander(f"Salt Bridges ({len(salt)})"):
            for s in salt:
                st.write(f"{s['pos_name']} **{s['pos_res']}** — {s['neg_name']} **{s['neg_res']}** ({s['distance']:.1f} Å)")
    if disordered:
        with st.expander(f"Disordered Regions ({len(disordered)})"):
            for d in disordered:
                st.write(f"Residues **{d['start']}-{d['end']}** ({d['length']} aa, mean pLDDT: {d['mean_plddt']:.1f})")


def show_annotations_tab(uniprot_data: dict):
    from api_clients import extract_uniprot_annotations
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
        for label, terms in [("Cellular Component", go_c), ("Molecular Function", go_f), ("Biological Process", go_p)]:
            if terms:
                with st.expander(f"{label} ({len(terms)})"):
                    for g in terms:
                        st.write(f"- {g['term'][2:]} (`{g['id']}`)")

    if ann["pdb_refs"]:
        st.markdown(f"**PDB Structures ({len(ann['pdb_refs'])})**")
        st.dataframe([{"PDB": p["id"], "Method": p["method"], "Resolution": p["resolution"], "Chains": p["chains"]}
                       for p in ann["pdb_refs"]], use_container_width=True)

    if ann["keywords"]:
        st.markdown("**Keywords**")
        kw_by_cat: dict[str, list[str]] = {}
        for kw in ann["keywords"]:
            kw_by_cat.setdefault(kw["category"] or "Other", []).append(kw["name"])
        for cat, names in sorted(kw_by_cat.items()):
            st.write(f"**{cat}:** {', '.join(names)}")


def show_external_databases_tab(uniprot_id: str, sequence: str = ""):
    """Display data from external databases: InterPro, RCSB PDB, STRING, Reactome, KEGG, MobiDB."""
    from api_clients import (fetch_interpro_domains, fetch_string_interactions, fetch_reactome_pathways,
                             fetch_kegg_pathways, fetch_mobidb_disorder, fetch_ebi_protein_features)

    # InterPro domains
    st.markdown("### InterPro Domain Classification")
    with st.spinner("Fetching InterPro data..."):
        interpro = fetch_interpro_domains(uniprot_id)
    if interpro:
        st.dataframe([{"Accession": d["accession"], "Name": d["name"], "Type": d.get("type", ""),
                        "Start": d["start"], "End": d["end"]} for d in interpro], use_container_width=True)
    else:
        st.caption("No InterPro data available.")

    # STRING interactions
    st.markdown("### Protein-Protein Interactions (STRING)")
    with st.spinner("Fetching STRING data..."):
        interactions = fetch_string_interactions(uniprot_id)
    if interactions:
        from viz import make_interaction_network_plot
        st.plotly_chart(make_interaction_network_plot(interactions), use_container_width=True)
        st.dataframe([{"Partner": i.get("preferredName_B", i.get("stringId_B", "")),
                        "Score": f"{i.get('score', 0):.3f}",
                        "Experimental": f"{i.get('escore', 0):.3f}",
                        "Database": f"{i.get('dscore', 0):.3f}"}
                       for i in interactions[:20]], use_container_width=True)
    else:
        st.caption("No STRING interaction data available.")

    # Reactome pathways
    st.markdown("### Reactome Pathways")
    with st.spinner("Fetching Reactome data..."):
        reactome = fetch_reactome_pathways(uniprot_id)
    if reactome:
        for p in reactome[:20]:
            name = p.get("displayName", p.get("name", "Unknown"))
            st_id = p.get("stId", "")
            st.write(f"- **{name}** (`{st_id}`)")
    else:
        st.caption("No Reactome pathway data available.")

    # KEGG pathways
    st.markdown("### KEGG Pathways")
    with st.spinner("Fetching KEGG data..."):
        kegg = fetch_kegg_pathways(uniprot_id)
    if kegg:
        for p in kegg[:20]:
            st.write(f"- **{p['name']}** (`{p['pathway_id']}`)")
    else:
        st.caption("No KEGG pathway data available.")

    # MobiDB disorder
    st.markdown("### MobiDB Disorder Consensus")
    with st.spinner("Fetching MobiDB data..."):
        mobidb = fetch_mobidb_disorder(uniprot_id)
    if mobidb:
        # MobiDB returns complex structure, show summary
        acc = mobidb.get("acc", uniprot_id)
        st.write(f"**Accession:** {acc}")
        consensus = mobidb.get("consensus", {})
        if consensus:
            for key, val in consensus.items():
                if isinstance(val, dict) and "regions" in val:
                    regions = val["regions"]
                    st.write(f"**{key}:** {len(regions)} regions")
    else:
        st.caption("No MobiDB data available.")


def show_compare_alignment_tab(seq_a: str, seq_b: str, label_a: str, label_b: str):
    """Show sequence alignment in Compare mode."""
    from analysis import align_sequences
    from viz import make_alignment_viz

    alignment = align_sequences(seq_a, seq_b)
    if alignment:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Identity", f"{alignment['identity']:.1f}%")
        c2.metric("Similarity", f"{alignment['similarity']:.1f}%")
        c3.metric("Gaps", str(alignment['gaps']))
        c4.metric("Score", f"{alignment['score']:.0f}")

        viz_text = make_alignment_viz(alignment)
        st.code(viz_text, language="text")


def show_sasa_tab(pdb_text: str):
    """Display SASA analysis."""
    try:
        from analysis import calculate_sasa
        from viz import make_sasa_plot
        with st.spinner("Calculating SASA..."):
            sasa_data = calculate_sasa(pdb_text)
        if sasa_data:
            total_sasa = sum(d["sasa"] for d in sasa_data)
            avg_rel = sum(d.get("relative_sasa", 0) for d in sasa_data) / len(sasa_data) if sasa_data else 0
            exposed = sum(1 for d in sasa_data if d.get("relative_sasa", 0) > 0.5)
            buried = sum(1 for d in sasa_data if d.get("relative_sasa", 0) < 0.2)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total SASA", f"{total_sasa:.0f} Å²")
            c2.metric("Avg Relative SASA", f"{avg_rel:.2f}")
            c3.metric("Exposed Residues", exposed)
            c4.metric("Buried Residues", buried)
            st.plotly_chart(make_sasa_plot(sasa_data), use_container_width=True)
        else:
            st.warning("SASA calculation failed. Install `freesasa` for accurate SASA.")
    except ImportError:
        st.info("Install `freesasa` for solvent accessible surface area analysis.")


def show_nma_tab(pdb_text: str, residues: list[dict]):
    """Display Normal Mode Analysis results."""
    try:
        from analysis import run_normal_mode_analysis, calculate_gnm_bfactors
        from viz import make_nma_fluctuations_plot, make_nma_cross_correlation, make_gnm_bfactor_comparison
        with st.spinner("Running Normal Mode Analysis (may take a moment)..."):
            nma = run_normal_mode_analysis(pdb_text)
        if nma:
            res_nums = [r["residue_num"] for r in residues]
            st.markdown("**Anisotropic Network Model (ANM) — Mean Square Fluctuations**")
            st.plotly_chart(make_nma_fluctuations_plot(nma["sqflucts"], res_nums), use_container_width=True)
            st.markdown("**Cross-Correlation Map**")
            st.plotly_chart(make_nma_cross_correlation(nma["cross_correlations"]))
            # GNM B-factor comparison
            with st.spinner("Calculating GNM B-factors..."):
                gnm_bfactors = calculate_gnm_bfactors(pdb_text)
            if gnm_bfactors:
                st.markdown("**GNM-Predicted vs Experimental B-factors**")
                st.plotly_chart(make_gnm_bfactor_comparison(residues, gnm_bfactors), use_container_width=True)
        else:
            st.warning("NMA failed. Install `prody` for normal mode analysis.")
    except ImportError:
        st.info("Install `prody` for normal mode analysis.")


def show_topology_tab(pdb_text: str):
    """Display topology diagram."""
    try:
        from analysis import generate_topology_data
        from viz import make_topology_diagram
        topo = generate_topology_data(pdb_text)
        if topo:
            st.markdown(f"**{len([t for t in topo if t['type'] == 'helix'])} helices, "
                        f"{len([t for t in topo if t['type'] == 'strand'])} strands**")
            fig = make_topology_diagram(topo)
            st.pyplot(fig)
            with st.expander("Secondary Structure Elements"):
                for t in topo:
                    icon = "🔴" if t["type"] == "helix" else "🔵" if t["type"] == "strand" else "⚪"
                    st.write(f"{icon} **{t['type'].title()}** {t['start']}-{t['end']} ({t['length']} residues)")
        else:
            st.info("Could not determine topology from structure.")
    except ImportError:
        st.info("Install `matplotlib` for topology diagrams.")


def show_compare_structural_tab(pdb_a: str, pdb_b: str, label_a: str, label_b: str,
                                 res_a: list[dict], res_b: list[dict]):
    """Enhanced structural comparison with RMSD and TM-align."""
    from analysis import (calculate_radius_of_gyration, detect_disulfide_bonds,
                          detect_salt_bridges, detect_hydrogen_bonds)

    # Basic metrics comparison
    st.markdown("### Structural Metrics")
    metrics_a = {
        "Rg": f"{calculate_radius_of_gyration(pdb_a):.1f} Å",
        "SS bonds": len(detect_disulfide_bonds(pdb_a)),
        "Salt bridges": len(detect_salt_bridges(pdb_a)),
        "H-bonds": len(detect_hydrogen_bonds(pdb_a)),
    }
    metrics_b = {
        "Rg": f"{calculate_radius_of_gyration(pdb_b):.1f} Å",
        "SS bonds": len(detect_disulfide_bonds(pdb_b)),
        "Salt bridges": len(detect_salt_bridges(pdb_b)),
        "H-bonds": len(detect_hydrogen_bonds(pdb_b)),
    }
    rows = [{"Metric": k, label_a: str(metrics_a[k]), label_b: str(metrics_b[k])} for k in metrics_a]
    st.dataframe(rows, use_container_width=True)

    # TM-align / RMSD
    try:
        from analysis import run_tm_align
        from viz import make_rmsd_summary
        with st.spinner("Running TM-align..."):
            tm_result = run_tm_align(pdb_a, pdb_b)
        if tm_result:
            st.markdown("### TM-align Structural Alignment")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSD", f"{tm_result['rmsd']:.2f} Å")
            c2.metric("TM-score (→A)", f"{tm_result['tm_score_a']:.3f}")
            c3.metric("TM-score (→B)", f"{tm_result['tm_score_b']:.3f}")
            c4.metric("Aligned Length", tm_result["aligned_length"])
            st.code(make_rmsd_summary(tm_result), language="text")
        else:
            st.caption("TM-align requires `tmtools`.")
    except ImportError:
        st.info("Install `tmtools` for RMSD and structural alignment.")
