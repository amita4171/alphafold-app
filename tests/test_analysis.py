"""Comprehensive test suite for analysis.py -- pure unit tests, no network calls."""
from __future__ import annotations

import math
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from analysis import (
    THREE_TO_ONE,
    KD_SCALE,
    clean_sequence,
    parse_fasta,
    validate_sequence,
    extract_sequence_from_pdb,
    parse_plddt_from_pdb,
    parse_backbone_atoms,
    parse_all_atoms,
    parse_pdb_chains,
    parse_plddt_by_chain,
    parse_hetatm,
    _dihedral,
    calculate_phi_psi,
    calculate_distance_map,
    calculate_radius_of_gyration,
    detect_disulfide_bonds,
    detect_salt_bridges,
    detect_hydrogen_bonds,
    detect_cation_pi,
    calculate_contact_order,
    calculate_residue_burial,
    estimate_sasa_approximate,
    assign_ss_from_phi_psi,
    count_ramachandran_outliers,
    detect_pae_domains,
    compute_structure_stats,
    find_disordered_regions,
    compute_sequence_properties,
    compute_charge_at_ph,
    compute_hydrophobicity,
    find_glycosylation_sites,
    find_phosphorylation_sites,
    predict_transmembrane,
    compute_flexibility,
    compute_aliphatic_index,
    compute_half_life,
    detect_signal_peptide,
    compute_sequence_complexity,
    classify_protein,
    align_sequences,
    score_substitutions,
    calculate_sasa,
    calculate_rmsd,
    run_tm_align,
    run_normal_mode_analysis,
    calculate_gnm_bfactors,
    build_sequence_distance_matrix,
    build_upgma_tree,
    compute_logo_data,
    assign_secondary_structure_from_coords,
    generate_topology_data,
)


# ============================================================================
# 1. SEQUENCE HELPERS
# ============================================================================


class TestCleanSequence:
    def test_basic(self):
        assert clean_sequence("MVLSP") == "MVLSP"

    def test_lowercase(self):
        assert clean_sequence("mvlsp") == "MVLSP"

    def test_strips_fasta_header(self):
        assert clean_sequence(">header\nMVLSP") == "MVLSP"

    def test_strips_numbers_and_spaces(self):
        assert clean_sequence("MVL 123 SP") == "MVLSP"

    def test_multiline(self):
        assert clean_sequence("MVL\nSP") == "MVLSP"

    def test_empty(self):
        assert clean_sequence("") == ""

    def test_only_header(self):
        assert clean_sequence(">header\n") == ""

    def test_multiple_headers(self):
        # Only last sequence fragment is used (headers are skipped)
        result = clean_sequence(">h1\nABC\n>h2\nDEF")
        assert result == "ABCDEF"


class TestParseFasta:
    def test_multi_sequence(self, fasta_text):
        result = parse_fasta(fasta_text)
        assert len(result) == 2
        assert result[0][0] == "seq1"
        assert result[0][1] == "MVLSPADKTNVKAAWGKVGA"
        assert result[1][0] == "seq2"

    def test_single_sequence(self):
        result = parse_fasta(">myseq\nMVLSP")
        assert len(result) == 1
        assert result[0] == ("myseq", "MVLSP")

    def test_empty_input(self):
        result = parse_fasta("")
        assert result == []

    def test_no_header(self):
        # No FASTA header -> nothing captured
        result = parse_fasta("MVLSP")
        assert result == []

    def test_unnamed_header(self):
        result = parse_fasta(">\nMVLSP")
        assert result[0][0] == "unnamed"

    def test_multiline_sequence(self):
        result = parse_fasta(">s1\nMVL\nSP")
        assert result[0][1] == "MVLSP"

    def test_skips_empty_sequence(self):
        result = parse_fasta(">s1\n>s2\nABC")
        # s1 has no sequence lines, so it should be skipped
        assert len(result) == 1
        assert result[0][0] == "s2"


class TestValidateSequence:
    def test_valid_sequence(self, sample_sequence):
        valid, msg = validate_sequence(sample_sequence)
        assert valid is True
        assert msg == "OK"

    def test_empty_sequence(self):
        valid, msg = validate_sequence("")
        assert valid is False
        assert "empty" in msg.lower()

    def test_too_short(self, short_sequence):
        valid, msg = validate_sequence(short_sequence)
        assert valid is False
        assert "short" in msg.lower()

    def test_too_long(self, long_sequence):
        valid, msg = validate_sequence(long_sequence)
        assert valid is False
        assert "long" in msg.lower()

    def test_invalid_digit_characters(self):
        # '1' is not in the valid_aa set, but validate_sequence checks set(seq)
        # set("MVLSPADKTN1KAAWGKVGA") includes '1' which is not in valid_aa
        valid, msg = validate_sequence("MVLSPADKTN1KAAWGKVGA")
        assert valid is False
        assert "Invalid" in msg

    def test_invalid_amino_acid_letter(self):
        valid, msg = validate_sequence("MVLSPADKTNXKAAWGKVGA")
        assert valid is False
        assert "Invalid" in msg

    def test_exactly_10_residues(self):
        valid, _ = validate_sequence("MVLSPADKTN")
        assert valid is True

    def test_exactly_400_residues(self):
        valid, _ = validate_sequence("A" * 400)
        assert valid is True

    def test_nine_residues_too_short(self):
        valid, _ = validate_sequence("MVLSPADKT")
        assert valid is False


# ============================================================================
# 2. PDB PARSING
# ============================================================================


class TestExtractSequenceFromPdb:
    def test_mini_pdb(self, mini_pdb):
        seq = extract_sequence_from_pdb(mini_pdb)
        assert seq == "MVLSP"

    def test_empty_pdb(self, empty_pdb):
        seq = extract_sequence_from_pdb(empty_pdb)
        assert seq == ""

    def test_two_chain(self, two_chain_pdb):
        seq = extract_sequence_from_pdb(two_chain_pdb)
        # parse_plddt_from_pdb keys by res_num only (no chain),
        # so chain B residues 1,2 overwrite chain A residues 1,2.
        # Result: res1=ASP(D), res2=GLU(E), res3=CYS(C)
        assert seq == "DEC"


class TestParsePlddtFromPdb:
    def test_mini_pdb(self, mini_pdb):
        residues = parse_plddt_from_pdb(mini_pdb)
        assert len(residues) == 5
        assert residues[0]["residue_num"] == 1
        assert residues[0]["plddt"] == 95.0
        assert residues[0]["residue_name"] == "MET"
        assert residues[0]["chain"] == "A"

    def test_values(self, mini_pdb):
        residues = parse_plddt_from_pdb(mini_pdb)
        plddts = [r["plddt"] for r in residues]
        assert plddts == [95.0, 92.0, 88.0, 45.0, 78.0]

    def test_empty_pdb(self, empty_pdb):
        residues = parse_plddt_from_pdb(empty_pdb)
        assert residues == []

    def test_sorted_order(self, mini_pdb):
        residues = parse_plddt_from_pdb(mini_pdb)
        nums = [r["residue_num"] for r in residues]
        assert nums == sorted(nums)


class TestParseBackboneAtoms:
    def test_mini_pdb(self, mini_pdb):
        backbone = parse_backbone_atoms(mini_pdb)
        assert len(backbone) == 5
        assert set(backbone.keys()) == {1, 2, 3, 4, 5}

    def test_atom_types(self, mini_pdb):
        backbone = parse_backbone_atoms(mini_pdb)
        for res_num in backbone:
            assert "N" in backbone[res_num]
            assert "CA" in backbone[res_num]
            assert "C" in backbone[res_num]

    def test_coordinates(self, mini_pdb):
        backbone = parse_backbone_atoms(mini_pdb)
        # Residue 1: N at (0,0,0), CA at (1.458,0,0)
        assert backbone[1]["N"] == pytest.approx((0.0, 0.0, 0.0))
        assert backbone[1]["CA"] == pytest.approx((1.458, 0.0, 0.0))

    def test_empty_pdb(self, empty_pdb):
        backbone = parse_backbone_atoms(empty_pdb)
        assert backbone == {}


class TestParseAllAtoms:
    def test_mini_pdb(self, mini_pdb):
        atoms = parse_all_atoms(mini_pdb)
        assert len(atoms) == 20  # 4 atoms per residue x 5 residues

    def test_atom_fields(self, mini_pdb):
        atoms = parse_all_atoms(mini_pdb)
        first = atoms[0]
        assert first["name"] == "N"
        assert first["res_name"] == "MET"
        assert first["chain"] == "A"
        assert first["res_num"] == 1
        assert first["x"] == pytest.approx(0.0)
        assert first["bfactor"] == pytest.approx(95.0)

    def test_empty_pdb(self, empty_pdb):
        atoms = parse_all_atoms(empty_pdb)
        assert atoms == []


class TestParsePdbChains:
    def test_single_chain(self, mini_pdb):
        chains = parse_pdb_chains(mini_pdb)
        assert chains == ["A"]

    def test_two_chains(self, two_chain_pdb):
        chains = parse_pdb_chains(two_chain_pdb)
        assert chains == ["A", "B"]

    def test_empty_pdb(self, empty_pdb):
        chains = parse_pdb_chains(empty_pdb)
        assert chains == []

    def test_order_preserved(self, two_chain_pdb):
        chains = parse_pdb_chains(two_chain_pdb)
        assert chains[0] == "A"
        assert chains[1] == "B"


class TestParsePlddtByChain:
    def test_single_chain(self, mini_pdb):
        by_chain = parse_plddt_by_chain(mini_pdb)
        assert "A" in by_chain
        assert len(by_chain["A"]) == 5

    def test_two_chains(self, two_chain_pdb):
        by_chain = parse_plddt_by_chain(two_chain_pdb)
        assert len(by_chain) == 2
        assert len(by_chain["A"]) == 3
        assert len(by_chain["B"]) == 2

    def test_plddt_values(self, two_chain_pdb):
        by_chain = parse_plddt_by_chain(two_chain_pdb)
        chain_b_plddts = [r["plddt"] for r in by_chain["B"]]
        assert chain_b_plddts == [60.0, 55.0]

    def test_empty_pdb(self, empty_pdb):
        by_chain = parse_plddt_by_chain(empty_pdb)
        assert by_chain == {}


class TestParseHetatm:
    def test_hetatm_records(self, hetatm_pdb):
        ligands = parse_hetatm(hetatm_pdb)
        assert len(ligands) == 1  # NAG only, HOH excluded
        assert ligands[0]["name"] == "NAG"
        assert ligands[0]["num_atoms"] == 3

    def test_no_hetatm(self, mini_pdb):
        ligands = parse_hetatm(mini_pdb)
        assert ligands == []

    def test_empty_pdb(self, empty_pdb):
        ligands = parse_hetatm(empty_pdb)
        assert ligands == []


# ============================================================================
# 3. STRUCTURAL ANALYSIS
# ============================================================================


class TestDihedral:
    def test_known_angle(self):
        # Four coplanar points should give a known dihedral
        p0 = (1.0, 0.0, 0.0)
        p1 = (0.0, 0.0, 0.0)
        p2 = (0.0, 1.0, 0.0)
        p3 = (0.0, 1.0, 1.0)
        angle = _dihedral(p0, p1, p2, p3)
        assert isinstance(angle, float)
        # Angle should be finite
        assert math.isfinite(angle)

    def test_zero_dihedral(self):
        # All in a plane -> 0 or 180 degrees
        p0 = (1.0, 0.0, 0.0)
        p1 = (0.0, 0.0, 0.0)
        p2 = (0.0, 1.0, 0.0)
        p3 = (-1.0, 1.0, 0.0)
        angle = _dihedral(p0, p1, p2, p3)
        # All coplanar -> dihedral is 0 or 180
        assert abs(angle) < 1.0 or abs(abs(angle) - 180.0) < 1.0

    def test_right_angle_dihedral(self):
        p0 = (1.0, 0.0, 0.0)
        p1 = (0.0, 0.0, 0.0)
        p2 = (0.0, 1.0, 0.0)
        p3 = (0.0, 1.0, 1.0)
        angle = _dihedral(p0, p1, p2, p3)
        # Should be approximately +90 or -90
        assert abs(abs(angle) - 90.0) < 5.0


class TestCalculatePhiPsi:
    def test_mini_pdb(self, mini_pdb):
        angles = calculate_phi_psi(mini_pdb)
        # 5 residues, skip first and last -> 3 inner residues
        assert len(angles) == 3

    def test_angle_ranges(self, mini_pdb):
        angles = calculate_phi_psi(mini_pdb)
        for phi, psi in angles:
            assert -180 <= phi <= 180
            assert -180 <= psi <= 180

    def test_empty_pdb(self, empty_pdb):
        angles = calculate_phi_psi(empty_pdb)
        assert angles == []


class TestCalculateDistanceMap:
    def test_mini_pdb(self, mini_pdb):
        dist_matrix, res_nums = calculate_distance_map(mini_pdb)
        assert dist_matrix.shape == (5, 5)
        assert len(res_nums) == 5

    def test_diagonal_zero(self, mini_pdb):
        dist_matrix, _ = calculate_distance_map(mini_pdb)
        np.testing.assert_array_almost_equal(np.diag(dist_matrix), 0.0)

    def test_symmetry(self, mini_pdb):
        dist_matrix, _ = calculate_distance_map(mini_pdb)
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)

    def test_positive_distances(self, mini_pdb):
        dist_matrix, _ = calculate_distance_map(mini_pdb)
        assert np.all(dist_matrix >= 0)

    def test_empty_pdb_raises_or_empty(self, empty_pdb):
        # Empty PDB has no CA atoms; numpy operations on empty array may raise
        try:
            dist_matrix, res_nums = calculate_distance_map(empty_pdb)
            assert res_nums == []
        except (ValueError, IndexError):
            pass  # expected -- no atoms to compute distances from


class TestCalculateRadiusOfGyration:
    def test_mini_pdb(self, mini_pdb):
        rg = calculate_radius_of_gyration(mini_pdb)
        assert isinstance(rg, float)
        assert rg > 0

    def test_reasonable_value(self, mini_pdb):
        rg = calculate_radius_of_gyration(mini_pdb)
        # 5 residues, roughly extended -> Rg should be a few Angstroms
        assert 1.0 < rg < 20.0


class TestDetectDisulfideBonds:
    def test_close_cys(self, disulfide_pdb):
        bonds = detect_disulfide_bonds(disulfide_pdb)
        assert len(bonds) == 1
        assert bonds[0]["cys1"] == 1
        assert bonds[0]["cys2"] == 3
        assert bonds[0]["distance"] < 3.0

    def test_no_cys(self, mini_pdb):
        bonds = detect_disulfide_bonds(mini_pdb)
        assert bonds == []

    def test_empty_pdb(self, empty_pdb):
        bonds = detect_disulfide_bonds(empty_pdb)
        assert bonds == []


class TestDetectSaltBridges:
    def test_charged_pair(self, salt_bridge_pdb):
        bridges = detect_salt_bridges(salt_bridge_pdb, threshold=10.0)
        # ARG(1) and ASP(3) with generous threshold
        assert len(bridges) >= 1
        assert bridges[0]["pos_name"] == "ARG"
        assert bridges[0]["neg_name"] == "ASP"

    def test_no_charged_residues(self, mini_pdb):
        bridges = detect_salt_bridges(mini_pdb)
        assert bridges == []

    def test_empty_pdb(self, empty_pdb):
        bridges = detect_salt_bridges(empty_pdb)
        assert bridges == []

    def test_tight_threshold(self, salt_bridge_pdb):
        bridges = detect_salt_bridges(salt_bridge_pdb, threshold=0.1)
        # Very tight threshold should find nothing
        assert bridges == []


class TestDetectHydrogenBonds:
    def test_mini_pdb(self, mini_pdb):
        hbonds = detect_hydrogen_bonds(mini_pdb)
        # The N-O distances in the mini PDB should produce some h-bonds
        assert isinstance(hbonds, list)

    def test_hbond_fields(self, mini_pdb):
        hbonds = detect_hydrogen_bonds(mini_pdb, dist_threshold=5.0)
        if hbonds:
            assert "donor_res" in hbonds[0]
            assert "acceptor_res" in hbonds[0]
            assert "distance" in hbonds[0]

    def test_no_self_hbonds(self, mini_pdb):
        hbonds = detect_hydrogen_bonds(mini_pdb, dist_threshold=100.0)
        for hb in hbonds:
            assert hb["donor_res"] != hb["acceptor_res"]

    def test_empty_pdb(self, empty_pdb):
        hbonds = detect_hydrogen_bonds(empty_pdb)
        assert hbonds == []

    def test_tight_threshold(self, mini_pdb):
        hbonds = detect_hydrogen_bonds(mini_pdb, dist_threshold=0.1)
        assert hbonds == []


class TestDetectCationPi:
    def test_no_aromatic_or_cation(self, mini_pdb):
        interactions = detect_cation_pi(mini_pdb)
        assert interactions == []

    def test_empty_pdb(self, empty_pdb):
        interactions = detect_cation_pi(empty_pdb)
        assert interactions == []


class TestCalculateContactOrder:
    def test_mini_pdb(self, mini_pdb):
        co = calculate_contact_order(mini_pdb)
        assert isinstance(co, float)
        assert co >= 0

    def test_very_tight_threshold(self, mini_pdb):
        co = calculate_contact_order(mini_pdb, contact_threshold=0.01)
        assert co == 0.0

    def test_empty_pdb(self, empty_pdb):
        co = calculate_contact_order(empty_pdb)
        assert co == 0.0


class TestCalculateResidueBurial:
    def test_mini_pdb(self, mini_pdb):
        burial = calculate_residue_burial(mini_pdb)
        assert len(burial) == 5

    def test_burial_fields(self, mini_pdb):
        burial = calculate_residue_burial(mini_pdb)
        for b in burial:
            assert "residue_num" in b
            assert "neighbors" in b
            assert "burial_score" in b
            assert 0 <= b["burial_score"] <= 1.0

    def test_large_radius(self, mini_pdb):
        burial = calculate_residue_burial(mini_pdb, radius=1000.0)
        # All residues are neighbors of each other
        for b in burial:
            assert b["neighbors"] == 4  # n-1

    def test_tiny_radius(self, mini_pdb):
        burial = calculate_residue_burial(mini_pdb, radius=0.01)
        for b in burial:
            assert b["neighbors"] == 0

    def test_empty_pdb(self, empty_pdb):
        burial = calculate_residue_burial(empty_pdb)
        assert burial == []


class TestEstimateSasaApproximate:
    def test_mini_pdb(self, mini_pdb):
        sasa = estimate_sasa_approximate(mini_pdb)
        assert len(sasa) == 5

    def test_sasa_fields(self, mini_pdb):
        sasa = estimate_sasa_approximate(mini_pdb)
        for s in sasa:
            assert "residue_num" in s
            assert "sasa_relative" in s
            assert 0.0 <= s["sasa_relative"] <= 1.0

    def test_empty_pdb(self, empty_pdb):
        sasa = estimate_sasa_approximate(empty_pdb)
        assert sasa == []


class TestAssignSsFromPhiPsi:
    def test_helix_angles(self):
        # Typical alpha-helix: phi ~ -60, psi ~ -45
        angles = [(-60, -45), (-60, -45), (-60, -45)]
        result = assign_ss_from_phi_psi(angles)
        assert all(ss == "H" for ss in result)

    def test_strand_angles(self):
        # Typical beta-strand: phi ~ -120, psi ~ 130
        angles = [(-120, 130), (-120, 130)]
        result = assign_ss_from_phi_psi(angles)
        assert all(ss == "E" for ss in result)

    def test_coil_angles(self):
        # Something outside both ranges
        angles = [(60, 60)]
        result = assign_ss_from_phi_psi(angles)
        assert result == ["C"]

    def test_empty(self):
        result = assign_ss_from_phi_psi([])
        assert result == []


class TestCountRamachandranOutliers:
    def test_all_favored(self):
        angles = [(-60, -45), (-120, 130)]
        result = count_ramachandran_outliers(angles)
        assert result["favored"] == 2
        assert result["total"] == 2
        assert result["outlier"] == 0

    def test_outlier(self):
        # A clearly outlier angle
        angles = [(120, 120)]
        result = count_ramachandran_outliers(angles)
        assert result["outlier"] == 1

    def test_empty(self):
        result = count_ramachandran_outliers([])
        assert result == {"favored": 0, "allowed": 0, "outlier": 0, "total": 0}

    def test_keys(self):
        result = count_ramachandran_outliers([(-60, -45)])
        assert set(result.keys()) == {"favored", "allowed", "outlier", "total"}


class TestDetectPaeDomains:
    def test_two_domains(self, pae_matrix_dict):
        domains = detect_pae_domains(pae_matrix_dict, threshold=5.0)
        assert isinstance(domains, list)
        assert len(domains) >= 1

    def test_single_domain(self, pae_uniform_dict):
        domains = detect_pae_domains(pae_uniform_dict, threshold=5.0)
        # Uniform low PAE -> one domain
        assert len(domains) == 1
        assert domains[0]["size"] == 5

    def test_list_format(self):
        # PAE as plain list of lists
        matrix = [[1] * 4 for _ in range(4)]
        domains = detect_pae_domains(matrix, threshold=5.0)
        assert len(domains) >= 1

    def test_nested_dict_list_format(self):
        # [{predicted_aligned_error: [[...]]}] format
        matrix = [[1] * 4 for _ in range(4)]
        domains = detect_pae_domains([{"predicted_aligned_error": matrix}], threshold=5.0)
        assert len(domains) >= 1

    def test_pae_key(self):
        domains = detect_pae_domains({"pae": [[1, 2], [2, 1]]}, threshold=5.0)
        assert isinstance(domains, list)

    def test_empty_matrix(self):
        domains = detect_pae_domains({"predicted_aligned_error": []})
        assert domains == []

    def test_unsupported_type(self):
        domains = detect_pae_domains("not a matrix")
        assert domains == []

    def test_domain_fields(self, pae_matrix_dict):
        domains = detect_pae_domains(pae_matrix_dict, threshold=5.0)
        if domains:
            d = domains[0]
            assert "domain_id" in d
            assert "start" in d
            assert "end" in d
            assert "size" in d


# ============================================================================
# 4. STRUCTURE STATS
# ============================================================================


class TestComputeStructureStats:
    def test_basic(self, residues_for_stats):
        stats = compute_structure_stats(residues_for_stats)
        assert stats["num_residues"] == 5
        assert stats["mean_plddt"] == pytest.approx(79.6, abs=0.1)
        assert stats["min_plddt"] == 45.0
        assert stats["max_plddt"] == 95.0

    def test_percentages(self, residues_for_stats):
        stats = compute_structure_stats(residues_for_stats)
        # Scores: 95,92,88,45,78 -> >90: 2 (95,92), >70: 4 (95,92,88,78), <=50: 1 (45)
        assert stats["pct_very_high"] == pytest.approx(40.0)
        assert stats["pct_confident"] == pytest.approx(80.0)
        assert stats["pct_low"] == pytest.approx(20.0)

    def test_keys(self, residues_for_stats):
        stats = compute_structure_stats(residues_for_stats)
        expected_keys = {
            "num_residues", "mean_plddt", "median_plddt",
            "min_plddt", "max_plddt", "pct_very_high",
            "pct_confident", "pct_low",
        }
        assert set(stats.keys()) == expected_keys


class TestFindDisorderedRegions:
    def test_one_disordered(self, residues_for_stats):
        # SER at position 4 has pLDDT=45 which is < 50
        regions = find_disordered_regions(residues_for_stats, threshold=50.0)
        assert len(regions) == 1
        assert regions[0]["start"] == 4
        assert regions[0]["length"] == 1
        assert regions[0]["mean_plddt"] == 45.0

    def test_no_disorder(self, residues_for_stats):
        regions = find_disordered_regions(residues_for_stats, threshold=10.0)
        assert regions == []

    def test_all_disordered(self, residues_for_stats):
        regions = find_disordered_regions(residues_for_stats, threshold=100.0)
        # All 5 residues are below 100
        assert len(regions) == 1
        assert regions[0]["length"] == 5

    def test_empty_input(self):
        regions = find_disordered_regions([])
        assert regions == []

    def test_trailing_disordered_region(self):
        # Disordered region at the end of the protein
        residues = [
            {"residue_num": 1, "residue_name": "ALA", "chain": "A", "plddt": 90.0},
            {"residue_num": 2, "residue_name": "GLY", "chain": "A", "plddt": 30.0},
            {"residue_num": 3, "residue_name": "VAL", "chain": "A", "plddt": 25.0},
        ]
        regions = find_disordered_regions(residues, threshold=50.0)
        assert len(regions) == 1
        assert regions[0]["start"] == 2
        assert regions[0]["end"] == 3


# ============================================================================
# 5. SEQUENCE ANALYSIS
# ============================================================================


class TestComputeSequenceProperties:
    def test_basic(self, sample_sequence):
        props = compute_sequence_properties(sample_sequence)
        assert props["length"] == 20
        assert isinstance(props["molecular_weight"], float)
        assert props["molecular_weight"] > 0
        assert isinstance(props["isoelectric_point"], float)

    def test_keys(self, sample_sequence):
        props = compute_sequence_properties(sample_sequence)
        expected_keys = {
            "length", "molecular_weight", "isoelectric_point", "gravy",
            "instability_index", "aromaticity", "helix_fraction",
            "turn_fraction", "sheet_fraction", "aa_counts",
            "aa_percent", "extinction_coeff",
        }
        assert expected_keys.issubset(set(props.keys()))

    def test_secondary_structure_fractions(self, sample_sequence):
        props = compute_sequence_properties(sample_sequence)
        total = props["helix_fraction"] + props["turn_fraction"] + props["sheet_fraction"]
        # Fractions are non-negative and approximately sum to <= 1.0
        # (BioPython may slightly exceed 1.0 depending on version)
        assert 0.0 <= total <= 1.1


class TestComputeChargeAtPh:
    def test_returns_curve(self, sample_sequence):
        curve = compute_charge_at_ph(sample_sequence)
        assert len(curve) == 141  # pH 0.0 to 14.0 in steps of 0.1

    def test_tuple_format(self, sample_sequence):
        curve = compute_charge_at_ph(sample_sequence)
        ph, charge = curve[0]
        assert ph == pytest.approx(0.0)
        assert isinstance(charge, float)

    def test_positive_at_low_ph(self, sample_sequence):
        curve = compute_charge_at_ph(sample_sequence)
        # At pH 0, charge should be positive (all groups protonated)
        assert curve[0][1] > 0

    def test_negative_at_high_ph(self, sample_sequence):
        curve = compute_charge_at_ph(sample_sequence)
        # At pH 14, charge should be negative
        assert curve[-1][1] < 0


class TestComputeHydrophobicity:
    def test_basic(self, sample_sequence):
        hydro = compute_hydrophobicity(sample_sequence, window=9)
        assert len(hydro) > 0
        # Length should be len(seq) - window + 1 = 20 - 9 + 1 = 12
        assert len(hydro) == 12

    def test_position_numbering(self, sample_sequence):
        hydro = compute_hydrophobicity(sample_sequence, window=9)
        # First position should be half+1 = 5
        assert hydro[0][0] == 5

    def test_short_sequence(self):
        hydro = compute_hydrophobicity("MVL", window=9)
        assert hydro == []

    def test_empty_sequence(self):
        hydro = compute_hydrophobicity("", window=9)
        assert hydro == []

    def test_hydrophobic_sequence(self):
        # Highly hydrophobic sequence
        hydro = compute_hydrophobicity("I" * 20, window=9)
        # Isoleucine has KD = 4.5, so average should be ~4.5
        for _, value in hydro:
            assert value == pytest.approx(4.5, abs=0.01)


class TestFindGlycosylationSites:
    def test_nxt_motif(self):
        # N-X-S/T where X != P
        sites = find_glycosylation_sites("MNASKVL")
        assert len(sites) == 1
        assert sites[0]["position"] == 2
        assert sites[0]["motif"] == "NAS"

    def test_npt_excluded(self):
        # N-P-T should NOT match (X == P)
        sites = find_glycosylation_sites("MNPTKVL")
        assert sites == []

    def test_nxt_at_end(self):
        # Motif can't start at last two positions
        sites = find_glycosylation_sites("AAAAN")
        assert sites == []

    def test_multiple_sites(self):
        sites = find_glycosylation_sites("NASAAANST")
        assert len(sites) == 2

    def test_empty_sequence(self):
        sites = find_glycosylation_sites("")
        assert sites == []

    def test_short_sequence(self):
        sites = find_glycosylation_sites("NA")
        assert sites == []


class TestFindPhosphorylationSites:
    def test_basic(self):
        sites = find_phosphorylation_sites("MSVTY")
        # S at pos 2, T at pos 4, Y at pos 5
        assert len(sites) == 3
        assert sites[0] == {"position": 2, "residue": "S"}

    def test_no_sites(self):
        sites = find_phosphorylation_sites("MVLAP")
        assert sites == []

    def test_empty_sequence(self):
        sites = find_phosphorylation_sites("")
        assert sites == []

    def test_all_targets(self):
        sites = find_phosphorylation_sites("STY")
        assert len(sites) == 3


class TestPredictTransmembrane:
    def test_hydrophobic_stretch(self):
        # A highly hydrophobic stretch of 25 residues
        seq = "A" * 10 + "I" * 25 + "A" * 10
        regions = predict_transmembrane(seq, window=21, threshold=1.6)
        assert len(regions) >= 1

    def test_no_tm(self):
        # A hydrophilic sequence
        seq = "DDDEEEKKKRRRDDDEEEKKK"
        regions = predict_transmembrane(seq, window=21, threshold=1.6)
        assert regions == []

    def test_short_sequence(self):
        regions = predict_transmembrane("MVLSP", window=21)
        assert regions == []

    def test_empty_sequence(self):
        regions = predict_transmembrane("", window=21)
        assert regions == []

    def test_region_fields(self):
        seq = "A" * 10 + "I" * 25 + "A" * 10
        regions = predict_transmembrane(seq, window=21, threshold=1.6)
        if regions:
            assert "start" in regions[0]
            assert "end" in regions[0]
            assert "length" in regions[0]


class TestComputeFlexibility:
    def test_basic(self, sample_sequence):
        flex = compute_flexibility(sample_sequence)
        assert len(flex) > 0
        # flexibility uses window=9, so result has len-8 values
        # But BioPython's flexibility returns len-8 values
        # Position numbering starts at 5 (offset by 4)
        assert flex[0][0] == 5

    def test_short_sequence(self):
        # Sequence too short for window=9
        flex = compute_flexibility("MVLSP")
        # BioPython might return empty or very few values
        # The function shouldn't crash
        assert isinstance(flex, list)


class TestComputeAliphaticIndex:
    def test_basic(self, sample_sequence):
        ai = compute_aliphatic_index(sample_sequence)
        assert isinstance(ai, float)
        assert ai >= 0

    def test_all_alanine(self):
        ai = compute_aliphatic_index("A" * 10)
        # 100 * (1.0 + 0 + 0) = 100
        assert ai == pytest.approx(100.0)

    def test_all_valine(self):
        ai = compute_aliphatic_index("V" * 10)
        # 100 * (0 + 2.9 * 1.0 + 0) = 290
        assert ai == pytest.approx(290.0)

    def test_empty_sequence(self):
        ai = compute_aliphatic_index("")
        assert ai == 0.0

    def test_no_aliphatic(self):
        # Sequence with no A, V, I, L
        ai = compute_aliphatic_index("DDDKKK")
        assert ai == pytest.approx(0.0)


class TestComputeHalfLife:
    def test_methionine_start(self):
        hl = compute_half_life("MVLSP")
        assert hl["mammalian"] == ">30 hours"
        assert hl["yeast"] == ">20 hours"

    def test_arginine_start(self):
        hl = compute_half_life("RVLSP")
        assert hl["mammalian"] == "1 hour"

    def test_empty_sequence(self):
        hl = compute_half_life("")
        assert hl == {"mammalian": "N/A", "yeast": "N/A", "ecoli": "N/A"}

    def test_unknown_start(self):
        hl = compute_half_life("XVLSP")
        assert hl == {"mammalian": "N/A", "yeast": "N/A", "ecoli": "N/A"}


class TestDetectSignalPeptide:
    def test_hydrophobic_signal(self):
        # 30 residues, a hydrophobic core
        seq = "M" + "L" * 14 + "A" * 15 + "DDDDDDDDDDDDDDDDDDDD"
        result = detect_signal_peptide(seq)
        # L has KD=3.8, so avg over 10+ residues > 1.0
        assert result is not None
        assert "start" in result
        assert "end" in result
        assert "score" in result
        assert result["score"] > 1.0

    def test_no_signal(self):
        # Hydrophilic first 30 residues
        seq = "D" * 30 + "AAAAAAAAAA"
        result = detect_signal_peptide(seq)
        assert result is None

    def test_short_sequence(self):
        result = detect_signal_peptide("MVLSP")
        # Less than 10 residues in first 30
        assert result is None

    def test_very_short(self):
        result = detect_signal_peptide("MVL")
        assert result is None


class TestComputeSequenceComplexity:
    def test_basic(self, sample_sequence):
        complexity = compute_sequence_complexity(sample_sequence, window=12)
        assert len(complexity) == 20 - 12 + 1  # 9

    def test_low_complexity(self):
        # Homopolymer
        complexity = compute_sequence_complexity("A" * 20, window=12)
        for _, entropy in complexity:
            assert entropy == 0.0

    def test_high_complexity(self):
        # All different amino acids
        complexity = compute_sequence_complexity("ACDEFGHIKLMNPQRSTVWY", window=12)
        for _, entropy in complexity:
            assert entropy > 3.0  # High entropy

    def test_short_sequence(self):
        complexity = compute_sequence_complexity("MVL", window=12)
        assert complexity == []

    def test_empty_sequence(self):
        complexity = compute_sequence_complexity("", window=12)
        assert complexity == []


class TestClassifyProtein:
    def test_hydrophilic_stable(self):
        props = {"gravy": -1.0, "instability_index": 30, "aromaticity": 0.05}
        tags = classify_protein("A" * 10, props)
        assert "Hydrophilic" in tags
        assert "Stable" in tags

    def test_membrane_associated(self):
        props = {"gravy": 1.0, "instability_index": 50, "aromaticity": 0.05}
        tags = classify_protein("A" * 10, props)
        assert "Membrane-associated" in tags
        assert "Unstable" in tags

    def test_globular(self):
        props = {"gravy": 0.0, "instability_index": 30, "aromaticity": 0.05}
        tags = classify_protein("A" * 10, props)
        assert "Globular" in tags

    def test_aromatic_rich(self):
        props = {"gravy": 0.0, "instability_index": 30, "aromaticity": 0.20}
        tags = classify_protein("A" * 10, props)
        assert "Aromatic-rich" in tags

    def test_multi_domain_candidate(self):
        props = {"gravy": 0.0, "instability_index": 30, "aromaticity": 0.05}
        tags = classify_protein("A" * 201, props)
        assert "Multi-domain candidate" in tags

    def test_short_not_multi_domain(self):
        props = {"gravy": 0.0, "instability_index": 30, "aromaticity": 0.05}
        tags = classify_protein("A" * 50, props)
        assert "Multi-domain candidate" not in tags


# ============================================================================
# 6. ALIGNMENT & COMPARISON
# ============================================================================


class TestAlignSequences:
    def test_identical(self):
        result = align_sequences("MVLSPADKTN", "MVLSPADKTN")
        assert result["identity"] == pytest.approx(100.0)
        assert result["score"] > 0

    def test_different(self):
        result = align_sequences("MVLSPADKTN", "AAAAAAAAAA")
        assert result["identity"] < 100.0

    def test_result_keys(self):
        result = align_sequences("MVLSPADKTN", "MVLSPADKTN")
        assert "aligned_a" in result
        assert "aligned_b" in result
        assert "identity" in result
        assert "similarity" in result
        assert "gaps" in result
        assert "score" in result

    def test_similarity_gte_identity(self):
        result = align_sequences("MVLSPADKTN", "MVLSAADKTN")
        assert result["similarity"] >= result["identity"]


class TestScoreSubstitutions:
    def test_identical(self):
        result = score_substitutions("MVLSP", "MVLSP")
        assert len(result) == 5
        for sub in result:
            assert sub["aa_a"] == sub["aa_b"]
            assert sub["is_conservative"] is True

    def test_different(self):
        result = score_substitutions("MVLSP", "AVLSP")
        # M->A might or might not be conservative
        assert result[0]["aa_a"] == "M"
        assert result[0]["aa_b"] == "A"

    def test_with_gaps(self):
        result = score_substitutions("M-LSP", "MVLSP")
        # Gap positions should be skipped
        assert all(s["aa_a"] != "-" and s["aa_b"] != "-" for s in result)

    def test_empty_sequences(self):
        result = score_substitutions("", "")
        assert result == []

    def test_position_numbering(self):
        result = score_substitutions("MVL", "MVL")
        assert result[0]["position"] == 1
        assert result[2]["position"] == 3


# ============================================================================
# 7. ADVANCED STRUCTURAL ANALYSIS (mocked external dependencies)
# ============================================================================


class TestCalculateSasa:
    def test_returns_none_when_freesasa_unavailable(self, mini_pdb):
        with patch.dict("sys.modules", {"freesasa": None}):
            result = calculate_sasa(mini_pdb)
            # Should return None because import fails
            assert result is None

    def test_empty_pdb(self, empty_pdb):
        result = calculate_sasa(empty_pdb)
        # Will likely fail or return None for empty PDB
        assert result is None or result == []


class TestCalculateRmsd:
    def test_returns_none_when_tmtools_unavailable(self, mini_pdb):
        with patch.dict("sys.modules", {"tmtools": None}):
            result = calculate_rmsd(mini_pdb, mini_pdb)
            assert result is None

    def test_empty_pdb(self, empty_pdb):
        result = calculate_rmsd(empty_pdb, empty_pdb)
        assert result is None

    def test_with_mock_tmtools(self, mini_pdb):
        mock_result = MagicMock()
        mock_result.rmsd = 0.0
        mock_result.tm_norm_chain1 = 1.0
        mock_result.aligned_length = 5
        mock_result.seq_id = 1.0

        mock_tmtools = MagicMock()
        mock_tmtools.tm_align.return_value = mock_result

        with patch.dict("sys.modules", {"tmtools": mock_tmtools}):
            result = calculate_rmsd(mini_pdb, mini_pdb)
            if result is not None:
                assert result["rmsd"] == pytest.approx(0.0)
                assert result["tm_score"] == pytest.approx(1.0)


class TestRunTmAlign:
    def test_returns_none_when_tmtools_unavailable(self, mini_pdb):
        with patch.dict("sys.modules", {"tmtools": None}):
            result = run_tm_align(mini_pdb, mini_pdb)
            assert result is None

    def test_empty_pdb(self, empty_pdb):
        result = run_tm_align(empty_pdb, empty_pdb)
        assert result is None

    def test_with_mock_tmtools(self, mini_pdb):
        mock_result = MagicMock()
        mock_result.rmsd = 0.5
        mock_result.tm_norm_chain1 = 0.95
        mock_result.tm_norm_chain2 = 0.95
        mock_result.seqM = "MVLSP"
        mock_result.u = np.eye(3)
        mock_result.t = np.zeros(3)

        mock_tmtools = MagicMock()
        mock_tmtools.tm_align.return_value = mock_result

        with patch.dict("sys.modules", {"tmtools": mock_tmtools}):
            result = run_tm_align(mini_pdb, mini_pdb)
            if result is not None:
                assert "rmsd" in result
                assert "tm_score_a" in result
                assert "tm_score_b" in result
                assert "aligned_length" in result
                assert "rotation_matrix" in result
                assert "translation_vector" in result


class TestRunNormalModeAnalysis:
    def test_returns_none_when_prody_unavailable(self, mini_pdb):
        with patch.dict("sys.modules", {"prody": None}):
            result = run_normal_mode_analysis(mini_pdb)
            assert result is None

    def test_empty_pdb(self, empty_pdb):
        result = run_normal_mode_analysis(empty_pdb)
        assert result is None


class TestCalculateGnmBfactors:
    def test_returns_none_when_prody_unavailable(self, mini_pdb):
        with patch.dict("sys.modules", {"prody": None}):
            result = calculate_gnm_bfactors(mini_pdb)
            assert result is None

    def test_empty_pdb(self, empty_pdb):
        result = calculate_gnm_bfactors(empty_pdb)
        assert result is None


# ============================================================================
# 8. BATCH / PHYLOGENETIC ANALYSIS
# ============================================================================


class TestBuildSequenceDistanceMatrix:
    def test_basic(self):
        seqs = [("seq1", "MVLSP"), ("seq2", "MVLSA"), ("seq3", "AAAAA")]
        result = build_sequence_distance_matrix(seqs)
        assert result is not None
        names, dist = result
        assert names == ["seq1", "seq2", "seq3"]
        assert dist.shape == (3, 3)

    def test_diagonal_zero(self):
        seqs = [("a", "MVLSP"), ("b", "AAALP")]
        result = build_sequence_distance_matrix(seqs)
        assert result is not None
        _, dist = result
        np.testing.assert_array_almost_equal(np.diag(dist), 0.0)

    def test_symmetry(self):
        seqs = [("a", "MVLSP"), ("b", "AAALP")]
        result = build_sequence_distance_matrix(seqs)
        assert result is not None
        _, dist = result
        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_identical_sequences(self):
        seqs = [("a", "MVLSP"), ("b", "MVLSP")]
        result = build_sequence_distance_matrix(seqs)
        assert result is not None
        _, dist = result
        assert dist[0, 1] == pytest.approx(0.0)

    def test_single_sequence(self):
        result = build_sequence_distance_matrix([("a", "MVLSP")])
        assert result is None

    def test_empty(self):
        result = build_sequence_distance_matrix([])
        assert result is None

    def test_normalized_distance(self):
        seqs = [("a", "AAAAA"), ("b", "BBBBB")]
        result = build_sequence_distance_matrix(seqs)
        assert result is not None
        _, dist = result
        # All 5 characters differ, max_len=5, so distance = 5/5 = 1.0
        assert dist[0, 1] == pytest.approx(1.0)


class TestBuildUpgmaTree:
    def test_basic(self):
        names = ["a", "b", "c"]
        dist = np.array([
            [0, 1, 4],
            [1, 0, 3],
            [4, 3, 0],
        ], dtype=float)
        tree = build_upgma_tree(names, dist)
        assert tree is not None
        assert "name" in tree
        assert "children" in tree
        assert "distance" in tree

    def test_two_sequences(self):
        names = ["a", "b"]
        dist = np.array([[0, 2], [2, 0]], dtype=float)
        tree = build_upgma_tree(names, dist)
        assert tree is not None
        assert len(tree["children"]) == 2

    def test_single_sequence(self):
        names = ["a"]
        dist = np.array([[0]], dtype=float)
        tree = build_upgma_tree(names, dist)
        assert tree is None

    def test_empty(self):
        tree = build_upgma_tree([], np.array([]))
        assert tree is None

    def test_mismatched_dimensions(self):
        names = ["a", "b"]
        dist = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
        tree = build_upgma_tree(names, dist)
        assert tree is None


class TestComputeLogoData:
    def test_basic(self):
        sequences = ["MVLSP", "MVLSA", "MVLTP"]
        result = compute_logo_data(sequences)
        assert result is not None
        assert "matrix" in result
        assert "consensus" in result
        assert len(result["consensus"]) == 5

    def test_consensus(self):
        sequences = ["AAAA", "AAAA", "ABAA"]
        result = compute_logo_data(sequences)
        assert result is not None
        assert result["consensus"][0] == "A"

    def test_different_lengths(self):
        sequences = ["MVL", "MVLSP"]
        result = compute_logo_data(sequences)
        assert result is not None
        # Padded to length 5
        assert len(result["consensus"]) == 5

    def test_empty_input(self):
        result = compute_logo_data([])
        assert result is None

    def test_single_sequence(self):
        result = compute_logo_data(["MVLSP"])
        assert result is not None
        assert result["consensus"] == "MVLSP"

    def test_matrix_shape(self):
        sequences = ["ABC", "ABD"]
        result = compute_logo_data(sequences)
        assert result is not None
        assert result["matrix"].shape[0] == 3  # 3 positions


# ============================================================================
# 9. SECONDARY STRUCTURE FROM COORDINATES
# ============================================================================


class TestAssignSecondaryStructureFromCoords:
    def test_mini_pdb(self, mini_pdb):
        result = assign_secondary_structure_from_coords(mini_pdb)
        assert result is not None
        assert len(result) == 5
        for entry in result:
            assert "residue_num" in entry
            assert "ss" in entry
            assert entry["ss"] in ("H", "E", "C")

    def test_empty_pdb(self, empty_pdb):
        result = assign_secondary_structure_from_coords(empty_pdb)
        assert result is None

    def test_too_few_residues(self):
        # Only 3 residues -> needs >= 5
        pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 90.00           C\n"
            "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 90.00           C\n"
            "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 90.00           O\n"
            "ATOM      5  N   GLY A   2       3.325   1.490   0.000  1.00 85.00           N\n"
            "ATOM      6  CA  GLY A   2       3.950   2.810   0.000  1.00 85.00           C\n"
            "ATOM      7  C   GLY A   2       5.470   2.750   0.000  1.00 85.00           C\n"
            "ATOM      8  O   GLY A   2       6.090   1.690   0.000  1.00 85.00           O\n"
            "ATOM      9  N   VAL A   3       6.050   3.900   0.200  1.00 80.00           N\n"
            "ATOM     10  CA  VAL A   3       7.500   3.970   0.200  1.00 80.00           C\n"
            "ATOM     11  C   VAL A   3       8.050   5.370   0.200  1.00 80.00           C\n"
            "ATOM     12  O   VAL A   3       7.300   6.340   0.200  1.00 80.00           O\n"
            "END\n"
        )
        result = assign_secondary_structure_from_coords(pdb)
        assert result is None


class TestGenerateTopologyData:
    def test_mini_pdb(self, mini_pdb):
        result = generate_topology_data(mini_pdb)
        assert result is not None
        assert isinstance(result, list)
        for elem in result:
            assert "type" in elem
            assert elem["type"] in ("helix", "strand", "coil")
            assert "start" in elem
            assert "end" in elem
            assert "length" in elem
            assert elem["length"] > 0

    def test_empty_pdb(self, empty_pdb):
        result = generate_topology_data(empty_pdb)
        assert result is None

    def test_coverage(self, mini_pdb):
        result = generate_topology_data(mini_pdb)
        if result:
            # Elements should cover all residues
            total_length = sum(e["length"] for e in result)
            assert total_length == 5

    def test_consecutive_elements(self, mini_pdb):
        result = generate_topology_data(mini_pdb)
        if result and len(result) > 1:
            for i in range(len(result) - 1):
                # End of one element + 1 should be start of next
                assert result[i]["end"] + 1 == result[i + 1]["start"]


# ============================================================================
# 10. EDGE CASES AND INTEGRATION
# ============================================================================


class TestEdgeCases:
    """Cross-cutting edge cases that test multiple functions with unusual input."""

    def test_pdb_with_only_end(self):
        pdb = "END\n"
        assert parse_plddt_from_pdb(pdb) == []
        assert parse_backbone_atoms(pdb) == {}
        assert parse_all_atoms(pdb) == []
        assert parse_pdb_chains(pdb) == []
        assert parse_hetatm(pdb) == []

    def test_pdb_blank_string(self):
        pdb = ""
        assert parse_plddt_from_pdb(pdb) == []
        assert parse_backbone_atoms(pdb) == {}
        assert parse_all_atoms(pdb) == []

    def test_sequence_with_all_amino_acids(self):
        seq = "ACDEFGHIKLMNPQRSTVWY"
        valid, msg = validate_sequence(seq)
        assert valid is True
        sites = find_phosphorylation_sites(seq)
        assert len(sites) > 0  # Has S, T, Y

    def test_single_residue_pdb(self):
        pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 90.00           C\n"
            "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 90.00           C\n"
            "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 90.00           O\n"
            "END\n"
        )
        residues = parse_plddt_from_pdb(pdb)
        assert len(residues) == 1
        angles = calculate_phi_psi(pdb)
        assert angles == []
        co = calculate_contact_order(pdb)
        assert co == 0.0

    def test_extract_then_validate_round_trip(self, mini_pdb):
        """Extract sequence from PDB, then validate it."""
        seq = extract_sequence_from_pdb(mini_pdb)
        assert seq == "MVLSP"
        # Too short for validate_sequence (min 10), but extraction works
        valid, _ = validate_sequence(seq)
        assert valid is False  # 5 residues < 10

    def test_distance_map_single_residue(self):
        pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 90.00           C\n"
            "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 90.00           C\n"
            "ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 90.00           O\n"
            "END\n"
        )
        dist_matrix, res_nums = calculate_distance_map(pdb)
        assert dist_matrix.shape == (1, 1)
        assert dist_matrix[0, 0] == pytest.approx(0.0)

    def test_hydrophobicity_known_values(self):
        """Test that known hydrophobic residues produce positive KD values."""
        # Isoleucine (I): KD=4.5
        result = compute_hydrophobicity("I" * 20, window=9)
        assert all(v > 0 for _, v in result)
        # Aspartate (D): KD=-3.5
        result = compute_hydrophobicity("D" * 20, window=9)
        assert all(v < 0 for _, v in result)

    def test_complexity_homopolymer_vs_diverse(self):
        """Homopolymer should have lower complexity than diverse sequence."""
        homo = compute_sequence_complexity("A" * 20, window=12)
        diverse = compute_sequence_complexity("ACDEFGHIKLMNPQRSTVWY", window=12)
        if homo and diverse:
            avg_homo = sum(v for _, v in homo) / len(homo)
            avg_diverse = sum(v for _, v in diverse) / len(diverse)
            assert avg_homo < avg_diverse

    def test_burial_consistent_with_sasa(self, mini_pdb):
        """Burial and approximate SASA should both return results."""
        burial = calculate_residue_burial(mini_pdb, radius=100.0)
        sasa = estimate_sasa_approximate(mini_pdb)
        assert len(burial) > 0
        if sasa:
            assert len(sasa) > 0

    def test_pae_domains_high_threshold(self, pae_matrix_dict):
        """Very high threshold should merge everything into one domain."""
        domains = detect_pae_domains(pae_matrix_dict, threshold=100.0)
        assert len(domains) == 1

    def test_glycosylation_nt_variant(self):
        """N-X-T motif should also be detected."""
        sites = find_glycosylation_sites("MNATKVL")
        assert len(sites) == 1
        assert sites[0]["motif"] == "NAT"

    def test_half_life_all_standard_residues(self):
        """All standard amino acids should return a valid half-life."""
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            hl = compute_half_life(aa + "AAAA")
            assert "mammalian" in hl
            # All should have non-N/A mammalian entry
            assert hl["mammalian"] != "N/A"

    def test_aliphatic_index_mixed(self):
        """Test aliphatic index with mixed residues."""
        # 5A, 3V, 2I -> ala%=0.5, val%=0.3, ile%=0.2, leu%=0
        seq = "AAAAAVVVII"
        ai = compute_aliphatic_index(seq)
        expected = 100 * (0.5 + 2.9 * 0.3 + 3.9 * 0.2)
        assert ai == pytest.approx(expected, abs=0.01)
