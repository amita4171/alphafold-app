"""Shared fixtures for analysis.py test suite."""
from __future__ import annotations

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Minimal PDB text: 5 residues (MET-VAL-LEU-SER-PRO), chain A
# Backbone atoms N, CA, C, O for each residue.
# B-factor column encodes pLDDT-like values: 95, 92, 88, 45, 78
# ---------------------------------------------------------------------------
MINI_PDB = """\
ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 95.00           N
ATOM      2  CA  MET A   1       1.458   0.000   0.000  1.00 95.00           C
ATOM      3  C   MET A   1       2.009   1.420   0.000  1.00 95.00           C
ATOM      4  O   MET A   1       1.251   2.390   0.000  1.00 95.00           O
ATOM      5  N   VAL A   2       3.325   1.490   0.000  1.00 92.00           N
ATOM      6  CA  VAL A   2       3.950   2.810   0.000  1.00 92.00           C
ATOM      7  C   VAL A   2       5.470   2.750   0.000  1.00 92.00           C
ATOM      8  O   VAL A   2       6.090   1.690   0.000  1.00 92.00           O
ATOM      9  N   LEU A   3       6.050   3.900   0.200  1.00 88.00           N
ATOM     10  CA  LEU A   3       7.500   3.970   0.200  1.00 88.00           C
ATOM     11  C   LEU A   3       8.050   5.370   0.200  1.00 88.00           C
ATOM     12  O   LEU A   3       7.300   6.340   0.200  1.00 88.00           O
ATOM     13  N   SER A   4       9.370   5.440   0.200  1.00 45.00           N
ATOM     14  CA  SER A   4      10.000   6.760   0.200  1.00 45.00           C
ATOM     15  C   SER A   4      11.520   6.700   0.200  1.00 45.00           C
ATOM     16  O   SER A   4      12.140   5.640   0.200  1.00 45.00           O
ATOM     17  N   PRO A   5      12.100   7.850   0.400  1.00 78.00           N
ATOM     18  CA  PRO A   5      13.550   7.920   0.400  1.00 78.00           C
ATOM     19  C   PRO A   5      14.100   9.320   0.400  1.00 78.00           C
ATOM     20  O   PRO A   5      13.350  10.290   0.400  1.00 78.00           O
END
"""

# Two-chain PDB (chain A residues 1-3, chain B residues 1-2)
TWO_CHAIN_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 90.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 90.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 90.00           O
ATOM      5  N   GLY A   2       3.325   1.490   0.000  1.00 85.00           N
ATOM      6  CA  GLY A   2       3.950   2.810   0.000  1.00 85.00           C
ATOM      7  C   GLY A   2       5.470   2.750   0.000  1.00 85.00           C
ATOM      8  O   GLY A   2       6.090   1.690   0.000  1.00 85.00           O
ATOM      9  N   CYS A   3       6.050   3.900   0.200  1.00 70.00           N
ATOM     10  CA  CYS A   3       7.500   3.970   0.200  1.00 70.00           C
ATOM     11  C   CYS A   3       8.050   5.370   0.200  1.00 70.00           C
ATOM     12  O   CYS A   3       7.300   6.340   0.200  1.00 70.00           O
ATOM     13  N   ASP B   1      20.000  20.000  20.000  1.00 60.00           N
ATOM     14  CA  ASP B   1      21.458  20.000  20.000  1.00 60.00           C
ATOM     15  C   ASP B   1      22.009  21.420  20.000  1.00 60.00           C
ATOM     16  O   ASP B   1      21.251  22.390  20.000  1.00 60.00           O
ATOM     17  N   GLU B   2      23.325  21.490  20.000  1.00 55.00           N
ATOM     18  CA  GLU B   2      23.950  22.810  20.000  1.00 55.00           C
ATOM     19  C   GLU B   2      25.470  22.750  20.000  1.00 55.00           C
ATOM     20  O   GLU B   2      26.090  21.690  20.000  1.00 55.00           O
END
"""

# PDB with two CYS SG atoms close enough for a disulfide bond (< 3.0 A)
DISULFIDE_PDB = """\
ATOM      1  N   CYS A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  CYS A   1       1.458   0.000   0.000  1.00 90.00           C
ATOM      3  C   CYS A   1       2.009   1.420   0.000  1.00 90.00           C
ATOM      4  O   CYS A   1       1.251   2.390   0.000  1.00 90.00           O
ATOM      5  SG  CYS A   1       1.000   1.000   0.500  1.00 90.00           S
ATOM      6  N   ALA A   2       3.325   1.490   0.000  1.00 88.00           N
ATOM      7  CA  ALA A   2       3.950   2.810   0.000  1.00 88.00           C
ATOM      8  C   ALA A   2       5.470   2.750   0.000  1.00 88.00           C
ATOM      9  O   ALA A   2       6.090   1.690   0.000  1.00 88.00           O
ATOM     10  N   CYS A   3       6.050   3.900   0.200  1.00 85.00           N
ATOM     11  CA  CYS A   3       7.500   3.970   0.200  1.00 85.00           C
ATOM     12  C   CYS A   3       8.050   5.370   0.200  1.00 85.00           C
ATOM     13  O   CYS A   3       7.300   6.340   0.200  1.00 85.00           O
ATOM     14  SG  CYS A   3       1.500   1.500   0.300  1.00 85.00           S
END
"""

# PDB with HETATM records (ligand)
HETATM_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 90.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 90.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00 90.00           O
HETATM   50  C1  NAG A 100      10.000  10.000  10.000  1.00 50.00           C
HETATM   51  C2  NAG A 100      11.000  10.000  10.000  1.00 50.00           C
HETATM   52  O1  NAG A 100      10.500  11.000  10.000  1.00 50.00           O
HETATM   53  O   HOH A 200      20.000  20.000  20.000  1.00 30.00           O
END
"""

# PDB with charged residues for salt bridge detection
SALT_BRIDGE_PDB = """\
ATOM      1  N   ARG A   1       0.000   0.000   0.000  1.00 90.00           N
ATOM      2  CA  ARG A   1       1.458   0.000   0.000  1.00 90.00           C
ATOM      3  C   ARG A   1       2.009   1.420   0.000  1.00 90.00           C
ATOM      4  O   ARG A   1       1.251   2.390   0.000  1.00 90.00           O
ATOM      5  NH1 ARG A   1       2.000   2.000   0.500  1.00 90.00           N
ATOM      6  NH2 ARG A   1       2.500   2.000   0.500  1.00 90.00           N
ATOM      7  NE  ARG A   1       2.250   1.800   0.500  1.00 90.00           N
ATOM      8  N   ALA A   2       3.325   1.490   0.000  1.00 88.00           N
ATOM      9  CA  ALA A   2       3.950   2.810   0.000  1.00 88.00           C
ATOM     10  C   ALA A   2       5.470   2.750   0.000  1.00 88.00           C
ATOM     11  O   ALA A   2       6.090   1.690   0.000  1.00 88.00           O
ATOM     12  N   ASP A   3       6.050   3.900   0.200  1.00 85.00           N
ATOM     13  CA  ASP A   3       7.500   3.970   0.200  1.00 85.00           C
ATOM     14  C   ASP A   3       8.050   5.370   0.200  1.00 85.00           C
ATOM     15  O   ASP A   3       7.300   6.340   0.200  1.00 85.00           O
ATOM     16  OD1 ASP A   3       3.000   2.500   0.500  1.00 85.00           O
ATOM     17  OD2 ASP A   3       3.200   2.700   0.500  1.00 85.00           O
END
"""


@pytest.fixture
def mini_pdb():
    """5-residue PDB text (MET-VAL-LEU-SER-PRO)."""
    return MINI_PDB


@pytest.fixture
def two_chain_pdb():
    """Two-chain PDB (A: ALA-GLY-CYS, B: ASP-GLU)."""
    return TWO_CHAIN_PDB


@pytest.fixture
def disulfide_pdb():
    """PDB with two CYS residues whose SG atoms are < 3.0 A apart."""
    return DISULFIDE_PDB


@pytest.fixture
def hetatm_pdb():
    """PDB with HETATM records (NAG ligand + HOH)."""
    return HETATM_PDB


@pytest.fixture
def salt_bridge_pdb():
    """PDB with ARG and ASP charged residues near each other."""
    return SALT_BRIDGE_PDB


@pytest.fixture
def empty_pdb():
    """PDB text with no ATOM records."""
    return "END\n"


@pytest.fixture
def sample_sequence():
    """A 20-residue protein sequence covering many amino acid types."""
    return "MVLSPADKTNVKAAWGKVGA"


@pytest.fixture
def short_sequence():
    """A 5-residue sequence (too short for validate_sequence)."""
    return "MVLSP"


@pytest.fixture
def long_sequence():
    """A sequence that exceeds the 400-residue limit."""
    return "A" * 401


@pytest.fixture
def fasta_text():
    """Multi-sequence FASTA text."""
    return (
        ">seq1 hemoglobin alpha\n"
        "MVLSPADKTNVKAAWGKVGA\n"
        ">seq2 hemoglobin beta\n"
        "MVHLTPEEKSAVTALWGKVN\n"
    )


@pytest.fixture
def residues_for_stats():
    """Residue dicts matching parse_plddt_from_pdb output format."""
    return [
        {"residue_num": 1, "residue_name": "MET", "chain": "A", "plddt": 95.0},
        {"residue_num": 2, "residue_name": "VAL", "chain": "A", "plddt": 92.0},
        {"residue_num": 3, "residue_name": "LEU", "chain": "A", "plddt": 88.0},
        {"residue_num": 4, "residue_name": "SER", "chain": "A", "plddt": 45.0},
        {"residue_num": 5, "residue_name": "PRO", "chain": "A", "plddt": 78.0},
    ]


@pytest.fixture
def pae_matrix_dict():
    """A small PAE matrix (5x5) with two domains."""
    # Residues 0-2 form one domain (low PAE), residues 3-4 form another
    matrix = [
        [0, 2, 3, 15, 18],
        [2, 0, 2, 16, 17],
        [3, 2, 0, 14, 19],
        [15, 16, 14, 0, 2],
        [18, 17, 19, 2, 0],
    ]
    return {"predicted_aligned_error": matrix}


@pytest.fixture
def pae_uniform_dict():
    """A uniform low-PAE matrix (single domain)."""
    matrix = [[1] * 5 for _ in range(5)]
    return {"predicted_aligned_error": matrix}
