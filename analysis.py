"""Pure analysis/computation functions for AlphaFold Explorer. No Streamlit or Plotly."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────

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

HALF_LIFE_TABLE = {
    "M": {"mammalian": ">30 hours", "yeast": ">20 hours", "ecoli": ">10 hours"},
    "S": {"mammalian": ">30 hours", "yeast": ">20 hours", "ecoli": ">10 hours"},
    "A": {"mammalian": ">30 hours", "yeast": ">20 hours", "ecoli": ">10 hours"},
    "T": {"mammalian": ">30 hours", "yeast": ">20 hours", "ecoli": ">10 hours"},
    "V": {"mammalian": ">30 hours", "yeast": ">20 hours", "ecoli": ">10 hours"},
    "G": {"mammalian": ">30 hours", "yeast": ">20 hours", "ecoli": ">10 hours"},
    "C": {"mammalian": "1.2 hours", "yeast": ">20 hours", "ecoli": ">10 hours"},
    "P": {"mammalian": ">20 hours", "yeast": ">20 hours", "ecoli": "?"},
    "I": {"mammalian": "20 hours", "yeast": "30 min", "ecoli": ">10 hours"},
    "E": {"mammalian": "1 hour", "yeast": "30 min", "ecoli": ">10 hours"},
    "Y": {"mammalian": "2.8 hours", "yeast": "10 min", "ecoli": "2 min"},
    "Q": {"mammalian": "0.8 hours", "yeast": "10 min", "ecoli": ">10 hours"},
    "D": {"mammalian": "1.1 hours", "yeast": "3 min", "ecoli": ">10 hours"},
    "N": {"mammalian": "1.4 hours", "yeast": "3 min", "ecoli": ">10 hours"},
    "H": {"mammalian": "3.5 hours", "yeast": "10 min", "ecoli": ">10 hours"},
    "L": {"mammalian": "5.5 hours", "yeast": "3 min", "ecoli": "2 min"},
    "F": {"mammalian": "1.1 hours", "yeast": "3 min", "ecoli": "2 min"},
    "W": {"mammalian": "2.8 hours", "yeast": "3 min", "ecoli": "2 min"},
    "K": {"mammalian": "1.3 hours", "yeast": "3 min", "ecoli": "2 min"},
    "R": {"mammalian": "1 hour", "yeast": "2 min", "ecoli": "2 min"},
}


# ── Sequence helpers ─────────────────────────────────────────────────────

def clean_sequence(seq: str) -> str:
    """Strip FASTA headers, non-alpha chars, uppercase."""
    lines = seq.strip().split("\n")
    cleaned = []
    for line in lines:
        if line.startswith(">"):
            continue
        cleaned.append(re.sub(r"[^A-Za-z]", "", line))
    return "".join(cleaned).upper()


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse multi-sequence FASTA."""
    sequences: list[tuple[str, str]] = []
    current_name: str | None = None
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
    """Check 10-400 residues, valid AA."""
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


# ── PDB parsing ──────────────────────────────────────────────────────────

def extract_sequence_from_pdb(pdb_text: str) -> str:
    """Sequence from CA atoms."""
    residues = parse_plddt_from_pdb(pdb_text)
    return "".join(THREE_TO_ONE.get(r["residue_name"], "X") for r in residues)


def parse_plddt_from_pdb(pdb_text: str) -> list[dict]:
    """Per-residue pLDDT from B-factor column."""
    residues: dict[int, dict] = {}
    for line in pdb_text.split("\n"):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            res_num = int(line[22:26].strip())
            bfactor = float(line[60:66].strip())
            chain = line[21:22].strip()
            res_name = line[17:20].strip()
            residues[res_num] = {"residue_num": res_num, "residue_name": res_name, "chain": chain, "plddt": bfactor}
    return [residues[k] for k in sorted(residues.keys())]


def parse_backbone_atoms(pdb_text: str) -> dict[int, dict[str, tuple[float, float, float]]]:
    """N, CA, C coords per residue."""
    backbone: dict[int, dict[str, tuple[float, float, float]]] = {}
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
    """All ATOM records."""
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


def parse_pdb_chains(pdb_text: str) -> list[str]:
    """Return list of unique chain IDs from ATOM records."""
    chains: list[str] = []
    seen: set[str] = set()
    for line in pdb_text.split("\n"):
        if line.startswith("ATOM"):
            ch = line[21:22].strip()
            if ch and ch not in seen:
                seen.add(ch)
                chains.append(ch)
    return chains


def parse_plddt_by_chain(pdb_text: str) -> dict[str, list[dict]]:
    """pLDDT data grouped by chain ID."""
    by_chain: dict[str, dict[int, dict]] = {}
    for line in pdb_text.split("\n"):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            res_num = int(line[22:26].strip())
            chain = line[21:22].strip()
            if chain not in by_chain:
                by_chain[chain] = {}
            if res_num not in by_chain[chain]:
                by_chain[chain][res_num] = {
                    "residue_num": res_num, "residue_name": line[17:20].strip(),
                    "chain": chain, "plddt": float(line[60:66].strip()),
                }
    return {ch: [v[k] for k in sorted(v.keys())] for ch, v in by_chain.items()}


def parse_hetatm(pdb_text: str) -> list[dict]:
    """Parse HETATM records for ligands (exclude HOH)."""
    groups: dict[tuple[str, str, int], int] = {}
    for line in pdb_text.split("\n"):
        if not line.startswith("HETATM"):
            continue
        res_name = line[17:20].strip()
        if res_name == "HOH":
            continue
        chain = line[21:22].strip()
        res_num = int(line[22:26].strip())
        key = (res_name, chain, res_num)
        groups[key] = groups.get(key, 0) + 1
    return [{"name": k[0], "chain": k[1], "res_num": k[2], "num_atoms": v} for k, v in groups.items()]


# ── Structural analysis ──────────────────────────────────────────────────

def _dihedral(p0: tuple, p1: tuple, p2: tuple, p3: tuple) -> float:
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
    """Backbone dihedral angles."""
    backbone = parse_backbone_atoms(pdb_text)
    res_nums = sorted(backbone.keys())
    angles = []
    for i in range(1, len(res_nums) - 1):
        prev = backbone.get(res_nums[i - 1], {})
        curr = backbone.get(res_nums[i], {})
        nxt = backbone.get(res_nums[i + 1], {})
        if "C" in prev and all(k in curr for k in ("N", "CA", "C")) and "N" in nxt:
            phi = _dihedral(prev["C"], curr["N"], curr["CA"], curr["C"])
            psi = _dihedral(curr["N"], curr["CA"], curr["C"], nxt["N"])
            angles.append((phi, psi))
    return angles


def calculate_distance_map(pdb_text: str) -> tuple[np.ndarray, list[int]]:
    """CA-CA distance matrix."""
    backbone = parse_backbone_atoms(pdb_text)
    res_nums = sorted(r for r in backbone if "CA" in backbone[r])
    coords = np.array([backbone[r]["CA"] for r in res_nums])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist_matrix, res_nums


def calculate_radius_of_gyration(pdb_text: str) -> float:
    backbone = parse_backbone_atoms(pdb_text)
    ca_coords = np.array([backbone[r]["CA"] for r in sorted(backbone) if "CA" in backbone[r]])
    center = np.mean(ca_coords, axis=0)
    return float(np.sqrt(np.mean(np.sum((ca_coords - center) ** 2, axis=1))))


def detect_disulfide_bonds(pdb_text: str) -> list[dict]:
    """SG-SG < 3.0 A."""
    sg_atoms: dict[int, np.ndarray] = {}
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
    """Charged residue pairs."""
    atoms = parse_all_atoms(pdb_text)
    charged_sc = {
        "ARG": {"NH1", "NH2", "NE"}, "LYS": {"NZ"}, "HIS": {"ND1", "NE2"},
        "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
    }
    res_charged_coords: dict[int, list] = {}
    res_names: dict[int, str] = {}
    for a in atoms:
        if a["res_name"] in charged_sc and a["name"] in charged_sc[a["res_name"]]:
            rn = a["res_num"]
            res_names[rn] = a["res_name"]
            res_charged_coords.setdefault(rn, []).append([a["x"], a["y"], a["z"]])
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


def detect_hydrogen_bonds(pdb_text: str, dist_threshold: float = 3.5, angle_threshold: float = 120) -> list[dict]:
    """Backbone N-H...O=C hydrogen bonds (N-O distance check)."""
    n_atoms: dict[int, np.ndarray] = {}
    o_atoms: dict[int, np.ndarray] = {}
    for line in pdb_text.split("\n"):
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        res_num = int(line[22:26].strip())
        coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if atom_name == "N":
            n_atoms[res_num] = coord
        elif atom_name == "O":
            o_atoms[res_num] = coord
    hbonds = []
    for donor_res, n_coord in n_atoms.items():
        for acceptor_res, o_coord in o_atoms.items():
            if donor_res == acceptor_res:
                continue
            dist = float(np.linalg.norm(n_coord - o_coord))
            if dist < dist_threshold:
                hbonds.append({"donor_res": donor_res, "acceptor_res": acceptor_res, "distance": round(dist, 2)})
    return hbonds


def detect_cation_pi(pdb_text: str, threshold: float = 6.0) -> list[dict]:
    """Cation-pi interactions between charged and aromatic residues."""
    atoms = parse_all_atoms(pdb_text)
    # Collect cation atoms
    cation_targets = {"ARG": {"NZ", "NH1", "NH2"}, "LYS": {"NZ"}}
    cation_atoms: list[dict] = []
    for a in atoms:
        if a["res_name"] in cation_targets and a["name"] in cation_targets[a["res_name"]]:
            cation_atoms.append(a)
    # Collect aromatic ring atoms and compute centroids
    aromatic_rings = {
        "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
        "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
        "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    }
    ring_coords: dict[tuple[str, int], list[list[float]]] = {}
    ring_names: dict[tuple[str, int], str] = {}
    for a in atoms:
        if a["res_name"] in aromatic_rings and a["name"] in aromatic_rings[a["res_name"]]:
            key = (a["res_name"], a["res_num"])
            ring_coords.setdefault(key, []).append([a["x"], a["y"], a["z"]])
            ring_names[key] = a["res_name"]
    ring_centroids = {k: np.mean(v, axis=0) for k, v in ring_coords.items()}
    interactions = []
    for cat in cation_atoms:
        cat_coord = np.array([cat["x"], cat["y"], cat["z"]])
        for key, centroid in ring_centroids.items():
            dist = float(np.linalg.norm(cat_coord - centroid))
            if dist < threshold:
                interactions.append({
                    "cation_res": cat["res_num"], "cation_name": cat["res_name"],
                    "aromatic_res": key[1], "aromatic_name": key[0], "distance": round(dist, 2),
                })
    return interactions


def calculate_contact_order(pdb_text: str, contact_threshold: float = 8.0) -> float:
    """Relative contact order."""
    backbone = parse_backbone_atoms(pdb_text)
    res_nums = sorted(r for r in backbone if "CA" in backbone[r])
    n = len(res_nums)
    if n < 2:
        return 0.0
    coords = np.array([backbone[r]["CA"] for r in res_nums])
    total_sep = 0.0
    num_contacts = 0
    for i in range(n):
        for j in range(i + 2, n):  # skip adjacent
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist < contact_threshold:
                total_sep += abs(j - i)
                num_contacts += 1
    if num_contacts == 0:
        return 0.0
    return total_sep / (n * num_contacts)


def calculate_residue_burial(pdb_text: str, radius: float = 10.0) -> list[dict]:
    """Per-CA neighbor count within radius."""
    backbone = parse_backbone_atoms(pdb_text)
    res_nums = sorted(r for r in backbone if "CA" in backbone[r])
    if not res_nums:
        return []
    coords = np.array([backbone[r]["CA"] for r in res_nums])
    n = len(res_nums)
    # Pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    max_possible = n - 1
    result = []
    for i in range(n):
        neighbors = int(np.sum(dists[i] < radius)) - 1  # exclude self
        result.append({
            "residue_num": res_nums[i], "neighbors": neighbors,
            "burial_score": neighbors / max_possible if max_possible > 0 else 0.0,
        })
    return result


def estimate_sasa_approximate(pdb_text: str) -> list[dict]:
    """Rough per-residue SASA estimate (inverse of burial)."""
    burial = calculate_residue_burial(pdb_text)
    if not burial:
        return []
    max_b = max(b["burial_score"] for b in burial) if burial else 1.0
    if max_b == 0:
        max_b = 1.0
    return [{"residue_num": b["residue_num"], "sasa_relative": round(1.0 - b["burial_score"] / max_b, 3)} for b in burial]


def assign_ss_from_phi_psi(phi_psi: list[tuple[float, float]]) -> list[str]:
    """Assign H/E/C per residue from phi/psi angles."""
    result = []
    for phi, psi in phi_psi:
        if -160 < phi < -20 and -80 < psi < 40:
            result.append("H")
        elif -180 < phi < -40 and 80 < psi < 180:
            result.append("E")
        else:
            result.append("C")
    return result


def count_ramachandran_outliers(phi_psi: list[tuple[float, float]]) -> dict:
    """Count residues in favored, allowed, outlier Ramachandran regions."""
    favored = allowed = outlier = 0
    for phi, psi in phi_psi:
        # Favored: alpha or beta core
        if (-160 < phi < -20 and -80 < psi < 40) or (-180 < phi < -40 and 80 < psi < 180):
            favored += 1
        # Allowed: broader range
        elif (-180 <= phi <= 0 and -180 <= psi <= 180) or (20 < phi < 100 and -60 < psi < 60):
            allowed += 1
        else:
            outlier += 1
    return {"favored": favored, "allowed": allowed, "outlier": outlier, "total": len(phi_psi)}


def detect_pae_domains(pae_data: dict, threshold: float = 5.0) -> list[dict]:
    """Cluster PAE matrix to find domain boundaries."""
    # Support both formats: list-of-lists or predicted_aligned_error key
    if isinstance(pae_data, dict) and "predicted_aligned_error" in pae_data:
        matrix = np.array(pae_data["predicted_aligned_error"])
    elif isinstance(pae_data, dict) and "pae" in pae_data:
        matrix = np.array(pae_data["pae"])
    elif isinstance(pae_data, list):
        # Handle [{predicted_aligned_error: [[...]]}] format
        if len(pae_data) > 0 and isinstance(pae_data[0], dict):
            matrix = np.array(pae_data[0].get("predicted_aligned_error", []))
        else:
            matrix = np.array(pae_data)
    else:
        return []
    if matrix.size == 0:
        return []
    n = matrix.shape[0]
    # Binary contact matrix: low PAE = same domain
    contact = (matrix < threshold).astype(int)
    # Walk along diagonal to find domain boundaries
    domains = []
    domain_start = 0
    domain_id = 0
    for i in range(1, n):
        # Check if residue i has low PAE with previous domain block
        block_score = np.mean(contact[domain_start:i, i])
        if block_score < 0.5:
            # New domain boundary
            if i - domain_start >= 3:
                domains.append({"domain_id": domain_id, "start": domain_start + 1,
                                "end": i, "size": i - domain_start})
                domain_id += 1
            domain_start = i
    # Last domain
    if n - domain_start >= 3:
        domains.append({"domain_id": domain_id, "start": domain_start + 1,
                        "end": n, "size": n - domain_start})
    return domains


# ── Structure stats ──────────────────────────────────────────────────────

def compute_structure_stats(residues: list[dict]) -> dict:
    """Mean/median/min/max pLDDT, percent tiers."""
    scores = [r["plddt"] for r in residues]
    n = len(scores)
    return {
        "num_residues": n, "mean_plddt": float(np.mean(scores)),
        "median_plddt": float(np.median(scores)),
        "min_plddt": float(np.min(scores)), "max_plddt": float(np.max(scores)),
        "pct_very_high": sum(1 for s in scores if s > 90) / n * 100,
        "pct_confident": sum(1 for s in scores if s > 70) / n * 100,
        "pct_low": sum(1 for s in scores if s <= 50) / n * 100,
    }


def find_disordered_regions(residues: list[dict], threshold: float = 50.0) -> list[dict]:
    """Find contiguous stretches where pLDDT < threshold."""
    regions: list[dict] = []
    start = None
    scores: list[float] = []
    for r in residues:
        if r["plddt"] < threshold:
            if start is None:
                start = r["residue_num"]
                scores = []
            scores.append(r["plddt"])
        else:
            if start is not None:
                regions.append({"start": start, "end": r["residue_num"] - 1,
                                "length": len(scores), "mean_plddt": round(float(np.mean(scores)), 1)})
                start = None
    if start is not None:
        regions.append({"start": start, "end": residues[-1]["residue_num"],
                        "length": len(scores), "mean_plddt": round(float(np.mean(scores)), 1)})
    return regions


# ── Sequence analysis ────────────────────────────────────────────────────

def compute_sequence_properties(sequence: str) -> dict:
    """MW, pI, GRAVY, instability, aromaticity, ext_coeff, SS fractions, aa_counts, aa_percent."""
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
        "extinction_coeff": analysis.molar_extinction_coefficient(),
    }


def compute_charge_at_ph(sequence: str) -> list[tuple[float, float]]:
    """Charge vs pH 0-14."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    analysis = ProteinAnalysis(sequence)
    return [(ph / 10, analysis.charge_at_pH(ph / 10)) for ph in range(0, 141)]


def compute_hydrophobicity(sequence: str, window: int = 9) -> list[tuple[int, float]]:
    """Kyte-Doolittle sliding window."""
    if len(sequence) < window:
        return []
    half = window // 2
    return [(i + 1, sum(KD_SCALE.get(sequence[j], 0) for j in range(i - half, i + half + 1)) / window)
            for i in range(half, len(sequence) - half)]


def find_glycosylation_sites(sequence: str) -> list[dict]:
    """N-X-S/T motif where X != P."""
    sites = []
    for i in range(len(sequence) - 2):
        if sequence[i] == "N" and sequence[i + 1] != "P" and sequence[i + 2] in "ST":
            sites.append({"position": i + 1, "motif": sequence[i:i + 3]})
    return sites


def find_phosphorylation_sites(sequence: str) -> list[dict]:
    """S, T, Y positions."""
    return [{"position": i + 1, "residue": aa} for i, aa in enumerate(sequence) if aa in "STY"]


def predict_transmembrane(sequence: str, window: int = 21, threshold: float = 1.6) -> list[dict]:
    """Hydrophobic stretches."""
    if len(sequence) < window:
        return []
    regions = []
    in_tm = False
    start = 0
    for i in range(len(sequence) - window + 1):
        avg = sum(KD_SCALE.get(aa, 0) for aa in sequence[i:i + window]) / window
        if avg > threshold and not in_tm:
            in_tm = True
            start = i
        elif avg <= threshold and in_tm:
            in_tm = False
            regions.append({"start": start + 1, "end": i + window, "length": i + window - start})
    if in_tm:
        regions.append({"start": start + 1, "end": len(sequence), "length": len(sequence) - start})
    return regions


def compute_flexibility(sequence: str) -> list[tuple[int, float]]:
    """BioPython flexibility with position numbers."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    analysis = ProteinAnalysis(sequence)
    flex = analysis.flexibility()
    # flexibility() uses a window of 9, result is offset by 4
    return [(i + 5, v) for i, v in enumerate(flex)]


def compute_aliphatic_index(sequence: str) -> float:
    """100 * (Ala% + 2.9*Val% + 3.9*(Ile%+Leu%))."""
    n = len(sequence)
    if n == 0:
        return 0.0
    counts = Counter(sequence)
    ala = counts.get("A", 0) / n
    val = counts.get("V", 0) / n
    ile = counts.get("I", 0) / n
    leu = counts.get("L", 0) / n
    return 100 * (ala + 2.9 * val + 3.9 * (ile + leu))


def compute_half_life(sequence: str) -> dict:
    """N-end rule half-life lookup."""
    if not sequence:
        return {"mammalian": "N/A", "yeast": "N/A", "ecoli": "N/A"}
    first = sequence[0].upper()
    return HALF_LIFE_TABLE.get(first, {"mammalian": "N/A", "yeast": "N/A", "ecoli": "N/A"})


def detect_signal_peptide(sequence: str) -> dict | None:
    """Check first 30 residues for hydrophobic stretch (>10 residues with avg KD > 1.0)."""
    region = sequence[:30]
    if len(region) < 10:
        return None
    best = None
    for start in range(len(region) - 10 + 1):
        for end in range(start + 10, len(region) + 1):
            seg = region[start:end]
            avg_kd = sum(KD_SCALE.get(aa, 0) for aa in seg) / len(seg)
            if avg_kd > 1.0:
                if best is None or (end - start) > (best["end"] - best["start"]):
                    best = {"start": start + 1, "end": end, "score": round(avg_kd, 2)}
    return best


def compute_sequence_complexity(sequence: str, window: int = 12) -> list[tuple[int, float]]:
    """Shannon entropy sliding window."""
    if len(sequence) < window:
        return []
    result = []
    for i in range(len(sequence) - window + 1):
        seg = sequence[i:i + window]
        counts = Counter(seg)
        entropy = -sum((c / window) * math.log2(c / window) for c in counts.values())
        result.append((i + 1, round(entropy, 3)))
    return result


def classify_protein(sequence: str, props: dict) -> list[str]:
    """Classify based on properties. Return tags."""
    tags = []
    gravy = props.get("gravy", 0)
    instability = props.get("instability_index", 0)
    if gravy > 0.5:
        tags.append("Membrane-associated")
    elif gravy < -0.5:
        tags.append("Hydrophilic")
    else:
        tags.append("Globular")
    if instability < 40:
        tags.append("Stable")
    else:
        tags.append("Unstable")
    if props.get("aromaticity", 0) > 0.15:
        tags.append("Aromatic-rich")
    tm = predict_transmembrane(sequence)
    if tm:
        tags.append("Potential transmembrane")
    # Check for disorder propensity via low-complexity
    low_complex = compute_sequence_complexity(sequence)
    if low_complex:
        avg_entropy = sum(v for _, v in low_complex) / len(low_complex)
        if avg_entropy < 2.5:
            tags.append("Low complexity")
    if len(sequence) > 200:
        tags.append("Multi-domain candidate")
    return tags


# ── Alignment & comparison ───────────────────────────────────────────────

def align_sequences(seq_a: str, seq_b: str) -> dict:
    """Pairwise alignment using BioPython PairwiseAligner with BLOSUM62."""
    from Bio import Align
    from Bio.Align import substitution_matrices
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(seq_a, seq_b)
    best = alignments[0]
    # Build gapped alignment from aligned blocks
    blocks_a = best.aligned[0]  # tuple of (start, end) for seq_a
    blocks_b = best.aligned[1]  # tuple of (start, end) for seq_b
    aligned_a_chars, aligned_b_chars = [], []
    prev_a, prev_b = 0, 0
    for (a_s, a_e), (b_s, b_e) in zip(blocks_a, blocks_b):
        # Gaps before this block
        gap_a = a_s - prev_a
        gap_b = b_s - prev_b
        if gap_a > 0:
            aligned_a_chars.extend(list(seq_a[prev_a:a_s]))
            aligned_b_chars.extend(["-"] * gap_a)
        if gap_b > 0:
            aligned_a_chars.extend(["-"] * gap_b)
            aligned_b_chars.extend(list(seq_b[prev_b:b_s]))
        # Aligned block
        aligned_a_chars.extend(list(seq_a[a_s:a_e]))
        aligned_b_chars.extend(list(seq_b[b_s:b_e]))
        prev_a, prev_b = a_e, b_e
    # Trailing unaligned residues
    if prev_a < len(seq_a):
        aligned_a_chars.extend(list(seq_a[prev_a:]))
        aligned_b_chars.extend(["-"] * (len(seq_a) - prev_a))
    if prev_b < len(seq_b):
        aligned_a_chars.extend(["-"] * (len(seq_b) - prev_b))
        aligned_b_chars.extend(list(seq_b[prev_b:]))
    aligned_a = "".join(aligned_a_chars)
    aligned_b = "".join(aligned_b_chars)
    # Calculate stats
    matrix = substitution_matrices.load("BLOSUM62")
    identity = gaps = similar = 0
    length = min(len(aligned_a), len(aligned_b))
    for i in range(length):
        a, b = aligned_a[i], aligned_b[i]
        if a == "-" or b == "-":
            gaps += 1
        elif a == b:
            identity += 1
            similar += 1
        elif a in matrix.alphabet and b in matrix.alphabet and matrix[a, b] > 0:
            similar += 1
    return {
        "aligned_a": aligned_a, "aligned_b": aligned_b,
        "identity": 100 * identity / length if length else 0,
        "similarity": 100 * similar / length if length else 0,
        "gaps": gaps, "score": float(best.score),
    }


def score_substitutions(seq_a: str, seq_b: str) -> list[dict]:
    """Compare two aligned sequences, score each position with BLOSUM62."""
    from Bio.Align import substitution_matrices
    matrix = substitution_matrices.load("BLOSUM62")
    result = []
    for i in range(min(len(seq_a), len(seq_b))):
        a, b = seq_a[i], seq_b[i]
        if a == "-" or b == "-":
            continue
        try:
            score = int(matrix[a, b])
        except (KeyError, IndexError):
            score = 0
        result.append({"position": i + 1, "aa_a": a, "aa_b": b,
                        "score": score, "is_conservative": score > 0})
    return result


# ── Advanced structural analysis ─────────────────────────────────────────


def calculate_sasa(pdb_text: str) -> list[dict] | None:
    """Accurate per-residue solvent accessible surface area using freesasa.

    Returns list of {residue_num, residue_name, sasa, relative_sasa} or None on failure.
    """
    try:
        import freesasa
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_text)
            tmppath = f.name
        try:
            structure = freesasa.Structure(tmppath)
            result = freesasa.calc(structure)
            residue_areas = result.residueAreas()
            sasa_data = []
            for chain_key, residues in residue_areas.items():
                for res_num_str, area in residues.items():
                    sasa_data.append({
                        "residue_num": int(res_num_str) if isinstance(res_num_str, str) and res_num_str.lstrip('-').isdigit() else res_num_str,
                        "residue_name": area.residueType,
                        "sasa": area.total,
                        "relative_sasa": area.relativeSASA if hasattr(area, 'relativeSASA') else area.total,
                    })
            sasa_data.sort(key=lambda x: x["residue_num"] if isinstance(x["residue_num"], int) else 0)
            return sasa_data
        finally:
            os.unlink(tmppath)
    except Exception:
        return None


def calculate_rmsd(pdb_a: str, pdb_b: str) -> dict | None:
    """Calculate RMSD between two structures using tmtools.

    Returns {rmsd, tm_score, aligned_length, seq_identity} or None on failure.
    """
    try:
        import tmtools

        def _extract_ca(pdb_text: str):
            coords = []
            seq = []
            for line in pdb_text.split("\n"):
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                    res_name = line[17:20].strip()
                    seq.append(THREE_TO_ONE.get(res_name, "X"))
            return np.array(coords), "".join(seq)

        coords_a, seq_a = _extract_ca(pdb_a)
        coords_b, seq_b = _extract_ca(pdb_b)
        if len(coords_a) == 0 or len(coords_b) == 0:
            return None
        res = tmtools.tm_align(coords_a, coords_b, seq_a, seq_b)
        return {
            "rmsd": float(res.rmsd),
            "tm_score": float(res.tm_norm_chain1),
            "aligned_length": int(res.aligned_length),
            "seq_identity": float(res.seq_id),
        }
    except Exception:
        return None


def run_tm_align(pdb_a: str, pdb_b: str) -> dict | None:
    """Full TM-align structural alignment.

    Extract CA coordinates from both PDBs, run tmtools.tm_align().
    Returns {tm_score_a, tm_score_b, rmsd, aligned_length, rotation_matrix,
    translation_vector, aligned_residues_a, aligned_residues_b} or None on failure.
    """
    try:
        import tmtools

        def _extract_ca_with_residues(pdb_text: str):
            coords = []
            seq = []
            res_nums = []
            for line in pdb_text.split("\n"):
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                    res_name = line[17:20].strip()
                    seq.append(THREE_TO_ONE.get(res_name, "X"))
                    res_nums.append(int(line[22:26].strip()))
            return np.array(coords), "".join(seq), res_nums

        coords_a, seq_a, res_nums_a = _extract_ca_with_residues(pdb_a)
        coords_b, seq_b, res_nums_b = _extract_ca_with_residues(pdb_b)
        if len(coords_a) == 0 or len(coords_b) == 0:
            return None
        res = tmtools.tm_align(coords_a, coords_b, seq_a, seq_b)
        aligned_len = sum(1 for c in res.seqM if c != ' ') if hasattr(res, 'seqM') else min(len(coords_a), len(coords_b))
        return {
            "tm_score_a": float(res.tm_norm_chain1),
            "tm_score_b": float(res.tm_norm_chain2),
            "rmsd": float(res.rmsd),
            "aligned_length": aligned_len,
            "rotation_matrix": res.u.tolist() if hasattr(res, 'u') else None,
            "translation_vector": res.t.tolist() if hasattr(res, 't') else None,
            "aligned_residues_a": res_nums_a,
            "aligned_residues_b": res_nums_b,
        }
    except Exception:
        return None


def run_normal_mode_analysis(pdb_text: str, n_modes: int = 10) -> dict | None:
    """Elastic network model NMA using ProDy.

    Returns {eigenvalues, sqflucts, cross_correlations} or None on failure.
    """
    try:
        import prody
        import tempfile
        import os

        prody.confProDy(verbosity='none')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_text)
            tmppath = f.name
        try:
            atoms = prody.parsePDB(tmppath)
            calphas = atoms.select('calpha')
            if calphas is None or len(calphas) < 3:
                return None
            anm = prody.ANM('protein')
            anm.buildHessian(calphas)
            anm.calcModes(n_modes)
            sqflucts = prody.calcSqFlucts(anm).tolist()
            eigenvalues = anm.getEigvals().tolist()
            cross_corr = prody.calcCrossCorr(anm)
            return {
                "eigenvalues": eigenvalues,
                "sqflucts": sqflucts,
                "cross_correlations": cross_corr.tolist() if cross_corr is not None else None,
            }
        finally:
            os.unlink(tmppath)
    except Exception:
        return None


def calculate_gnm_bfactors(pdb_text: str) -> list[tuple[int, float]] | None:
    """GNM-predicted B-factors using ProDy.

    Returns list of (residue_num, predicted_bfactor) or None on failure.
    """
    try:
        import prody
        import tempfile
        import os

        prody.confProDy(verbosity='none')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_text)
            tmppath = f.name
        try:
            atoms = prody.parsePDB(tmppath)
            calphas = atoms.select('calpha')
            if calphas is None or len(calphas) < 3:
                return None
            gnm = prody.GNM('protein')
            gnm.buildKirchhoff(calphas)
            gnm.calcModes()
            sqflucts = prody.calcSqFlucts(gnm)
            res_nums = calphas.getResnums().tolist()
            return list(zip(res_nums, sqflucts.tolist()))
        finally:
            os.unlink(tmppath)
    except Exception:
        return None


def build_sequence_distance_matrix(sequences: list[tuple[str, str]]) -> tuple[list[str], np.ndarray] | None:
    """Compute all-vs-all sequence distance matrix for batch fold results.

    Uses length-normalized edit distance (Levenshtein-like) between sequence pairs.
    Returns (names, distance_matrix) or None on failure.
    """
    try:
        if not sequences or len(sequences) < 2:
            return None
        names = [s[0] for s in sequences]
        seqs = [s[1] for s in sequences]
        n = len(seqs)
        dist_matrix = np.zeros((n, n))

        def _edit_distance(s1: str, s2: str) -> int:
            """Simple dynamic programming edit distance."""
            m, k = len(s1), len(s2)
            dp = list(range(k + 1))
            for i in range(1, m + 1):
                prev = dp[0]
                dp[0] = i
                for j in range(1, k + 1):
                    temp = dp[j]
                    if s1[i - 1] == s2[j - 1]:
                        dp[j] = prev
                    else:
                        dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                    prev = temp
            return dp[k]

        for i in range(n):
            for j in range(i + 1, n):
                max_len = max(len(seqs[i]), len(seqs[j]))
                if max_len == 0:
                    dist = 0.0
                else:
                    dist = _edit_distance(seqs[i], seqs[j]) / max_len
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return names, dist_matrix
    except Exception:
        return None


def build_upgma_tree(names: list[str], dist_matrix: np.ndarray) -> dict | None:
    """Build UPGMA phylogenetic tree from distance matrix.

    Returns nested dict structure: {name, distance, children: [...]} or None on failure.
    """
    try:
        n = len(names)
        if n < 2 or dist_matrix.shape != (n, n):
            return None
        # Initialize clusters: each is a leaf node
        clusters: list[dict] = [{"name": names[i], "distance": 0.0, "children": []} for i in range(n)]
        # Track cluster sizes for weighted averaging
        sizes = [1] * n
        # Copy distance matrix so we can modify it
        dm = dist_matrix.astype(float).copy()
        np.fill_diagonal(dm, np.inf)
        active = list(range(n))

        while len(active) > 1:
            # Find minimum distance pair among active clusters
            min_dist = np.inf
            mi, mj = -1, -1
            for ii in range(len(active)):
                for jj in range(ii + 1, len(active)):
                    d = dm[active[ii], active[jj]]
                    if d < min_dist:
                        min_dist = d
                        mi, mj = ii, jj
            if mi == -1:
                break
            ci, cj = active[mi], active[mj]
            # Create new merged cluster
            new_node = {
                "name": f"({clusters[ci]['name']},{clusters[cj]['name']})",
                "distance": min_dist / 2.0,
                "children": [clusters[ci], clusters[cj]],
            }
            # Update distance matrix: add new row/column by extending
            new_idx = len(clusters)
            clusters.append(new_node)
            # Extend dm
            new_dm = np.full((new_idx + 1, new_idx + 1), np.inf)
            new_dm[:new_idx, :new_idx] = dm
            si, sj = sizes[ci], sizes[cj]
            for k in active:
                if k == ci or k == cj:
                    continue
                new_d = (dm[ci, k] * si + dm[cj, k] * sj) / (si + sj)
                new_dm[new_idx, k] = new_d
                new_dm[k, new_idx] = new_d
            dm = new_dm
            sizes.append(si + sj)
            # Update active list
            active = [k for k in active if k != ci and k != cj]
            active.append(new_idx)

        return clusters[-1] if clusters else None
    except Exception:
        return None


def compute_logo_data(sequences: list[str]) -> dict | None:
    """Prepare data for sequence logo from a list of aligned sequences.

    Compute per-position frequency matrix.
    Returns {matrix: pd.DataFrame, consensus: str} where matrix has amino acids
    as columns and positions as rows, or None on failure.
    """
    try:
        import pandas as pd

        if not sequences:
            return None
        # Ensure all sequences have the same length (pad shorter ones with '-')
        max_len = max(len(s) for s in sequences)
        padded = [s.ljust(max_len, '-') for s in sequences]
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY-")
        n_seqs = len(padded)
        freq_data = []
        consensus = []
        for pos in range(max_len):
            counts = Counter(s[pos] for s in padded)
            freqs = {aa: counts.get(aa, 0) / n_seqs for aa in amino_acids}
            freq_data.append(freqs)
            # Consensus is the most frequent non-gap character
            non_gap = {aa: c for aa, c in counts.items() if aa != '-'}
            if non_gap:
                consensus.append(max(non_gap, key=non_gap.get))
            else:
                consensus.append('-')
        matrix = pd.DataFrame(freq_data)
        matrix.index = list(range(1, max_len + 1))
        matrix.index.name = "position"
        return {"matrix": matrix, "consensus": "".join(consensus)}
    except Exception:
        return None


def assign_secondary_structure_from_coords(pdb_text: str) -> list[dict] | None:
    """Assign secondary structure (H/E/C) from 3D coordinates.

    Uses phi/psi angle-based assignment refined with backbone hydrogen bond
    patterns: i->i+4 H-bonds indicate helices, i->i+2 for turns.
    Returns list of {residue_num, ss} or None on failure.
    """
    try:
        backbone = parse_backbone_atoms(pdb_text)
        res_nums = sorted(backbone.keys())
        if len(res_nums) < 5:
            return None

        # Get phi/psi-based initial assignment
        phi_psi = calculate_phi_psi(pdb_text)
        ss_initial = assign_ss_from_phi_psi(phi_psi)

        # Build residue number -> ss index mapping
        # phi_psi skips first and last residue, so it starts at res_nums[1]
        inner_res = res_nums[1:-1] if len(res_nums) > 2 else res_nums
        ss_map: dict[int, str] = {}
        for i, rn in enumerate(inner_res):
            if i < len(ss_initial):
                ss_map[rn] = ss_initial[i]

        # Detect backbone H-bonds for refinement
        hbonds = detect_hydrogen_bonds(pdb_text, dist_threshold=3.5)
        hbond_set: set[tuple[int, int]] = set()
        for hb in hbonds:
            hbond_set.add((hb["donor_res"], hb["acceptor_res"]))

        # Refine: i->i+4 H-bond pattern => helix
        for rn in res_nums:
            if (rn, rn + 4) in hbond_set or (rn + 4, rn) in hbond_set:
                for k in range(rn, rn + 5):
                    if k in ss_map:
                        ss_map[k] = "H"

        # Refine: i->i+2 H-bond pattern => turn (keep as coil if not already helix)
        for rn in res_nums:
            if (rn, rn + 2) in hbond_set or (rn + 2, rn) in hbond_set:
                for k in range(rn, rn + 3):
                    if k in ss_map and ss_map[k] == "C":
                        ss_map[k] = "C"  # turns remain coil

        # Build full result including first/last residue as coil
        result = []
        for rn in res_nums:
            result.append({"residue_num": rn, "ss": ss_map.get(rn, "C")})
        return result
    except Exception:
        return None


def generate_topology_data(pdb_text: str) -> list[dict] | None:
    """Parse secondary structure elements into topology diagram data.

    Returns list of {type: 'helix'|'strand'|'coil', start, end, length}
    for drawing a 2D topology diagram, or None on failure.
    """
    try:
        ss_data = assign_secondary_structure_from_coords(pdb_text)
        if not ss_data:
            return None

        elements: list[dict] = []
        current_type = None
        current_start = None

        type_map = {"H": "helix", "E": "strand", "C": "coil"}

        prev_rn = None
        for entry in ss_data:
            ss = entry["ss"]
            rn = entry["residue_num"]
            elem_type = type_map.get(ss, "coil")

            if elem_type != current_type:
                if current_type is not None and current_start is not None and prev_rn is not None:
                    elements.append({
                        "type": current_type,
                        "start": current_start,
                        "end": prev_rn,
                        "length": prev_rn - current_start + 1,
                    })
                current_type = elem_type
                current_start = rn
            prev_rn = rn

        # Add the last element
        if current_type is not None and current_start is not None and ss_data:
            last_rn = ss_data[-1]["residue_num"]
            elements.append({
                "type": current_type,
                "start": current_start,
                "end": last_rn,
                "length": last_rn - current_start + 1,
            })

        return elements
    except Exception:
        return None
