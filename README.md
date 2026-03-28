# AlphaFold Explorer

Comprehensive protein structure analysis suite built with Streamlit. 3,433 lines across 6 modules. 100+ analysis functions. 12 external API integrations.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Docker

```bash
docker build -t alphafold-explorer .
docker compose up
# Open http://localhost:8501
```

No API keys needed.

## Modes

| Mode | Tabs |
|------|------|
| **Lookup** | pLDDT, 3D, PAE, Properties, Ramachandran, Structural, SASA, Dynamics, Topology, Annotations, Databases, Export |
| **Fold** | pLDDT, 3D, Properties, Ramachandran, Structural, SASA, Dynamics, Topology, Export |
| **Batch Fold** | Upload FASTA, fold up to 20 sequences, CSV export |
| **Compare** | pLDDT overlay, Stats, 3D (side-by-side/overlay), Sequence alignment, Structural (TM-align/RMSD) |
| **Upload PDB** | All analysis tabs for your own structures |

## Analysis Features

### Confidence & Quality
Per-residue pLDDT, PAE heatmap, B-factor histogram, Ramachandran (favored/allowed/outlier), secondary structure from phi/psi

### Sequence Properties
MW, pI, GRAVY, instability, aromaticity, extinction coefficient, aliphatic index, half-life, signal peptide, protein classification, flexibility profile, sequence complexity

### Composition & Profiles
AA composition (colored by property), hydrophobicity (Kyte-Doolittle), charge vs pH, flexibility, Shannon entropy

### Structural Analysis
CA-CA distance map, radius of gyration, disulfide bonds, salt bridges, hydrogen bonds, contact order, residue burial, disordered regions

### Solvent Accessibility (SASA)
Per-residue solvent accessible surface area via FreeSASA. Exposed/buried classification.

### Protein Dynamics (NMA)
Anisotropic Network Model fluctuations, cross-correlation map, GNM-predicted vs experimental B-factors (via ProDy)

### Topology
2D topology diagram: helices (red), strands (blue), coils (grey)

### Structural Alignment (Compare)
TM-align via tmtools: RMSD, TM-score, aligned length. Sequence alignment with BLOSUM62 scoring.

### PTM Prediction
N-glycosylation (NXS/T), phosphorylation (S/T/Y), transmembrane regions

### UniProt Annotations
Gene, function, subcellular location, disease, GO terms, PDB cross-refs, keywords

### External Databases
InterPro domains, STRING interactions, Reactome pathways, KEGG pathways, MobiDB disorder, EBI Proteins features

### 3D Visualization
5 styles (cartoon/stick/sphere/surface/line), 5 color schemes, structure overlay

### Export
PDB, FASTA, PAE JSON, PDF report, full analysis JSON, batch CSV

## Architecture

```
app.py              516 lines — Streamlit UI and routing
analysis.py       1,172 lines — 49 analysis functions
api_clients.py      397 lines — 22 cached API clients
viz.py              825 lines — 27 visualization functions
ui_components.py    414 lines — 13 shared UI components
export_utils.py     109 lines — PDF + JSON export
───────────────────────────────────────────────
Total:            3,433 lines
```

## Optional: Local ESM-2

```bash
pip install torch fair-esm
```

## Dependencies

Core: streamlit, plotly, biopython, numpy, requests, py3Dmol, stmol
Analysis: freesasa, tmtools, prody, networkx, logomaker, matplotlib
Export: reportlab, kaleido
Optional: torch, fair-esm
