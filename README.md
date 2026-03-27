# AlphaFold Explorer

Comprehensive protein structure analysis suite. Search AlphaFold DB, fold sequences, compare proteins, and run deep biophysical analysis — all in a Streamlit web app.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

No API keys needed. All external APIs are free and public.

## Modes

| Mode | Description |
|------|-------------|
| **Lookup** | AlphaFold DB search + UniProt annotations + InterPro + STRING + Reactome + KEGG |
| **Fold** | ESMFold API or local ESM-2. Full analysis suite |
| **Batch Fold** | Upload FASTA, fold up to 20 sequences, CSV export |
| **Compare** | pLDDT overlay, sequence alignment, 3D overlay, structural metrics |
| **Upload PDB** | Analyze your own PDB files |

## Analysis Features

### Confidence & Quality
- Per-residue pLDDT chart with domain overlays
- PAE heatmap, B-factor histogram
- Ramachandran plot with favored/allowed/outlier stats
- Per-residue secondary structure from phi/psi angles

### Sequence Properties
- Molecular weight, isoelectric point, GRAVY, instability index, aromaticity
- Extinction coefficient (reduced/oxidized), aliphatic index
- Half-life estimation (mammalian/yeast/E. coli)
- Signal peptide detection
- Protein classification (globular/membrane/IDP/stable/unstable)

### Composition & Profiles
- Amino acid composition chart (colored by property)
- Kyte-Doolittle hydrophobicity plot
- Charge vs pH titration curve
- Flexibility profile (BioPython)
- Shannon entropy / sequence complexity plot

### Structural Analysis
- CA-CA distance map
- Radius of gyration
- Disulfide bond detection
- Salt bridge detection
- Backbone hydrogen bond detection
- Relative contact order
- Residue burial score
- Disordered region annotation (pLDDT < 50)

### Post-Translational Modifications
- N-linked glycosylation sites (NXS/T motif)
- Phosphorylation sites (S/T/Y)
- Transmembrane region prediction

### UniProt Annotations
- Gene name, protein function, subcellular location
- Disease associations
- GO terms (component/function/process)
- PDB cross-references
- Keywords

### External Database Integration
- **InterPro**: domain family classification
- **STRING**: protein-protein interaction network
- **Reactome**: biological pathway mapping
- **KEGG**: metabolic pathway data
- **MobiDB**: disorder consensus

### 3D Visualization
- 5 styles: cartoon, stick, sphere, surface, line
- 5 color schemes: pLDDT, chain, hydrophobicity, secondary structure, uniform
- Structure overlay for comparison

### Compare Mode
- Overlaid pLDDT charts
- Sequence alignment with identity/similarity/gaps (BLOSUM62)
- Side-by-side or overlay 3D views
- Structural metric comparison (Rg, SS bonds, salt bridges, H-bonds)

### Export
- PDB download, FASTA export
- PDF report with stats + chart
- Full analysis JSON (all metrics, bonds, annotations)
- Batch results CSV

## Optional: Local ESM-2

```bash
pip install torch fair-esm
```

## Architecture

```
app.py              — Streamlit UI and routing (504 lines)
analysis.py         — 39 analysis functions (741 lines)
api_clients.py      — 22 cached API functions (397 lines)
viz.py              — 18 visualization functions (410 lines)
ui_components.py    — 8 shared UI components (301 lines)
export_utils.py     — PDF + JSON export (109 lines)
────────────────────────────────────────────────────────
Total: 2,462 lines across 6 modules
```

## Example Proteins

| UniProt ID | Protein | Notable |
|-----------|---------|---------|
| P69905 | Hemoglobin alpha | High confidence, salt bridges |
| P04637 | p53 tumor suppressor | 286 PDB structures, 173 GO terms |
| P68871 | Hemoglobin beta | Compare with P69905 (45% identity) |
| Q9BYF1 | ACE2 receptor | TM prediction, disease associations |
| P0DTC2 | SARS-CoV-2 spike | Glycosylation sites, large protein |

## Tech Stack

Streamlit, Plotly, BioPython, py3Dmol/stmol, NumPy, ReportLab, Kaleido, Requests
