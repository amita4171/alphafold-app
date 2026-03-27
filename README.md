# AlphaFold Explorer

Comprehensive protein structure analysis suite built with Streamlit. Search AlphaFold DB, fold sequences with ESMFold, and run in-depth biophysical and structural analysis.

## Modes

### Lookup (AlphaFold DB)
Search 200M+ predicted structures by UniProt ID or protein name.

### Fold (ESMFold)
Fold novel amino acid sequences (up to 400 residues). Optional local ESM-2 support.

### Batch Fold
Upload FASTA files with up to 20 sequences. Progress tracking, CSV export.

### Compare
Compare two proteins side-by-side or with 3D structure overlay. Structural metrics comparison.

### Upload PDB
Analyze your own PDB files with the full analysis suite.

## Analysis Features

| Category | Features |
|----------|----------|
| **Confidence** | Per-residue pLDDT chart with domain overlays, PAE heatmap, B-factor histogram |
| **3D Viewer** | Interactive viewer with 5 styles (cartoon/stick/sphere/surface/line) and 5 color schemes (pLDDT/chain/hydrophobicity/secondary structure/uniform) |
| **Sequence Properties** | Molecular weight, isoelectric point, GRAVY, instability index, aromaticity, extinction coefficient (reduced/oxidized) |
| **Composition** | Amino acid composition chart (colored by property), charge vs pH plot |
| **Hydrophobicity** | Kyte-Doolittle sliding window plot, transmembrane region prediction |
| **Structure** | Ramachandran plot, CA-CA distance map, radius of gyration, disulfide bond detection, salt bridge detection |
| **PTM Sites** | N-linked glycosylation sites (NXS/T motif), phosphorylation sites (S/T/Y) |
| **Annotations** | Gene name, protein function, subcellular location, disease associations, GO terms (component/function/process), PDB cross-references, UniProt keywords |
| **Domain Overlay** | UniProt domain/region/motif boundaries on pLDDT chart |
| **Export** | PDB download, PAE JSON, PDF report (with chart + stats), comprehensive analysis JSON |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

No API keys needed — all APIs are free and public.

## Optional: Local ESM-2

```bash
pip install torch fair-esm
```

Enables offline contact map analysis using `esm2_t6_8M_UR50D` (~30MB). GPU users with full ESMFold weights get local structure prediction.

## Example Proteins

| UniProt ID | Protein | Notable Features |
|-----------|---------|-----------------|
| P69905 | Hemoglobin alpha | High confidence, salt bridges |
| P0DTC2 | SARS-CoV-2 spike | Large, glycosylation sites |
| P04637 | p53 tumor suppressor | 286 PDB structures, 173 GO terms |
| P68871 | Hemoglobin beta | Good for comparison with P69905 |
| Q9BYF1 | ACE2 receptor | Transmembrane, disease associations |

## Performance

All API calls cached with `@st.cache_data` (1h TTL, ESMFold 10min). Repeated lookups are instant.

## Tech Stack

Streamlit, Plotly, BioPython, py3Dmol/stmol, ReportLab, Kaleido, NumPy, Requests
