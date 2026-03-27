# AlphaFold Explorer

Streamlit app for exploring protein structures from the AlphaFold Database, folding novel sequences with ESMFold, and comprehensive protein analysis.

## Features

### Lookup Mode
Search 200M+ structures in the AlphaFold DB by UniProt ID or protein name:
- Per-residue pLDDT confidence chart with domain annotation overlays
- Interactive 3D structure viewer (colored by confidence)
- PAE (Predicted Aligned Error) heatmap
- Sequence properties: molecular weight, isoelectric point, GRAVY, instability index, aromaticity
- Amino acid composition chart (colored by property: hydrophobic/charged/polar)
- Kyte-Doolittle hydrophobicity plot
- Ramachandran plot (phi/psi backbone angles)
- Download PDB, PAE JSON, and PDF reports

### Fold Mode
Fold novel amino acid sequences (up to 400 residues) using Meta's ESMFold:
- Paste raw sequence or FASTA format
- Optional local ESM-2 analysis (contact map) when torch + fair-esm installed
- Full analysis suite: pLDDT, properties, Ramachandran, hydrophobicity
- PDF export with all stats

### Batch Fold
Fold multiple sequences at once:
- Upload a FASTA file (up to 20 sequences)
- Progress tracking with per-sequence results
- Expandable pLDDT charts per sequence
- Export summary stats as CSV

### Compare
Side-by-side protein comparison:
- Compare any two proteins (UniProt ID or pasted sequence)
- Overlaid pLDDT charts on a single plot
- Stats comparison table
- Side-by-side 3D structure viewers OR structure overlay in one viewer

### Upload PDB
Analyze your own PDB files:
- B-factor / pLDDT visualization
- 3D structure viewer
- Sequence properties and amino acid composition
- Ramachandran plot
- PDF export

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

No API keys needed — AlphaFold DB and ESMFold are free public APIs.

## Optional: Local ESM-2

For offline ESM-2 analysis (contact map prediction), install PyTorch and fair-esm:

```bash
pip install torch fair-esm
```

This enables a "Use local ESM-2 model" checkbox in Fold mode. The smallest model (`esm2_t6_8M_UR50D`, ~30MB) is used for CPU-feasible analysis. If you have a GPU and the full ESMFold weights, local structure prediction is also supported.

## Try It

- **Lookup**: `P69905` (human hemoglobin alpha)
- **Fold**: `MKTAYIAKQRQISFVKSHFSRQDLDALK` (short test sequence)
- **Batch**: Upload a FASTA file with multiple sequences
- **Compare**: `P69905` vs `P68871` (hemoglobin alpha vs beta)
- **Upload**: Any `.pdb` file from RCSB PDB or your own predictions

## More Example Proteins

| UniProt ID | Protein |
|-----------|---------|
| P69905 | Human hemoglobin subunit alpha |
| P0DTC2 | SARS-CoV-2 spike protein |
| P04637 | Human p53 tumor suppressor |
| P68871 | Human hemoglobin subunit beta |
| Q9BYF1 | Human ACE2 receptor |

## Performance

API responses are cached with `@st.cache_data` (1-hour TTL) so repeated lookups are instant. ESMFold results are cached for 10 minutes.

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [AlphaFold DB API](https://alphafold.ebi.ac.uk/) — Predicted structures (DeepMind/EBI)
- [ESMFold API](https://esmatlas.com/) — Sequence folding (Meta AI)
- [UniProt API](https://www.uniprot.org/) — Protein search + domain annotations
- [py3Dmol](https://github.com/avirshup/py3dmol) + [stmol](https://github.com/napoles-uach/stmol) — 3D visualization
- [Plotly](https://plotly.com/) — Charts (pLDDT, Ramachandran, hydrophobicity, composition)
- [BioPython](https://biopython.org/) — Sequence analysis (ProtParam)
- [ReportLab](https://www.reportlab.com/) — PDF generation
- [Kaleido](https://github.com/nicholasmckinney/kaleido) — Static chart export
