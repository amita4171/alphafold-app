# AlphaFold Explorer

Streamlit app for exploring protein structures from the AlphaFold Database and folding novel sequences with ESMFold.

## Features

**Lookup mode** — Search 200M+ structures in the AlphaFold DB by UniProt ID or protein name:
- Per-residue pLDDT confidence chart with domain annotation overlays
- Interactive 3D structure viewer (colored by confidence)
- PAE (Predicted Aligned Error) heatmap
- Download PDB, PAE JSON, and PDF reports

**Fold mode** — Fold novel amino acid sequences (up to 400 residues) using Meta's ESMFold:
- Paste raw sequence or FASTA format
- Optional local ESM-2 analysis (contact map) when torch + fair-esm are installed
- Same visualizations as lookup mode plus PDF export

**Batch Fold** — Fold multiple sequences at once:
- Upload a FASTA file (up to 20 sequences)
- Progress tracking with per-sequence results
- Expandable pLDDT charts per sequence
- Export summary stats as CSV

**Compare** — Side-by-side protein comparison:
- Compare any two proteins (UniProt ID or pasted sequence)
- Overlaid pLDDT charts on a single plot
- Stats comparison table
- Side-by-side 3D structure viewers

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

## More Example Proteins

| UniProt ID | Protein |
|-----------|---------|
| P69905 | Human hemoglobin subunit alpha |
| P0DTC2 | SARS-CoV-2 spike protein |
| P04637 | Human p53 tumor suppressor |
| P68871 | Human hemoglobin subunit beta |
| Q9BYF1 | Human ACE2 receptor |

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [AlphaFold DB API](https://alphafold.ebi.ac.uk/) — Predicted structures (DeepMind/EBI)
- [ESMFold API](https://esmatlas.com/) — Sequence folding (Meta AI)
- [UniProt API](https://www.uniprot.org/) — Protein search + domain annotations
- [py3Dmol](https://github.com/avirshup/py3dmol) + [stmol](https://github.com/napoles-uach/stmol) — 3D visualization
- [Plotly](https://plotly.com/) — Charts
- [ReportLab](https://www.reportlab.com/) — PDF generation
- [Kaleido](https://github.com/nicholasmckinney/kaleido) — Static chart export
