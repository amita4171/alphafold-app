# AlphaFold Explorer

Streamlit app for exploring protein structures from the AlphaFold Database and folding novel sequences with ESMFold.

## Features

**Lookup mode** — Search 200M+ structures in the AlphaFold DB by UniProt ID or protein name:
- Per-residue pLDDT confidence chart
- Interactive 3D structure viewer (colored by confidence)
- PAE (Predicted Aligned Error) heatmap
- Download PDB and PAE JSON files

**Fold mode** — Fold novel amino acid sequences (up to 400 residues) using Meta's ESMFold:
- Paste raw sequence or FASTA format
- Same visualizations as lookup mode
- Download predicted PDB

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

No API keys needed — AlphaFold DB and ESMFold are free public APIs.

## Try It

- **Lookup**: `P69905` (human hemoglobin alpha)
- **Fold**: `MKTAYIAKQRQISFVKSHFSRQDLDALK` (short test sequence)

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
- [UniProt API](https://www.uniprot.org/) — Protein search
- [py3Dmol](https://github.com/avirshup/py3dmol) + [stmol](https://github.com/napoles-uach/stmol) — 3D visualization
- [Plotly](https://plotly.com/) — Charts
