# AlphaFold Explorer — Engineering Guide

## Project

Single-file Streamlit protein analysis app. Repo: `amita4171/alphafold-app`.

## Architecture

- **Single file**: `app.py` (~1400 lines) — all helpers, UI, routing
- **No backend**: Pure Streamlit with external API calls
- **Caching**: All API calls use `@st.cache_data` with TTL (1h general, 10min ESMFold)
- **Optional deps**: `torch` and `fair-esm` detected at import time via try/except

## Code Style

- Python 3.9+ with `from __future__ import annotations`
- Type hints on all functions
- Plotly for all charts
- py3Dmol + stmol for 3D rendering
- BioPython `ProteinAnalysis` for sequence properties
- ReportLab for PDF generation

## Conventions

- Helper functions organized by category (sequence, API, PDB parsing, analysis, visualization, export)
- Shared UI components prefixed with `show_` (e.g., `show_properties_tab`, `show_3d_tab`)
- API helpers take primitive args (strings/URLs) for cache-friendliness
- Mode routing via emoji matching (`"🔍" in mode`)
- New features: add helpers in appropriate section, add UI in mode's elif block

## Testing

No test suite — verify with:
```bash
python3 -c "from app import *; print('OK')"
```

End-to-end verification against live APIs:
```bash
python3 -c "from app import *; pred = fetch_alphafold_prediction('P69905'); print(pred is not None)"
```

## Dependencies

Required: `streamlit requests plotly numpy py3Dmol stmol biopython reportlab kaleido`
Optional: `torch fair-esm` (for local ESM-2 analysis)

## External APIs (no auth)

- AlphaFold DB: `alphafold.ebi.ac.uk/api` — structure predictions
- ESMFold: `api.esmatlas.com` — sequence folding (rate-limited)
- UniProt: `rest.uniprot.org` — search, annotations, domains, GO terms
