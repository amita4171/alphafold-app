# AlphaFold Explorer — Engineering Guide

## Project

Modular Streamlit protein analysis suite. Repo: `amita4171/alphafold-app`.

## Architecture

6 modules, 2462 lines total:
- `app.py` (504) — Streamlit UI, routing, sidebar
- `analysis.py` (741) — 39 pure analysis functions (no Streamlit deps)
- `api_clients.py` (397) — 22 cached API functions (@st.cache_data)
- `viz.py` (410) — 18 Plotly visualization functions
- `ui_components.py` (301) — 8 shared Streamlit UI components
- `export_utils.py` (109) — PDF + JSON export

## Conventions

- `from __future__ import annotations` in all files
- API helpers take primitive args (strings/URLs) for cache-friendliness
- Shared UI components prefixed with `show_`
- Mode routing via emoji matching (`"🔍" in mode`)
- New analysis: add to `analysis.py`, new API: add to `api_clients.py`, new chart: add to `viz.py`
- New UI sections: add to `ui_components.py`, wire in `app.py`

## Dependencies

Required: `streamlit requests plotly numpy py3Dmol stmol biopython reportlab kaleido`
Optional: `torch fair-esm` (for local ESM-2)

## External APIs (no auth)

- AlphaFold DB, ESMFold, UniProt, InterPro, STRING, Reactome, KEGG, MobiDB, EBI Proteins
