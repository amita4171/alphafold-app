# AlphaFold Explorer — Engineering Guide

## Project
Modular Streamlit protein analysis suite. Repo: `amita4171/alphafold-app`.

## Architecture
6 modules, 3,433 lines:
- `app.py` (516) — Streamlit UI, routing, sidebar, 5 modes
- `analysis.py` (1,172) — 49 pure analysis functions (no Streamlit)
- `viz.py` (825) — 27 Plotly + matplotlib visualization functions
- `api_clients.py` (397) — 22 cached API functions (@st.cache_data)
- `ui_components.py` (414) — 13 shared Streamlit UI components
- `export_utils.py` (109) — PDF + JSON export

## Conventions
- `from __future__ import annotations` in all files
- API helpers take primitives for cache-friendliness
- UI components prefixed with `show_`
- Mode routing: `"🔍" in mode`
- Heavy deps (freesasa, tmtools, prody) imported inside functions
- New analysis → `analysis.py`, API → `api_clients.py`, chart → `viz.py`, UI → `ui_components.py`

## Dependencies
Core: `streamlit requests plotly numpy py3Dmol stmol biopython`
Analysis: `freesasa tmtools prody networkx logomaker matplotlib`
Export: `reportlab kaleido`
Optional: `torch fair-esm`

## External APIs (no auth)
AlphaFold DB, ESMFold, UniProt, InterPro, STRING, Reactome, KEGG, MobiDB, RCSB PDB, PDBe, EBI Proteins, NCBI BLAST
