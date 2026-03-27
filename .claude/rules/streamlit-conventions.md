---
description: Streamlit app conventions for AlphaFold Explorer
globs: ["*.py"]
---

# Streamlit App Conventions

- All API calls must be decorated with `@st.cache_data(ttl=N, show_spinner=False)` — 3600s for lookups, 600s for ESMFold
- API helpers take primitive args (strings, ints) not dicts/objects for cache-friendliness
- Shared UI components go in `ui_components.py`, prefixed with `show_`
- Mode routing uses emoji matching: `if "🔍" in mode:`
- New analysis functions go in `analysis.py` (no streamlit imports)
- New API clients go in `api_clients.py` (only streamlit for caching)
- New visualizations go in `viz.py` (plotly + py3Dmol)
- Use `from __future__ import annotations` in every file
- BioPython imports inside functions to avoid import errors if missing
- All external API calls wrapped in try/except returning None or []
