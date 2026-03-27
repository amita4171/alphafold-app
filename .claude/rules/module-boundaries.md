---
description: Module boundary rules — what goes where
globs: ["*.py"]
---

# Module Boundaries

- `analysis.py` — Pure computation. NO streamlit, NO plotly, NO requests. Only numpy, math, re, collections, biopython.
- `api_clients.py` — HTTP calls + caching. Only streamlit (for @st.cache_data) + requests.
- `viz.py` — Chart creation. Only plotly + numpy. May import from analysis.py. Uses streamlit only for showmol.
- `ui_components.py` — Streamlit widgets. Imports from analysis, api_clients, viz. This is the glue layer.
- `export_utils.py` — PDF (reportlab) + JSON export. No streamlit.
- `app.py` — Routing + sidebar. Imports from all modules. Keep thin — delegate to ui_components.
