# Building AlphaFold Explorer: A Comprehensive Protein Analysis Suite in 3,400 Lines of Python

## The Problem

Protein analysis is fragmented. To study a single protein, a researcher typically needs to visit 5-10 different websites and tools:

- **AlphaFold DB** for predicted structures
- **UniProt** for annotations and function
- **InterPro** for domain classification
- **ExPASy ProtParam** for sequence properties
- **RCSB PDB** for experimental structures
- **STRING** for interaction networks
- **Reactome/KEGG** for pathway context
- **PyMOL or ChimeraX** for 3D visualization

Each tool has its own interface, data format, and learning curve. Results can't be easily compared across tools. And none of them combine structural analysis (SASA, NMA, contact order) with sequence analysis (hydrophobicity, PTM sites, disorder) in one place.

**AlphaFold Explorer** solves this by integrating 12 external APIs and 50+ analysis functions into a single Streamlit web application.

## What It Does

The app has five modes, each offering the full analysis suite:

### Lookup Mode
Enter a UniProt ID (e.g., `P04637` for p53) and get:
- AlphaFold predicted structure with per-residue confidence (pLDDT)
- Interactive 3D viewer with 5 rendering styles and 5 color schemes
- Predicted Aligned Error (PAE) heatmap
- Sequence properties: molecular weight, isoelectric point, GRAVY, extinction coefficient, half-life
- Amino acid composition, hydrophobicity profile, charge-vs-pH curve, flexibility profile
- Ramachandran plot with secondary structure assignment
- Structural analysis: distance map, radius of gyration, disulfide bonds, salt bridges, hydrogen bonds, contact order, residue burial, disordered regions
- Solvent accessible surface area (SASA) via FreeSASA
- Normal mode analysis (ANM/GNM) via ProDy
- 2D topology diagram
- UniProt annotations: gene, function, subcellular location, disease, GO terms, PDB cross-references
- External databases: InterPro domains, STRING interactions, Reactome/KEGG pathways, MobiDB disorder consensus

### Fold Mode
Paste an amino acid sequence and fold it with ESMFold. Same full analysis suite as Lookup, plus optional local ESM-2 contact map analysis.

### Batch Fold
Upload a FASTA file with up to 20 sequences. Progress tracking, per-sequence pLDDT charts, CSV export.

### Compare Mode
Compare two proteins side by side:
- Overlaid pLDDT confidence charts
- Sequence alignment with BLOSUM62 scoring (identity, similarity, gaps)
- Side-by-side or overlay 3D structures
- TM-align structural alignment (RMSD, TM-score)
- Structural metrics comparison table

### Upload PDB
Bring your own PDB file and get the full analysis suite applied to it.

## Architecture

The app is split into 6 focused modules:

```
app.py              516 lines   Streamlit routing, sidebar, 5 modes
analysis.py       1,184 lines   53 pure analysis functions
api_clients.py      397 lines   22 cached API functions
viz.py              825 lines   30 visualization functions
ui_components.py    414 lines   12 shared UI components
export_utils.py     109 lines   PDF + JSON export
──────────────────────────────────────────────────────
Total             3,445 lines
```

### Design Principles

**Strict module boundaries.** `analysis.py` has zero Streamlit or Plotly imports — it's pure computation with numpy and BioPython. `api_clients.py` only uses Streamlit for `@st.cache_data` decorators. `viz.py` only creates Plotly figures. This separation means analysis functions are independently testable and reusable.

**Cache everything.** Every API call is decorated with `@st.cache_data`. AlphaFold, UniProt, InterPro, STRING, Reactome, KEGG — all cached with 1-hour TTL. ESMFold (the most expensive call at 30-60 seconds) is cached for 10 minutes. This makes repeated lookups instant and dramatically reduces API load.

**Primitive arguments for cache keys.** API functions take strings (`fetch_alphafold_pdb(pdb_url: str)`) not dicts (`fetch_alphafold_pdb(prediction: dict)`). Streamlit's cache hashes arguments — strings are fast to hash, dicts require serialization. This small design choice makes caching reliable.

**Graceful degradation.** Every optional dependency (FreeSASA, tmtools, ProDy, torch) is imported inside the function that uses it, wrapped in try/except. If FreeSASA isn't installed, the SASA tab shows "Install `freesasa` for SASA analysis" instead of crashing. This means the app works with just the core dependencies and gets richer as you add optional packages.

**Heavy computation on demand.** Normal mode analysis, SASA calculation, and TM-align don't run until the user clicks the tab. Each expensive operation has its own `st.spinner()`. The Ramachandran plot, distance map, and structural analysis all compute lazily.

## Technical Decisions

### Why Streamlit?

For a data-heavy scientific app with many charts and interactive elements, Streamlit offers the best effort-to-result ratio. Alternatives considered:

- **Dash** — more control over layout, but much more boilerplate
- **Gradio** — great for ML demos, but limited for multi-tab complex UIs
- **Flask/FastAPI + React** — maximum flexibility, 10x the development effort

Streamlit's tab system, column layout, metrics display, file uploaders, and session state handle 90% of what a protein analysis app needs. The remaining 10% (interactive 3D structure picking, for example) isn't feasible in any web framework without a custom WebGL viewer.

### Why Single-Page, Not Multi-Page?

Streamlit supports multi-page apps via the `pages/` directory. We chose a single entry point (`app.py`) with radio-button mode selection because:

1. All modes share the same helper functions and UI components
2. Session state is simpler with one page
3. The sidebar radio gives instant mode switching without page reload
4. Docker deployment is simpler with a single entry point

### The 12 API Integration Challenge

Each external API has its own quirks:

| API | Quirk | Solution |
|-----|-------|----------|
| UniProt | Comment types are uppercase strings ("FUNCTION", "SUBCELLULAR LOCATION") | Exact string matching |
| UniProt | Cross-refs field is `uniProtKBCrossReferences` not `dbReferences` | Discovered by reading actual response |
| KEGG | Returns tab-delimited text, not JSON | Custom line parser |
| STRING | Protein names need species ID (9606 for human) | Default to human |
| Reactome | Pathway hierarchy requires multiple calls | Fetch flat list instead |
| ESMFold | 120-second timeout, intermittent failures | Cached results, graceful error |
| NCBI BLAST | Asynchronous (submit → poll → retrieve) | Three-function workflow |
| InterPro | Paginated results | Request page_size=100 |

The key insight: **every API call is wrapped in try/except returning None or empty list.** The UI checks for None and shows a graceful message. No single API failure can crash the app.

### SASA: FreeSASA vs Approximate

We implemented two SASA methods:

1. **`estimate_sasa_approximate`** — counts CA neighbors within a radius, inverts to get surface exposure. No dependencies. Fast but crude.
2. **`calculate_sasa`** — uses FreeSASA's C library for accurate Lee-Richards SASA calculation. Requires writing PDB to a temp file (FreeSASA reads from disk).

The UI prefers FreeSASA when available and falls back to the approximate method.

### Normal Mode Analysis

ProDy's Anisotropic Network Model (ANM) treats the protein as an elastic network of CA atoms connected by springs. It's the cheapest way to predict protein dynamics without running molecular dynamics.

We compute:
- **Mean square fluctuations** — which regions are flexible vs rigid
- **Cross-correlation map** — which regions move together
- **GNM B-factor prediction** — compare predicted vs experimental B-factors

The entire NMA runs in <2 seconds for a 500-residue protein.

### Structural Alignment with TM-align

The `tmtools` Python package wraps the TM-align algorithm. We extract CA coordinates from both PDBs, call `tmtools.tm_align()`, and get:
- **RMSD** — root mean square deviation (lower = more similar)
- **TM-score** — size-independent structural similarity (>0.5 = same fold, >0.9 = very similar)

For hemoglobin alpha vs beta: RMSD = 1.30 Å, TM-score = 0.918 — correctly identifying them as structurally near-identical despite only 45% sequence identity.

## Testing

223 unit tests cover all 53 analysis functions. Key testing decisions:

- **No network calls.** All tests use mock PDB fixtures (5-residue structures with realistic geometry).
- **External libraries mocked.** FreeSASA, tmtools, and ProDy are mocked with `unittest.mock.patch` so tests run without those packages installed.
- **BioPython tested directly.** It's a core dependency, so we test real BioPython behavior (ProteinAnalysis, PairwiseAligner).
- **Cross-version compatibility.** BioPython's alignment `str()` format changed between versions 1.79 and 1.85. We use the `aligned` block indices directly instead of parsing formatted strings.

Tests run in 1.5 seconds.

## CI/CD

GitHub Actions runs on every push:
1. **Syntax check** — parse all 6 Python modules with `ast.parse()`
2. **Unit tests** — `pytest tests/ -v`
3. **Docker build** — full image build
4. **Health check** — start container, verify `/_stcore/health` responds

## Deployment

```bash
# Local
pip install -r requirements.txt
streamlit run app.py

# Docker
docker compose up
# → http://localhost:8501

# Streamlit Cloud
# Connect repo at share.streamlit.io → deploys automatically
```

## What's Missing (Honestly)

- **No persistent storage.** Analysis results are lost on page refresh. Session state survives rerenders but not browser closes.
- **No authentication.** Anyone with the URL can use it.
- **No full MSA.** Pairwise alignment only (BLAST search is async but not integrated into the alignment workflow).
- **No DSSP.** The DSSP binary isn't available everywhere. We use phi/psi-based secondary structure assignment instead.
- **2.2 GB Docker image.** NumPy, SciPy, ProDy, and BioPython are large. A multi-stage build could help but the science packages are the bulk.
- **No interactive atom picking.** py3Dmol is view-only in Streamlit — you can't click atoms to measure distances.

## Numbers

| Metric | Value |
|--------|-------|
| Total lines of code | 3,445 |
| Analysis functions | 53 |
| API integrations | 12 |
| Visualization functions | 30 |
| Unit tests | 223 |
| Test runtime | 1.5 seconds |
| External APIs | AlphaFold DB, ESMFold, UniProt, InterPro, STRING, Reactome, KEGG, MobiDB, RCSB PDB, PDBe, EBI Proteins, NCBI BLAST |
| Python dependencies | 15 required + 2 optional |
| Docker image size | 2.2 GB |
| Time to build (6 phases) | 1 day |

## Try It

```bash
git clone https://github.com/amita4171/alphafold-app.git
cd alphafold-app
pip install -r requirements.txt
streamlit run app.py
```

Search for `P04637` (p53 tumor suppressor) — it has 286 PDB structures, 173 GO terms, 26 domain annotations, and disease associations. It's the best demo protein.

---

*Built with Claude Code. Source: [github.com/amita4171/alphafold-app](https://github.com/amita4171/alphafold-app)*
