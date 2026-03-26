# Phase 2 Enhancements — AlphaFold Explorer

**Date:** 2026-03-26
**Status:** Approved
**Approach:** Single-file (Approach A) — all enhancements added to `app.py`

---

## Overview

Five enhancements to the existing AlphaFold Explorer Streamlit app:

1. **Batch Fold** — Upload FASTA, fold all sequences, export CSV
2. **Sequence Comparison** — Side-by-side pLDDT charts for two proteins
3. **Domain Annotation** — UniProt domain boundaries on pLDDT chart
4. **PDF Export** — Downloadable report with charts and stats
5. **Local ESM Model** — Optional offline analysis using ESM-2

All changes go into the existing `app.py`. Expected final size: ~1000-1100 lines.

---

## 1. Batch Fold

### Sidebar Mode
New radio option: "📦 Batch Fold"

### User Flow
1. User uploads a `.fasta` file via `st.file_uploader`
2. App parses all sequences using `parse_fasta()`
3. Validates each sequence (10-400 residues, valid amino acids)
4. Shows validation summary: N valid, M skipped (with reasons)
5. User clicks "Fold All" button
6. Sequential folding via ESMFold API with `st.progress()` bar
7. Results stored in `st.session_state["batch_results"]`
8. Summary table: sequence name, length, mean pLDDT, median pLDDT, % very high, % confident, % low
9. Each row expandable (`st.expander`) to show pLDDT chart
10. "Download CSV" button exports stats table

### Constraints
- Max 20 sequences per batch (validation error if exceeded)
- Sequential folding (API rate limits)
- Each sequence capped at 400 residues (same as single fold)
- Results persist in session_state across rerenders

### New Helper
```python
def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into list of (name, sequence) tuples."""
```

### CSV Columns
`sequence_name, length, mean_plddt, median_plddt, pct_very_high, pct_confident, pct_low`

---

## 2. Sequence Comparison

### Sidebar Mode
New radio option: "⚖️ Compare"

### User Flow
1. Two input columns side by side
2. Each has a selectbox: "UniProt ID" or "Paste Sequence"
3. UniProt ID → fetches from AlphaFold DB (existing `fetch_alphafold_prediction` + `fetch_alphafold_pdb`)
4. Paste Sequence → folds via ESMFold API (existing `fold_with_esmfold`)
5. "Compare" button triggers both fetches/folds
6. Results displayed:
   - **Overlaid pLDDT chart**: single Plotly figure, two traces (blue = Protein A, orange = Protein B), shared x-axis
   - **Stats comparison table**: two columns showing all metrics side by side
   - **3D structures**: side by side in `st.columns([1, 1])`

### Key Details
- Each input independently validated
- If both are UniProt IDs, both fetch in sequence (no parallelism needed for two requests)
- Labels default to UniProt ID or "Sequence A" / "Sequence B" if pasted

---

## 3. Domain Annotation

### Where It Appears
Lookup mode only (requires UniProt ID for annotation data).

### User Flow
1. After fetching AlphaFold prediction (existing flow), also fetch UniProt features
2. Domain boundaries rendered as semi-transparent colored vertical spans (`fig.add_vrect`) on the pLDDT chart
3. Legend entries show domain name + residue range
4. Toggle checkbox: "Show domain annotations" (default: on)

### Data Source
UniProt REST API: `GET https://rest.uniprot.org/uniprotkb/{id}.json`
Parse `features` array, filter for types: `Domain`, `Region`, `Motif`, `Zinc finger`, `DNA binding`

### New Helper
```python
def fetch_uniprot_domains(uniprot_id: str) -> list[dict]:
    """Fetch domain annotations from UniProt.
    Returns list of {name: str, start: int, end: int, type: str}
    """
```

### Visualization
- Each domain gets a distinct color from a Plotly qualitative palette (e.g., `plotly.colors.qualitative.Set2`)
- Vertical spans with `opacity=0.15`
- Annotation labels positioned at the top of each span
- Modified `make_plddt_chart` gets optional `domains` parameter

---

## 4. PDF Export

### Where It Appears
Download tab in Lookup and Fold modes. Button label: "📄 Export PDF Report"

### Contents
1. **Header**: "AlphaFold Explorer Report" + protein name/ID + generation date
2. **Stats table**: all metrics from `compute_structure_stats`
3. **pLDDT chart**: Plotly figure exported as PNG via `fig.to_image(format="png")`, embedded in PDF
4. **Footer**: data source attribution (AlphaFold DB / ESMFold / UniProt)

### Implementation
- Library: `reportlab` (pure Python, lightweight)
- Chart export: `kaleido` (Plotly's static image export engine)
- No 3D structure screenshot (would require headless browser — noted in PDF as "view 3D structure in the app")

### New Dependencies
- `reportlab>=4.0` — added to `requirements.txt`
- `kaleido>=0.2.1` — added to `requirements.txt`

### New Helper
```python
def generate_pdf_report(
    title: str,
    stats: dict,
    plddt_fig: go.Figure,
    uniprot_id: str | None = None,
    domains: list[dict] | None = None,
) -> bytes:
    """Generate PDF report as bytes for st.download_button."""
```

### Batch Mode Export
Batch mode uses CSV export (already covered in Section 1), not per-sequence PDFs.

---

## 5. Local ESM Model (Optional)

### Where It Appears
Fold mode — checkbox: "Use local ESM-2 model (offline)"

### Dependency Detection
```python
try:
    import torch
    import esm
    LOCAL_ESM_AVAILABLE = True
except ImportError:
    LOCAL_ESM_AVAILABLE = False
```

### Two Capability Levels

**Level 1 — ESM-2 Analysis (embeddings + contact map):**
- Available if `torch` + `esm` are installed
- Uses `esm2_t6_8M_UR50D` (smallest, ~30MB, CPU-feasible)
- Outputs: per-residue embeddings, predicted contact map
- Displayed as: contact map heatmap (Plotly), no 3D structure
- Labeled clearly as "ESM-2 Analysis" (not structure prediction)

**Level 2 — Full Local Folding:**
- Available if `esmfold_v1` model is loadable (requires GPU + ~2GB VRAM)
- Outputs: PDB string (same as API)
- Same visualization pipeline as API fold

### Detection Logic
```python
def check_local_esm_capabilities() -> str:
    """Returns 'full_fold', 'analysis_only', or 'unavailable'"""
```

### UI Behavior
- If `unavailable`: checkbox disabled, tooltip shows install instructions
- If `analysis_only`: checkbox enabled, shows contact map + embedding viz, note that full structure requires API
- If `full_fold`: checkbox enabled, full fold pipeline runs locally

### New Helper
```python
def fold_local_esm(sequence: str) -> str | None:
    """Attempt local folding. Returns PDB string or None."""

def analyze_local_esm(sequence: str) -> dict | None:
    """Run ESM-2 analysis. Returns {contact_map: np.ndarray, ...} or None."""
```

### Not Added to requirements.txt
`torch` and `fair-esm` are optional. Documented in README under "Optional: Local ESM-2".

---

## Sidebar Layout (Final)

```
🧬 AlphaFold Explorer
─────────────────────
Mode:
  ○ 🔍 Lookup (AlphaFold DB)
  ○ 🧪 Fold (ESMFold)
  ○ 📦 Batch Fold
  ○ ⚖️ Compare
─────────────────────
About
  ...existing about text...
─────────────────────
Example proteins
  ...existing examples...
```

---

## New Dependencies

**Required (add to requirements.txt):**
- `reportlab>=4.0`
- `kaleido>=0.2.1`

**Optional (document in README only):**
- `torch>=2.0`
- `fair-esm>=2.0`

---

## Error Handling

- All API calls already have timeouts (15s for AlphaFold/UniProt, 120s for ESMFold)
- Batch mode: individual sequence failures don't abort the batch — failed sequences noted in results table
- Local ESM: graceful fallback at every level — missing deps → disabled, load failure → show error + suggest API
- Domain annotation: if UniProt features fetch fails, pLDDT chart renders without annotations (no error shown)
- PDF export: if kaleido fails to render chart, PDF still generates with stats table only

---

## What's NOT Changing

- Existing Lookup and Fold mode behavior (only additive changes: domain overlay, PDF button, ESM toggle)
- Existing helper functions (signatures unchanged, `make_plddt_chart` gets optional `domains` param)
- File structure (single `app.py`)
- Existing dependencies and their versions
