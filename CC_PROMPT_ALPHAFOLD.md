# Claude Code Build Prompt — AlphaFold Explorer

## What Exists
`app.py` — Full Streamlit app. Two modes:
1. **Lookup**: Search AlphaFold DB by UniProt ID or protein name. Shows pLDDT chart, 3D structure (py3Dmol), PAE heatmap, download PDB/JSON.
2. **Fold**: Fold novel sequences via ESMFold (Meta). Max 400 residues. Same visualizations.

All core functions tested and working. No API key needed (AlphaFold DB and ESMFold are free).

## What Needs to Be Done

### Phase 1: Run + Verify (10 min)
```bash
pip install streamlit requests plotly numpy py3Dmol stmol biopython
streamlit run app.py
```
Test with:
- Lookup: `P69905` (human hemoglobin alpha)
- Fold: `MKTAYIAKQRQISFVKSHFSRQDLDALK` (short test)

### Phase 2: Enhancements (optional, 30-60 min)
1. **Batch mode**: Upload a FASTA file with multiple sequences, fold all, export CSV of pLDDT stats
2. **Sequence comparison**: Side-by-side pLDDT charts for two proteins
3. **Domain annotation**: Overlay UniProt domain boundaries on pLDDT chart
4. **Export**: PDF report with structure image + pLDDT chart + stats summary
5. **Local ESM model**: If you have ESM-2 weights locally, add an offline fold option using Qwen/ESM

### Phase 3: Ship (15 min)
1. Add README.md with screenshots
2. Push to GitHub
3. Deploy to Streamlit Cloud (free) if desired

## Ship Checklist
- [ ] `streamlit run app.py` works
- [ ] Lookup mode: P69905 shows pLDDT chart + 3D structure
- [ ] Fold mode: Short sequence folds and displays
- [ ] README with usage instructions
- [ ] Pushed to GitHub
