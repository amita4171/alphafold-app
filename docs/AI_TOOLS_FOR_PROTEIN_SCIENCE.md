# AI Tools for Protein Science

A curated list of AI models, tools, and platforms relevant to protein structure, function, and design.

---

## Structure Prediction

| Tool | What It Does | Access |
|------|-------------|--------|
| **AlphaFold2** | Protein structure prediction from sequence + MSA | [GitHub](https://github.com/google-deepmind/alphafold), [ColabFold](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb) |
| **AlphaFold3** | Structures of proteins, DNA, RNA, ligands, ions, and modifications | [AlphaFold Server](https://alphafoldserver.com/) |
| **ESMFold** | Fast single-sequence structure prediction (no MSA needed) | [API](https://esmatlas.com/), [GitHub](https://github.com/facebookresearch/esm) |
| **OpenFold** | Open-source AlphaFold2 reimplementation with training code | [GitHub](https://github.com/aqlaboratory/openfold) |
| **RoseTTAFold** | Structure prediction, also does protein complexes | [GitHub](https://github.com/RosettaCommons/RoseTTAFold) |
| **OmegaFold** | Single-sequence structure prediction | [GitHub](https://github.com/HeliXonProtein/OmegaFold) |
| **Chai-1** | Multi-modal structure prediction (protein, DNA, RNA, small molecules) | [GitHub](https://github.com/chaidiscovery/chai-lab) |
| **Boltz-1** | Open biomolecular interaction modeling | [GitHub](https://github.com/jwohlwend/boltz) |

## Protein Language Models

| Tool | What It Does | Access |
|------|-------------|--------|
| **ESM-2** | Protein embeddings, contact prediction, zero-shot variant effects | [GitHub](https://github.com/facebookresearch/esm) |
| **ESM-3** | Generative protein language model (sequence, structure, function) | [EvolutionaryScale](https://www.evolutionaryscale.ai/) |
| **ProtTrans** | Protein language models (ProtBERT, ProtT5, ProtXL) | [GitHub](https://github.com/agemagician/ProtTrans) |
| **Ankh** | Large protein language model optimized for downstream tasks | [GitHub](https://github.com/agemagician/Ankh) |
| **SaProt** | Structure-aware protein language model | [GitHub](https://github.com/westlake-repl/SaProt) |
| **ProGen** | Protein generation conditioned on function | [GitHub](https://github.com/salesforce/progen) |

## Protein Design

| Tool | What It Does | Access |
|------|-------------|--------|
| **ProteinMPNN** | Sequence design for given backbone structure (inverse folding) | [GitHub](https://github.com/dauparas/ProteinMPNN) |
| **RFdiffusion** | De novo protein structure generation via diffusion | [GitHub](https://github.com/RosettaCommons/RFdiffusion) |
| **Chroma** | Generative model for proteins and protein complexes | [GitHub](https://github.com/generatebio/chroma) |
| **FrameDiff** | SE(3) diffusion model for protein backbone generation | [GitHub](https://github.com/jasonkyuyim/se3_diffusion) |
| **ESM-Design** | Language model-based protein design | via ESM-3 |
| **LigandMPNN** | Sequence design accounting for ligand contacts | [GitHub](https://github.com/dauparas/LigandMPNN) |
| **EvoDiff** | Evolution-guided protein generation | [GitHub](https://github.com/microsoft/evodiff) |

## Variant Effect Prediction

| Tool | What It Does | Access |
|------|-------------|--------|
| **AlphaMissense** | Missense variant pathogenicity prediction | [Paper](https://www.science.org/doi/10.1126/science.adg7492), [Data](https://zenodo.org/records/8208688) |
| **ESM-1v** | Zero-shot variant effect prediction from protein LM | [GitHub](https://github.com/facebookresearch/esm) |
| **EVE** | Evolutionary model of variant effects | [GitHub](https://github.com/OATML-Markslab/EVE) |
| **ProteinGym** | Benchmark for variant effect predictors | [GitHub](https://github.com/OATML-Markslab/ProteinGym) |
| **CADD** | Combined annotation dependent depletion | [Web](https://cadd.gs.washington.edu/) |

## Function Prediction

| Tool | What It Does | Access |
|------|-------------|--------|
| **DeepFRI** | Function prediction from structure (GO terms) | [GitHub](https://github.com/flatironinstitute/DeepFRI) |
| **ProteInfer** | GO term and EC number prediction from sequence | [GitHub](https://github.com/google-research/proteinfer) |
| **InterProScan** | Domain/family/site annotation | [Web](https://www.ebi.ac.uk/interpro/) |
| **SignalP 6.0** | Signal peptide prediction with deep learning | [Web](https://services.healthtech.dtu.dk/services/SignalP-6.0/) |
| **DeepTMHMM** | Transmembrane topology prediction | [Web](https://dtu.biolib.com/DeepTMHMM) |
| **NetPhos 3.1** | Phosphorylation site prediction | [Web](https://services.healthtech.dtu.dk/services/NetPhos-3.1/) |

## Docking & Binding

| Tool | What It Does | Access |
|------|-------------|--------|
| **DiffDock** | Molecular docking with diffusion models | [GitHub](https://github.com/gcorso/DiffDock) |
| **AutoDock Vina** | Classical molecular docking | [GitHub](https://github.com/ccsb-scripps/AutoDock-Vina) |
| **AlphaFold3** | Predicts protein-ligand/DNA/RNA complexes | [Server](https://alphafoldserver.com/) |
| **GNINA** | Deep learning scoring for molecular docking | [GitHub](https://github.com/gnina/gnina) |
| **HADDOCK** | Information-driven protein-protein docking | [Web](https://wenmr.science.uu.nl/haddock2.4/) |
| **P2Rank** | Ligand binding site prediction | [GitHub](https://github.com/rdk/p2rank) |

## Molecular Dynamics & Simulation

| Tool | What It Does | Access |
|------|-------------|--------|
| **OpenMM** | GPU-accelerated MD simulation | [GitHub](https://github.com/openmm/openmm) |
| **GROMACS** | High-performance MD simulation | [Web](https://www.gromacs.org/) |
| **MDAnalysis** | MD trajectory analysis in Python | [GitHub](https://github.com/MDAnalysis/mdanalysis) |
| **OpenFold/AlphaFlow** | MD-like conformational sampling | [GitHub](https://github.com/microsoft/AlphaFlow) |

## Drug Discovery

| Tool | What It Does | Access |
|------|-------------|--------|
| **ChEMBL** | Bioactivity database for drug-like molecules | [Web](https://www.ebi.ac.uk/chembl/) |
| **AlphaFold + Virtual Screening** | Structure-based virtual screening pipeline | Various |
| **MolMIM** | Molecular generation for drug design | NVIDIA BioNeMo |
| **TorchDrug** | ML platform for drug discovery | [GitHub](https://github.com/DeepGraphLearning/torchdrug) |
| **DrugBank** | Drug and drug target database | [Web](https://go.drugbank.com/) |

## Databases & Platforms

| Resource | What It Provides |
|----------|-----------------|
| **AlphaFold DB** | 200M+ predicted structures | [Web](https://alphafold.ebi.ac.uk/) |
| **UniProt** | Protein sequences and annotations | [Web](https://www.uniprot.org/) |
| **RCSB PDB** | Experimental protein structures | [Web](https://www.rcsb.org/) |
| **STRING** | Protein-protein interaction networks | [Web](https://string-db.org/) |
| **Reactome** | Biological pathway database | [Web](https://reactome.org/) |
| **InterPro** | Domain family classification | [Web](https://www.ebi.ac.uk/interpro/) |
| **ESM Atlas** | ESMFold predictions for MGnify | [Web](https://esmatlas.com/) |
| **Human Protein Atlas** | Protein expression data | [Web](https://www.proteinatlas.org/) |
| **KEGG** | Metabolic and signaling pathways | [Web](https://www.genome.jp/kegg/) |
| **Open Targets** | Disease-target associations | [Web](https://platform.opentargets.org/) |

## Multi-Modal / Emerging

| Tool | What It Does | Access |
|------|-------------|--------|
| **Genie 2** | De novo protein design with higher diversity | [GitHub](https://github.com/aqlaboratory/genie2) |
| **Distributional Graphormer** | Predict equilibrium distributions of molecular systems | [GitHub](https://github.com/microsoft/DiG) |
| **Protenix** | Open-source AlphaFold3 implementation | [GitHub](https://github.com/bytedance/protenix) |
| **ESMFold + ProteinMPNN pipeline** | Predict → design → validate loop | Composable |
| **ColabDesign** | Hallucination-based protein design with AF2 | [GitHub](https://github.com/sokrypton/ColabDesign) |
