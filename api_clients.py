from __future__ import annotations

import re

import requests
import streamlit as st

# ── Constants ────────────────────────────────────────────────────────────
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api"
ESMFOLD_API = "https://api.esmatlas.com/foldSequence/v1/pdb/"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"
DOMAIN_FEATURE_TYPES = {"Domain", "Region", "Motif", "Zinc finger", "DNA binding"}

# ═════════════════════════════════════════════════════════════════════════
# EXISTING API FUNCTIONS (1-6)
# ═════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alphafold_prediction(uniprot_id: str) -> dict | None:
    try:
        resp = requests.get(f"{ALPHAFOLD_API}/prediction/{uniprot_id}", timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data[0] if isinstance(data, list) else data
        return None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alphafold_pdb(pdb_url: str) -> str | None:
    try:
        resp = requests.get(pdb_url, timeout=15)
        return resp.text if resp.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alphafold_pae(pae_url: str) -> dict | None:
    try:
        resp = requests.get(pae_url, timeout=15)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def fold_with_esmfold(sequence: str) -> str | None:
    try:
        resp = requests.post(
            ESMFOLD_API, data=sequence,
            headers={"Content-Type": "text/plain"}, timeout=120,
        )
        return resp.text if resp.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def search_uniprot(query: str, limit: int = 10) -> list[dict]:
    try:
        resp = requests.get(
            f"{UNIPROT_API}/search",
            params={"query": query, "format": "json", "size": limit},
            timeout=15,
        )
        if resp.status_code == 200:
            results = []
            for entry in resp.json().get("results", []):
                results.append({
                    "accession": entry.get("primaryAccession", ""),
                    "name": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "Unknown"),
                    "organism": entry.get("organism", {}).get("scientificName", "Unknown"),
                    "length": entry.get("sequence", {}).get("length", 0),
                })
            return results
        return []
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_uniprot_full(uniprot_id: str) -> dict | None:
    try:
        resp = requests.get(f"{UNIPROT_API}/{uniprot_id}.json", timeout=15)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════
# EXTRACTION HELPERS (7-8)
# ═════════════════════════════════════════════════════════════════════════


def extract_uniprot_domains(data: dict) -> list[dict]:
    domains = []
    for feat in data.get("features", []):
        if feat.get("type") in DOMAIN_FEATURE_TYPES:
            loc = feat.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            desc = feat.get("description", feat.get("type", "Unknown"))
            if start is not None and end is not None:
                domains.append({"name": desc, "start": int(start), "end": int(end), "type": feat["type"]})
    return domains


def extract_uniprot_annotations(data: dict) -> dict:
    ann = {
        "gene_name": "", "synonyms": [], "function": "", "subcellular_location": "",
        "disease": "", "keywords": [], "go_terms": [], "pdb_refs": [],
        "protein_existence": data.get("proteinExistence", ""),
    }
    genes = data.get("genes", [])
    if genes:
        ann["gene_name"] = genes[0].get("geneName", {}).get("value", "")
        ann["synonyms"] = [s.get("value", "") for s in genes[0].get("synonyms", [])]
    for c in data.get("comments", []):
        ct = c.get("commentType", "")
        if ct == "FUNCTION":
            texts = c.get("texts", [])
            if texts:
                ann["function"] = texts[0].get("value", "")
        elif ct == "SUBCELLULAR LOCATION":
            locs = c.get("subcellularLocations", [])
            if locs:
                loc_strs = []
                for loc in locs:
                    loc_val = loc.get("location", {}).get("value", "")
                    if loc_val:
                        loc_strs.append(loc_val)
                ann["subcellular_location"] = "; ".join(loc_strs)
            else:
                note = c.get("note", {})
                texts = note.get("texts", [])
                if texts:
                    ann["subcellular_location"] = texts[0].get("value", "")
        elif ct == "DISEASE":
            disease_obj = c.get("disease", {})
            if disease_obj:
                ann["disease"] = disease_obj.get("diseaseId", "")
            else:
                note = c.get("note", {})
                texts = note.get("texts", [])
                if texts:
                    ann["disease"] = texts[0].get("value", "")[:500]
    ann["keywords"] = [{"name": kw.get("name", ""), "category": kw.get("category", "")} for kw in data.get("keywords", [])]
    for xref in data.get("uniProtKBCrossReferences", []):
        db = xref.get("database", "")
        if db == "GO":
            props = {p["key"]: p["value"] for p in xref.get("properties", [])}
            ann["go_terms"].append({"id": xref["id"], "term": props.get("GoTerm", "")})
        elif db == "PDB":
            props = {p["key"]: p["value"] for p in xref.get("properties", [])}
            ann["pdb_refs"].append({
                "id": xref["id"], "method": props.get("Method", ""),
                "resolution": props.get("Resolution", ""), "chains": props.get("Chains", ""),
            })
    return ann


# ═════════════════════════════════════════════════════════════════════════
# NEW API FUNCTIONS (9-22)
# ═════════════════════════════════════════════════════════════════════════


def blast_search_submit(sequence: str) -> str | None:
    try:
        resp = requests.post(
            "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi",
            data={"CMD": "Put", "PROGRAM": "blastp", "DATABASE": "nr",
                  "QUERY": sequence, "FORMAT_TYPE": "JSON2"},
            timeout=15,
        )
        if resp.status_code == 200:
            match = re.search(r"RID\s*=\s*(\S+)", resp.text)
            return match.group(1) if match else None
        return None
    except Exception:
        return None


def blast_search_check(rid: str) -> str:
    try:
        resp = requests.get(
            "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi",
            params={"CMD": "Get", "RID": rid, "FORMAT_OBJECT": "SearchInfo"},
            timeout=15,
        )
        if resp.status_code == 200:
            match = re.search(r"Status=(\S+)", resp.text)
            return match.group(1) if match else "UNKNOWN"
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def blast_search_results(rid: str) -> dict | None:
    try:
        resp = requests.get(
            "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi",
            params={"CMD": "Get", "RID": rid, "FORMAT_TYPE": "JSON2"},
            timeout=15,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_interpro_domains(uniprot_id: str) -> list[dict]:
    try:
        resp = requests.get(
            f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{uniprot_id}",
            params={"page_size": 100}, timeout=15,
        )
        if resp.status_code != 200:
            return []
        domains = []
        for entry in resp.json().get("results", []):
            meta = entry.get("metadata", {})
            name = meta.get("name", "")
            accession = meta.get("accession", "")
            entry_type = meta.get("type", "")
            source_db = meta.get("source_database", "")
            for protein in entry.get("proteins", []):
                for loc in protein.get("entry_protein_locations", []):
                    for frag in loc.get("fragments", []):
                        domains.append({
                            "accession": accession, "name": name, "type": entry_type,
                            "source_database": source_db,
                            "start": frag.get("start"), "end": frag.get("end"),
                        })
        return domains
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rcsb_pdb_search(query: str, limit: int = 10) -> list[dict]:
    try:
        body = {
            "query": {
                "type": "terminal", "service": "full_text",
                "parameters": {"value": query},
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": limit}},
        }
        resp = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=body, timeout=15)
        if resp.status_code == 200:
            return [
                {"pdb_id": r.get("identifier", ""), "score": r.get("score", 0.0)}
                for r in resp.json().get("result_set", [])
            ]
        return []
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rcsb_entry(pdb_id: str) -> dict | None:
    try:
        resp = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}", timeout=15)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pdbe_validation(pdb_id: str) -> dict | None:
    try:
        resp = requests.get(
            f"https://www.ebi.ac.uk/pdbe/api/validation/global-percentiles/entry/{pdb_id}",
            timeout=15,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pdbe_secondary_structure(pdb_id: str) -> dict | None:
    try:
        resp = requests.get(
            f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/secondary_structure/{pdb_id}",
            timeout=15,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_string_interactions(protein_name: str, species: int = 9606, limit: int = 10) -> list[dict]:
    try:
        resp = requests.get(
            "https://string-db.org/api/json/interaction_partners",
            params={"identifiers": protein_name, "species": species, "limit": limit},
            timeout=15,
        )
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_string_enrichment(proteins: list[str], species: int = 9606) -> list[dict]:
    try:
        resp = requests.get(
            "https://string-db.org/api/json/enrichment",
            params={"identifiers": "%0d".join(proteins), "species": species},
            timeout=15,
        )
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_kegg_pathways(uniprot_id: str) -> list[dict]:
    try:
        # Step 1: convert UniProt ID to KEGG gene ID
        resp = requests.get(f"https://rest.kegg.jp/conv/genes/uniprot:{uniprot_id}", timeout=15)
        if resp.status_code != 200 or not resp.text.strip():
            return []
        kegg_gene_id = resp.text.strip().split("\t")[-1]

        # Step 2: get pathway links
        resp = requests.get(f"https://rest.kegg.jp/link/pathway/{kegg_gene_id}", timeout=15)
        if resp.status_code != 200 or not resp.text.strip():
            return []
        pathway_ids = []
        for line in resp.text.strip().split("\n"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pathway_ids.append(parts[1])
        pathway_ids = pathway_ids[:20]

        # Step 3: get pathway names
        pathways = []
        for pid in pathway_ids:
            try:
                resp = requests.get(f"https://rest.kegg.jp/get/{pid}", timeout=15)
                if resp.status_code == 200:
                    for line in resp.text.split("\n"):
                        if line.startswith("NAME"):
                            name = line.replace("NAME", "").strip()
                            pathways.append({"pathway_id": pid, "name": name})
                            break
                    else:
                        pathways.append({"pathway_id": pid, "name": ""})
            except Exception:
                pathways.append({"pathway_id": pid, "name": ""})
        return pathways
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_reactome_pathways(uniprot_id: str) -> list[dict]:
    try:
        resp = requests.get(
            f"https://reactome.org/ContentService/data/pathways/low/entity/{uniprot_id}",
            params={"species": 9606}, timeout=15,
        )
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ebi_protein_features(uniprot_id: str) -> list[dict]:
    try:
        resp = requests.get(
            f"https://www.ebi.ac.uk/proteins/api/features/{uniprot_id}", timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("features", [])
        return []
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_mobidb_disorder(uniprot_id: str) -> dict | None:
    try:
        resp = requests.get(
            "https://mobidb.org/api/download",
            params={"acc": uniprot_id, "format": "json"}, timeout=15,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None
