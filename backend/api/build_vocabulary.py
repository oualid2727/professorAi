# backend/api/build_vocabulary.py
#
# Pedagogical RAG — Phase 2: concept normalizer / vocabulary builder.
#
# One-time batch job. Run from the api container AFTER chunks are indexed
# in ChromaDB:
#
#   docker compose exec api python -m api.build_vocabulary
#
# What it does:
#   1. Read every chunk (id + text) from ChromaDB.
#   2. Run the Phase 1 extractor over each chunk (free-form mode) to get raw
#      concept phrases. Cache the per-chunk results to disk so Phase 3 can
#      reuse them without re-calling the LLM.
#   3. Collect all unique raw phrases across the corpus.
#   4. Embed each unique phrase with nomic-embed-text (via Ollama).
#   5. Cluster the embeddings (agglomerative, cosine distance threshold).
#   6. Pick a canonical label per cluster (phrase nearest the centroid).
#   7. Write the canonical vocabulary to concept_vocabulary (Phase 0 table).
#   8. Save a phrase -> canonical_id mapping to disk for Phase 3.
#
# Outputs (under DATA_DIR, default /app/data):
#   concept_cache.json    {chunk_id: {"prerequisites": [...], "introduces": [...]}}
#   concept_mapping.json  {raw_phrase: canonical_id}
#
# These two files are the hand-off to Phase 3 (gold integration).

import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

# Phase 0 + Phase 1 modules (same package)
try:
    from api.concept_extractor import extract_concepts, slugify_concept
    from api import concepts_db
except ImportError:
    # Allow running as a plain script (tests, local dev)
    from concept_extractor import extract_concepts, slugify_concept
    import concepts_db


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ai_professor")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
CACHE_PATH   = os.path.join(DATA_DIR, "concept_cache.json")
MAPPING_PATH = os.path.join(DATA_DIR, "concept_mapping.json")

# Cosine DISTANCE threshold for agglomerative clustering.
# distance = 1 - cosine_similarity. 0.15 distance ≈ 0.85 similarity.
# Lower  → more aggressive merging (fewer, broader concepts).
# Higher → more clusters (more, finer concepts).
CLUSTER_DISTANCE_THRESHOLD = float(os.getenv("CLUSTER_DISTANCE_THRESHOLD", "0.15"))

# Skip extremely rare phrases that appear only once across the whole corpus?
# Keep them by default (min_freq=1) so we don't lose niche concepts.
MIN_PHRASE_FREQ = int(os.getenv("MIN_PHRASE_FREQ", "1"))


# ── ChromaDB read ─────────────────────────────────────────────────────────────

def read_all_chunks() -> List[Tuple[str, str]]:
    """Return [(chunk_id, chunk_text), ...] for every chunk in the collection."""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_collection(CHROMA_COLLECTION)
    # include documents; ids come back by default
    result = col.get(include=["documents"])
    ids = result.get("ids", []) or []
    docs = result.get("documents", []) or []
    return list(zip(ids, docs))


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_phrase(phrase: str) -> List[float]:
    """Embed one phrase via Ollama nomic-embed-text. Returns [] on failure."""
    import httpx
    try:
        resp = httpx.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": phrase},
            timeout=60,
        )
        resp.raise_for_status()
        return [float(x) for x in resp.json()["embedding"]]
    except Exception as e:
        print(f"[vocab] embed error for {phrase!r}: {e}")
        return []


def embed_phrases(phrases: List[str]) -> Tuple[List[str], np.ndarray]:
    """
    Embed a list of phrases. Returns (kept_phrases, matrix) where matrix is
    (n_kept, dim). Phrases that fail to embed are dropped (and reported).
    """
    kept, vectors = [], []
    dim = None
    for i, p in enumerate(phrases, 1):
        v = embed_phrase(p)
        if not v:
            continue
        if dim is None:
            dim = len(v)
        elif len(v) != dim:
            print(f"[vocab] dim mismatch for {p!r} ({len(v)} != {dim}); skipping")
            continue
        kept.append(p)
        vectors.append(v)
        if i % 50 == 0:
            print(f"[vocab] embedded {i}/{len(phrases)} phrases...", flush=True)
    if not vectors:
        return [], np.zeros((0, 0))
    return kept, np.array(vectors, dtype=np.float32)


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_phrases(matrix: np.ndarray) -> np.ndarray:
    """
    Agglomerative clustering with a cosine-distance threshold.
    Returns an array of cluster labels (one per row of matrix).
    """
    from sklearn.cluster import AgglomerativeClustering

    n = matrix.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
    )
    return clustering.fit_predict(matrix)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row so dot product == cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def pick_canonical(
    phrases: List[str],
    matrix: np.ndarray,
    labels: np.ndarray,
    phrase_freq: Counter,
) -> Dict[int, Dict]:
    """
    For each cluster, choose the canonical label: the phrase whose embedding
    is nearest the cluster centroid. Frequency breaks ties (prefer the phrase
    that appears in more chunks).

    Returns: { cluster_label: {
        "canonical_phrase": str,
        "members": [phrases...],
    }}
    """
    normed = _normalize_rows(matrix)
    clusters: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(idx)

    out: Dict[int, Dict] = {}
    for lab, idxs in clusters.items():
        sub = normed[idxs]                      # (k, dim) normalized
        centroid = sub.mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm
        sims = sub @ centroid                   # cosine sim to centroid

        # Rank by (similarity, frequency) — both higher is better
        best_local = max(
            range(len(idxs)),
            key=lambda k: (round(float(sims[k]), 4), phrase_freq[phrases[idxs[k]]]),
        )
        canonical_phrase = phrases[idxs[best_local]]
        members = [phrases[i] for i in idxs]
        out[lab] = {"canonical_phrase": canonical_phrase, "members": members}
    return out


# ── Extraction pass (with caching) ────────────────────────────────────────────

def run_extraction_pass(chunks: List[Tuple[str, str]], use_cache: bool = True) -> Dict[str, Dict]:
    """
    Run the Phase 1 extractor over every chunk. Returns and caches:
      { chunk_id: {"prerequisites": [...], "introduces": [...]} }
    """
    if use_cache and os.path.isfile(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                cached = json.load(f)
            print(f"[vocab] loaded cached extractions for {len(cached)} chunks from {CACHE_PATH}")
            return cached
        except Exception as e:
            print(f"[vocab] could not read cache ({e}); re-extracting")

    cache: Dict[str, Dict] = {}
    total = len(chunks)
    for i, (chunk_id, text) in enumerate(chunks, 1):
        cache[chunk_id] = extract_concepts(text)        # free-form mode
        if i % 25 == 0 or i == total:
            print(f"[vocab] extracted {i}/{total} chunks...", flush=True)

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"[vocab] cached extractions → {CACHE_PATH}")
    except Exception as e:
        print(f"[vocab] could not write cache: {e}")

    return cache


# ── Collect unique phrases ────────────────────────────────────────────────────

def collect_phrases(extractions: Dict[str, Dict]) -> Counter:
    """
    Count how many chunks each raw phrase appears in (across both roles).
    Frequency = number of chunks the phrase shows up in.
    """
    freq: Counter = Counter()
    for chunk_id, ex in extractions.items():
        seen = set(ex.get("prerequisites", [])) | set(ex.get("introduces", []))
        for phrase in seen:
            freq[phrase] += 1
    return freq


# ── Main orchestration ────────────────────────────────────────────────────────

def build_vocabulary(use_cache: bool = True) -> Dict:
    """
    Full Phase 2 pipeline. Returns a summary dict.
    """
    print("[vocab] === Phase 2: concept normalizer ===")

    # 1. Read chunks
    chunks = read_all_chunks()
    print(f"[vocab] {len(chunks)} chunks read from ChromaDB")
    if not chunks:
        print("[vocab] no chunks found — run the pipeline + indexer first.")
        return {"concepts": 0, "phrases": 0, "chunks": 0}

    # 2. Extract concepts (cached)
    extractions = run_extraction_pass(chunks, use_cache=use_cache)

    # 3. Collect unique phrases with frequency
    freq = collect_phrases(extractions)
    phrases = [p for p, c in freq.items() if c >= MIN_PHRASE_FREQ]
    print(f"[vocab] {len(phrases)} unique phrases (min_freq={MIN_PHRASE_FREQ})")
    if not phrases:
        print("[vocab] no phrases extracted — check the extractor / Ollama.")
        return {"concepts": 0, "phrases": 0, "chunks": len(chunks)}

    # 4. Embed
    kept_phrases, matrix = embed_phrases(phrases)
    print(f"[vocab] embedded {len(kept_phrases)} phrases → matrix {matrix.shape}")
    if not kept_phrases:
        print("[vocab] no embeddings — is nomic-embed-text pulled in Ollama?")
        return {"concepts": 0, "phrases": len(phrases), "chunks": len(chunks)}

    # 5. Cluster
    labels = cluster_phrases(matrix)
    n_clusters = len(set(labels.tolist()))
    print(f"[vocab] clustered into {n_clusters} concepts (threshold={CLUSTER_DISTANCE_THRESHOLD})")

    # 6. Canonical label per cluster
    clusters = pick_canonical(kept_phrases, matrix, labels, freq)

    # 7. Build phrase → canonical_id map and write vocabulary
    phrase_to_id: Dict[str, str] = {}
    centroid_by_id: Dict[str, List[float]] = {}
    normed = _normalize_rows(matrix)

    # Precompute index lookup for embeddings by phrase
    phrase_index = {p: i for i, p in enumerate(kept_phrases)}

    used_ids = set()
    for lab, info in clusters.items():
        canonical_phrase = info["canonical_phrase"]
        concept_id = slugify_concept(canonical_phrase)

        # Avoid ID collisions between different clusters that slugify the same
        base_id = concept_id
        suffix = 2
        while concept_id in used_ids:
            concept_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(concept_id)

        members = info["members"]
        # Cluster centroid embedding (mean of normalized member vectors)
        member_idxs = [phrase_index[m] for m in members]
        centroid = normed[member_idxs].mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm

        # Persist to concept_vocabulary
        concepts_db.upsert_concept(
            concept_id=concept_id,
            canonical_label=canonical_phrase,
            description="",
            embedding=centroid.tolist(),
            raw_terms=sorted(members),
        )

        for m in members:
            phrase_to_id[m] = concept_id

    # 8. Save phrase → id mapping for Phase 3
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(MAPPING_PATH, "w") as f:
            json.dump(phrase_to_id, f, ensure_ascii=False, indent=2)
        print(f"[vocab] saved phrase→id mapping ({len(phrase_to_id)} phrases) → {MAPPING_PATH}")
    except Exception as e:
        print(f"[vocab] could not write mapping: {e}")

    print(f"[vocab] === done: {n_clusters} concepts from {len(kept_phrases)} phrases ===")
    return {
        "concepts": n_clusters,
        "phrases": len(kept_phrases),
        "chunks": len(chunks),
        "mapping_path": MAPPING_PATH,
        "cache_path": CACHE_PATH,
    }


if __name__ == "__main__":
    use_cache = "--no-cache" not in sys.argv
    summary = build_vocabulary(use_cache=use_cache)
    print(f"[vocab] summary: {summary}")