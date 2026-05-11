"""
catalog.py
Loads the SHL catalog from catalog.json and provides BM25-based search.
"""
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

# ── Load once at import time ────────────────────────────────────────────────
_CATALOG_PATH = Path(__file__).parent / "shl_product_catalog.json"


def _split_csv_field(s: str) -> list[str]:
    parts = [p.strip() for p in (s or "").split(",")]
    return [p for p in parts if p]


def _normalize_item(raw: dict) -> dict:
    """Normalize raw scraped catalog record to the schema used by the agent/API."""
    keys = raw.get("keys")
    if not isinstance(keys, list):
        keys = []

    job_levels = raw.get("job_levels")
    if not isinstance(job_levels, list):
        job_levels = _split_csv_field(str(raw.get("job_levels_raw") or ""))

    languages = raw.get("languages")
    if not isinstance(languages, list):
        languages = _split_csv_field(str(raw.get("languages_raw") or ""))

    duration = str(raw.get("duration") or "").strip()

    return {
        "id": str(raw.get("entity_id") or raw.get("id") or ""),
        "name": str(raw.get("name") or "").strip(),
        "url": str(raw.get("link") or raw.get("url") or "").strip(),
        "description": str(raw.get("description") or "").strip(),
        "keys": keys,
        "test_type": str(raw.get("test_type") or ""),
        "job_levels": job_levels,
        "languages": languages,
        "duration": duration,
    }


def _load_catalog_json(path: Path) -> list[dict]:
    """Load catalog JSON, tolerating occasional illegal newlines inside quoted strings."""
    raw = path.read_text(encoding="utf-8", errors="replace")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Repair strategy: replace literal newlines occurring inside strings with '\n'.
        out_chars: list[str] = []
        in_str = False
        escape = False
        for ch in raw:
            if escape:
                out_chars.append(ch)
                escape = False
                continue

            if ch == "\\":
                out_chars.append(ch)
                escape = True
                continue

            if ch == '"':
                out_chars.append(ch)
                in_str = not in_str
                continue

            if in_str and ch == "\n":
                out_chars.append("\\n")
                continue

            out_chars.append(ch)

        return json.loads("".join(out_chars))


_RAW_CATALOG: list[dict] = _load_catalog_json(_CATALOG_PATH)

CATALOG: list[dict] = [_normalize_item(item) for item in (_RAW_CATALOG or [])]

# ── Build BM25 index ────────────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

# Concatenate all searchable fields into a single document per item
_DOCS: list[list[str]] = []
for item in CATALOG:
    combined = " ".join([
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
        " ".join(item.get("languages", [])),
        item.get("duration", ""),
    ])
    _DOCS.append(_tokenize(combined))

_BM25 = BM25Okapi(_DOCS)


def search(query: str, top_k: int = 20) -> list[dict]:
    """
    Return up to top_k catalog items ranked by BM25 relevance to query.
    Returns the raw catalog dicts (not copies) — callers must not mutate them.
    """
    if not query.strip():
        return CATALOG[:top_k]

    tokens = _tokenize(query)
    scores = _BM25.get_scores(tokens)

    # Get indices sorted by score descending
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked[:top_k]:
        if scores[idx] > 0:
            results.append(CATALOG[idx])

    # If BM25 finds nothing meaningful, return top items (fallback)
    if not results:
        results = CATALOG[:top_k]

    return results


def search_with_scores(query: str, top_k: int = 20) -> list[tuple[dict, float]]:
    """Return up to top_k catalog items with their BM25 scores.

    This is intended for evaluation/debugging and hybrid reranking strategies.
    Items are returned in descending BM25 score order.
    """
    if not query.strip():
        return [(item, 0.0) for item in CATALOG[:top_k]]

    tokens = _tokenize(query)
    scores = _BM25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results: list[tuple[dict, float]] = []
    for idx in ranked[:top_k]:
        score = float(scores[idx])
        if score > 0:
            results.append((CATALOG[idx], score))

    if not results:
        results = [(item, 0.0) for item in CATALOG[:top_k]]

    return results


def get_by_name(name: str) -> dict | None:
    """Exact (case-insensitive) name lookup. Used for comparison queries."""
    name_lower = name.lower().strip()
    for item in CATALOG:
        if item["name"].lower() == name_lower:
            return item
    return None


def get_all_names() -> list[str]:
    """Return all catalog names — used to validate recommendations."""
    return [item["name"] for item in CATALOG]


# Build a name→item map for O(1) lookup during validation
_NAME_MAP: dict[str, dict] = {item["name"]: item for item in CATALOG}
_URL_SET: set[str] = {item["url"] for item in CATALOG}


def is_valid_url(url: str) -> bool:
    """Return True only if url is present in the catalog."""
    return url in _URL_SET
