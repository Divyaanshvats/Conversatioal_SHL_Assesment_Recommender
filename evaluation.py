import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from agent import chat
from catalog import CATALOG, search


@dataclass
class RetrievalCase:
    query: str
    expected_names: List[str]


def _normalize_name(s: str) -> str:
    return (s or "").strip().lower()


def _recall_at_k(predicted_names: List[str], expected_names: List[str], k: int) -> float:
    """Mathematically correct Recall@K on names.

    Recall@K = (# relevant in predicted_top_k) / (total relevant)
    """
    if not expected_names:
        return 0.0

    normalized_expected = {_normalize_name(n) for n in expected_names if (n or "").strip()}
    predicted_top_k = [
        _normalize_name(n)
        for n in (predicted_names or [])[:k]
        if (n or "").strip()
    ]

    relevant_found = sum(1 for n in predicted_top_k if n in normalized_expected)
    return relevant_found / max(1, len(normalized_expected))


def evaluate_reranking_pipeline(
    cases: List[RetrievalCase],
    k: int = 10,
    top_k_retrieve: int = 30,
) -> Dict[str, Any]:
    """Evaluate Recall@K on final reranked+validated recommendations.

    Also computes a BM25-only baseline Recall@K on the first K retrieved items.
    """
    rows: List[Dict[str, Any]] = []
    bm25_recalls: List[float] = []
    final_recalls: List[float] = []

    for idx, c in enumerate(cases, start=1):
        # BM25 baseline
        retrieved = search(c.query, top_k=top_k_retrieve)
        bm25_top_names = [x.get("name", "") for x in (retrieved or [])[:k]]
        bm25_r = _recall_at_k(bm25_top_names, c.expected_names, k)
        bm25_recalls.append(bm25_r)

        # Final pipeline output: BM25 -> LLM rerank -> grounding validation (inside agent.chat)
        result = chat([{"role": "user", "content": c.query}])
        final_recs = result.get("recommendations") or []
        final_names = [r.get("name", "") for r in final_recs]
        if not final_recs:
            print("RERANK DEBUG:", "Agent returned empty recommendations")
            print("RERANK DEBUG RESULT:", result)
            # Provide explicit BM25 fallback view to confirm retrieval had candidates.
            bm25_fallback_names = [x.get("name", "") for x in (retrieved or [])[:k]]
            print("RERANK DEBUG BM25 FALLBACK TOP NAMES:", bm25_fallback_names)
        final_r = _recall_at_k(final_names, c.expected_names, k)
        final_recalls.append(final_r)

        print("=" * 80)
        print(f"Case {idx}/{len(cases)}")
        print("QUERY:", c.query)
        print("EXPECTED NAMES:", c.expected_names)
        print("BM25 TOP NAMES:", bm25_top_names)
        print("FINAL RERANKED NAMES:", final_names[:k])
        print(f"BM25 RECALL@{k}:", round(bm25_r, 4))
        print(f"FINAL RECALL@{k}:", round(final_r, 4))

        rows.append(
            {
                "query": c.query,
                "expected": c.expected_names,
                "bm25_top_names": bm25_top_names,
                "final_top_names": final_names[:k],
                f"bm25_recall@{k}": bm25_r,
                f"final_recall@{k}": final_r,
            }
        )

    mean_bm25_recall = sum(bm25_recalls) / max(1, len(bm25_recalls))
    mean_final_recall = sum(final_recalls) / max(1, len(final_recalls))

    print("=" * 80)
    print(f"MEAN BM25 RECALL@{k}:", round(mean_bm25_recall, 4))
    print(f"MEAN FINAL RECALL@{k}:", round(mean_final_recall, 4))

    return {
        "k": k,
        "top_k_retrieve": top_k_retrieve,
        "num_cases": len(cases),
        f"mean_bm25_recall@{k}": mean_bm25_recall,
        f"mean_final_recall@{k}": mean_final_recall,
        "cases": rows,
    }


def default_cases() -> List[RetrievalCase]:
    """Starter suite.

    IMPORTANT: expected_names MUST exactly match your catalog assessment names.
    Update these to reflect your catalog.json; this file intentionally includes
    recruiter-style queries that are harder than single-keyword matches.
    """
    return [
        RetrievalCase(
            query=(
                "Hiring a mid-level Java backend engineer for a banking platform: REST APIs, SQL performance tuning, "
                "secure coding mindset, and Agile collaboration with stakeholders. Need technical + behavioral fit."
            ),
            expected_names=[
                "Automata - SQL (New)",
                "Java Platform Enterprise Edition 7 (Java EE 7)",
                "Business Communication (adaptive)",
                "Interpersonal Communications",
            ],
        ),
        RetrievalCase(
            query=(
                "Customer-facing implementation consultant: gathers requirements, manages stakeholders, writes light SQL, "
                "presents to clients, and works in Agile squads. Need communication + problem solving + fit."
            ),
            expected_names=[
                "Business Communication (adaptive)",
                "Interpersonal Communications",
                "OPQ Manager Plus Report",
            ],
        ),
        RetrievalCase(
            query=(
                "Tech lead for a regulated fintech: mentoring, prioritization, stakeholder communication, and driving delivery in Agile. "
                "Need leadership potential + personality + communication."
            ),
            expected_names=[
                "OPQ Manager Plus Report",
                "Business Communication (adaptive)",
                "Interpersonal Communications",
            ],
        ),
    ]


if __name__ == "__main__":
    start = time.time()
    report = evaluate_reranking_pipeline(default_cases(), k=10, top_k_retrieve=30)
    report["catalog_size"] = len(CATALOG)
    report["elapsed_s"] = time.time() - start
    print(json.dumps(report, indent=2))
