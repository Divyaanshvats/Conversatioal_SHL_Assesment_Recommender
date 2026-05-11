"""
agent.py
Builds the system prompt with injected catalog results and calls Groq.
Returns a parsed dict: { reply, recommendations, end_of_conversation }.
"""
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq
from dotenv import load_dotenv

from catalog import get_by_name, search_with_scores

load_dotenv()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"

# ── System prompt template ──────────────────────────────────────────────────
_SYSTEM_TEMPLATE = """
You are the SHL Assessment Advisor — a domain expert that helps hiring managers and
recruiters select the right SHL assessments from the official product catalog.

═══════════════════════════════════════════════════════
ROLE AND SCOPE
═══════════════════════════════════════════════════════
You ONLY discuss SHL assessments from the catalog below.
You REFUSE to give general hiring advice, legal opinions, compensation guidance, or
any topic unrelated to SHL assessments. When off-topic, say exactly:
"I can only help with SHL assessment selection. [redirect question back to assessments]"

You NEVER recommend an assessment whose name or URL is not in CATALOG SEARCH RESULTS.
If a relevant test doesn't exist in the catalog, say so explicitly — do not invent one.

═══════════════════════════════════════════════════════
CONVERSATIONAL BEHAVIORS (follow strictly)
═══════════════════════════════════════════════════════

1. CLARIFY before recommending.
   If the query is vague (e.g., "I need an assessment", "hiring an engineer"),
   ask one focused clarifying question. Do NOT output recommendations yet.
   A vague query lacks at minimum: role/function OR job level.
   Once you have enough context, recommend immediately — do not keep asking.

2. RECOMMEND 1–10 assessments once you have enough context.
   Use only items from CATALOG SEARCH RESULTS. Pick the most relevant ones.
   Structure: technical skills tests first, then cognitive (Verify G+), then
   personality (OPQ32r), then simulations/SJTs. 
   
   Recommendation strategy by role type:
   - Technical roles: relevant K (Knowledge & Skills) tests + Verify G+ + OPQ32r
   - Leadership/executive: OPQ32r + relevant reports (UCF, Leadership Report)
   - Sales: OPQ32r + OPQ MQ Sales Report + Sales Transformation
   - Contact centre/ops: relevant simulation + personality (DSI or Entry Level)
   - Graduate/entry-level: Verify G+ or Verify Numerical + Graduate Scenarios + OPQ32r
   - Safety-critical: DSI + Safety & Dependability 8.0 + domain knowledge test

3. REFINE when the user changes constraints mid-conversation.
   "Add X", "Remove Y", "Drop Z" → update the shortlist in place.
   Never start over. Carry forward confirmed items.
   When user says "that works", "confirmed", "that's good", "that covers it",
   "locking it in", "keep it as is" → echo the final list and set end_of_conversation=true.

4. COMPARE when asked.
   "What's the difference between X and Y?" → answer from catalog data only.
   During a compare turn, keep recommendations empty ([]). The shortlist doesn't change.

5. TURN BUDGET: The conversation is capped at 8 turns total. By turn 3 at latest,
   if you have reasonable context, provide a shortlist. Never keep clarifying indefinitely.

═══════════════════════════════════════════════════════
TEST TYPE CODES (use these in recommendations)
═══════════════════════════════════════════════════════
A = Ability & Aptitude | K = Knowledge & Skills | P = Personality & Behavior
B = Biodata & Situational Judgment | S = Simulations | C = Competencies
D = Development & 360 | E = Assessment Exercises
Each assessment has a pre-computed test_type field — use it as-is.

═══════════════════════════════════════════════════════
CATALOG SEARCH RESULTS (use ONLY these for recommendations)
═══════════════════════════════════════════════════════
{catalog_block}

═══════════════════════════════════════════════════════
OUTPUT FORMAT — CRITICAL — DO NOT DEVIATE
═══════════════════════════════════════════════════════
You MUST end every response with a JSON block between <RESPONSE> tags.
The block must be valid JSON. No trailing commas.

When clarifying or comparing (no shortlist yet):
<RESPONSE>
{{
  "reply": "Your conversational reply here.",
  "recommendations": [],
  "end_of_conversation": false
}}
</RESPONSE>

When recommending (always include URL from catalog, never fabricate):
<RESPONSE>
{{
  "reply": "Your conversational reply here.",
  "recommendations": [
    {{
      "name": "Exact name from catalog",
      "url": "https://www.shl.com/products/product-catalog/view/exact-slug/",
      "test_type": "K"
    }}
  ],
  "end_of_conversation": false
}}
</RESPONSE>

When conversation is done (user confirmed the final list):
<RESPONSE>
{{
  "reply": "Confirmed. Final shortlist above.",
  "recommendations": [ ... same final list ... ],
  "end_of_conversation": true
}}
</RESPONSE>

RULES FOR THE JSON BLOCK:
- recommendations is ALWAYS a list (never null, never omitted)
- Empty when clarifying, comparing, or refusing — NOT when recommending
- end_of_conversation is ONLY true when user explicitly confirms they are done
- Every url must be verbatim from the catalog — not approximated, not invented
- Max 10 items in recommendations
- The <RESPONSE> block must be the LAST thing in your message
""".strip()


def safe_response(
    reply: str,
    recommendations: Optional[list] = None,
    end_of_conversation: bool = False,
) -> dict:
    """Return a response dict that strictly matches the assignment schema."""
    if recommendations is None or not isinstance(recommendations, list):
        recommendations = []
    return {
        "reply": str(reply or ""),
        "recommendations": recommendations,
        "end_of_conversation": bool(end_of_conversation),
    }


def _is_vague_user_query(text: str) -> bool:
    """Deterministic heuristic: vague if too short / generic and lacks role or level signals."""
    t = (text or "").strip().lower()
    if not t:
        return True

    generic_phrases = (
        "i need an assessment",
        "need an assessment",
        "recommend a test",
        "recommend a assessment",
        "help me hire",
        "help with hiring",
        "suggest a test",
        "suggest an assessment",
        "need a test",
        "assessment",
        "recommendations",
    )

    # Very short or explicitly generic.
    if len(t.split()) <= 4:
        return True
    if any(p in t for p in generic_phrases):
        pass

    # Must contain at least some role/function signal.
    role_terms = (
        "engineer",
        "developer",
        "software",
        "data",
        "analyst",
        "scientist",
        "manager",
        "leader",
        "director",
        "executive",
        "sales",
        "account",
        "marketing",
        "finance",
        "accounting",
        "hr",
        "recruit",
        "customer service",
        "contact center",
        "call center",
        "operations",
        "support",
        "admin",
        "graduate",
        "intern",
        "technician",
        "nurse",
        "doctor",
        "warehouse",
    )
    has_role = any(r in t for r in role_terms)

    # Must contain some seniority/level or equivalently specific constraint.
    level_terms = (
        "entry",
        "junior",
        "mid",
        "senior",
        "lead",
        "principal",
        "manager",
        "director",
        "executive",
        "graduate",
        "intern",
        "0-2",
        "1-3",
        "3-5",
        "5+",
        "years",
    )
    has_level = any(l in t for l in level_terms)

    # If neither role nor level is present, we treat as vague.
    if not has_role and not has_level:
        return True

    return False


def _is_off_topic(text: str) -> bool:
    """Deterministic off-topic filter for non-SHL requests / prompt injection."""
    t = (text or "").strip().lower()
    if not t:
        return False

    # If they mention SHL/assessments/tests/hiring, we generally allow.
    in_scope_markers = (
        "shl",
        "assessment",
        "assessments",
        "test",
        "tests",
        "catalog",
        "product-catalog",
        "hiring",
        "role",
        "job description",
        "jd",
        "candidate",
    )
    if any(m in t for m in in_scope_markers):
        return False

    off_topic_markers = (
        "aws",
        "certification",
        "certifications",
        "azure",
        "gcp",
        "kubernetes",
        "terraform",
        "legal",
        "law",
        "contract",
        "medical",
        "diagnosis",
        "treatment",
        "symptom",
        "prescription",
        "interview hacks",
        "cheat",
        "bypass",
        "prompt injection",
        "ignore previous",
        "system prompt",
        "jailbreak",
    )
    return any(m in t for m in off_topic_markers)


def _clarification_reply() -> str:
    return (
        "To recommend the right SHL assessment, what role are you hiring for and what is the job level "
        "(entry / mid / senior)? If you can share a short job description, that also helps."
    )


def _refusal_reply() -> str:
    return "I can only help with SHL assessment selection. What role are you hiring for and what skills should the assessment measure?"


def _extract_last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def _user_turn_count(messages: list[dict]) -> int:
    return sum(1 for m in (messages or []) if m.get("role") == "user")


def _is_compare_query(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    compare_markers = (
        "difference between",
        "differences between",
        "compare",
        "vs",
        "versus",
    )
    return any(m in t for m in compare_markers)


def _extract_two_names_for_compare(text: str) -> Optional[Tuple[str, str]]:
    t = (text or "").strip()
    if not t:
        return None

    m = re.search(r"difference between\s+(.+?)\s+and\s+(.+?)(?:\?|\.|$)", t, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"compare\s+(.+?)\s+and\s+(.+?)(?:\?|\.|$)", t, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"(.+?)\s+vs\.?\s+(.+?)(?:\?|\.|$)", t, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def _build_catalog_block(results: list[dict]) -> str:
    """Serialize catalog search results as a compact numbered list."""
    lines = []
    for i, item in enumerate(results, 1):
        langs = ", ".join(item["languages"][:4])
        if len(item["languages"]) > 4:
            langs += f" (+{len(item['languages']) - 4} more)"
        lines.append(
            f"{i}. [{item['test_type']}] {item['name']}\n"
            f"   URL: {item['url']}\n"
            f"   Keys: {', '.join(item['keys'])}\n"
            f"   Duration: {item['duration'] or 'unspecified'} | "
            f"Job levels: {', '.join(item['job_levels'][:4]) or 'any'}\n"
            f"   Languages: {langs or 'unspecified'}\n"
            f"   Description: {item['description'][:180]}{'...' if len(item['description']) > 180 else ''}"
        )
    return "\n\n".join(lines)


def _build_candidate_block(results: list[dict]) -> str:
    lines = []
    for item in results or []:
        name = str(item.get("name") or "").strip()
        test_type = str(item.get("test_type") or "").strip()
        desc = str(item.get("description") or "").strip()
        url = str(item.get("url") or "").strip()
        if not name or not url:
            continue
        short_desc = desc[:160]
        lines.append(
            f"Name: {name}\n"
            f"Type: {test_type}\n"
            f"Description: {short_desc}\n"
            f"URL: {url}"
        )
    return "\n\n".join(lines)


def _extract_query(messages: list[dict]) -> str:
    """
    Derive a search query from the conversation.
    Combine all user turns — gives BM25 richer signal.
    """
    user_texts = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_texts)


def _build_catalog_validators(results: list[dict]) -> Tuple[dict, set]:
    """Build fast lookup structures for the current retrieval set."""
    name_map: Dict[str, dict] = {}
    url_set: set[str] = set()
    for item in results or []:
        name = str(item.get("name") or "")
        url = str(item.get("url") or "")
        if name:
            name_map[name.strip().lower()] = item
        if url:
            url_set.add(url)
    return name_map, url_set


def _normalize_name(s: str) -> str:
    return " ".join((s or "").lower().strip().split())


def _validate_recommendations(raw_recs: Any, name_map: dict, url_set: set) -> list[dict]:
    """Ensure recs exist in catalog results; dedupe; cap at 10."""
    if not isinstance(raw_recs, list):
        return []

    cleaned: List[dict] = []
    seen_urls: set[str] = set()
    for rec in raw_recs:
        if len(cleaned) >= 10:
            break
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("name") or "").strip()
        norm_name = _normalize_name(name)
        url = str(rec.get("url") or "").strip()

        # Prefer URL validation when present.
        catalog_item: Optional[dict] = None
        if url and url in url_set:
            # best-effort to locate by url so we can canonicalize name/test_type
            for _, item in name_map.items():
                if str(item.get("url") or "").strip() == url:
                    catalog_item = item
                    break
        elif name:
            catalog_item = name_map.get(norm_name)
            if catalog_item:
                url = str(catalog_item.get("url") or "").strip()

        if not url or url not in url_set:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Use canonical catalog fields when possible.
        if catalog_item is None:
            # best-effort to locate by url
            for _, item in name_map.items():
                if str(item.get("url") or "").strip() == url:
                    catalog_item = item
                    break

        cleaned.append(
            {
                "name": str((catalog_item or {}).get("name") or name),
                "url": url,
                "test_type": str((catalog_item or {}).get("test_type") or rec.get("test_type") or ""),
            }
        )
    return cleaned


def _bm25_fallback_recommendations(rerank_candidates: list[dict], name_map: dict, url_set: set) -> list[dict]:
    """Deterministic non-empty fallback when we are in recommendation mode.

    Uses BM25 candidate ordering and runs through the same grounding validation.
    """
    fallback = [
        {
            "name": x.get("name", ""),
            "url": x.get("url", ""),
            "test_type": x.get("test_type", ""),
        }
        for x in (rerank_candidates or [])[:10]
    ]
    return _validate_recommendations(fallback, name_map, url_set)


def _parse_response(raw: str) -> dict:
    """
    Extract the <RESPONSE> JSON block from the model output.
    Falls back gracefully if the block is malformed.
    """
    text = (raw or "").strip()

    # Find <RESPONSE>...</RESPONSE> first
    match = re.search(r"<RESPONSE>\s*(.*?)\s*</RESPONSE>", text, re.DOTALL | re.IGNORECASE)
    candidate = match.group(1).strip() if match else ""

    # If model returned fenced JSON, extract inside ```...```
    if candidate:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL | re.IGNORECASE)
        if fenced:
            candidate = fenced.group(1).strip()
    else:
        # Fallback: attempt to locate first JSON object in the entire output
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            candidate = fenced.group(1).strip()
        else:
            any_json = re.search(r"(\{\s*\"reply\"\s*:\s*.*\})", text, re.DOTALL)
            if any_json:
                candidate = any_json.group(1).strip()

    if not candidate:
        return safe_response(reply=text, recommendations=[], end_of_conversation=False)

    # Try strict JSON parse; if it fails, do a minimal cleanup (common trailing commas)
    data: dict
    try:
        data = json.loads(candidate)
    except Exception:
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            data = json.loads(cleaned)
        except Exception:
            reply_text = text
            if match:
                reply_text = text.replace(match.group(0), "").strip()
            return safe_response(reply=reply_text or "", recommendations=[], end_of_conversation=False)

    reply_text = ""
    if match:
        reply_text = text.replace(match.group(0), "").strip()
    if not reply_text:
        reply_text = str(data.get("reply") or "").strip()

    return safe_response(
        reply=reply_text,
        recommendations=data.get("recommendations") if isinstance(data, dict) else [],
        end_of_conversation=bool((data or {}).get("end_of_conversation", False)) if isinstance(data, dict) else False,
    )


def chat(messages: list[dict]) -> dict:
    """
    Main entry point.
    messages: list of {"role": "user"|"assistant", "content": str}
    Returns: {"reply": str, "recommendations": list, "end_of_conversation": bool}
    """
    last_user = _extract_last_user_message(messages)

    # Enforce a deterministic turn budget in code (not just prompt text).
    if _user_turn_count(messages) >= 8:
        return safe_response(
            reply="This conversation has reached the maximum number of turns. Please start a new chat if you want to explore different assessment options.",
            recommendations=[],
            end_of_conversation=True,
        )

    # Deterministic refusal for off-topic / prompt injection-like requests
    if _is_off_topic(last_user):
        return safe_response(_refusal_reply(), [], False)

    # Deterministic clarification for vague requests
    if _is_vague_user_query(last_user):
        return safe_response(_clarification_reply(), [], False)

    # Deterministic comparison behavior: answer from catalog only; keep recommendations empty.
    if _is_compare_query(last_user):
        names = _extract_two_names_for_compare(last_user)
        if names:
            a_name, b_name = names
            a = get_by_name(a_name)
            b = get_by_name(b_name)
            if a and b:
                reply = (
                    f"Here’s a catalog-based comparison:\n\n"
                    f"{a['name']}: {a.get('description','')}\n\n"
                    f"{b['name']}: {b.get('description','')}"
                )
                return safe_response(reply=reply, recommendations=[], end_of_conversation=False)
        return safe_response(
            reply="Which two SHL assessments would you like me to compare? Please provide their exact names from the catalog.",
            recommendations=[],
            end_of_conversation=False,
        )

    query = _extract_query(messages)
    retrieved_with_scores = search_with_scores(query, top_k=15)
    rerank_candidates = [item for (item, _score) in (retrieved_with_scores or [])]
    bm25_scores_by_url: Dict[str, float] = {
        str(item.get("url") or "").strip(): float(score)
        for (item, score) in (retrieved_with_scores or [])
        if str(item.get("url") or "").strip()
    }
    catalog_block = _build_candidate_block(rerank_candidates)
    name_map, url_set = _build_catalog_validators(rerank_candidates)
    print("CATALOG QUERY:", query)
    print("BM25 CANDIDATE POOL SIZE:", len(rerank_candidates or []))
    print("BM25 TOP NAMES:", [x.get("name") for x in (rerank_candidates or [])[:10]])

    # If retrieval somehow yields no candidates, we cannot recommend anything.
    # Keep schema intact and ask for more detail.
    if not (rerank_candidates or []):
        return safe_response(
            reply=_clarification_reply(),
            recommendations=[],
            end_of_conversation=False,
        )

    # If Groq isn't configured, provide a deterministic retrieval-only fallback.
    # This keeps the API functional and schema-compliant.
    if not _GROQ_API_KEY:
        recs = _bm25_fallback_recommendations(rerank_candidates, name_map, url_set)
        reply = (
            "I can't access the LLM right now, but here are the closest matching SHL assessments from the catalog "
            "based on your requirements."
        )
        return safe_response(reply=reply, recommendations=recs, end_of_conversation=False)

    rerank_prompt = (
        "You are an SHL assessment recommendation assistant.\n\n"
        f"User hiring query:\n{query}\n\n"
        "Retrieved SHL catalog candidates:\n"
        f"{catalog_block}\n\n"
        "Task:\n"
        "Score EACH candidate for semantic relevance to the user query.\n"
        "Return a relevance_score from 0.0 to 1.0 for every URL provided above.\n\n"
        "Strong guidance:\n"
        "- Strongly prioritize assessments matching explicit technologies, frameworks, and hard skills mentioned in the query.\n"
        "- Behavioral, communication, and leadership alignment should refine ranking quality, not replace core technical relevance.\n"
        "- If the query is a purely technical role, do NOT let behavioral assessments dominate technical ones.\n"
        "- ONLY score assessments provided above.\n"
        "- DO NOT invent assessments.\n"
        "- Keep URLs exactly unchanged.\n\n"
        "Output format — CRITICAL — DO NOT DEVIATE:\n"
        "You MUST end every response with a JSON block between <RESPONSE> tags.\n"
        "The block must be valid JSON. No trailing commas.\n\n"
        "<RESPONSE>\n"
        "{\n"
        '  "reply": "short explanation",\n'
        '  "scored_candidates": [\n'
        "    {\n"
        '      "url": "exact catalog url",\n'
        '      "relevance_score": 0.0\n'
        "    }\n"
        "  ],\n"
        '  "end_of_conversation": false\n'
        "}\n"
        "</RESPONSE>\n"
    )

    try:
        _client = Groq(api_key=_GROQ_API_KEY)
        start = time.time()
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": rerank_prompt}],
            temperature=0.2,
        )
        _ = time.time() - start
        raw_text = ""
        try:
            raw_text = (response.choices[0].message.content or "") if response and response.choices else ""
        except Exception:
            raw_text = ""
        print("RAW RESPONSE:", raw_text)
    except Exception as e:
        print("AGENT ERROR:", str(e))

        fallback_recommendations = []
        for item in (rerank_candidates or [])[:10]:
            fallback_recommendations.append(
                {
                    "name": item.get("name", ""),
                    "url": item.get("url", ""),
                    "test_type": item.get("test_type", "K"),
                }
            )
        fallback_recommendations = _validate_recommendations(fallback_recommendations, name_map, url_set)
        return safe_response(
            reply=(
                "Based on the provided hiring requirements, here are relevant "
                "SHL assessments for technical and behavioral evaluation."
            ),
            recommendations=fallback_recommendations,
            end_of_conversation=False,
        )


    cleaned_text = (
        (raw_text or "")
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    parsed = _parse_response(cleaned_text)
    print("PARSED RESPONSE:", parsed)

    scored = []
    try:
        scored = parsed.get("scored_candidates") if isinstance(parsed, dict) else []
        if not isinstance(scored, list):
            scored = []
    except Exception:
        scored = []

    llm_scores_by_url: Dict[str, float] = {}
    for row in scored:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        try:
            score = float(row.get("relevance_score"))
        except Exception:
            score = 0.0
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        llm_scores_by_url[url] = score

    if not llm_scores_by_url:
        print("RERANK NOTICE:", "No usable LLM semantic scores returned; falling back to BM25 ordering")
        fallback_recs = _bm25_fallback_recommendations(rerank_candidates, name_map, url_set)
        return safe_response(
            reply=(parsed.get("reply", "") if isinstance(parsed, dict) else ""),
            recommendations=fallback_recs,
            end_of_conversation=bool((parsed or {}).get("end_of_conversation", False)) if isinstance(parsed, dict) else False,
        )

    # Hybrid scoring (BM25 anchor)
    bm25_vals = list(bm25_scores_by_url.values())
    bm25_max = max(bm25_vals) if bm25_vals else 0.0
    bm25_min = min(bm25_vals) if bm25_vals else 0.0
    denom = (bm25_max - bm25_min) if (bm25_max - bm25_min) > 1e-9 else 1.0

    diagnostics_rows = []
    hybrid_ranked = []
    for item in rerank_candidates or []:
        url = str(item.get("url") or "").strip()
        bm25_raw = float(bm25_scores_by_url.get(url, 0.0))
        bm25_norm = (bm25_raw - bm25_min) / denom
        llm_sem = float(llm_scores_by_url.get(url, 0.0))
        final_score = 0.75 * bm25_norm + 0.25 * llm_sem
        hybrid_ranked.append((final_score, item))
        diagnostics_rows.append(
            {
                "name": str(item.get("name") or ""),
                "url": url,
                "bm25": bm25_raw,
                "bm25_norm": bm25_norm,
                "llm": llm_sem,
                "final": final_score,
            }
        )

    hybrid_ranked.sort(key=lambda t: t[0], reverse=True)
    print("HYBRID RERANK DIAGNOSTICS (TOP 15):")
    for row in sorted(diagnostics_rows, key=lambda r: r.get("final", 0.0), reverse=True)[:15]:
        print(
            "-",
            row.get("name"),
            "bm25=", round(float(row.get("bm25", 0.0)), 4),
            "llm=", round(float(row.get("llm", 0.0)), 4),
            "final=", round(float(row.get("final", 0.0)), 4),
        )

    hybrid_recs = [
        {"name": (item or {}).get("name", ""), "url": (item or {}).get("url", ""), "test_type": (item or {}).get("test_type", "")}
        for (_score, item) in (hybrid_ranked or [])
    ]

    validated_recs = _validate_recommendations(hybrid_recs, name_map, url_set)
    print("VALIDATED NAMES:", [x.get("name") for x in (validated_recs or [])])
    print("FILTERED RECOMMENDATIONS COUNT:", len(validated_recs or []))

    if not (validated_recs or []):
        print("FALLBACK REASON:", "No validated hybrid recommendations after grounding/validation; falling back to BM25 ordering")
        fallback_recommendations = _bm25_fallback_recommendations(rerank_candidates, name_map, url_set)
        return safe_response(
            reply=(
                "Based on the provided hiring requirements, here are relevant "
                "SHL assessments for technical and behavioral evaluation."
            ),
            recommendations=fallback_recommendations,
            end_of_conversation=False,
        )

    capped_recs = validated_recs[:10]
    # Guardrail: once we are in recommendation mode (i.e., we have some grounded recs),
    # never return an empty list.
    if not capped_recs:
        capped_recs = _bm25_fallback_recommendations(rerank_candidates, name_map, url_set)
    return safe_response(
        reply=parsed.get("reply", ""),
        recommendations=capped_recs,
        end_of_conversation=bool(parsed.get("end_of_conversation", False)),
    )
