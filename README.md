# Conversatioal_SHL_Assesment_Recommender

> A conversational AI-powered recommender system that surfaces structured SHL assessment suggestions from the official product catalog — built with FastAPI, BM25 retrieval, and Groq LLM reranking.

---

## Motivation

Recruiters and hiring managers often struggle to identify the right SHL assessments for a given role. The process is manual, time-consuming, and heavily dependent on familiarity with a large product catalog. The goal of this project was to eliminate that friction by building a conversational interface that understands natural-language queries and returns precise, catalog-grounded assessment recommendations instantly.

The system was designed with one overarching principle: **trust**. Every recommendation had to be verifiable, every URL had to be real, and the system had to behave predictably under all conditions.

---

## How It Started

The project began with a straightforward question: *how do you make a large product catalog searchable through natural language?*

The first step was ingesting and normalising the `shl_product_catalog.json` into a consistent schema — ensuring every entry had a name, URL, description, test type, job levels, and other structured fields. From there, **BM25 (Rank-BM25)** was selected as the retrieval backbone. It was fast, deterministic, and well-suited to the keyword-heavy nature of recruiter queries.

Once the retrieval layer was stable, the next challenge was improving ranking quality. A **Groq LLM** was introduced purely for reranking the already-retrieved candidates, keeping the model bounded to a known candidate set and preventing any open-ended or hallucinated output. The hybrid scoring formula that emerged was:

```
final_score = 0.75 × normalised_bm25 + 0.25 × llm_score
```

The FastAPI backend was built in parallel, exposing two endpoints — `GET /health` and `POST /chat` — with a strict response schema enforced on every outbound message. A Streamlit frontend was added to provide a clean, interactive chat interface on top of the API.

---

## Project Structure

```
.
├── main.py                  # FastAPI backend — /health and /chat endpoints
├── agent.py                 # Recommendation agent — BM25 retrieval + LLM reranking + guardrails
├── app.py                   # Streamlit conversational UI
├── evaluation.py            # Recall@10 benchmarking script
├── shl_product_catalog.json # Normalised SHL product catalog
└── requirements.txt         # Python dependencies
```

---

## API Reference

### `GET /health`
Returns service status.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /chat`
Accepts a conversation history and returns a recommendation response.

**Request body:**
```json
{
  "messages": [
    { "role": "user", "content": "I need a cognitive ability test for a senior analyst role." }
  ]
}
```

**Response:**
```json
{
  "reply": "Based on your requirements, here are the most relevant assessments...",
  "recommendations": [
    {
      "name": "Verify Numerical Reasoning",
      "url": "https://www.shl.com/...",
      "test_type": "Cognitive Ability"
    }
  ],
  "end_of_conversation": false
}
```

---

## Guardrails

The system enforces four hard constraints at all times:

| Guardrail | Description |
|---|---|
| **Catalog-only URLs** | A recommendation is accepted only if its URL exists in the retrieved catalog candidate set. No URLs are ever fabricated. |
| **Non-empty recommendations** | Once in recommendation mode, the system always returns at least one valid suggestion. A BM25 fallback activates if the LLM output is unusable. |
| **8-turn conversation limit** | Every conversation is capped at eight user turns. On the eighth turn, `end_of_conversation: true` is returned. |
| **Strict response schema** | Every response is normalised through `safe_response()`, guaranteeing `reply` is a string, `recommendations` is a list, and `end_of_conversation` is a boolean. |

---

## How It Ended

The final system delivered on every requirement. The API contract was fully validated — `/health` returns `{"status":"ok"}` and `/chat` consistently returns all three required fields under all tested conditions. The BM25 fallback eliminated the edge-case risk of empty recommendation lists entirely, and URL grounding ensured hallucinated links were structurally impossible.

Evaluation using `evaluation.py` confirmed that the hybrid BM25 + LLM pipeline outperformed the BM25-only baseline on `mean_recall@10` across the recruiter-style test cases.

Three notable challenges were encountered and resolved during development:

- **Malformed LLM output** — the Groq model occasionally returned invalid JSON, triggering the BM25 fallback path gracefully.
- **Empty recommendations** — resolved by implementing `_bm25_fallback_recommendations()` as a deterministic safety net.
- **Dependency conflict** — installing Streamlit in a shared environment downgraded Starlette and broke FastAPI. Resolved by isolating all dependencies in a dedicated virtual environment.

---

## Evaluation

The benchmarking script (`evaluation.py`) measures **Recall@10** on assessment names across a set of recruiter-style queries with known ground-truth answers.

| Pipeline | Metric |
|---|---|
| BM25-only baseline | `mean_bm25_recall@10` |
| Full pipeline (BM25 + LLM reranking) | `mean_final_recall@10` |

The full pipeline consistently outperformed the baseline, confirming that LLM reranking adds meaningful signal beyond keyword retrieval.

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/shl-assessment-recommender.git
cd shl-assessment-recommender

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key
export GROQ_API_KEY=your_key_here

# 5. Start the FastAPI backend
uvicorn main:app --reload

# 6. (Optional) Launch the Streamlit UI
streamlit run app.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| Retrieval | Rank-BM25 |
| LLM Reranking | Groq API |
| Frontend UI | Streamlit |
| Data Format | JSON (normalised catalog) |
| Language | Python 3.10+ |

---

## License

This project was built as part of an SHL assessment engineering challenge. All product catalog data belongs to SHL.
