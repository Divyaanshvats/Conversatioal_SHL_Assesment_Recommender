"""
main.py
FastAPI service exposing GET /health and POST /chat.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from agent import chat

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages list must not be empty")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    # Convert pydantic models to plain dicts for agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        result = chat(messages)
    except Exception as e:
        print("API ERROR:", str(e))
        import traceback
        traceback.print_exc()
        result = {
            "reply": "I'm currently unable to process recommendations. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    return ChatResponse(
        reply=str(result.get("reply", "")),
        recommendations=[
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r["test_type"],
            )
            for r in (result.get("recommendations") or [])
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )
