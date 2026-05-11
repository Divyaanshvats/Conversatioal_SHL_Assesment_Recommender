import os
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://conversatioal-shl-assesment-recommender-3.onrender.com")
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"


def _init_page() -> None:
    st.set_page_config(
        page_title="SHL Assessment Recommender",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _apply_dark_styles() -> None:
    st.markdown(
        """
<style>
    /* App container */
    .stApp {
        background: radial-gradient(1200px 800px at 20% 10%, rgba(124, 92, 255, 0.14), transparent 60%),
                    radial-gradient(1200px 800px at 80% 30%, rgba(0, 209, 255, 0.10), transparent 55%),
                    #0b0f17;
        color: #e7eaf0;
    }

    /* Constrain main column a bit for a "dashboard" feel */
    section.main > div.block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1100px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f18 0%, #080c13 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* Chat input: make it feel larger */
    [data-testid="stChatInput"] textarea {
        min-height: 84px !important;
        padding-top: 0.9rem !important;
        padding-bottom: 0.9rem !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #e7eaf0 !important;
    }

    /* Make buttons slightly more modern */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.06);
        color: #e7eaf0;
    }
    .stButton > button:hover {
        border-color: rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.10);
    }

    /* Recommendation card */
    .rec-card {
        border-radius: 16px;
        padding: 14px 14px;
        background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.03) 100%);
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        height: 100%;
    }
    .rec-title {
        font-size: 0.98rem;
        font-weight: 650;
        margin: 0;
        color: #f3f5f9;
        line-height: 1.25rem;
    }
    .rec-meta {
        margin-top: 6px;
        color: rgba(231,234,240,0.75);
        font-size: 0.86rem;
    }
    .rec-link {
        margin-top: 10px;
        display: inline-block;
        font-size: 0.88rem;
        color: #9ad7ff;
        text-decoration: none;
        border: 1px solid rgba(154,215,255,0.25);
        padding: 7px 10px;
        border-radius: 12px;
        background: rgba(154,215,255,0.08);
    }
    .rec-link:hover {
        border-color: rgba(154,215,255,0.45);
        background: rgba(154,215,255,0.12);
    }

    /* Subtle separators */
    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 1rem 0;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _get_health(timeout_s: float = 1.2) -> Dict[str, Any]:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=timeout_s)
        ok = r.status_code == 200
        data: Dict[str, Any] = {}
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}
        return {"ok": ok, "status_code": r.status_code, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []

    if "last_recommendations" not in st.session_state:
        st.session_state.last_recommendations: List[Dict[str, str]] = []

    if "end_of_conversation" not in st.session_state:
        st.session_state.end_of_conversation = False


def _clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.last_recommendations = []
    st.session_state.end_of_conversation = False


def _post_chat(messages: List[Dict[str, str]], timeout_s: float = 90.0) -> Dict[str, Any]:
    payload = {"messages": messages}
    r = requests.post(CHAT_ENDPOINT, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _render_header() -> None:
    st.markdown(
        """
<div style="display:flex; align-items:flex-end; justify-content:space-between; gap:12px;">
  <div>
    <div style="font-size:1.45rem; font-weight:750; letter-spacing:-0.02em;">SHL Assessment Recommender</div>
    <div style="color:rgba(231,234,240,0.70); margin-top:4px;">Describe a role or hiring needs. Chat to refine. Get structured assessment recommendations.</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr />", unsafe_allow_html=True)


def _render_recommendations(recommendations: List[Dict[str, Any]]) -> None:
    if not recommendations:
        st.caption("No recommendations returned for the latest turn.")
        return

    st.markdown("#### Recommendations")

    cols = st.columns(2)
    for i, rec in enumerate(recommendations):
        name = str(rec.get("name", ""))
        url = str(rec.get("url", ""))
        test_type = str(rec.get("test_type", ""))

        with cols[i % 2]:
            st.markdown(
                f"""
<div class="rec-card">
  <div class="rec-title">{name if name else "(Unnamed assessment)"}</div>
  <div class="rec-meta">Test type: <b>{test_type if test_type else "—"}</b></div>
  {f'<a class="rec-link" href="{url}" target="_blank" rel="noopener noreferrer">Open assessment</a>' if url else '<div class="rec-meta" style="margin-top:10px;">No URL provided.</div>'}
</div>
                """,
                unsafe_allow_html=True,
            )


def _render_chat() -> None:
    for msg in st.session_state.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role not in ("user", "assistant"):
            continue
        with st.chat_message(role):
            st.markdown(content)


def _normalize_recs(recs: Any) -> List[Dict[str, Any]]:
    if not isinstance(recs, list):
        return []
    out: List[Dict[str, Any]] = []
    for r in recs:
        if isinstance(r, dict):
            out.append(r)
    return out


def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
<div style="font-size:1.05rem; font-weight:750;">SHL Recommender</div>
<div style="color:rgba(231,234,240,0.70); margin-top:2px;">Conversational dashboard</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr />", unsafe_allow_html=True)

        health = _get_health()
        if health.get("ok"):
            st.success("Backend: Online")
        else:
            st.error("Backend: Offline")

        st.caption(f"Backend URL: `{BACKEND_URL}`")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear chat", use_container_width=True):
                _clear_chat()
                st.rerun()
        with col2:
            if st.button("Refresh", use_container_width=True):
                st.rerun()

        st.markdown("<hr />", unsafe_allow_html=True)

        with st.expander("Settings", expanded=False):
            st.text_input(
                "Backend URL",
                value=BACKEND_URL,
                key="_backend_url_display",
                disabled=True,
            )
            st.caption(
                "To change backend URL, set environment variable `BACKEND_URL` before running Streamlit."
            )


def main() -> None:
    _init_page()
    _apply_dark_styles()
    _ensure_state()

    _sidebar()

    _render_header()

    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        st.markdown("#### Conversation")
        _render_chat()

        user_text = st.chat_input(
            "Enter hiring requirements or paste a job description…",
            disabled=bool(st.session_state.end_of_conversation),
        )

        if st.session_state.end_of_conversation:
            st.info("Conversation marked as complete by the backend. Clear chat to start a new one.")

        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        data = _post_chat(st.session_state.messages)
                        reply = str(data.get("reply", ""))
                        recs = _normalize_recs(data.get("recommendations"))
                        eoc = bool(data.get("end_of_conversation", False))

                        if reply:
                            st.markdown(reply)
                        else:
                            st.markdown("(No reply returned.)")

                        if recs:
                            st.markdown("---")
                            _render_recommendations(recs)

                        st.session_state.messages.append({"role": "assistant", "content": reply or ""})
                        st.session_state.last_recommendations = recs
                        st.session_state.end_of_conversation = eoc

                    except requests.HTTPError as e:
                        st.error("Request failed.")
                        try:
                            st.code(e.response.text)
                        except Exception:
                            st.code(str(e))
                    except Exception as e:
                        st.error("Could not reach the backend.")
                        st.code(str(e))

            time.sleep(0.05)
            st.rerun()

    with right:
        _render_recommendations(st.session_state.last_recommendations)


if __name__ == "__main__":
    main()
