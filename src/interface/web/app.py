"""Streamlit multipage web UI for the 1337 RAG Assistant."""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

from src.interface.web.api_client import RAGAPIClient
from src.interface.web.i18n import SUPPORTED_LANGUAGES, load_translations

__all__ = ["main"]

_logger = logging.getLogger(__name__)

_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
_LANGFUSE_URL = os.getenv("RAG_LANGFUSE_URL", "http://localhost:3000")
_EVAL_PATH = Path("evals/runs/latest.json")


def _client() -> RAGAPIClient:
    return RAGAPIClient(base_url=_API_URL, langfuse_base_url=_LANGFUSE_URL)


def _t(key: str, **kwargs: object) -> str:
    lang = st.session_state.get("lang", "en")
    translations = load_translations(lang)
    template = translations.get(key, key)
    return template.format(**kwargs) if kwargs else template


def _init_session_state() -> None:
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "citations" not in st.session_state:
        st.session_state["citations"] = []
    if "last_trace_id" not in st.session_state:
        st.session_state["last_trace_id"] = None


def _render_sidebar() -> str:
    with st.sidebar:
        st.title(_t("app_title"))
        st.caption(_t("app_subtitle"))
        st.divider()

        lang_labels = [_t(f"lang_{code}") for code in SUPPORTED_LANGUAGES]
        lang_codes = list(SUPPORTED_LANGUAGES.keys())
        current_idx = lang_codes.index(st.session_state.get("lang", "en"))
        selected_idx = st.selectbox(
            _t("sidebar_language"),
            options=range(len(lang_codes)),
            format_func=lambda i: lang_labels[i],
            index=current_idx,
        )
        st.session_state["lang"] = lang_codes[selected_idx]  # type: ignore[index] — bounds-checked by Streamlit selectbox length invariant

        st.divider()
        sid = st.session_state.get("session_id", "")
        st.text_input(
            _t("sidebar_session_id_label"),
            value=sid or _t("sidebar_session_id_empty"),
            disabled=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(_t("sidebar_clear"), use_container_width=True):
                st.session_state["messages"] = []
                st.session_state["citations"] = []
                st.session_state["last_trace_id"] = None
                st.rerun()
        with col2:
            if st.button(_t("sidebar_new_session"), use_container_width=True):
                st.session_state["session_id"] = str(uuid.uuid4())
                st.session_state["messages"] = []
                st.session_state["citations"] = []
                st.session_state["last_trace_id"] = None
                st.rerun()

        st.divider()
        page = st.radio(
            "Navigation",
            options=[
                _t("nav_chat"),
                _t("nav_ingest"),
                _t("nav_sessions"),
                _t("nav_eval"),
            ],
            label_visibility="collapsed",
        )
    return page  # type: ignore[return-value] — st.radio returns str, mypy infers Any due to Streamlit's dynamic typing


def _render_citations(citations: list[Any]) -> None:
    if not citations:
        return
    with st.expander(_t("citations_expander"), expanded=False):
        for cit in citations:
            cols = st.columns([3, 1, 1])
            cols[0].markdown(f"**{_t('citation_source')}:** {cit.get('source', '—')}")
            cols[1].markdown(f"**{_t('citation_page')}:** {cit.get('page', '—')}")
            score_str = f"{cit['score']:.3f}" if isinstance(cit.get("score"), float) else "—"
            cols[2].markdown(f"**{_t('citation_score')}:** {score_str}")


def _page_chat() -> None:
    messages: list[dict[str, Any]] = st.session_state["messages"]

    if not messages:
        st.info(_t("chat_no_messages"))

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                _render_citations(msg["citations"])
            if msg["role"] == "assistant" and msg.get("trace_id"):
                trace_url = _client().langfuse_trace_url(msg["trace_id"])
                st.markdown(f"[{_t('langfuse_trace')}]({trace_url})")

    prompt = st.chat_input(_t("chat_input_placeholder"))
    if not prompt:
        return

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    citations_out: list[Any] = []
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        try:
            with st.spinner(_t("chat_thinking")):
                for token in _client().chat_stream_sync(
                    query=prompt,
                    session_id=st.session_state.get("session_id"),
                    language=st.session_state.get("lang"),
                    citations_out=citations_out,
                ):
                    full_text += token
                    placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)
        except Exception as exc:
            _logger.exception("Chat stream error")
            st.error(_t("error_generic", error=str(exc)))
            return

        _render_citations(citations_out)

    messages.append(
        {
            "role": "assistant",
            "content": full_text,
            "citations": citations_out,
            "trace_id": st.session_state.get("last_trace_id"),
        }
    )
    st.session_state["citations"] = citations_out


def _page_ingest() -> None:
    st.header(_t("ingest_title"))
    uploaded = st.file_uploader(
        _t("ingest_upload_label"),
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button(_t("ingest_button")) and uploaded:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            for uf in uploaded:
                dest = Path(tmpdir) / uf.name
                dest.write_bytes(uf.read())
            try:
                import asyncio

                result = asyncio.run(
                    _client().ingest(
                        path=tmpdir,
                        language_hint=st.session_state.get("lang"),
                    )
                )
                st.success(
                    _t(
                        "ingest_success",
                        files_processed=result.get("files_processed", 0),
                        chunks_created=result.get("chunks_created", 0),
                        duration_seconds=result.get("duration_seconds", 0.0),
                    )
                )
            except Exception as exc:
                _logger.exception("Ingest error")
                st.error(_t("ingest_error", error=str(exc)))


def _page_sessions() -> None:
    st.header(_t("sessions_title"))
    session_id = st.text_input(_t("sessions_select"), value="")
    if not session_id:
        st.info(_t("sessions_no_sessions"))
        return
    try:
        import asyncio

        data = asyncio.run(_client().get_session(session_id))
        messages = data.get("messages", [])
        for msg in messages:
            role_label = (
                _t("sessions_role_user")
                if msg.get("role") == "user"
                else _t("sessions_role_assistant")
            )
            with st.chat_message(msg.get("role", "user")):
                st.markdown(f"**{role_label}:** {msg.get('content', '')}")
    except Exception as exc:
        _logger.exception("Sessions fetch error")
        st.error(_t("error_generic", error=str(exc)))


def _page_eval() -> None:
    st.header(_t("eval_title"))
    if not _EVAL_PATH.exists():
        st.info(_t("eval_no_runs"))
        return
    try:
        with _EVAL_PATH.open(encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
    except Exception as exc:
        _logger.exception("Eval load error")
        st.error(_t("error_generic", error=str(exc)))
        return

    aggregate = data.get("aggregate", {})
    thresholds = data.get("thresholds", {})

    for metric, score in aggregate.items():
        threshold = thresholds.get(metric)
        passed = threshold is None or score >= threshold
        col1, col2, col3 = st.columns([2, 1, 1])
        col1.markdown(f"**{metric}**")
        col2.metric(_t("eval_metric_score"), f"{score:.3f}")
        if threshold is not None:
            col3.metric(
                _t("eval_threshold"),
                f"{threshold:.3f}",
                delta=f"{score - threshold:+.3f}",
                delta_color="normal" if passed else "inverse",
            )

    if docs_url := data.get("docs_url"):
        st.markdown(f"[{_t('eval_docs_link')}]({docs_url})")


def main() -> None:
    """Run the Streamlit multipage RAG web UI."""
    st.set_page_config(
        page_title=_t("app_title"),
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()
    page = _render_sidebar()

    chat_label = _t("nav_chat")
    ingest_label = _t("nav_ingest")
    sessions_label = _t("nav_sessions")
    eval_label = _t("nav_eval")

    if page == chat_label:
        _page_chat()
    elif page == ingest_label:
        _page_ingest()
    elif page == sessions_label:
        _page_sessions()
    elif page == eval_label:
        _page_eval()
    else:
        _page_chat()


if __name__ == "__main__":
    main()
