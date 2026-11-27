"""Chat over the official infant-feeding guides.

Run from the repo root, after the package is installed, the PDFs are
downloaded, and the index exists:

    streamlit run app/streamlit_app.py

Each turn searches on its own. The weekly menu is phase 4b.
"""

from __future__ import annotations

from inspect import signature
from pathlib import Path

import streamlit as st

from baby_food_rag.embed import load_embedder
from baby_food_rag.generate import (
    DEFAULT_MODEL,
    DISCLAIMER,
    EXAMPLE_QUESTIONS,
    MissingAPIKey,
    OpenRouterLLM,
    button_width_kwargs,
    indexed_guides,
    question_to_send,
    run_question,
)
from baby_food_rag.index import load_index

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"
MANIFEST = ROOT / "sources" / "manifest.yaml"
_KEY_ERROR = "Defina OPENROUTER_API_KEY no ambiente. A chave não é gravada aqui."
_ASK_ERROR = "Não foi possível consultar os guias. A pergunta não entrou no chat."

_CSS = """
<style>
.block-container { max-width: 40rem; padding-top: 1rem; padding-bottom: 5rem; }
h1 { font-size: 1.45rem !important; line-height: 1.25; margin-bottom: 0.25rem; }
[data-testid="stCaptionContainer"] { font-size: 0.95rem; }
[data-testid="stChatMessage"] { font-size: 1.05rem; line-height: 1.45; }
div.stButton > button { white-space: normal; text-align: left; height: auto; }
</style>
"""


@st.cache_resource
def cached_embedder(name: str):
    """One e5 model for the whole chat. A hash index needs none."""
    if name in {"", "hash"}:
        return None
    return load_embedder(name)


def _keep_failed_question(question: str, reason: str) -> None:
    st.session_state.failed_question = question
    st.session_state.failed_reason = reason
    st.rerun()


def _clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.failed_question = None
    st.session_state.failed_reason = None
    st.rerun()


def _render_assistant(message: dict) -> None:
    segments = message.get("segments")
    if segments:
        for part in segments:
            if part.get("type") == "citation":
                with st.expander(part["label"]):
                    if part.get("title"):
                        st.markdown(f"**{part['title']}**")
                    st.caption(f"Página {part['page']}")
                    for excerpt in part.get("excerpts") or []:
                        st.markdown(excerpt)
                    if part.get("url"):
                        st.markdown(f"[Abrir o PDF nesta página]({part['url']})")
            elif str(part.get("text", "")).strip():
                st.markdown(part["text"])
    else:
        st.markdown(message.get("content", ""))
    st.caption(DISCLAIMER)
    if message.get("passages"):
        with st.expander("Trechos recuperados"):
            st.markdown(message["passages"])


def main() -> None:
    st.set_page_config(page_title="Alimentação do bebê", layout="centered")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title("Alimentação do bebê")
    st.caption(DISCLAIMER)
    try:
        index = load_index(INDEX_DIR)
    except FileNotFoundError:
        st.error("Índice não encontrado. Baixe os PDFs e rode `baby-food index` antes.")
        return
    with st.sidebar:
        st.header("Guias no índice")
        for guide in indexed_guides(index.meta, MANIFEST):
            year = f" ({guide['year']})" if guide.get("year") else ""
            if guide["url"]:
                st.markdown(f"- [{guide['id']}]({guide['url']}){year}")
            else:
                st.markdown(f"- {guide['id']}{year}")
        st.caption("O cardápio da semana fica para uma fase seguinte.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.session_state.messages and st.button("Limpar conversa"):
        _clear_chat()
    picked = None
    if not st.session_state.messages:
        st.caption("Perguntas de exemplo")
        width_kwargs = button_width_kwargs(set(signature(st.button).parameters))
        for index_q, question in enumerate(EXAMPLE_QUESTIONS):
            if st.button(question, key=f"example-{index_q}", **width_kwargs):
                picked = question
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant(message)
            else:
                st.markdown(message["content"])
    failed = st.session_state.get("failed_question")
    retry = False
    if failed:
        st.error(st.session_state.get("failed_reason") or _ASK_ERROR)
        st.caption(failed)
        retry = st.button("Tentar de novo")
    typed = st.chat_input("Pergunte sobre a alimentação do bebê")
    question = question_to_send(typed, failed, retry) or picked
    if not question:
        return
    try:
        with st.spinner("Consultando os guias..."):
            st.session_state.messages = run_question(
                st.session_state.messages,
                question,
                index,
                cached_embedder,
                lambda: OpenRouterLLM(model=DEFAULT_MODEL),
            )
    except MissingAPIKey:
        _keep_failed_question(question, _KEY_ERROR)
    except Exception:
        _keep_failed_question(question, _ASK_ERROR)
    else:
        st.session_state.failed_question = None
        st.session_state.failed_reason = None
        st.rerun()


if __name__ == "__main__":
    main()
