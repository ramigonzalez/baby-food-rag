"""Chat over the official infant-feeding guides.

Run from the repo root, after the package is installed, the PDFs are
downloaded, and the index exists:

    streamlit run app/streamlit_app.py

Weekly menus are phase 4b. This screen only answers one question at a time.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from baby_food_rag.embed import load_embedder
from baby_food_rag.generate import (
    DEFAULT_MODEL,
    DISCLAIMER,
    MissingAPIKey,
    OpenRouterLLM,
    append_successful_turn,
    indexed_guides,
    question_to_send,
)
from baby_food_rag.index import load_index

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"
MANIFEST = ROOT / "sources" / "manifest.yaml"
_KEY_ERROR = "Defina OPENROUTER_API_KEY no ambiente. A chave não é gravada aqui."
_ASK_ERROR = "Não foi possível consultar os guias. A pergunta não entrou no chat."


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


def main() -> None:
    st.set_page_config(page_title="Alimentação do bebê", layout="centered")
    st.markdown(f"**{DISCLAIMER}**")
    st.title("Alimentação do bebê")
    try:
        index = load_index(INDEX_DIR)
    except FileNotFoundError:
        st.error("Índice não encontrado. Baixe os PDFs e rode `baby-food index` antes.")
        return
    with st.sidebar:
        st.header("Guias no índice")
        for guide in indexed_guides(index.meta, MANIFEST):
            if guide["url"]:
                st.markdown(f"- [{guide['id']}]({guide['url']}) ({guide['year']})")
            else:
                st.markdown(f"- {guide['id']} ({guide['year']})")
        st.caption("O cardápio da semana fica para uma fase seguinte.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("passages"):
                with st.expander("Trechos recuperados"):
                    st.markdown(message["passages"])
    failed = st.session_state.get("failed_question")
    retry = False
    if failed:
        st.error(st.session_state.get("failed_reason") or _ASK_ERROR)
        st.caption(failed)
        retry = st.button("Tentar de novo")
    typed = st.chat_input("Pergunte sobre a alimentação do bebê")
    question = question_to_send(typed, failed, retry)
    if not question:
        return
    embedder = cached_embedder(index.embedder_name)
    try:
        llm = OpenRouterLLM(model=DEFAULT_MODEL)
        with st.spinner("Consultando os guias..."):
            st.session_state.messages = append_successful_turn(
                st.session_state.messages,
                question,
                index,
                llm,
                embedder=embedder,
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
