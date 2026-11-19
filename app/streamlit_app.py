"""Chat over the official infant-feeding guides.

Run from the repo root, after the package is installed, the PDFs are
downloaded, and the index exists:

    streamlit run app/streamlit_app.py

Weekly menus are phase 4b. This screen only answers one question at a time.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from baby_food_rag.generate import (
    DEFAULT_MODEL,
    DISCLAIMER,
    MissingAPIKey,
    OpenRouterLLM,
    ask,
    format_answer_markdown,
    indexed_guides,
    passages_markdown,
)
from baby_food_rag.index import load_index

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"
MANIFEST = ROOT / "sources" / "manifest.yaml"


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
    question = st.chat_input("Pergunte sobre a alimentação do bebê")
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        llm = OpenRouterLLM(model=DEFAULT_MODEL)
    except MissingAPIKey:
        st.error("Defina OPENROUTER_API_KEY no ambiente. A chave não é gravada aqui.")
        return
    with st.spinner("Consultando os guias..."):
        answer = ask(index, question, llm)
    content = format_answer_markdown(answer)
    passages = passages_markdown(answer.hits)
    st.session_state.messages.append(
        {"role": "assistant", "content": content, "passages": passages}
    )
    st.rerun()


if __name__ == "__main__":
    main()
