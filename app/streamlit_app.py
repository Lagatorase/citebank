"""Demo UI. Talks to the FastAPI service if API_URL is set, otherwise runs the pipeline in-process.

  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import streamlit as st

# makes "import citebank" work when streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_URL = os.getenv("API_URL", "").rstrip("/")

st.set_page_config(page_title="CiteBank", page_icon=":bank:", layout="wide")
st.title("CiteBank")
st.caption("Ask questions about DBS, OCBC and UOB annual reports. Every answer cites the page it came from.")


@st.cache_resource(show_spinner="Loading models and index...")
def local_pipeline():
    # cached so the models load once per session, not on every question
    from citebank.pipeline import RAGPipeline

    return RAGPipeline()


def ask(question: str, k: int) -> dict:
    if API_URL:
        resp = httpx.post(f"{API_URL}/ask", json={"question": question, "top_k": k}, timeout=300)
        resp.raise_for_status()
        return resp.json()
    return local_pipeline().ask(question, k).to_dict()


with st.sidebar:
    st.header("Settings")
    k = st.slider("Sources to retrieve", 1, 10, 5)
    st.markdown("**Try asking**")
    examples = [
        "What was DBS's net profit for 2025?",
        "How did OCBC's net profit change from 2024 to 2025?",
        "What was UOB's non-performing loan ratio at the end of 2025?",
        "What does DBS say about its use of AI?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

question = st.text_input("Your question", key="question", placeholder="e.g. What was DBS's CET1 ratio in 2025?")

if question:
    with st.spinner("Searching the reports..."):
        try:
            result = ask(question, k)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.stop()

    if result["refused"]:
        st.warning(result["answer"])
    else:
        st.markdown(result["answer"])

    if result.get("invalid_citations"):
        st.error(f"The model cited sources that don't exist: {result['invalid_citations']}")

    st.caption(f"{result['model']} · retrieval {result['retrieval_s']:.2f}s · generation {result['generation_s']:.2f}s")

    st.subheader("Sources")
    cited = {c["number"] for c in result.get("citations", [])}
    for n, src in enumerate(result["sources"], start=1):
        # cited sources are marked so it's easy to see what the answer actually relied on
        mark = "cited" if n in cited else "not cited"
        label = f"[{n}] {src['bank']} {src['year']}, page {src['page']} · {src['section'] or src['kind']} · {mark}"
        with st.expander(label, expanded=n in cited):
            st.markdown(src["text"])
