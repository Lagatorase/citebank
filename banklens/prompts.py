"""Prompt text kept in one place, so changes show up clearly in git and can be re-tested with the eval suite."""

from __future__ import annotations

from .retrieval import Hit

# the exact phrase the model must use when the sources don't cover the question; the eval checks for it
REFUSAL = "I could not find this in the annual reports provided."

SYSTEM_PROMPT = f"""You answer questions about bank annual reports using only the numbered sources provided.

Rules:
1. Use only facts stated in the sources. Do not use outside knowledge, even if you think you know the answer.
2. Put a citation like [1] or [2] right after every sentence that states a fact, pointing to the source it came from.
3. Give figures exactly as written in the source, with their currency, unit and year.
4. If the sources do not contain the answer, reply with exactly: {REFUSAL}
5. If the sources only partly answer the question, answer the part they cover and say what is missing.
6. If two sources disagree, say so and cite both.
7. Keep the answer short: two to five sentences, or a short list for comparisons."""


def source_label(hit: Hit) -> str:
    # the label the model sees for each source, which also tells it the bank, year and page
    c = hit.chunk
    section = f", section: {c.section}" if c.section else ""
    return f"{c.bank} annual report {c.year or ''}, page {c.page}{section}".replace("  ", " ")


def chunk_body(hit: Hit) -> str:
    # chunk text starts with a header line that I already show in the label, so I strip it here
    text = hit.chunk.text
    return text.split("\n", 1)[1] if text.startswith("[") and "\n" in text else text


# small local models tend to forget the citation rule after a long list of sources, so I repeat it last
# (I leave the refusal phrase out here, because repeating it made the 3b model refuse questions it could answer)
CITATION_REMINDER = "Remember: end every sentence that states a fact with a citation like [1], using the source numbers above."


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f"[{n}] {source_label(h)}\n{chunk_body(h)}" for n, h in enumerate(hits, start=1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n{CITATION_REMINDER}\nAnswer:"
