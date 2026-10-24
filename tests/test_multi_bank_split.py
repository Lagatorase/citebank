from citebank.embeddings import HashEmbedder
from citebank.retrieval import Hit, Retriever

QUESTION = "Which of Demo Bank and Acme reported the higher net profit in 2025?"


class PageReranker:
    # stands in for the cross-encoder: ranks later pages first, so I can tell the rerank really ran
    name = "page"

    def rerank(self, query, hits):
        ranked = sorted(hits, key=lambda h: -h.chunk.page)
        return [Hit(h.chunk, float(h.chunk.page), h.source + "+rerank") for h in ranked]


def test_split_brings_in_the_bank_one_search_misses(two_bank_store, settings):
    # this is the problem the split fixes: without it, Acme never shows up
    off = Retriever(two_bank_store, HashEmbedder(), settings.with_overrides(multi_bank_split=False))
    assert {h.chunk.bank for h in off.search(QUESTION, k=3, mode="bm25")} == {"DEMO"}
    # the same search with the split on brings Acme in
    on = Retriever(two_bank_store, HashEmbedder(), settings)
    assert {h.chunk.bank for h in on.search(QUESTION, k=3, mode="bm25")} == {"DEMO", "ACME"}


def test_split_gives_every_bank_at_least_two(two_bank_store, settings):
    retriever = Retriever(two_bank_store, HashEmbedder(), settings)
    hits = retriever.search(QUESTION, k=3, mode="hybrid")
    banks = [h.chunk.bank for h in hits]
    # 2 banks with k=3 means 2 each, taken in turn
    assert banks == ["ACME", "DEMO", "ACME", "DEMO"]
    assert all(h.chunk.year == 2025 for h in hits)


def test_split_keeps_total_at_k_when_shares_allow(two_bank_store, settings):
    retriever = Retriever(two_bank_store, HashEmbedder(), settings)
    hits = retriever.search(QUESTION, k=5, mode="bm25")
    banks = [h.chunk.bank for h in hits]
    assert len(hits) == 5 and banks.count("ACME") >= 2 and banks.count("DEMO") >= 2


def test_split_still_finds_the_right_page(two_bank_store, settings):
    retriever = Retriever(two_bank_store, HashEmbedder(), settings)
    hits = retriever.search(QUESTION, k=4, mode="hybrid")
    demo_pages = [h.chunk.page for h in hits if h.chunk.bank == "DEMO"]
    # net profit is in the chairman's statement (page 1) and the highlights table (page 2)
    assert set(demo_pages) & {1, 2}


def test_single_bank_question_is_not_split(two_bank_store, settings):
    retriever = Retriever(two_bank_store, HashEmbedder(), settings)
    hits = retriever.search("What was Demo Bank's net profit in 2025?", k=3, mode="hybrid")
    assert len(hits) == 3 and {h.chunk.bank for h in hits} == {"DEMO"}


def test_split_reranks_within_each_bank(two_bank_store, settings):
    retriever = Retriever(two_bank_store, HashEmbedder(), settings, reranker=PageReranker())
    hits = retriever.search(QUESTION, k=4, mode="hybrid")
    for bank in ("ACME", "DEMO"):
        pages = [h.chunk.page for h in hits if h.chunk.bank == bank]
        assert len(pages) == 2 and pages == sorted(pages, reverse=True)
    assert all(h.source.endswith("+rerank") for h in hits)
