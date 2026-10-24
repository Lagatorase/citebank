from citebank.chunking import Chunk
from citebank.embeddings import HashEmbedder
from citebank.llm import EchoLLM, LLMResult
from citebank.pipeline import RAGPipeline, is_empty_answer, is_refusal, parse_citations
from citebank.prompts import REFUSAL
from citebank.retrieval import Hit, Retriever, detect_filters, expand_acronyms, reciprocal_rank_fusion
from citebank.store import VectorStore, tokenize


def test_tokenize_keeps_figures():
    assert tokenize("Net profit of S$10,300.5 million in 2025") == ["net", "profit", "s", "10300.5", "million", "2025"]


def test_rrf_rewards_agreement():
    # item 2 is near the top of both lists, so it should beat items that only one list liked
    fused = reciprocal_rank_fusion([[1, 2, 3], [4, 2, 5]])
    assert fused[0][0] == 2


def test_expand_acronyms():
    assert "common equity tier 1" in expand_acronyms("What was the CET1 ratio?")
    assert expand_acronyms("total income") == "total income"


def test_detect_filters(store):
    banks, years = detect_filters("What did Demo Bank earn in 2024 and 2031?", store)
    assert banks == {"DEMO"} and years == {2024}


def test_year_filter_restricts_results(store, settings):
    retriever = Retriever(store, HashEmbedder(), settings)
    hits = retriever.search("How many people did Demo Bank employ in 2024?", k=3, mode="hybrid")
    assert hits and all(h.chunk.year == 2024 for h in hits)
    assert hits[0].chunk.page == 5


def test_filter_falls_back_when_nothing_matches(store, settings):
    retriever = Retriever(store, HashEmbedder(), settings)
    # 2031 isn't in the index, so the search should still return results from all years
    assert retriever.search("net profit 2031", k=3, mode="bm25")


def test_acronym_query_finds_table(store, settings):
    retriever = Retriever(store, HashEmbedder(), settings)
    hits = retriever.search("CET1 ratio 2025", k=1, mode="bm25")
    assert hits[0].chunk.kind == "table"


def test_store_roundtrip(store, tmp_path):
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert [c.id for c in loaded.chunks] == [c.id for c in store.chunks]
    assert loaded.vectors.shape == store.vectors.shape


def _hit(page):
    return Hit(Chunk(f"id{page}", f"[X | page {page}]\nbody {page}", "DBS", 2025, page, "S", "text", "dbs_2025.pdf"), 1.0, "test")


def test_parse_citations_flags_made_up_sources():
    hits = [_hit(3), _hit(9)]
    cites, invalid = parse_citations("Profit rose [1]. Margin fell [2][5].", hits)
    assert [c.page for c in cites] == [3, 9]
    assert invalid == [5]


def test_refusal_detection():
    assert is_refusal(REFUSAL)
    assert is_refusal("Sorry. " + REFUSAL.lower())
    assert is_refusal("I could not find this in the provided annual reports.")
    assert not is_refusal("Net profit was SGD 4.57 billion [1].")


def test_pipeline_end_to_end(settings, store):
    rag = RAGPipeline(settings, embedder=HashEmbedder(), llm=EchoLLM(), store=store)
    ans = rag.ask("How many employees did Demo Bank have in 2025?")
    assert not ans.refused
    assert ans.citations and ans.citations[0].page == 5
    assert "19,050" in ans.answer


def test_pipeline_refuses_without_sources(settings):
    empty = VectorStore([], HashEmbedder().embed_documents([]), "hash")
    rag = RAGPipeline(settings, embedder=HashEmbedder(), llm=EchoLLM(), store=empty)
    ans = rag.ask("anything at all")
    assert ans.refused and ans.answer == REFUSAL


class ScriptedLLM:
    # stands in for a model that gives the replies I script, in order, and repeats the last one after that
    name = "scripted"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, system, user, temperature=0.0, max_tokens=700):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return LLMResult(reply, self.name, 0.1)


def test_is_empty_answer():
    assert is_empty_answer("") and is_empty_answer("[1]") and is_empty_answer(" [1][2]. ")
    assert not is_empty_answer("Net profit was SGD 4.57 billion [1].")


def test_empty_answer_is_retried_once(settings, store):
    llm = ScriptedLLM("[1]", "Net profit was SGD 4.57 billion [1].")
    rag = RAGPipeline(settings, embedder=HashEmbedder(), llm=llm, store=store)
    ans = rag.ask("What was Demo Bank's net profit in 2025?")
    assert llm.calls == 2 and ans.retried and not ans.refused
    assert ans.answer == "Net profit was SGD 4.57 billion [1]." and ans.citations
    # both calls count towards the generation time
    assert abs(ans.generation_s - 0.2) < 1e-9


def test_answer_that_stays_empty_becomes_a_refusal(settings, store):
    llm = ScriptedLLM("[1]")
    rag = RAGPipeline(settings, embedder=HashEmbedder(), llm=llm, store=store)
    ans = rag.ask("What was Demo Bank's net profit in 2025?")
    assert llm.calls == 2 and ans.retried
    assert ans.answer == REFUSAL and ans.refused and not ans.citations


def test_normal_answer_is_not_retried(settings, store):
    llm = ScriptedLLM("Net profit was SGD 4.57 billion [1].")
    ans = RAGPipeline(settings, embedder=HashEmbedder(), llm=llm, store=store).ask("What was Demo Bank's net profit in 2025?")
    assert llm.calls == 1 and not ans.retried
