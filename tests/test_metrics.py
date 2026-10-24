import json

from citebank.chunking import Chunk
from eval.metrics import load_items, number_match, retrieval_scores


def _chunk(bank, year, page):
    return Chunk("x", "t", bank, year, page, "", "text", "f.pdf")


def test_number_match_normalises_formats():
    assert number_match("Net profit was S$10.30 billion [1].", "SGD 10.3 billion") == 1.0
    assert number_match("It employed 18400 people.", "18,400 employees") == 1.0
    assert number_match("It rose from 2.05% to 2.20%.", "2.05% in 2024 to 2.11% in 2025") == 0.5


def test_number_match_ignores_years_and_handles_no_numbers():
    assert number_match("Profit was 4.57 billion", "SGD 4.57 billion in 2025") == 1.0
    assert number_match("anything", "Through stress testing.") is None


def test_retrieval_scores():
    evidence = [{"bank": "DBS", "year": 2025, "page": 10}, {"bank": "DBS", "year": 2025, "page": 40}]
    chunks = [_chunk("DBS", 2024, 10), _chunk("DBS", 2025, 10), _chunk("OCBC", 2025, 40)]
    s = retrieval_scores(chunks, evidence)
    assert s == {"hit": 1.0, "mrr": 0.5, "recall": 0.5}
    # with a tolerance of one page, a neighbouring page also counts
    assert retrieval_scores([_chunk("DBS", 2025, 41)], evidence, page_tolerance=1)["hit"] == 1.0


def test_load_items_skips_comments_and_unverified(tmp_path):
    path = tmp_path / "q.jsonl"
    rows = [{"id": "a", "question": "q1", "type": "numeric", "verified": True},
            {"id": "b", "question": "q2", "type": "numeric", "verified": False}]
    path.write_text("// comment\n\n" + "\n".join(json.dumps(r) for r in rows))
    assert [i.id for i in load_items(path)] == ["a"]
    assert len(load_items(path, include_unverified=True)) == 2


def test_shipped_question_files_parse():
    from citebank.config import ROOT

    assert len(load_items(ROOT / "eval" / "demo_questions.jsonl")) == 9
    real = load_items(ROOT / "eval" / "questions.jsonl", include_unverified=True)
    assert len(real) == 21 and all(i.type in {"numeric", "factual", "comparison", "unanswerable"} for i in real)
