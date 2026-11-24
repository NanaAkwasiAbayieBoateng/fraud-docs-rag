from rag_app.index import Chunk, SearchResult
from rag_app.rerank import LexicalReranker


def _result(text, score):
    return SearchResult(
        chunk=Chunk(text=text, source="doc.md", chunk_index=0),
        score=score, vector_score=score, bm25_score=score,
    )


def test_reranker_promotes_full_coverage_chunk():
    query = "money mule recruitment red flags"
    weak = _result("Money is moved onward by criminals.", score=0.95)
    strong = _result(
        "Recruitment red flags: money mule recruitment uses job offers.", score=0.40
    )
    out = LexicalReranker().rerank(query, [weak, strong], k=2)
    assert out[0].chunk is strong.chunk


def test_reranker_respects_k_and_phrase_bonus():
    query = "confirmation of payee"
    phrase = _result("Confirmation of Payee checks the account name.", score=0.1)
    other = _result("Payee details for online payments and transfers.", score=0.9)
    out = LexicalReranker().rerank(query, [other, phrase], k=1)
    assert len(out) == 1
    assert out[0].chunk is phrase.chunk
