"""API 키 없이도 실행되는 정책 검색 및 관계 참조 무결성 테스트."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from build_chunks import make_chunks
from rag import (
    EMBED_DIM,
    answer_with_openai,
    build_prompt_context,
    embed_corpus,
    graph_document_edges,
    load_bundle,
    retrieve,
)


def test_chunks_preserve_korean_law_sections():
    text = "1. 정책목표\n- 법률 제3조 (목적)\n· 저리 대출 지원\n" + "주택 공급 확대 " * 180
    parts = make_chunks(text)
    assert parts
    assert all(len(p) <= 1300 for p in parts)
    assert any("법률 제3조" in part for part in parts)


def test_bundle_document_and_relation_integrity():
    bundle = load_bundle()
    assert len(bundle.docs) == 262
    assert len(bundle.chunks) >= 262
    assert len(bundle.entities) == 5691
    assert len(bundle.relations) == 4381
    assert len(bundle.graph) > 100000
    assert set(bundle.chunks.doc_id) <= set(bundle.docs.ID)
    assert set(bundle.relations.source_id) <= set(bundle.docs.ID)


def test_retrieval_and_provenance():
    bundle = load_bundle()
    found = retrieve(bundle, "2023년 전세사기 피해자 지원 보완방안의 세부정책", top_k=6)
    assert any(item["doc_id"] == "75748" for item in found["chunks"])
    assert found["relations"]
    assert all(item["doc_id"] in bundle.doc_by_id for item in found["relations"])
    assert "[문서 75748]" in build_prompt_context(found)


def test_rdf_path_and_labels():
    bundle = load_bundle()
    edges = graph_document_edges(bundle, "75748")
    assert edges
    assert any(e["predicate"] == "describesPolicy" for e in edges)
    assert any(e["predicate"] == "hasMeasure" for e in edges)


def test_optional_semantic_search():
    bundle = load_bundle()
    matrix = np.zeros((len(bundle.chunks), 3), dtype=np.float32)
    matrix[:, 0] = 1
    question_vector = np.array([1, 0, 0], dtype=np.float32)
    found = retrieve(bundle, "주택공급 확대", embeddings=matrix, question_embedding=question_vector)
    assert "임베딩" in found["mode"]
    assert found["chunks"]


def test_mock_embedding_and_responses_calls():
    class MockClient:
        def __init__(self):
            self.embeddings = self
            self.responses = self
            self.kwargs = None
        def create(self, **kwargs):
            self.kwargs = kwargs
            if "dimensions" in kwargs:
                inputs = kwargs["input"]
                return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.0] * EMBED_DIM) for i in range(len(inputs))])
            return SimpleNamespace(output_text="근거 기반 답변 [문서 75748]")
    client = MockClient()
    bundle = load_bundle()
    matrix = embed_corpus(client, bundle.chunks.head(5))
    assert matrix.shape == (5, EMBED_DIM)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    found = retrieve(bundle, "2023년 전세사기 지원 보완")
    answer = answer_with_openai(client, "어떤 세부정책?", found)
    assert "75748" in answer
    assert client.kwargs["store"] is False
