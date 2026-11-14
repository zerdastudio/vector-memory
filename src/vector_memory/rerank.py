"""Rerankers: a second, slower look at the closest candidates.

Vector search ranks by direction in the embedding space; a reranker reads the query and each candidate together,
and usually orders them better. Any object with `async rerank(query, documents) -> list[float]` works.
"""
from __future__ import annotations

from typing import Any


class CohereReranker:
    """Cohere's rerank API (`pip install cohere`)."""

    def __init__(self, model: str = "rerank-v3.5", *, api_key: str | None = None, client: Any = None):
        self.model = model
        if client is None:
            import cohere
            client = cohere.AsyncClientV2(api_key=api_key)
        self._client = client

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        reply = await self._client.rerank(model=self.model, query=query, documents=documents,
                                          top_n=len(documents))
        scores = [0.0] * len(documents)
        for r in reply.results:
            scores[r.index] = float(r.relevance_score)
        return scores


class OverlapReranker:
    """Scores by the share of query words found in the document. Offline; for tests and demos."""

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        from .embedders import _tokens
        wanted = set(_tokens(query))
        if not wanted:
            return [0.0] * len(documents)
        return [len(wanted & set(_tokens(d))) / len(wanted) for d in documents]
