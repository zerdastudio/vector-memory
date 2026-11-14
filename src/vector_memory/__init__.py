"""Long-term memory for LLM agents: stamped embeddings, add/update/merge dedup, rerank, threshold calibration."""
from .calibrate import Calibration, calibrate
from .embedders import (DOCUMENT, QUERY, Embedder, EmbedderChain, EmbeddingError, HashingEmbedder,
                        OpenAIEmbedder)
from .rerank import CohereReranker, OverlapReranker
from .store import ADD, MERGE, UPDATE, Decision, Hit, MemoryStore, Reranker, StoreClosed

__version__ = "2.0.1"

__all__ = [
    "Calibration", "calibrate", "DOCUMENT", "QUERY", "Embedder", "EmbedderChain", "EmbeddingError",
    "HashingEmbedder", "OpenAIEmbedder", "CohereReranker", "OverlapReranker", "ADD", "MERGE", "UPDATE",
    "Decision", "Hit", "MemoryStore", "Reranker", "StoreClosed",
]
