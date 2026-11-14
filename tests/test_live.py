"""Live checks of the real clients, against the providers' APIs. They cost a fraction of a cent.

Skipped unless LIVE_TESTS=1, so CI and a plain `pytest` never touch the network. To run them:

    pip install -e ".[dev,openai,cohere]"
    LIVE_TESTS=1 OPENAI_API_KEY=... LIVE_EMBED_MODEL=<an embedding model> \
    COHERE_API_KEY=... LIVE_RERANK_MODEL=<a rerank model> pytest -v tests/test_live.py

LIVE_EMBED_BASE_URL points the embedder at another OpenAI-compatible provider (with that provider's key and
model). A provider whose key or model is missing is skipped, and the reason names the missing variable.
"""
import asyncio
import os

import numpy as np
import pytest

from vector_memory import CohereReranker, EmbedderChain, EmbeddingError, MemoryStore, OpenAIEmbedder

pytestmark = pytest.mark.skipif(os.environ.get("LIVE_TESTS") != "1", reason="live tests: set LIVE_TESTS=1")

FACT = "The client pays its invoices within thirty days."
PARAPHRASE = "Invoices are settled by the customer in under a month."
UNRELATED = "The office plants are watered every Tuesday."


def need(*names):
    missing = [n for n in names if not os.environ.get(n, "").strip()]
    if missing:
        pytest.skip("missing " + ", ".join(missing))
    return [os.environ[n] for n in names]


def embedder(key, model, dimensions=None):
    return OpenAIEmbedder(model, api_key=key, base_url=os.environ.get("LIVE_EMBED_BASE_URL") or None,
                          dimensions=dimensions)


def distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_a_real_model_puts_a_paraphrase_closer_than_an_unrelated_fact():
    key, model = need("OPENAI_API_KEY", "LIVE_EMBED_MODEL")
    vectors = asyncio.run(embedder(key, model).embed([FACT, PARAPHRASE, UNRELATED], "document"))
    assert len(vectors) == 3 and len({len(v) for v in vectors}) == 1
    assert distance(vectors[0], vectors[1]) < distance(vectors[0], vectors[2])


def test_shortened_dimensions_come_back_with_their_own_stamp():
    key, model = need("OPENAI_API_KEY", "LIVE_EMBED_MODEL")
    e = embedder(key, model, dimensions=256)
    vectors, stamp, _ = asyncio.run(EmbedderChain([e]).embed_from([FACT], "document"))
    assert len(vectors[0]) == 256 and stamp == f"{model}@256"


def test_a_store_on_a_real_model_recalls_the_right_memory(tmp_path):
    key, model = need("OPENAI_API_KEY", "LIVE_EMBED_MODEL")

    async def go():
        async with MemoryStore(str(tmp_path / "live.db"), embedder(key, model)) as store:
            for text in (FACT, UNRELATED, "Deploys are frozen on Fridays after 6 pm."):
                await store.remember(text)
            hits = await store.recall("When does the client pay?", k=1)
            assert hits[0].text == FACT and hits[0].model_id == f"{model}@native"
    asyncio.run(go())


def test_a_real_reranker_scores_the_relevant_document_highest():
    key, model = need("COHERE_API_KEY", "LIVE_RERANK_MODEL")
    scores = asyncio.run(CohereReranker(model, api_key=key).rerank(
        "When does the client pay?", [UNRELATED, FACT, "Deploys are frozen on Fridays after 6 pm."]))
    assert len(scores) == 3 and max(range(3), key=lambda i: scores[i]) == 1


def test_a_wrong_key_makes_the_chain_fail_with_an_embedding_error():
    (model,) = need("LIVE_EMBED_MODEL")
    with pytest.raises(EmbeddingError) as e:
        asyncio.run(EmbedderChain([embedder("not-a-real-key", model)]).embed([FACT]))
    # refused for the key (401 at OpenAI, 400 "API key not valid" at Gemini), not unreachable
    assert any(sign in str(e.value) for sign in ("AuthenticationError", "401", "API key"))
