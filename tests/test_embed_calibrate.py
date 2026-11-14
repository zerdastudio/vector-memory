import asyncio
import os
from types import SimpleNamespace as NS

import pytest

from vector_memory import CohereReranker, EmbedderChain, EmbeddingError, HashingEmbedder, OpenAIEmbedder, calibrate
from vector_memory.calibrate import load_pairs, main as calibrate_main

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(HERE, "..", "examples", "pairs.toml")


def run(coro):
    return asyncio.run(coro)


class Table:
    def __init__(self, table, model_id="table@2"):
        self.table, self.model_id = table, model_id

    async def embed(self, texts, kind):
        return [self.table[t] for t in texts]


def test_hashing_embedder_is_deterministic_and_unit_length():
    e = HashingEmbedder(32)
    [a], [b] = run(e.embed(["Deploys on Fridays"], "document")), run(e.embed(["deploys on friday"], "query"))
    assert a == b                                   # case, plural and stop words do not matter
    assert sum(x * x for x in a) == pytest.approx(1.0)
    assert e.model_id == "hashing-bow@32"


def test_chain_falls_back_and_stamps_who_answered():
    class Down:
        model_id = "down@2"

        async def embed(self, texts, kind):
            raise TimeoutError("slow")

    chain = EmbedderChain([Down(), Table({"x": [1.0, 0.0]})])
    vectors, model_id = run(chain.embed(["x"]))
    assert (vectors, model_id, chain.primary) == ([[1.0, 0.0]], "table@2", "down@2")


def test_chain_rejects_half_answers():
    class Short:
        model_id = "short@2"

        async def embed(self, texts, kind):
            return [[1.0, 0.0]]

    class Uneven:
        model_id = "uneven@2"

        async def embed(self, texts, kind):
            return [[1.0, 0.0], [1.0]]

    with pytest.raises(EmbeddingError) as info:
        run(EmbedderChain([Short(), Uneven()]).embed(["a", "b"]))
    assert "short@2" in str(info.value) and "uneven@2" in str(info.value)
    with pytest.raises(ValueError):
        EmbedderChain([])


def test_openai_embedder_with_a_stand_in_client():
    seen = {}

    async def create(**kw):
        seen.update(kw)
        return NS(data=[NS(index=1, embedding=[0.0, 1.0]), NS(index=0, embedding=[1.0, 0.0])])

    e = OpenAIEmbedder("text-embedding-3-small", dimensions=2, client=NS(embeddings=NS(create=create)))
    assert e.model_id == "text-embedding-3-small@2"
    assert run(e.embed(["a", "b"], "document")) == [[1.0, 0.0], [0.0, 1.0]]   # put back in input order
    assert seen == {"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": 2}
    plain = OpenAIEmbedder("m", client=NS(embeddings=NS(create=create)))
    seen.clear()
    run(plain.embed(["a", "b"], "query"))
    assert "dimensions" not in seen and plain.model_id == "m@native"


@pytest.mark.parametrize("indexes", [[None, None, None], [0, None, 2], [0, 0, 1], [1, 1, 0]])
def test_an_endpoint_without_usable_indexes_keeps_the_reply_order(indexes):
    # Gemini's OpenAI-compatible endpoint answers a batch with `index` left empty (found by tests/test_live.py)
    async def create(**kw):
        return NS(data=[NS(index=i, embedding=[float(n), 1.0]) for n, i in enumerate(indexes)])

    e = OpenAIEmbedder("m", client=NS(embeddings=NS(create=create)))
    assert run(e.embed(["a", "b", "c"], "document")) == [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]


def test_cohere_reranker_with_a_stand_in_client():
    async def rerank(**kw):
        assert kw["top_n"] == 3
        return NS(results=[NS(index=2, relevance_score=0.9), NS(index=0, relevance_score=0.1)])

    r = CohereReranker(client=NS(rerank=rerank))
    assert run(r.rerank("q", ["a", "b", "c"])) == [0.1, 0.0, 0.9]
    assert run(r.rerank("q", [])) == []


SEPARATES = {
    "dup a": [1.0, 0.0], "dup b": [0.999, 0.045],          # ~0.001
    "para a": [1.0, 0.0], "para b": [0.8, 0.6],            # 0.2
    "far a": [1.0, 0.0], "far b": [0.0, 1.0],              # 1.0
}


def test_calibration_that_separates():
    result = run(calibrate(Table(SEPARATES), duplicates=[("dup a", "dup b")], paraphrases=[("para a", "para b")],
                           unrelated=[("far a", "far b")], update_distance=0.1, merge_distance=0.5))
    assert result.discriminates and result.problems == []
    assert result.suggested_merge == pytest.approx(0.6)
    assert result.suggested_update == pytest.approx((0.001 + 0.2) / 2, abs=0.002)
    assert "the model separates them" in result.report()


def test_calibration_that_does_not_separate_and_bad_thresholds():
    table = {"p a": [1.0, 0.0], "p b": [0.0, 1.0], "u a": [1.0, 0.0], "u b": [0.8, 0.6]}
    result = run(calibrate(Table(table), paraphrases=[("p a", "p b")], unrelated=[("u a", "u b")],
                           merge_distance=0.3))
    assert not result.discriminates and result.suggested_merge is None
    assert any("no merge threshold" in p for p in result.problems)
    assert any("merge_distance 0.3" in p for p in result.problems)
    assert "do not trust" in result.report()


def test_calibration_refuses_to_measure_a_fallback():
    class Down:
        model_id = "down@2"

        async def embed(self, texts, kind):
            raise TimeoutError

    with pytest.raises(RuntimeError, match="wrong model"):
        run(calibrate(EmbedderChain([Down(), Table(SEPARATES)]), paraphrases=[("para a", "para b")],
                      unrelated=[("far a", "far b")]))
    with pytest.raises(ValueError):
        run(calibrate(Table(SEPARATES), paraphrases=[], unrelated=[("far a", "far b")]))


def test_example_pairs_load_and_the_toy_embedder_fails_them():
    pairs = load_pairs(PAIRS)
    assert all(len(pairs[g]) >= 3 for g in ("duplicates", "paraphrases", "unrelated"))
    result = run(calibrate(HashingEmbedder(), **pairs))
    assert not result.discriminates   # a bag of words cannot see that "kilometres" means "metric"


def test_cli_without_a_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert calibrate_main([PAIRS]) == 2
    assert "OPENAI_API_KEY is empty" in capsys.readouterr().err


def test_calibration_looks_at_the_farthest_paraphrase():
    # one paraphrase is close (0.2), the other far (0.5); an unrelated pair sits at 0.4, between them
    table = {"p1 a": [1.0, 0.0], "p1 b": [0.8, 0.6], "p2 a": [0.0, 1.0], "p2 b": [0.866, 0.5],
             "u a": [1.0, 0.0], "u b": [0.6, 0.8]}
    result = run(calibrate(Table(table), paraphrases=[("p1 a", "p1 b"), ("p2 a", "p2 b")],
                           unrelated=[("u a", "u b")]))
    assert not result.discriminates and result.suggested_merge is None


@pytest.mark.parametrize("bad", [[float("nan"), 1.0], [0.0, 0.0], [], ["x", 1.0], [float("inf"), 0.0]])
def test_chain_falls_back_on_a_broken_vector(bad):
    class Bad:
        model_id = "bad@2"

        async def embed(self, texts, kind):
            return [bad for _ in texts]

    vectors, model_id = run(EmbedderChain([Bad(), HashingEmbedder(8)]).embed(["hello"]))
    assert model_id == "hashing-bow@8" and len(vectors[0]) == 8


def test_chain_accepts_numpy_rows():
    import numpy as np

    class Numpy:
        model_id = "np@2"

        async def embed(self, texts, kind):
            return np.array([[0.6, 0.8] for _ in texts], dtype=np.float32)

    vectors, model_id = run(EmbedderChain([Numpy()]).embed(["a", "b"]))
    assert model_id == "np@2" and vectors[1] == pytest.approx([0.6, 0.8])
