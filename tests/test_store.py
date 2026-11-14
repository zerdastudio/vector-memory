import asyncio
import logging
import sqlite3

import pytest

from vector_memory import (ADD, MERGE, UPDATE, EmbedderChain, EmbeddingError, HashingEmbedder, MemoryStore,
                           OverlapReranker)


def run(coro):
    return asyncio.run(coro)


class Fixed:
    """An embedder that returns the vectors it is told to, by text."""

    def __init__(self, table: dict, model_id: str = "fixed@3"):
        self.table, self.model_id, self.calls = table, model_id, []

    async def embed(self, texts, kind):
        self.calls.append((list(texts), kind))
        return [self.table[t] for t in texts]


class Broken:
    model_id = "broken@3"

    async def embed(self, texts, kind):
        raise ConnectionError("down")


# Unit vectors with known cosine distances from "base": 0.0, ~0.10, ~0.29, 1.0.
TABLE = {
    "base": [1.0, 0.0, 0.0],
    "base again": [1.0, 0.0, 0.0],
    "near": [0.9, 0.436, 0.0],        # distance ~0.10
    "similar": [0.71, 0.704, 0.0],    # distance ~0.29
    "other": [0.0, 0.0, 1.0],         # distance 1
    "merged": [0.95, 0.312, 0.0],
}


def store(tmp_path, embedder=None, **kw):
    kw.setdefault("update_distance", 0.15)
    kw.setdefault("merge_distance", 0.35)
    return MemoryStore(str(tmp_path / "m.db"), embedder or Fixed(TABLE), **kw)


async def merger(old, new):
    return "merged"


def test_add_update_merge_decisions(tmp_path):
    async def go():
        async with store(tmp_path, merger=merger) as s:
            first = await s.remember("base")
            assert (first.op, first.distance, first.neighbor_id) == (ADD, None, None)
            again = await s.remember("near")
            assert again.op == UPDATE and again.memory_id == first.memory_id and again.text == "base"
            assert again.distance == pytest.approx(0.10, abs=0.01)
            merged = await s.remember("similar")
            assert merged.op == MERGE and merged.memory_id == first.memory_id and merged.text == "merged"
            new = await s.remember("other")
            assert new.op == ADD and new.memory_id != first.memory_id and new.distance == pytest.approx(1.0)
            row = await s.get(first.memory_id)
            assert row["text"] == "merged" and row["confidence"] == pytest.approx(0.55)
            hist = await s.history(first.memory_id)
            assert [h["text"] for h in hist] == ["base"] and "similar" in hist[0]["reason"]
    run(go())


def test_without_merger_a_similar_text_is_added(tmp_path):
    async def go():
        async with store(tmp_path) as s:
            a = await s.remember("base")
            b = await s.remember("similar")
            assert b.op == ADD and b.neighbor_id == a.memory_id and b.memory_id != a.memory_id
    run(go())


def test_confidence_is_capped(tmp_path):
    async def go():
        async with store(tmp_path, reinforce=0.4) as s:
            a = await s.remember("base", confidence=0.5)
            for _ in range(3):
                await s.remember("base again")
            assert (await s.get(a.memory_id))["confidence"] == 1.0
    run(go())


def test_empty_text_and_empty_merge_are_errors(tmp_path):
    async def empty_merger(old, new):
        return "  "

    async def go():
        async with store(tmp_path, merger=empty_merger) as s:
            with pytest.raises(ValueError):
                await s.remember("   ")
            await s.remember("base")
            with pytest.raises(ValueError):
                await s.remember("similar")
            assert (await s.history(1)) == []   # nothing half-written
    run(go())


def test_thresholds_are_validated(tmp_path):
    for bad in [(0.3, 0.3), (0.4, 0.2), (-0.1, 0.2), (0.1, 2.5)]:
        with pytest.raises(ValueError):
            store(tmp_path, update_distance=bad[0], merge_distance=bad[1])


def test_recall_orders_by_distance_counts_hits_and_filters_confidence(tmp_path):
    table = dict(TABLE, q=[1.0, 0.0, 0.0])

    async def go():
        async with store(tmp_path, Fixed(table)) as s:
            await s.remember("other", confidence=0.9)
            await s.remember("similar", confidence=0.2)
            await s.remember("base", confidence=0.5)   # distance 0.29 from similar: added (no merger)
            hits = await s.recall("q", k=2)
            assert [h.text for h in hits] == ["base", "similar"]
            assert (await s.get(hits[0].id))["hits"] == 1
            assert [h.text for h in await s.recall("q", k=2, min_confidence=0.4)] == ["base", "other"]
            with pytest.raises(ValueError):
                await s.recall("q", k=0)
    run(go())


def test_recall_embeds_the_query_as_a_query(tmp_path):
    emb = Fixed(dict(TABLE, q=[1.0, 0.0, 0.0]))

    async def go():
        async with store(tmp_path, emb) as s:
            await s.remember("base")
            await s.recall("q")
    run(go())
    assert [kind for _, kind in emb.calls] == ["document", "query"]


def test_reranker_reorders_the_candidates(tmp_path):
    class Reverse:
        async def rerank(self, query, docs):
            return list(range(len(docs)))   # the farthest candidate scores highest

    table = dict(TABLE, q=[1.0, 0.0, 0.0])

    async def go():
        async with store(tmp_path, Fixed(table)) as s:
            for t in ("base", "similar", "other"):
                await s.remember(t)
            hits = await s.recall("q", k=1, reranker=Reverse(), candidates=3)
            assert [h.text for h in hits] == ["other"] and hits[0].score == 2

            class Wrong:
                async def rerank(self, query, docs):
                    return [1.0]

            with pytest.raises(ValueError):
                await s.recall("q", k=1, reranker=Wrong(), candidates=3)
    run(go())


def test_vectors_from_another_model_are_never_compared(tmp_path, caplog):
    primary = HashingEmbedder(64, name="primary")
    backup = HashingEmbedder(64, name="backup")   # same dimensions: only the stamp tells them apart

    async def go():
        async with MemoryStore(str(tmp_path / "m.db"), primary) as s:
            await s.remember("The user prefers metric units.")
        chain = EmbedderChain([Broken(), backup])
        async with MemoryStore(str(tmp_path / "m.db"), chain) as s:
            with caplog.at_level(logging.WARNING, logger="vector_memory"):
                d = await s.remember("The user prefers metric units.")
            assert d.op == ADD and d.distance is None and d.model_id == "backup@64"
            assert any("other spaces" in r.message for r in caplog.records)
            assert (await s.stats())["by_model"] == {"backup@64": 1, "primary@64": 1}
            hits = await s.recall("metric units")
            assert [h.model_id for h in hits] == ["backup@64"]
            assert await s.reembed(backup) == 1
            assert (await s.stats())["by_model"] == {"backup@64": 2}
            assert (await s.remember("The user prefers metric units.")).op == UPDATE
    run(go())


def test_every_embedder_failing_raises_and_writes_nothing(tmp_path):
    async def go():
        async with MemoryStore(str(tmp_path / "m.db"), EmbedderChain([Broken()])) as s:
            with pytest.raises(EmbeddingError):
                await s.remember("anything")
            assert (await s.stats())["active"] == 0
    run(go())


def test_forget_keeps_the_row_and_its_history(tmp_path):
    async def go():
        async with store(tmp_path) as s:
            a = await s.remember("base")
            await s.forget(a.memory_id, "user asked")
            assert (await s.get(a.memory_id))["status"] == "forgotten"
            assert (await s.history(a.memory_id))[0]["reason"] == "user asked"
            assert await s.recall("base") == []
            assert (await s.stats()) == {"active": 0, "forgotten": 1, "by_model": {}}
            assert (await s.remember("base")).op == ADD    # a forgotten memory is not a duplicate
            with pytest.raises(KeyError):
                await s.forget(999)
    run(go())


def test_concurrent_copies_of_one_fact_are_stored_once(tmp_path):
    async def go():
        async with MemoryStore(str(tmp_path / "m.db"), HashingEmbedder()) as s:
            decisions = await asyncio.gather(*(s.remember("Deploys are frozen on Fridays.") for _ in range(10)))
            assert sorted(d.op for d in decisions) == [ADD] + [UPDATE] * 9
            assert (await s.stats())["active"] == 1
    run(go())


def test_the_file_is_plain_sqlite_and_survives_reopening(tmp_path):
    async def go():
        async with store(tmp_path) as s:
            await s.remember("base")
        async with store(tmp_path) as s:
            assert (await s.remember("near")).op == UPDATE
    run(go())
    with sqlite3.connect(tmp_path / "m.db") as conn:
        assert conn.execute("SELECT model_id, dims FROM memories").fetchall() == [("fixed@3", 3)]


def test_store_reopens_after_close(tmp_path):
    async def go():
        s = store(tmp_path)
        await s.remember("base")
        await s.close()
        assert (await s.remember("near")).op == UPDATE
        await s.close()
    run(go())


def test_overlap_reranker():
    scores = run(OverlapReranker().rerank("metric units report", ["units in reports", "coffee", ""]))
    assert scores == [pytest.approx(2 / 3), 0.0, 0.0]
    assert run(OverlapReranker().rerank("the a", ["x"])) == [0.0]


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), "0.1", True])
def test_reinforce_is_validated(tmp_path, bad):
    with pytest.raises(ValueError, match="reinforce"):
        store(tmp_path, reinforce=bad)


def test_a_merger_that_returns_no_text_is_a_clear_error(tmp_path):
    async def sloppy(old, new):
        return None

    s = store(tmp_path, merger=sloppy)
    run(s.remember("base"))
    with pytest.raises(ValueError, match="merger must return"):
        run(s.remember("similar"))
    assert run(s.stats())["active"] == 1


def test_a_reranker_with_bad_scores_is_refused(tmp_path):
    class NaNs:
        async def rerank(self, query, documents):
            return [float("nan")] * len(documents)

    s = store(tmp_path)
    run(s.remember("base"))
    run(s.remember("other"))
    with pytest.raises(ValueError, match="not finite"):
        run(s.recall("base", 1, reranker=NaNs()))


def test_cancelled_remember_leaves_the_store_usable(tmp_path):
    class Slow(Fixed):
        async def embed(self, texts, kind):
            await asyncio.sleep(0.05)
            return await super().embed(texts, kind)

    async def go():
        s = store(tmp_path, Slow(TABLE))
        task = asyncio.create_task(s.remember("base"))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await s.remember("other")).op == ADD
        assert (await s.stats())["active"] == 1
        await s.close()
    run(go())
