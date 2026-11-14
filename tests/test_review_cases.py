"""The cases of the first external review: sizes under one stamp, closing, the command line, and the edges that
the first tests did not pin down."""
import asyncio
import logging
import sqlite3
import threading

import numpy as np
import pytest

from test_store import TABLE, Broken, Fixed, merger, run, store

from vector_memory import (ADD, MERGE, EmbedderChain, EmbeddingError, HashingEmbedder, MemoryStore,
                           StoreClosed, calibrate)
from vector_memory.calibrate import main as calibrate_main


class Switch:
    """An embedder that can be turned off, and answers with vectors of `dims` numbers. Texts whose lengths differ
    modulo `dims` come out orthogonal."""

    def __init__(self, model_id, dims, up=True):
        self.model_id, self.dims, self.up = model_id, dims, up

    async def embed(self, texts, kind):
        if not self.up:
            raise ConnectionError("down")
        out = []
        for t in texts:
            v = [0.0] * self.dims
            v[len(t) % self.dims] = 1.0
            out.append(v)
        return out


# ---- one stamp, two sizes -------------------------------------------------------------------------------------

def test_a_fallback_with_the_same_stamp_and_another_size_is_refused_and_the_store_keeps_working(tmp_path, caplog):
    primary, backup = Switch("small@native", 4), Switch("small@native", 3)
    s = MemoryStore(str(tmp_path / "m.db"), EmbedderChain([primary, backup]))

    async def go():
        await s.remember("first fact")                  # 10 characters: slot 2 of 4
        primary.up = False
        with caplog.at_level(logging.WARNING, logger="vector_memory"):
            with pytest.raises(EmbeddingError, match="whose vectors here have 4"):
                await s.remember("second fact")
        primary.up = True
        assert (await s.remember("fact number 3")).op == ADD   # 13 characters: slot 1
        assert len(await s.recall("first fact", k=5)) == 2
        assert (await s.stats())["by_model"] == {"small@native": 2}
        await s.close()
    run(go())


def test_a_file_with_two_sizes_under_one_stamp_still_works_and_reembed_repairs_it(tmp_path, caplog):
    path = str(tmp_path / "m.db")

    async def seed():
        async with MemoryStore(path, Switch("small@native", 4)) as s:
            await s.remember("first fact")
            await s.remember("another one, quite different")   # 28 characters: slot 0
    run(seed())
    conn = sqlite3.connect(path)   # an older version of the library could leave this behind
    v3 = (np.array([1.0, 0.0, 0.0], dtype=np.float32)).tobytes()
    conn.execute("INSERT INTO memories (text, model_id, dims, vector, confidence, created, updated) "
                 "VALUES ('odd one', 'small@native', 3, ?, 0.5, 0, 0)", (v3,))
    conn.commit()
    conn.close()

    async def go():
        with caplog.at_level(logging.WARNING, logger="vector_memory"):
            async with MemoryStore(path, Switch("small@native", 4)) as s:
                assert any("sizes in this file" in r.getMessage() for r in caplog.records)
                assert len(await s.recall("first fact", k=5)) == 2          # the 4-number space
                assert (await s.remember("yet another fact!")).op == ADD   # 17 characters: slot 1
                assert (await s.stats())["by_model"] == {"small@native (3 numbers)": 1, "small@native (4 numbers)": 3}
                assert await s.reembed(Switch("small@native", 4)) == 1      # it gives 4: only the odd one is redone
                assert (await s.stats())["by_model"] == {"small@native": 4}
    run(go())


def test_reembed_redoes_own_memories_whose_size_changed(tmp_path):
    async def go():
        async with MemoryStore(str(tmp_path / "m.db"), EmbedderChain([Switch("a@1", 4)])) as s:
            await s.remember("one fact")
            s.chain = EmbedderChain([Switch("b@1", 3)])
            await s.remember("other fact")
            assert await s.reembed(Switch("a@1", 5)) == 2   # b's memory moves, then a's own one changes size
            assert (await s.stats())["by_model"] == {"a@1": 2}
    run(go())


def test_calibrate_refuses_a_fallback_even_under_the_primarys_stamp():
    table = {t: [1.0, 0.0] for t in ("a", "b", "c", "d")}

    class Same(Fixed):
        pass

    with pytest.raises(RuntimeError, match="fallback #1"):
        run(calibrate(EmbedderChain([Broken(), Same(table, model_id="broken@3")]),
                      paraphrases=[("a", "b")], unrelated=[("c", "d")]))


# ---- input checks ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [5.0, -1.0, float("nan"), float("inf"), True, "0.7"])
def test_confidence_is_validated(tmp_path, bad):
    s = store(tmp_path)
    with pytest.raises(ValueError, match="confidence"):
        run(s.remember("base", confidence=bad))


@pytest.mark.parametrize("bad", [0, -1, True, 2.5])
def test_reembed_batch_is_validated(tmp_path, bad):
    with pytest.raises(ValueError, match="batch"):
        run(store(tmp_path).reembed(HashingEmbedder(8), batch=bad))


@pytest.mark.parametrize("bad", [[1e39, 1.0], [1e-46, 0.0], [1e20, 1e20]])
def test_vectors_broken_in_float32_fall_back(bad):
    class Big:
        model_id = "big@2"

        async def embed(self, texts, kind):
            return [bad for _ in texts]

    vectors, model_id = run(EmbedderChain([Big(), HashingEmbedder(8)]).embed(["hello"]))
    assert model_id == "hashing-bow@8"


# ---- opening and closing --------------------------------------------------------------------------------------

def test_concurrent_opens_make_one_connection(tmp_path):
    s = store(tmp_path)
    made = []
    original = s._connect

    def counting():
        made.append(1)
        return original()

    s._connect = counting

    async def go():
        await asyncio.gather(*(s.open() for _ in range(5)))
        await s.close()
    run(go())
    assert len(made) == 1


def test_close_waits_for_a_merge_in_flight_and_refuses_what_waits(tmp_path):
    gate = asyncio.Event()

    async def slow_merger(old, new):
        await gate.wait()
        return "merged"

    async def go():
        s = store(tmp_path, merger=slow_merger)
        await s.remember("base")
        merging = asyncio.create_task(s.remember("similar"))
        await asyncio.sleep(0.01)                        # the merge holds the lock, waiting for its merger
        closing = asyncio.create_task(s.close())        # in line for the lock first
        await asyncio.sleep(0.01)
        waiting = asyncio.create_task(s.remember("other"))  # in line behind the close
        await asyncio.sleep(0.01)
        gate.set()
        assert (await merging).op == MERGE
        await closing
        with pytest.raises(StoreClosed):
            await waiting
        assert not [t for t in threading.enumerate() if t.name.startswith("vector-memory")]
        async with MemoryStore(s.path, Fixed(TABLE)) as again:   # what was written is there
            assert (await again.get(1))["text"] == "merged"
    run(go())


# ---- the edges of the decision --------------------------------------------------------------------------------

def _distance(tmp_path, a, b):
    async def go():
        async with store(tmp_path / "probe", update_distance=0.0, merge_distance=0.01) as s:
            await s.remember(a)
            return (await s.remember(b)).distance
    (tmp_path / "probe").mkdir(exist_ok=True)
    return run(go())


def test_a_distance_equal_to_update_distance_is_not_an_update(tmp_path):
    d = _distance(tmp_path, "base", "near")
    s = store(tmp_path, update_distance=d, merge_distance=0.5, merger=merger)
    run(s.remember("base"))
    assert run(s.remember("near")).op == MERGE


def test_a_distance_equal_to_merge_distance_is_an_add(tmp_path):
    d = _distance(tmp_path, "base", "similar")
    s = store(tmp_path, update_distance=0.05, merge_distance=d, merger=merger)
    run(s.remember("base"))
    assert run(s.remember("similar")).op == ADD


def test_a_merge_stores_the_vector_of_the_merged_text(tmp_path):
    async def go():
        async with store(tmp_path, merger=merger) as s:
            await s.remember("base")
            decision = await s.remember("similar")
            assert decision.op == MERGE
            [hit] = await s.recall("merged", k=1)
            assert hit.id == decision.memory_id and hit.distance == pytest.approx(0.0, abs=1e-6)
    run(go())


def test_history_comes_back_oldest_first(tmp_path):
    texts = iter(["merged", "merged again"])

    async def two_merges(old, new):
        return next(texts)

    table = dict(TABLE, **{"merged again": [0.95, 0.312, 0.0]})

    async def go():
        async with store(tmp_path, Fixed(table), update_distance=0.01, merge_distance=0.5, merger=two_merges) as s:
            await s.remember("base")
            await s.remember("similar")
            await s.remember("similar")
            assert [h["text"] for h in await s.history(1)] == ["base", "merged"]
    run(go())


def test_min_confidence_keeps_a_memory_exactly_at_the_floor(tmp_path):
    async def go():
        async with store(tmp_path) as s:
            await s.remember("base", confidence=0.5)
            assert len(await s.recall("base", k=1, min_confidence=0.5)) == 1
            assert await s.recall("base", k=1, min_confidence=0.51) == []
    run(go())


def test_the_reranker_sees_three_times_k_by_default(tmp_path):
    class Likes:
        async def rerank(self, query, docs):
            return [1.0 if d == "similar" else 0.0 for d in docs]

    async def go():
        async with store(tmp_path, update_distance=0.01, merge_distance=0.02) as s:
            for t in ("base", "near", "similar", "other"):
                await s.remember(t)
            [hit] = await s.recall("base", k=1, reranker=Likes())   # pool: base, near, similar
            assert hit.text == "similar"
    run(go())


def test_reembed_leaves_forgotten_memories_alone(tmp_path):
    async def go():
        async with store(tmp_path, update_distance=0.01, merge_distance=0.02) as s:
            await s.remember("base")
            await s.remember("other")
            await s.forget(2)
            assert await s.reembed(HashingEmbedder(16)) == 1
            assert (await s.get(2))["model_id"] == "fixed@3"
    run(go())


# ---- the warnings the README promises -------------------------------------------------------------------------

def test_the_chain_warns_when_a_fallback_answers(caplog):
    with caplog.at_level(logging.WARNING, logger="vector_memory"):
        run(EmbedderChain([Broken(), HashingEmbedder(8)]).embed(["hello"]))
    assert any("embedded with fallback hashing-bow@8" in r.getMessage() for r in caplog.records)


def test_recall_warns_about_memories_in_other_spaces(tmp_path, caplog):
    async def go():
        async with store(tmp_path) as s:
            await s.remember("base")
            s.chain = EmbedderChain([HashingEmbedder(8)])
            with caplog.at_level(logging.WARNING, logger="vector_memory"):
                await s.recall("hello")
    run(go())
    assert any("recall searched only memories made by hashing-bow@8" in r.getMessage() for r in caplog.records)


# ---- calibrate: thresholds and exit codes ---------------------------------------------------------------------

PAIRS_TABLE = {"p a": [1.0, 0.0], "p b": [0.8, 0.6], "u a": [1.0, 0.0], "u b": [0.0, 1.0],
               "d a": [1.0, 0.0], "d b": [0.99, 0.141]}
PAIRS = dict(paraphrases=[("p a", "p b")], unrelated=[("u a", "u b")])


def test_calibrate_checks_the_thresholds_it_is_given():
    fixed = Fixed(PAIRS_TABLE, model_id="t@2")
    ok = run(calibrate(fixed, **PAIRS))
    unrel = ok.worst("unrelated", min)
    assert run(calibrate(fixed, **PAIRS, merge_distance=unrel)).problems == []       # the edge is allowed
    bad_update = run(calibrate(fixed, **PAIRS, duplicates=[("d a", "d b")], update_distance=0.5))
    assert any("update_distance 0.5 is outside" in p for p in bad_update.problems)
    crossed = run(calibrate(fixed, **PAIRS, update_distance=0.5, merge_distance=0.4))
    assert any("0 <= update < merge" in p for p in crossed.problems)


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Runs the command line with a stand-in OpenAIEmbedder that answers from a table, or fails."""
    state = {"table": PAIRS_TABLE, "fail": None}

    class StandIn:
        def __init__(self, model, **kw):
            if state["fail"] == "sdk":
                raise ImportError("No module named 'openai'")
            self.model_id = f"{model}@native"

        async def embed(self, texts, kind):
            if state["fail"] == "network":
                raise ConnectionError("unreachable")
            return [state["table"][t] for t in texts]

    import vector_memory.embedders as embedders
    monkeypatch.setattr(embedders, "OpenAIEmbedder", StandIn)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    good = tmp_path / "pairs.toml"
    good.write_text('paraphrases = [["p a", "p b"]]\nunrelated = [["u a", "u b"]]\n')
    state["good"] = str(good)
    return state


def test_cli_exit_codes(cli, tmp_path, capsys):
    main = calibrate_main
    assert main([cli["good"]]) == 0
    assert main([cli["good"], "--merge-distance", "0.1"]) == 1          # measured: the threshold does not fit
    cli["table"] = dict(PAIRS_TABLE, **{"u b": [0.8, 0.6]})
    assert main([cli["good"]]) == 1                                     # measured: no separation
    cli["table"] = PAIRS_TABLE
    assert main([str(tmp_path / "missing.toml")]) == 2
    only = tmp_path / "only.toml"
    only.write_text('unrelated = [["u a", "u b"]]\n')
    assert main([str(only)]) == 2
    broken = tmp_path / "broken.toml"
    broken.write_text("paraphrases = [[")
    assert main([str(broken)]) == 2
    cli["fail"] = "sdk"
    assert main([cli["good"]]) == 2
    cli["fail"] = "network"
    assert main([cli["good"]]) == 3
    assert "could not measure" in capsys.readouterr().err


def test_a_refused_size_lets_the_next_embedder_answer(tmp_path):
    primary, backup = Switch("small@native", 4), Switch("small@native", 3)

    async def go():
        async with MemoryStore(str(tmp_path / "m.db"), EmbedderChain([primary, backup, HashingEmbedder(8)])) as s:
            await s.remember("first fact")
            primary.up = False
            assert (await s.remember("second fact")).model_id == "hashing-bow@8"
    run(go())


def test_two_sizes_racing_under_one_new_stamp_cannot_both_be_written(tmp_path):
    class TwoSizes:
        model_id = "racy@native"

        async def embed(self, texts, kind):
            dims = 4 if texts[0].startswith("long") else 3
            return [[1.0] + [0.0] * (dims - 1) for _ in texts]

    async def go():
        async with MemoryStore(str(tmp_path / "m.db"), TwoSizes()) as s:
            async with s._write_lock:   # a write in progress: both pass the chain's check, then wait here
                tasks = [asyncio.create_task(s.remember("long one")), asyncio.create_task(s.remember("short"))]
                await asyncio.sleep(0.01)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert [type(r).__name__ for r in results] == ["Decision", "EmbeddingError"]
            assert (await s.stats())["by_model"] == {"racy@native": 1}
    run(go())


def test_after_reembed_an_emptied_stamp_may_come_back_with_another_size(tmp_path):
    async def go():
        async with MemoryStore(str(tmp_path / "m.db"), Switch("a@1", 4)) as s:
            await s.remember("one fact")
            await s.reembed(Switch("b@1", 3))
            s.chain = EmbedderChain([Switch("a@1", 5)])
            assert (await s.remember("other fact")).model_id == "a@1"
    run(go())


def test_the_chain_says_which_check_refused_a_vector():
    class NaN:
        model_id = "nan@2"

        async def embed(self, texts, kind):
            return [[float("nan"), 1.0] for _ in texts]

    with pytest.raises(EmbeddingError, match="values that are not finite"):
        run(EmbedderChain([NaN()]).embed(["hello"]))


def test_reembed_with_the_primary_repairs_a_store_its_backup_started(tmp_path):
    primary, backup = Switch("small@native", 4), Switch("small@native", 3)

    async def go():
        async with MemoryStore(str(tmp_path / "m.db"), EmbedderChain([primary, backup])) as s:
            primary.up = False
            await s.remember("first fact")               # the backup writes first: the stamp gets 3 numbers
            primary.up = True
            await s.remember("fact no 3")                # the primary's 4 numbers are refused; the backup answers
            assert await s.reembed(Switch("small@native", 4)) == 2
            backup.up = False
            assert (await s.remember("a fact of size four")).op == ADD   # now the primary is the one that fits
            assert (await s.stats())["by_model"] == {"small@native": 3}
    run(go())


def test_the_chain_warns_when_a_fallback_answers_under_the_primarys_stamp(caplog):
    with caplog.at_level(logging.WARNING, logger="vector_memory"):
        run(EmbedderChain([Switch("same@1", 4, up=False), Switch("same@1", 4)]).embed(["hello"]))
    assert any("fallback #1 under the primary's own stamp same@1" in r.getMessage() for r in caplog.records)
