"""The memory store: SQLite rows, unit vectors, and a decision for every new memory."""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

import numpy as np

from .embedders import DOCUMENT, QUERY, Embedder, EmbedderChain, EmbeddingError

log = logging.getLogger("vector_memory")

ADD, UPDATE, MERGE = "add", "update", "merge"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY,
    text        TEXT    NOT NULL,
    model_id    TEXT    NOT NULL,   -- the stamp: which model made the vector
    dims        INTEGER NOT NULL,
    vector      BLOB    NOT NULL,   -- float32, unit length
    confidence  REAL    NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0,
    status      TEXT    NOT NULL DEFAULT 'active',   -- active | forgotten
    created     REAL    NOT NULL,
    updated     REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS memories_space ON memories (model_id, status);
CREATE TABLE IF NOT EXISTS history (
    id          INTEGER PRIMARY KEY,
    memory_id   INTEGER NOT NULL,
    text        TEXT    NOT NULL,   -- the text as it was before the change
    reason      TEXT    NOT NULL,
    ts          REAL    NOT NULL
);
"""

Merger = Callable[[str, str], Awaitable[str]]


class StoreClosed(RuntimeError):
    """The store was closed while a call was waiting for it. Open it again to go on."""


def _fraction(name: str, value) -> float:
    if not (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and 0 <= value <= 1):
        raise ValueError(f"{name} must be a number from 0 to 1, not {value!r}")
    return float(value)


class Reranker(Protocol):
    async def rerank(self, query: str, documents: list[str]) -> list[float]: ...


@dataclass
class Decision:
    op: str                      # add | update | merge
    memory_id: int
    text: str                    # the memory's text after the decision
    model_id: str
    distance: float | None       # to the nearest memory in the same space; None when there was none
    neighbor_id: int | None


@dataclass
class Hit:
    id: int
    text: str
    distance: float              # cosine distance: 0 same direction, 1 unrelated, 2 opposite
    confidence: float
    model_id: str
    score: float | None = None   # the reranker's score, when one was used


def _unit(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("cannot store a zero or non-finite vector")
    return v / norm


class MemoryStore:
    """Long-term memory for an agent.

    `remember` embeds a text and compares it with the nearest active memory made by the same model:
    - closer than `update_distance`: the same thing said again. The old memory is reinforced (confidence up),
      its text is kept;
    - closer than `merge_distance`: the same subject, framed differently. With a `merger` (usually an LLM call)
      the two texts become one, and the old text goes to the history; without one, the new text is added;
    - otherwise: something new, added.

    The thresholds depend on the embedding model. The defaults are a starting point, not a truth: measure yours
    with `calibrate`.

    Nothing is ever deleted: `forget` marks a memory, and every text that a merge replaced stays in `history`.
    All database work runs on one thread, so reads and writes never interleave.

    A vector space is a stamp and a size: the store only compares vectors that share both, and it refuses a
    vector whose size differs from the one its stamp already has here (the chain then tries the next embedder).
    """

    def __init__(self, path: str, embedder: Embedder | EmbedderChain, *, update_distance: float = 0.15,
                 merge_distance: float = 0.30, merger: Merger | None = None, reinforce: float = 0.05,
                 clock: Callable[[], float] = time.time):
        if not 0 <= update_distance < merge_distance <= 2:
            raise ValueError("need 0 <= update_distance < merge_distance <= 2")
        self.path = path
        self.chain = embedder if isinstance(embedder, EmbedderChain) else EmbedderChain([embedder])
        self.update_distance = update_distance
        self.merge_distance = merge_distance
        self.merger = merger
        self.reinforce = _fraction("reinforce", reinforce)
        self._clock = clock
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._conn: sqlite3.Connection | None = None
        self._dims: dict[str, int] = {}   # the vector size of each stamp in this file
        self._write_lock = asyncio.Lock()
        self._open_lock = asyncio.Lock()

    # ---- plumbing -------------------------------------------------------------------------------------------

    async def _run(self, fn, *args):
        if self._pool is None:  # one thread for every database call: no two ever interleave
            self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="vector-memory")
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    async def _db(self, fn, *args):
        if self._conn is None:
            raise StoreClosed("the store is closed")
        return await self._run(fn, *args)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.executescript(_SCHEMA)
        conn.commit()
        return conn

    def _load_dims(self) -> dict[str, int]:
        sizes: dict[str, list[tuple[int, int]]] = {}
        for model_id, dims, count in self._conn.execute(
                "SELECT model_id, dims, COUNT(*) FROM memories WHERE status = 'active' GROUP BY model_id, dims"):
            sizes.setdefault(model_id, []).append((count, dims))
        for model_id, found in sizes.items():
            if len(found) > 1:
                log.warning("%s has vectors of %s sizes in this file: only the most common is used; run reembed",
                            model_id, sorted(d for _, d in found))
        return {model_id: max(found)[1] for model_id, found in sizes.items()}

    async def open(self) -> "MemoryStore":
        async with self._open_lock:   # two calls at once open one connection, not two
            if self._conn is None:
                self._conn = await self._run(self._connect)
                self._dims = await self._db(self._load_dims)
        return self

    async def close(self) -> None:
        """Waits for the writes in flight, then closes. A call still waiting for the store gets StoreClosed."""
        async with self._write_lock:
            conn, self._conn = self._conn, None
            if conn is not None:
                await self._run(conn.close)
            if self._pool is not None:
                pool, self._pool = self._pool, None
                pool.shutdown(wait=True)   # its queue is empty by now: this does not block

    async def __aenter__(self):
        return await self.open()

    async def __aexit__(self, *exc):
        await self.close()

    def _space(self, model_id: str, dims: int) -> tuple[list[tuple], np.ndarray | None]:
        rows = self._conn.execute(
            "SELECT id, text, confidence, vector FROM memories WHERE model_id = ? AND dims = ? AND status = 'active'",
            (model_id, dims)).fetchall()
        if not rows:
            return [], None
        matrix = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        return rows, matrix

    def _spaces(self) -> dict[str, int]:
        """Active memories per space. A space is named by its stamp, plus its size when one stamp has two."""
        rows = self._conn.execute(
            "SELECT model_id, dims, COUNT(*) FROM memories WHERE status = 'active' GROUP BY model_id, dims").fetchall()
        stamps = [m for m, _, _ in rows]
        return {(m if stamps.count(m) == 1 else f"{m} ({d} numbers)"): c for m, d, c in rows}

    def _other_spaces(self, model_id: str, dims: int) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT model_id, dims, COUNT(*) FROM memories WHERE status = 'active' AND NOT (model_id = ? AND dims = ?)"
            " GROUP BY model_id, dims", (model_id, dims)).fetchall()
        return {(m if m != model_id else f"{m} ({d} numbers)"): c for m, d, c in rows}

    def _claim(self, model_id: str, dims: int) -> None:
        """Bind a stamp to a size, or refuse a second size for it (two vectors raced past the chain's check)."""
        known = self._dims.setdefault(model_id, dims)
        if known != dims:
            raise EmbeddingError(f"vectors of {dims} numbers under the stamp {model_id}, whose vectors here have "
                                 f"{known}")

    # ---- writing --------------------------------------------------------------------------------------------

    async def remember(self, text: str, *, confidence: float = 0.5) -> Decision:
        confidence = _fraction("confidence", confidence)
        text = text.strip()
        if not text:
            raise ValueError("an empty text is not a memory")
        await self.open()
        vectors, model_id, _ = await self.chain.embed_from([text], DOCUMENT, self._dims)
        vector = _unit(vectors[0])
        async with self._write_lock:   # decide and write as one step: two copies of a fact cannot both be added
            self._claim(model_id, len(vector))
            nearest = await self._db(self._nearest, vector, model_id)
            others = await self._db(self._other_spaces, model_id, len(vector))
            if others:
                log.warning("dedup only sees memories made by %s; %s are in other spaces and cannot be compared",
                            model_id, others)
            if nearest is None or nearest[2] >= self.merge_distance:
                mid = await self._db(self._insert, text, model_id, vector, confidence)
                return Decision(ADD, mid, text, model_id, nearest and nearest[2], nearest and nearest[0])
            neighbor_id, old_text, distance = nearest
            if distance < self.update_distance:
                await self._db(self._reinforce, neighbor_id)
                return Decision(UPDATE, neighbor_id, old_text, model_id, distance, neighbor_id)
            if self.merger is None:
                mid = await self._db(self._insert, text, model_id, vector, confidence)
                return Decision(ADD, mid, text, model_id, distance, neighbor_id)
            merged = await self.merger(old_text, text)
            if not isinstance(merged, str) or not merged.strip():
                raise ValueError(f"the merger must return a non-empty text, not {merged!r:.80}")
            merged = merged.strip()
            merged_vectors, merged_model, _ = await self.chain.embed_from([merged], DOCUMENT, self._dims)
            merged_vector = _unit(merged_vectors[0])
            self._claim(merged_model, len(merged_vector))
            await self._db(self._replace, neighbor_id, merged, merged_model, merged_vector,
                           f"merged with: {text}")
            return Decision(MERGE, neighbor_id, merged, merged_model, distance, neighbor_id)

    def _nearest(self, vector: np.ndarray, model_id: str):
        rows, matrix = self._space(model_id, vector.shape[0])
        if matrix is None:
            return None
        distances = np.clip(1.0 - matrix @ vector, 0.0, 2.0)   # rounding can dip just below zero
        i = int(np.argmin(distances))
        return rows[i][0], rows[i][1], float(distances[i])

    def _insert(self, text, model_id, vector, confidence) -> int:
        now = self._clock()
        cur = self._conn.execute(
            "INSERT INTO memories (text, model_id, dims, vector, confidence, created, updated) VALUES (?,?,?,?,?,?,?)",
            (text, model_id, int(vector.shape[0]), vector.astype(np.float32).tobytes(), float(confidence), now, now))
        self._conn.commit()
        return int(cur.lastrowid)

    def _reinforce(self, memory_id: int) -> None:
        self._conn.execute("UPDATE memories SET confidence = MIN(1.0, confidence + ?), updated = ? WHERE id = ?",
                           (self.reinforce, self._clock(), memory_id))
        self._conn.commit()

    def _replace(self, memory_id, text, model_id, vector, reason) -> None:
        now = self._clock()
        old = self._conn.execute("SELECT text FROM memories WHERE id = ?", (memory_id,)).fetchone()[0]
        self._conn.execute("INSERT INTO history (memory_id, text, reason, ts) VALUES (?,?,?,?)",
                           (memory_id, old, reason, now))
        self._conn.execute(
            "UPDATE memories SET text = ?, model_id = ?, dims = ?, vector = ?, updated = ? WHERE id = ?",
            (text, model_id, int(vector.shape[0]), vector.astype(np.float32).tobytes(), now, memory_id))
        self._conn.commit()

    async def forget(self, memory_id: int, reason: str = "forgotten") -> None:
        await self.open()
        async with self._write_lock:
            await self._db(self._forget, memory_id, reason)

    def _forget(self, memory_id: int, reason: str) -> None:
        row = self._conn.execute("SELECT text FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise KeyError(memory_id)
        self._conn.execute("INSERT INTO history (memory_id, text, reason, ts) VALUES (?,?,?,?)",
                           (memory_id, row[0], reason, self._clock()))
        self._conn.execute("UPDATE memories SET status = 'forgotten', updated = ? WHERE id = ?",
                           (self._clock(), memory_id))
        self._conn.commit()

    async def reembed(self, embedder: Embedder, batch: int = 64) -> int:
        """Re-embed every active memory with `embedder`, so they all share its space. Returns how many changed.

        Memories already stamped by `embedder` are kept, unless their size differs from what it gives now (a
        backup that wrote first under the same stamp): then they are redone too. Duplicates made while the spaces were split stay two memories: re-embedding moves vectors, it does not
        merge. If it fails halfway, the memories done so far are in the new space and the rest in the old ones.
        """
        if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
            raise ValueError(f"batch must be an integer >= 1, not {batch!r}")
        await self.open()
        chain = EmbedderChain([embedder])
        stamp = embedder.model_id
        changed = 0
        size: int | None = None

        async def move(rows):
            nonlocal changed, size
            for start in range(0, len(rows), batch):
                part = rows[start:start + batch]
                known = {stamp: size} if size is not None else None
                vectors, model_id, _ = await chain.embed_from([r[1] for r in part], DOCUMENT, known)
                for row, v in zip(part, vectors):
                    vector = _unit(v)
                    size = vector.shape[0]
                    await self._db(self._set_vector, row[0], model_id, vector)
                    changed += 1

        async with self._write_lock:
            rows = await self._db(lambda: self._conn.execute(
                "SELECT id, text, model_id, dims FROM memories WHERE status = 'active' ORDER BY id").fetchall())
            others = [r for r in rows if r[2] != stamp]
            own = [r for r in rows if r[2] == stamp]
            await move(others)
            if own and size is None:      # nothing else to move: ask the embedder what size it gives now
                vectors, _, _ = await chain.embed_from([own[0][1]], DOCUMENT)
                size = len(vectors[0])
            await move([r for r in own if r[3] != size])
            self._dims = await self._db(self._load_dims)
        return changed

    def _set_vector(self, memory_id, model_id, vector) -> None:
        self._conn.execute("UPDATE memories SET model_id = ?, dims = ?, vector = ? WHERE id = ?",
                           (model_id, int(vector.shape[0]), vector.astype(np.float32).tobytes(), memory_id))
        self._conn.commit()

    # ---- reading --------------------------------------------------------------------------------------------

    async def recall(self, query: str, k: int = 5, *, min_confidence: float = 0.0,
                     reranker: Reranker | None = None, candidates: int | None = None) -> list[Hit]:
        """The `k` memories closest to `query`. With a reranker, the closest `candidates` (default 3k) are
        reordered by its scores, and the best `k` of those come back."""
        if k < 1:
            raise ValueError("k must be at least 1")
        await self.open()
        vectors, model_id, _ = await self.chain.embed_from([query], QUERY, self._dims)
        vector = _unit(vectors[0])
        pool_size = max(candidates or 3 * k, k) if reranker else k
        hits = await self._db(self._search, vector, model_id, pool_size, min_confidence)
        others = await self._db(self._other_spaces, model_id, len(vector))
        if others:
            log.warning("recall searched only memories made by %s; %s are in other spaces", model_id, others)
        if reranker and hits:
            scores = await reranker.rerank(query, [h.text for h in hits])
            if len(scores) != len(hits):
                raise ValueError(f"the reranker returned {len(scores)} scores for {len(hits)} documents")
            scores = [float(s) for s in scores]
            if not all(math.isfinite(s) for s in scores):
                raise ValueError("the reranker returned scores that are not finite numbers")
            for h, s in zip(hits, scores):
                h.score = s
            hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:k]
        if hits:
            await self._db(self._count_hits, [h.id for h in hits])
        return hits

    def _search(self, vector, model_id, n, min_confidence) -> list[Hit]:
        rows, matrix = self._space(model_id, vector.shape[0])
        if matrix is None:
            return []
        distances = np.clip(1.0 - matrix @ vector, 0.0, 2.0)
        order = np.argsort(distances, kind="stable")
        hits = []
        for i in order:
            mid, text, confidence, _ = rows[int(i)]
            if confidence >= min_confidence:
                hits.append(Hit(mid, text, float(distances[i]), confidence, model_id))
            if len(hits) == n:
                break
        return hits

    def _count_hits(self, ids: list[int]) -> None:
        self._conn.executemany("UPDATE memories SET hits = hits + 1 WHERE id = ?", [(i,) for i in ids])
        self._conn.commit()

    async def get(self, memory_id: int) -> dict:
        await self.open()

        def read():
            row = self._conn.execute(
                "SELECT id, text, model_id, confidence, hits, status FROM memories WHERE id = ?",
                (memory_id,)).fetchone()
            if row is None:
                raise KeyError(memory_id)
            return dict(zip(("id", "text", "model_id", "confidence", "hits", "status"), row))
        return await self._db(read)

    async def history(self, memory_id: int) -> list[dict]:
        await self.open()
        rows = await self._db(lambda: self._conn.execute(
            "SELECT text, reason, ts FROM history WHERE memory_id = ? ORDER BY id", (memory_id,)).fetchall())
        return [{"text": t, "reason": r, "ts": ts} for t, r, ts in rows]

    async def stats(self) -> dict:
        """How many memories, and in which vector spaces: more than one active space means a fallback embedder
        answered at some point, and those memories cannot see each other until `reembed`."""
        await self.open()

        def read():
            by_model = self._spaces()
            forgotten = self._conn.execute("SELECT COUNT(*) FROM memories WHERE status != 'active'").fetchone()[0]
            return {"active": sum(by_model.values()), "forgotten": forgotten, "by_model": by_model}
        return await self._db(read)
