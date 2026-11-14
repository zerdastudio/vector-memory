"""Embedders, and the chain that falls back from one to the next.

Vectors from two different models live in two different spaces: comparing them gives numbers that look fine and
mean nothing. So every vector this library produces carries the id of the model that actually made it (a
"stamp"), and the store only ever compares vectors with the same stamp.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any, Protocol

import numpy as np

log = logging.getLogger("vector_memory")

DOCUMENT = "document"   # text being stored
QUERY = "query"         # text being searched for (some models embed the two differently)


class EmbeddingError(Exception):
    """Every embedder in the chain failed."""


class Embedder(Protocol):
    model_id: str   # names the vector space: model and dimensions

    async def embed(self, texts: list[str], kind: str) -> list[list[float]]: ...


class OpenAIEmbedder:
    """OpenAI's embeddings API, or any compatible endpoint (`base_url`), with optional shortened `dimensions`."""

    def __init__(self, model: str = "text-embedding-3-small", *, api_key: str | None = None,
                 base_url: str | None = None, dimensions: int | None = None, client: Any = None):
        self.model = model
        self.dimensions = dimensions
        self.model_id = f"{model}@{dimensions or 'native'}"
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=1)
        self._client = client

    async def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        extra = {"dimensions": self.dimensions} if self.dimensions else {}
        reply = await self._client.embeddings.create(model=self.model, input=texts, **extra)
        data = sorted(reply.data, key=lambda d: d.index)
        return [list(d.embedding) for d in data]


_STOP = frozenset(
    "a an the is are was were be been to of and or in on at for with by it its this that as from i you he she we "
    "they my your our their me us them do does did not no so if then than but into about".split())


def _tokens(text: str) -> list[str]:
    out = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in _STOP:
            continue
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        out.append(word)
    return out


class HashingEmbedder:
    """A bag-of-words embedder: offline, deterministic, free. For tests and demos only.

    Texts that share words come out close, and that is all it knows: "car" and "automobile" are as far apart as
    "car" and "banana". A real model is what puts paraphrases close together, and `calibrate` is how you check it.
    """

    def __init__(self, dims: int = 256, *, name: str = "hashing-bow"):
        self.dims = dims
        self.model_id = f"{name}@{dims}"

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dims
        for tok in _tokens(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dims] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        return [self._one(t) for t in texts]


def _checked(vector) -> list[float]:
    values = [float(x) for x in vector]
    if not all(math.isfinite(x) for x in values):
        raise EmbeddingError("a vector with values that are not finite")
    with np.errstate(all="ignore"):   # the store keeps float32: check the vector as it will be kept
        v32 = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(v32))
    if not np.isfinite(v32).all() or not math.isfinite(norm):
        raise EmbeddingError("a vector that overflows float32")
    if norm == 0:
        raise EmbeddingError("an empty vector, or one of zeros (in float32)")
    return values


class EmbedderChain:
    """Tries each embedder in order and returns the vectors with the stamp of the one that answered.

    A reply with the wrong number of vectors, vectors of uneven length, or a vector that is not a list of finite
    numbers with some length to it counts as a failure, and the next embedder is tried: a broken answer would
    end up stored next to text it does not describe. So does a reply whose size differs from the size `known`
    for its stamp (a store passes the sizes it already holds): two sizes under one stamp cannot be compared.
    """

    def __init__(self, embedders: list[Embedder]):
        if not embedders:
            raise ValueError("the chain needs at least one embedder")
        self.embedders = list(embedders)

    @property
    def primary(self) -> str:
        return self.embedders[0].model_id

    async def embed(self, texts: list[str], kind: str = DOCUMENT) -> tuple[list[list[float]], str]:
        vectors, model_id, _ = await self.embed_from(texts, kind)
        return vectors, model_id

    async def embed_from(self, texts: list[str], kind: str = DOCUMENT, known: dict[str, int] | None = None
                         ) -> tuple[list[list[float]], str, int]:
        """Like `embed`, plus the position in the chain of the embedder that answered (0: the primary)."""
        errors = []
        for position, embedder in enumerate(self.embedders):
            try:
                vectors = [_checked(v) for v in await embedder.embed(texts, kind)]
                if len(vectors) != len(texts):
                    raise EmbeddingError(f"{len(vectors)} vectors for {len(texts)} texts")
                if len({len(v) for v in vectors}) > 1:
                    raise EmbeddingError("vectors of uneven length")
                expected = (known or {}).get(embedder.model_id)
                if vectors and expected is not None and len(vectors[0]) != expected:
                    raise EmbeddingError(f"vectors of {len(vectors[0])} numbers under the stamp {embedder.model_id}, "
                                         f"whose vectors here have {expected}")
                if position and embedder.model_id == self.primary:
                    log.warning("embedded with fallback #%d under the primary's own stamp %s: its vectors are "
                                "compared with the primary's as if the same model made them", position,
                                embedder.model_id)
                elif position:
                    log.warning("embedded with fallback %s: its vectors are only compared with its own",
                                embedder.model_id)
                return vectors, embedder.model_id, position
            except Exception as e:
                errors.append(f"{embedder.model_id}: {type(e).__name__}: {e}")
                log.warning("embedder %s failed: %s", embedder.model_id, e)
        raise EmbeddingError("every embedder failed: " + "; ".join(errors))
