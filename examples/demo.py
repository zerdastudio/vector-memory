"""A tour of the store: no keys, no network, a second or two.

It uses the offline bag-of-words embedder, which is not a real model, and the first step proves it: the
calibration finds that it cannot tell a paraphrase from an unrelated fact. That is the point of calibrating.

    python examples/demo.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from vector_memory import EmbedderChain, HashingEmbedder, MemoryStore, OverlapReranker, calibrate
from vector_memory.calibrate import load_pairs

HERE = os.path.dirname(os.path.abspath(__file__))


async def merger(old: str, new: str) -> str:
    """In a real agent this is an LLM call: 'write one memory that keeps what both say'."""
    return f"{old.rstrip('.')}; also: {new[0].lower()}{new[1:]}"


class Down:
    """An embedder that is out of service, to show the fallback."""
    model_id = "primary-model@1024"

    async def embed(self, texts, kind):
        raise ConnectionError("provider unreachable")


def show(decision) -> None:
    dist = "-" if decision.distance is None else f"{decision.distance:.3f}"
    print(f"  {decision.op:6} #{decision.memory_id}  distance {dist:>5}  {decision.text}")


async def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="  [warning] %(message)s", stream=sys.stdout)
    embedder = HashingEmbedder()

    print("1. Calibrate before trusting a threshold (examples/pairs.toml)\n")
    result = await calibrate(embedder, **load_pairs(os.path.join(HERE, "pairs.toml")))
    print("\n".join("   " + line for line in result.report().splitlines()))

    print("\n2. Remember (update below 0.15, merge below 0.60: thresholds for this toy embedder only)\n")
    path = os.path.join(tempfile.mkdtemp(prefix="vector-memory-demo-"), "memory.db")
    async with MemoryStore(path, embedder, update_distance=0.15, merge_distance=0.60, merger=merger) as store:
        for text in [
            "The user prefers metric units in reports.",
            "The user prefers metric units in reports",                     # the same fact again
            "In reports the user prefers metric units and 24-hour times.",   # same subject, more detail
            "Deploys are frozen on Fridays after 6 pm.",
            "The staging database is reset every Monday night.",
            "The client pays invoices within 30 days.",
        ]:
            show(await store.remember(text))

        print("\n3. Recall\n")
        query = "Which units and times does the user want in reports?"
        print(f"  query: {query}")
        for hit in await store.recall(query, k=2):
            print(f"  {hit.distance:.3f}  #{hit.id}  {hit.text}")
        print("  with a reranker on the 6 closest:")
        for hit in await store.recall(query, k=2, reranker=OverlapReranker(), candidates=6):
            print(f"  score {hit.score:.2f}  #{hit.id}  {hit.text}")

        print("\n4. History: nothing is lost in a merge\n")
        for h in await store.history(1):
            print(f"  #1 was: {h['text']!r}  ({h['reason']})")

    print("\n5. The primary embedder goes down\n")
    chain = EmbedderChain([Down(), HashingEmbedder(128, name="backup-model")])
    async with MemoryStore(path, chain, update_distance=0.15, merge_distance=0.60) as store:
        show(await store.remember("The user prefers metric units in reports."))
        print("  -> #5 repeats #1, but #1 lives in the other space: dedup could not see it")
        print(f"  spaces: {(await store.stats())['by_model']}")
        print(f"  re-embedded with the backup: {await store.reembed(HashingEmbedder(128, name='backup-model'))}")
        print(f"  spaces: {(await store.stats())['by_model']}")
        show(await store.remember("The user prefers metric units in reports."))


if __name__ == "__main__":
    asyncio.run(main())
