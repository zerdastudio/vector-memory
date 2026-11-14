# vector-memory

[![tests](https://github.com/zerdastudio/vector-memory/actions/workflows/tests.yml/badge.svg)](https://github.com/zerdastudio/vector-memory/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Long-term memory for an LLM agent, in one SQLite file: the agent remembers what a user or a task taught it, and
recalls it later. Built so the memory does not quietly rot: **a fact told twice reinforces the old memory
instead of adding a copy, and a fact told with more detail is merged into it; every vector is stamped with the
model that made it, so a fallback model never gets compared with the main one; what is stored is never deleted
(forgetting marks a memory, a merge keeps the old text in its history); and a calibration tool tells you
whether your embedding model can tell a paraphrase from a different fact at all, before you trust a
threshold.**

SQLite and numpy. Any embedding model, with a fallback chain; OpenAI-compatible APIs and Cohere rerank through
optional SDKs.

**Where it comes from.** This is the memory layer of a multi-agent system I built, where it ran against live
embedding and rerank providers, rewritten here as a standalone library with its own test suite.

## When to use it

When one agent, in one process, needs a memory of thousands of facts that you can open and read: no server to
run, nothing to deploy, and every decision (add, update, merge) returned to you with its distance. If you need
millions of vectors, many processes writing at once, or a managed memory service, a vector database or a
hosted memory layer is the better choice. This library stays small on purpose, so that each step can be checked
and calibrated.

## What it looks like

`python examples/demo.py` runs offline with a toy bag-of-words embedder. Two moments from it:

```text
2. Remember (update below 0.15, merge below 0.60: thresholds for this toy embedder only)

  add    #1  distance     -  The user prefers metric units in reports.
  update #1  distance 0.000  The user prefers metric units in reports.
  merge  #1  distance 0.209  The user prefers metric units in reports; also: in reports the user prefers metric units and 24-hour times.
  add    #2  distance 1.000  Deploys are frozen on Fridays after 6 pm.
  add    #3  distance 1.000  The staging database is reset every Monday night.

5. The primary embedder goes down

  [warning] embedded with fallback backup-model@128: its vectors are only compared with its own
  [warning] dedup only sees memories made by backup-model@128; {'hashing-bow@256': 4} are in other spaces and cannot be compared
  add    #5  distance     -  The user prefers metric units in reports.
  -> #5 repeats #1, but #1 lives in the other space: dedup could not see it
  spaces: {'backup-model@128': 1, 'hashing-bow@256': 4}
  re-embedded with the backup: 4
  spaces: {'backup-model@128': 5}
```

The same fact said twice reinforces the old memory instead of adding a copy. The same subject with more detail
is merged, and the old text goes to the history. When the main embedding provider fails, the backup answers;
its vectors live in another space, so the store refuses to compare them with the old ones, says so, and shows
the damage (a duplicate that dedup could not see) until you re-embed.

And the demo's first step calibrates the toy embedder on [`examples/pairs.toml`](examples/pairs.toml), and fails
it on purpose:

```text
farthest paraphrase 0.817, closest unrelated 0.817, gap +0.000
PROBLEM: paraphrases reach 0.817 but unrelated pairs start at 0.817: no merge threshold separates them with this model
verdict: do not trust these thresholds
```

A bag of words cannot see that "kilometres and kilograms" means "metric units". That is the check to run on
your real model.

## Quick start

```bash
git clone https://github.com/zerdastudio/vector-memory
cd vector-memory
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                    # 76 tests, offline
python examples/demo.py   # the tour above
```

With a real model, first check that it separates your facts:

```bash
pip install -e ".[openai]"
export OPENAI_API_KEY=...
python -m vector_memory.calibrate examples/pairs.toml --model text-embedding-3-small
```

Then use it. This is a skeleton: `merge` is yours to write, usually one LLM call. Leave `merger` out and
similar facts are added side by side instead of merged; if the merger raises, the error reaches you and
nothing is written.

```python
import asyncio
from vector_memory import EmbedderChain, MemoryStore, OpenAIEmbedder

embedder = EmbedderChain([
    OpenAIEmbedder("text-embedding-3-small"),
    OpenAIEmbedder("text-embedding-3-small", base_url="https://your-backup-endpoint/v1"),
])

async def merge(old: str, new: str) -> str:
    ...  # an LLM call: "write one memory that keeps what both say"

async def main():
    async with MemoryStore("memory.db", embedder, update_distance=0.10, merge_distance=0.35,
                           merger=merge) as memory:
        print(await memory.remember("The client pays invoices within 30 days."))
        for hit in await memory.recall("When does the client pay?", k=3):
            print(f"{hit.distance:.3f}  {hit.text}")

asyncio.run(main())
```

Use the thresholds `calibrate` suggests for your model and your pairs, not the ones above. Without arguments
the store uses `update_distance=0.15` and `merge_distance=0.30`: a starting point, wrong for most models.

## How a memory is stored

1. **Embed** the text through the chain. The first embedder that answers with the right number of finite
   vectors of even length wins, and its id is the vector's stamp. An answer whose size differs from the size its
   stamp already has in the store is refused too, and the chain goes on: two sizes under one stamp (a backup
   endpoint serving another model under the same name) cannot be compared.
2. **Find the nearest** active memory **in the same space**, that is the same stamp and the same size. Vectors
   from different models are never compared: the numbers would look fine and mean nothing.
3. **Decide**, under one lock so two copies of a fact arriving together cannot both be added:
   - closer than `update_distance`: **update**. The old memory gains confidence; its text stays, and the new
     one is not kept anywhere (a fixed typo stays unfixed);
   - closer than `merge_distance`: **merge**, if you gave a merger: the two texts become one, re-embedded, and
     the old text goes to `history`. Without a merger the new text is added;
   - otherwise: **add**.
4. **Recall** embeds the query as a query (some models embed questions and documents differently), takes the
   closest memories in the same space, and optionally lets a reranker reorder the closest few.

`forget` marks a memory and keeps it; `history` returns every text a memory had, oldest first; `stats` counts
memories per vector space, so a split after a fallback is visible; `reembed` moves everything into one space.
It moves vectors, it does not merge: a duplicate added while the spaces were split stays two memories. If a
backup wrote first under the primary's stamp, the store keeps the backup's size until you `reembed` with the
primary.

## The API

All methods are async; `async with MemoryStore(...)` opens and closes the store.

| call | returns | raises |
|---|---|---|
| `remember(text, confidence=0.5)` | `Decision`: `op` (add, update, merge), `memory_id`, `text`, `model_id`, `distance`, `neighbor_id` | `ValueError` (empty text, confidence outside 0-1), `EmbeddingError` |
| `recall(query, k=5, min_confidence=0.0, reranker=None, candidates=None)` | the `k` closest `Hit`s: `id`, `text`, `distance`, `confidence`, `model_id`, `score` (with a reranker) | `ValueError` (k below 1), `EmbeddingError` |
| `forget(memory_id, reason="forgotten")` | `None` | `KeyError` (unknown id) |
| `get(memory_id)` | a dict: `id`, `text`, `model_id`, `confidence`, `hits`, `status` | `KeyError` (unknown id) |
| `history(memory_id)` | a list of dicts: `text`, `reason`, `ts`, oldest first (empty for an unknown id) | |
| `stats()` | a dict: `active`, `forgotten`, `by_model` (memories per vector space) | |
| `reembed(embedder, batch=64)` | how many memories changed | `ValueError` (batch), `EmbeddingError` |

A call still waiting for the store when it closes raises `StoreClosed`; a call made after `close` opens the
store again. `recall` returns the best `k` it has, however far: filter on
`distance` (or on `score`, with a reranker) if a poor match should not count.

## Calibrating

A dedup threshold is a distance, and what a distance means depends entirely on the embedding model. A wrong
threshold fails silently: everything is added, or everything merged, and nothing goes red. `calibrate` measures
three kinds of pairs you write from your own data:

| pairs | example | should be |
|---|---|---|
| duplicates | the same sentence with a typo fixed | update |
| paraphrases | the same fact in other words | merge (or update) |
| unrelated | two different facts | add |

The merge threshold must fall between the farthest paraphrase and the closest unrelated pair. If they overlap,
no threshold works with that model. It also refuses to calibrate if the primary embedder failed and a fallback
answered, even one with the same stamp, since that could measure the wrong model.

The command line exits with 0 when the model separates the pairs (and the thresholds you pass fit), 1 when it
does not, 2 when nothing was measured because of the setup (no key, no SDK, a missing or incomplete pairs file),
and 3 when the embedding calls failed.

## Limits

- Search is exact and brute force (numpy over the vectors of one space): fine for thousands of memories, not
  for millions. Swap in an index if you outgrow it.
- One process per database file.
- A merge holds the write lock while the merger (usually an LLM call) runs: other writes wait. `close` waits
  for them too; a call still waiting for the store when it closes gets `StoreClosed`.
- `recall` does not wait for `reembed`: during one, it sees the memories already moved and misses the others.
  A `reembed` that fails halfway leaves two spaces; run it again.
- A `remember` cancelled while its database write is running still completes that write: SQLite work runs on
  a thread, and a thread cannot be stopped halfway. The caller sees the cancellation; the memory is there.
- The test suite runs offline: in CI the OpenAI and Cohere clients run against stand-in clients.

## License

MIT
