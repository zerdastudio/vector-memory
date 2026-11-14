"""Do the numbers discriminate? Measure it before trusting a threshold.

A dedup threshold is a distance. Whether 0.30 means "same subject" or "anything at all" depends entirely on the
embedding model, and a wrong threshold fails silently: every new memory is added, or every one is merged, and
nothing goes red. So this measures, on pairs you choose:

- duplicates: the same fact with trivial edits             -> should be UPDATE
- paraphrases: the same fact in other words                -> should be MERGE (or UPDATE)
- unrelated: different facts                               -> should be ADD

The merge threshold has to fall between the farthest paraphrase and the closest unrelated pair. If paraphrases
and unrelated pairs overlap, no threshold can work with this model, and the report says so.

    python -m vector_memory.calibrate examples/pairs.toml --model text-embedding-3-small

Exit codes: 0 separates, 1 does not (or a threshold you gave does not fit), 2 nothing measured because of the
setup (key, SDK, pairs file), 3 nothing measured because the embedding calls failed.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tomllib
from dataclasses import dataclass, field

import numpy as np

from .embedders import DOCUMENT, Embedder, EmbedderChain, EmbeddingError

Pair = tuple[str, str]

# Exit codes of the command line tool.
EXIT_OK = 0                # the model separates the pairs, and the thresholds given (if any) fit
EXIT_NOT_SEPARATED = 1     # measured: the model does not separate them, or a threshold does not fit
EXIT_SETUP = 2             # nothing measured: no key, no SDK, an unreadable or incomplete pairs file
EXIT_FAILED = 3            # nothing measured: the embedding calls failed


@dataclass
class Calibration:
    model_id: str
    distances: dict[str, list[tuple[str, str, float]]]
    suggested_update: float | None
    suggested_merge: float | None
    problems: list[str] = field(default_factory=list)

    def worst(self, group: str, fn) -> float | None:
        values = [d for _, _, d in self.distances.get(group, [])]
        return fn(values) if values else None

    @property
    def discriminates(self) -> bool:
        para, unrel = self.worst("paraphrases", max), self.worst("unrelated", min)
        return para is not None and unrel is not None and para < unrel

    def report(self) -> str:
        lines = [f"model: {self.model_id}", ""]
        for group in ("duplicates", "paraphrases", "unrelated"):
            rows = self.distances.get(group, [])
            if not rows:
                continue
            lines.append(f"{group} (cosine distance)")
            for a, b, d in sorted(rows, key=lambda r: r[2]):
                lines.append(f"  {d:6.3f}  {a[:40]!r} ~ {b[:40]!r}")
            lines.append("")
        para, unrel = self.worst("paraphrases", max), self.worst("unrelated", min)
        if para is not None and unrel is not None:
            lines.append(f"farthest paraphrase {para:.3f}, closest unrelated {unrel:.3f}, gap {unrel - para:+.3f}")
        if self.suggested_update is not None:
            lines.append(f"suggested update_distance: {self.suggested_update:.3f}")
        if self.suggested_merge is not None:
            lines.append(f"suggested merge_distance:  {self.suggested_merge:.3f}")
        lines += [f"PROBLEM: {p}" for p in self.problems]
        lines.append("verdict: " + ("the model separates them" if self.discriminates and not self.problems
                                    else "do not trust these thresholds"))
        return "\n".join(lines)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), 0.0, 2.0))


async def calibrate(embedder: Embedder | EmbedderChain, *, paraphrases: list[Pair], unrelated: list[Pair],
                    duplicates: list[Pair] = (), update_distance: float | None = None,
                    merge_distance: float | None = None) -> Calibration:
    chain = embedder if isinstance(embedder, EmbedderChain) else EmbedderChain([embedder])
    if not paraphrases or not unrelated:
        raise ValueError("calibration needs at least one paraphrase pair and one unrelated pair")
    groups = {"duplicates": list(duplicates), "paraphrases": list(paraphrases), "unrelated": list(unrelated)}
    texts = sorted({t for pairs in groups.values() for pair in pairs for t in pair})
    vectors, model_id, position = await chain.embed_from(texts, DOCUMENT)
    if position:
        raise RuntimeError(f"the primary embedder failed and fallback #{position} ({model_id}) answered: this "
                           "would calibrate the wrong model")
    by_text = {t: np.asarray(v, dtype=np.float64) for t, v in zip(texts, vectors)}
    distances = {g: [(a, b, _distance(by_text[a], by_text[b])) for a, b in pairs] for g, pairs in groups.items()}

    def worst(group, fn):
        values = [d for _, _, d in distances[group]]
        return fn(values) if values else None

    problems = []
    para_max, unrel_min = worst("paraphrases", max), worst("unrelated", min)
    suggested_merge = None
    if para_max < unrel_min:
        suggested_merge = (para_max + unrel_min) / 2
    else:
        problems.append(f"paraphrases reach {para_max:.3f} but unrelated pairs start at {unrel_min:.3f}: "
                        "no merge threshold separates them with this model")
    suggested_update = None
    dup_max, para_min = worst("duplicates", max), worst("paraphrases", min)
    if dup_max is not None:
        if dup_max < para_min:
            suggested_update = (dup_max + para_min) / 2
        else:
            problems.append(f"duplicates reach {dup_max:.3f} and paraphrases start at {para_min:.3f}: "
                            "the update threshold cannot tell them apart")
    if update_distance is not None and merge_distance is not None and not (
            0 <= update_distance < merge_distance <= 2):
        problems.append(f"update_distance {update_distance} and merge_distance {merge_distance}: the store needs "
                        "0 <= update < merge <= 2")
    if merge_distance is not None and not (para_max < merge_distance <= unrel_min):
        problems.append(f"merge_distance {merge_distance} is outside ({para_max:.3f}, {unrel_min:.3f}]")
    if update_distance is not None and dup_max is not None and not (dup_max < update_distance <= para_min):
        problems.append(f"update_distance {update_distance} is outside ({dup_max:.3f}, {para_min:.3f}]")
    return Calibration(model_id, distances, suggested_update, suggested_merge, problems)


def load_pairs(path: str) -> dict[str, list[Pair]]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return {g: [tuple(p) for p in data.get(g, [])] for g in ("duplicates", "paraphrases", "unrelated")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that an embedding model separates paraphrases from "
                                                 "unrelated texts, and suggest dedup thresholds.")
    parser.add_argument("pairs", help="a TOML file with duplicates, paraphrases and unrelated pairs")
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--base-url", help="an OpenAI-compatible endpoint")
    parser.add_argument("--key-env", default="OPENAI_API_KEY", help="the variable that holds the API key")
    parser.add_argument("--update-distance", type=float)
    parser.add_argument("--merge-distance", type=float)
    args = parser.parse_args(argv)
    key = os.environ.get(args.key_env, "").strip()
    if not key:
        print(f"{args.key_env} is empty", file=sys.stderr)
        return EXIT_SETUP
    try:
        pairs = load_pairs(args.pairs)
        if not pairs["paraphrases"] or not pairs["unrelated"]:
            raise ValueError(f"{args.pairs} needs at least one paraphrases pair and one unrelated pair")
    except (OSError, ValueError, TypeError) as e:   # tomllib.TOMLDecodeError is a ValueError
        print(f"cannot use {args.pairs}: {e}", file=sys.stderr)
        return EXIT_SETUP
    try:
        from .embedders import OpenAIEmbedder
        embedder = OpenAIEmbedder(args.model, api_key=key, base_url=args.base_url, dimensions=args.dimensions)
        result = asyncio.run(calibrate(embedder, **pairs, update_distance=args.update_distance,
                                       merge_distance=args.merge_distance))
    except ImportError as e:
        print(f"the openai SDK is missing ({e}): pip install -e \".[openai]\"", file=sys.stderr)
        return EXIT_SETUP
    except (EmbeddingError, RuntimeError) as e:
        print(f"could not measure: {e}", file=sys.stderr)
        return EXIT_FAILED
    print(result.report())
    return EXIT_OK if result.discriminates and not result.problems else EXIT_NOT_SEPARATED


if __name__ == "__main__":
    raise SystemExit(main())
