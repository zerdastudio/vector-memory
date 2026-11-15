# Changelog

## 2.0.1 (2026-11-14)

- When an embedding endpoint leaves some of a batch's order indexes empty (seen on Gemini's OpenAI-compatible
  endpoint), the vectors keep the order of the reply, which is the order of the texts. Live batch-order test.

## 2.0.0 (2026-11-14)

First public release: the memory layer of a multi-agent system I built, rewritten as a standalone library with its
own test suite. Vectors stamped with the model that made them, add/update/merge by distance, nothing stored is
deleted, rerank, and a calibration tool for the thresholds. Live check against OpenAI, Gemini and Cohere on
14 November 2026.
