"""Retrieval + judge metrics.

* retrieval.py — hand-written hit@k, MRR, recall@k. Deterministic, LLM-free.
  These carry the PR gate (build step 2).
* judge.py — RAGAS wrapper, median-of-N. Build step 4 (not present yet).
"""
