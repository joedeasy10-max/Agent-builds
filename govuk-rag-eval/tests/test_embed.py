"""The offline embedder must be deterministic and normalised.

A failing case: identical text embeds to different vectors between runs (then
retrieval ranking is non-deterministic and the gate is worthless).
"""

import numpy as np

from src.embed import HashingEmbedder


def test_hashing_embedder_is_deterministic():
    e = HashingEmbedder(dimensions=64, seed=42)
    a = e.embed(["register for self assessment", "pay your tax bill"])
    b = e.embed(["register for self assessment", "pay your tax bill"])
    assert np.array_equal(a, b)


def test_hashing_embedder_shape_and_norm():
    e = HashingEmbedder(dimensions=64, seed=42)
    v = e.embed(["some text here", "more text"])
    assert v.shape == (2, 64)
    assert v.dtype == np.float32
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_similar_text_scores_higher_than_unrelated():
    e = HashingEmbedder(dimensions=512, seed=42)
    vs = e.embed(
        [
            "register for self assessment tax return deadline",  # doc A
            "self assessment registration deadline tax",          # query-ish, close to A
            "book a driving theory test appointment",             # unrelated
        ]
    )
    doc_a, close, unrelated = vs
    assert float(doc_a @ close) > float(doc_a @ unrelated)


def test_empty_text_gives_zero_vector_without_crashing():
    e = HashingEmbedder(dimensions=32, seed=1)
    v = e.embed([""])
    assert v.shape == (1, 32)
    assert np.all(v == 0.0)  # nothing to hash; normalisation leaves it zero
