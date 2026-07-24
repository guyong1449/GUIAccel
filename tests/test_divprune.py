"""Unit tests for skillreuse.model.divprune."""

from __future__ import annotations

import math

import pytest
import torch

from skillreuse.model.divprune import divprune_select, pairwise_cosine_distance


def _make_matrix(n: int = 10, d: int = 4, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, d)


# ---------------------------------------------------------------------------
# pairwise_cosine_distance
# ---------------------------------------------------------------------------


def test_cosine_distance_matrix():
    mat = _make_matrix(5, 4)
    dist = pairwise_cosine_distance(mat)

    assert dist.shape == (5, 5)
    assert torch.allclose(dist.diagonal(), torch.zeros(5), atol=1e-5)
    assert (dist >= -1e-6).all() and (dist <= 2.0 + 1e-6).all()
    assert torch.allclose(dist, dist.t(), atol=1e-5)


# ---------------------------------------------------------------------------
# divprune_select
# ---------------------------------------------------------------------------


def test_divprune_select_basic():
    mat = _make_matrix(10, 4)
    idx = divprune_select(mat, keep_ratio=0.3)

    assert idx.shape == (3,)
    assert (idx >= 0).all() and (idx < 10).all()
    assert idx.tolist() == sorted(idx.tolist())


def test_divprune_select_keep_all():
    n = 8
    mat = _make_matrix(n, 4)
    idx = divprune_select(mat, keep_ratio=1.0)

    assert idx.shape == (n,)
    assert idx.tolist() == list(range(n))


def test_divprune_select_keep_one():
    mat = _make_matrix(10, 4)
    idx = divprune_select(mat, keep_ratio=1e-9)

    assert idx.shape == (1,)


def test_divprune_select_no_duplicates():
    mat = _make_matrix(20, 8)
    for ratio in [0.1, 0.3, 0.5, 0.8]:
        idx = divprune_select(mat, keep_ratio=ratio)
        assert len(idx) == len(set(idx.tolist())), f"duplicates at ratio={ratio}"


def test_divprune_select_sorted():
    mat = _make_matrix(15, 6)
    for ratio in [0.2, 0.4, 0.6]:
        idx = divprune_select(mat, keep_ratio=ratio)
        assert idx.tolist() == sorted(idx.tolist()), f"not sorted at ratio={ratio}"


# ---------------------------------------------------------------------------
# grid_thw formula sanity check
# ---------------------------------------------------------------------------


def test_grid_thw_formula():
    for k in [1, 4, 9, 16, 25]:
        t, h, w = 1, 2, 2 * k
        assert (t * h * w) // 4 == k


# ---------------------------------------------------------------------------
# Matches original implementation
# ---------------------------------------------------------------------------


def _original_divprune(visual_feature_vectors, threshold_ratio):
    """Inline reimplementation of the original LLaVA DivPrune algorithm."""
    image_feature_length = visual_feature_vectors.shape[0]
    threshold_terms = int(round(threshold_ratio * image_feature_length))

    norm_matrix = visual_feature_vectors / visual_feature_vectors.norm(dim=1, keepdim=True)
    cosine_matrix = 1.0 - torch.mm(norm_matrix, norm_matrix.t())

    s = torch.empty(threshold_terms, dtype=torch.long, device=visual_feature_vectors.device)
    for i in range(threshold_terms):
        if i == 0:
            m2 = cosine_matrix
        else:
            m2 = torch.index_select(
                cosine_matrix, 0, torch.index_select(s, 0, torch.arange(0, i, device=cosine_matrix.device))
            )

        if i == 0:
            scores = torch.topk(m2, 2, dim=0, largest=False).values[1, :]
        else:
            scores = torch.min(m2, dim=0).values

        phrase_to_add_idx = torch.argmax(scores)
        s[i] = phrase_to_add_idx

    return s


def test_divprune_matches_original():
    torch.manual_seed(42)
    mat = torch.randn(20, 8)
    keep_ratio = 0.3

    original_idx = _original_divprune(mat, keep_ratio)
    new_idx = divprune_select(mat, keep_ratio=keep_ratio)

    assert set(original_idx.tolist()) == set(new_idx.tolist())
