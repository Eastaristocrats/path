"""Deterministic score identities and public input contracts."""

import math
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from pathtail import binary_bound, binary_pair, normalize_pair


def count_vectors(m):
    return np.array([c for c in product(range(m + 1), repeat=4) if sum(c) == m])


@pytest.mark.parametrize("method", ["monomial", "pooled"])
@pytest.mark.parametrize("m", [3, 4])
def test_finite_multinomial_pair_identity(method, m):
    counts = count_vectors(m)
    p = np.array([0.1, 0.2, 0.3, 0.4])
    weights = np.array(
        [math.factorial(m) / math.prod(math.factorial(int(x)) for x in c) for c in counts]
    ) * np.prod(p**counts, axis=1)
    chi = np.array([1, -1, -1, 1])
    residual = chi / (p * np.sum(1 / p))
    means = []
    companion = None
    for state in range(4):
        pair = binary_pair(counts, state, method=method)
        means.append(weights @ pair.score)
        companion = weights @ pair.companion
    np.testing.assert_allclose(means, companion * residual, atol=1e-14, rtol=1e-13)


def test_minimum_budget_normalized_scores_match():
    counts = count_vectors(3)
    for state in range(4):
        mono = binary_pair(counts, state)
        pooled = binary_pair(counts, state, method="pooled")
        np.testing.assert_allclose(
            normalize_pair(mono, bound=binary_bound(3), tolerance=0),
            normalize_pair(pooled, bound=binary_bound(3, method="pooled")),
        )


@pytest.mark.parametrize("method", ["monomial", "pooled"])
@pytest.mark.parametrize("tau", [0, 0.01, 0.5])
def test_global_bound_contains_every_small_count_vector(method, tau):
    counts = count_vectors(5)
    bound = binary_bound(5, method=method, tolerance=tau)
    for state in range(4):
        z = normalize_pair(binary_pair(counts, state, method=method), bound=bound, tolerance=tau)
        assert np.max(np.abs(z)) <= 1


def test_batch_broadcasting_matches_scalar_calls():
    counts = np.array([[[3, 1, 1, 3], [2, 2, 2, 2]], [[1, 3, 2, 2], [8, 0, 0, 0]]])
    outcomes = np.array([0, 3])
    pair = binary_pair(counts, outcomes, method="pooled")
    for row, col in product(range(2), repeat=2):
        scalar = binary_pair(counts[row, col], outcomes[col], method="pooled")
        assert pair.score[row, col] == scalar.score
        assert pair.companion[row, col] == scalar.companion


@pytest.mark.parametrize(
    "counts,outcome",
    [
        ([], 0),
        ([1, 2, 3], 0),
        ([1.0, 1.0, 1.0, 1.0], 0),
        ([True, True, True, True], 0),
        ([-1, 1, 1, 3], 0),
        ([2**32, 0, 0, 0], 0),
        ([2**30, 2**30, 0, 0], 0),
        ([1, 1, 0, 0], 0),
        ([[1, 1, 1, 1], [2, 1, 1, 1]], 0),
        ([1, 1, 1, 1], 4),
        ([1, 1, 1, 1], -1),
        ([1, 1, 1, 1], 1.0),
        ([1, 1, 1, 1], True),
        ([1, 1, 1, 1], [1, 2]),
    ],
)
def test_invalid_count_and_outcome_inputs(counts, outcome):
    with pytest.raises(ValueError):
        binary_pair(counts, outcome)


@pytest.mark.parametrize("budget", [True, 2, 3.0, -1, 2**32])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        binary_bound(budget)


def test_unknown_method_and_nonfinite_bound_are_rejected():
    with pytest.raises(ValueError, match="method"):
        binary_pair([1, 1, 1, 1], 0, method="other")
    with pytest.raises(ValueError, match="method"):
        binary_bound(4, method="other")
    with pytest.raises(ValueError, match="finite normalization"):
        binary_bound(4, method="pooled", tolerance=1e308)


@pytest.mark.parametrize("which", ["counts", "outcome"])
def test_masked_count_or_outcome_is_rejected(which):
    counts, outcome = [2, 2, 2, 2], 0
    if which == "counts":
        counts = np.ma.array(counts, mask=[False, True, False, False])
    else:
        outcome = np.ma.array(outcome, mask=True)
    with pytest.raises(ValueError, match="Masked"):
        binary_pair(counts, outcome)


@pytest.mark.parametrize("method", ["monomial", "pooled"])
@pytest.mark.parametrize("m", [3, 5, 8])
def test_tight_bound_attained_and_conservative_bound_contains_it(method, m):
    tolerance = 0.005
    counts = count_vectors(m)
    largest = max(
        np.max(
            np.abs(binary_pair(counts, state, method=method).score)
            + tolerance * binary_pair(counts, state, method=method).companion
        )
        for state in range(4)
    )
    tight = binary_bound(m, method=method, tolerance=tolerance)
    conservative = binary_bound(m, method=method, tolerance=tolerance, normalization="conservative")
    assert tight == pytest.approx(largest, rel=1e-14)
    assert tight <= conservative


def test_bound_rounding_and_enumeration_limit():
    assert Fraction.from_float(binary_bound(3)) >= Fraction(1, 6)
    with pytest.raises(ValueError, match="normalization"):
        binary_bound(8, normalization="invalid")
    with pytest.raises(ValueError, match="100,000"):
        binary_bound(83, tolerance=0.01)
    assert binary_bound(83, tolerance=0.01, normalization="conservative") > 0
