import subprocess
import sys
from fractions import Fraction
from math import factorial

import pytest
import sympy as sp

from pathtail import primitive_degree, volume_pair


def test_volume_is_exact_and_invariant_to_forecast_permutation():
    pair = volume_pair([[1, 0], [1, 1], [1, 2]], [0, 1, 4], [1, 3], 9)
    assert pair.companion == 1
    assert pair.score == sp.Rational(10, 3)
    reverse = volume_pair([[1, 2], [1, 1], [1, 0]], [4, 1, 0], [1, 3], 9)
    assert reverse == pair


def test_volume_singular_gram_has_no_inverse_or_regularization():
    pair = volume_pair([[1, 0], [1, 0]], [0, 1], [1, 1], 1)
    assert pair.score == pair.companion == 0


def test_volume_accepts_exact_fraction_strings_and_fraction_objects():
    pair = volume_pair([[1], [1]], ["1/3", Fraction(2, 3)], [1], "1")
    assert pair.score == sp.Rational(1, 2) and pair.companion == 1


@pytest.mark.parametrize(
    "features,targets,outcome_features,outcome_target",
    [
        ([], [], [1], 0),
        ([[1], [1, 2]], [0, 1], [1], 0),
        ([[1, 0]], [0], [1, 0], 0),
        ([[1], [1]], [0], [1], 0),
        ([[1]], [0], [1, 2], 0),
        (1, [0], [1], 0),
        ([[1.0]], [0], [1], 0),
        ([[True]], [0], [1], 0),
        ([["1/0"]], [0], [1], 0),
        ([[1]], [0], [1], "invalid"),
        ([[1]], 0, [1], 0),
        ([[1]], [0], 1, 0),
        ([[1]], [0], [1], sp.oo),
    ],
)
def test_invalid_exact_inputs(features, targets, outcome_features, outcome_target):
    with pytest.raises(ValueError):
        volume_pair(features, targets, outcome_features, outcome_target)


@pytest.mark.parametrize(
    "target,degree", [([0, 0, 0, 0, 1], 1), ([0, 0, 1, 0, 0], 2), ([0, 0, 1, 0, 1], 3)]
)
def test_fixed_nuisance_different_targets_and_coefficient_identity(target, degree):
    nuisance = [[1, 0, 0], [1, 1, 0], [1, 2, 0], [0, 0, 1], [0, 0, 1]]
    result = primitive_degree(nuisance, target)
    assert result["q"] == degree
    assert result["exact_interior_identity_check"]
    p = [sp.Rational(i, 31) for i in [2, 3, 5, 8, 13]]

    def evaluate(terms):
        return sum(
            sp.Rational(term["polynomial_coefficient"])
            * sp.prod(prob**count for prob, count in zip(p, term["counts"], strict=True))
            for term in terms
        )

    design, f = sp.Matrix(nuisance), sp.Matrix(target)
    weighted = sp.diag(*p)
    residual = f - design * (design.T * weighted * design).inv() * design.T * weighted * f
    companion = evaluate(result["companion_score"])
    assert companion > 0
    assert [evaluate(poly) for poly in result["primitive_scores"]] == list(companion * residual)
    for polynomial in [*result["primitive_scores"], result["companion_score"]]:
        for term in polynomial:
            counts = term["counts"]
            assert sum(counts) == degree
            multinomial = sp.Rational(factorial(degree), sp.prod(factorial(c) for c in counts))
            assert sp.Rational(term["score_value"]) * multinomial == sp.Rational(
                term["polynomial_coefficient"]
            )


@pytest.mark.parametrize(
    "nuisance,target",
    [
        ([[1], [1]], [0]),
        ([[1], [1]], None),
        ([[1], [1]], [1, 1]),
        ([[1, 1], [1, 1]], [0, 1]),
        ([[0], [1]], [1, 0]),
    ],
)
def test_invalid_degree_designs(nuisance, target):
    with pytest.raises(ValueError):
        primitive_degree(nuisance, target)


@pytest.mark.parametrize("options", [[], ["-O"]])
def test_failed_identity_check_is_not_reported_as_success(options):
    code = """
from unittest.mock import patch
from pathtail import primitive_degree
with patch('pathtail.degree.sp.simplify', return_value=1):
    try:
        primitive_degree([[1], [1]], [0, 1])
    except ArithmeticError:
        pass
    else:
        raise SystemExit('Failed identity check was reported as success')
"""
    subprocess.run(
        [sys.executable, *options, "-c", code], check=True, capture_output=True, text=True
    )
