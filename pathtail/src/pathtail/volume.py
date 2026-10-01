"""Exact rational volume pairs, including singular forecast designs."""

from math import prod
from numbers import Rational, Real

import sympy as sp

from .pairs import ScorePair


def rational(value) -> sp.Rational:
    """Accept exact rationals; require strings for decimal input."""
    if (
        isinstance(value, (bool, sp.Float))
        or isinstance(value, Real)
        and not isinstance(value, Rational)
    ):
        raise ValueError("Use integers or rational strings instead of floats or booleans")
    try:
        result = sp.Rational(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("Input entries must be finite rational numbers") from exc
    if not result.is_finite:
        raise ValueError("Input entries must be finite rational numbers")
    return result


def rational_matrix(rows) -> sp.Matrix:
    try:
        values = [list(row) for row in rows]
    except TypeError as exc:
        raise ValueError("Expected a nonempty rectangular matrix") from exc
    if not values or not values[0] or any(len(row) != len(values[0]) for row in values):
        raise ValueError("Expected a nonempty rectangular matrix")
    return sp.Matrix([[rational(value) for value in row] for row in values])


def volume_pair(features, targets, outcome_features, outcome_target) -> ScorePair[sp.Rational]:
    """Compute the adjugate construction divided by the falling factorial (M)_p.

    ``features`` contains one nuisance row per forecast; ``targets`` contains
    the corresponding target values. The realized sample Gram matrix may be
    singular. This reference implementation uses exact
    rational arithmetic rather than a numerically regularized inverse.
    """
    design = rational_matrix(features)
    m, rank = design.shape
    if m < rank:
        raise ValueError("Forecast budget must be at least the declared nuisance dimension")
    try:
        target = sp.Matrix([rational(value) for value in targets])
        row = [rational(value) for value in outcome_features]
    except TypeError as exc:
        raise ValueError("Targets and outcome features must be vectors") from exc
    if len(target) != m or len(row) != rank:
        raise ValueError("Target or outcome feature dimensions do not match the design")
    value = rational(outcome_target)
    gram = design.T * design
    determinant = gram.det()
    correction = (sp.Matrix([row]) * gram.adjugate() * design.T * target)[0]
    falling = prod(range(m - rank + 1, m + 1))
    return ScorePair((determinant * value - correction) / falling, determinant / falling)
