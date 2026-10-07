"""Primitive polynomial degree and exact count-score coefficients."""

import math
from collections.abc import Iterable
from itertools import combinations
from numbers import Integral
from typing import TypedDict

import sympy as sp

from .volume import ExactInput, rational, rational_matrix


class PolynomialTerm(TypedDict):
    """One degree-q monomial and its unbiased multinomial count value."""

    counts: list[int]
    polynomial_coefficient: str
    score_value: str


class DegreeResult(TypedDict):
    """JSON-compatible exact solver output; absent count vectors have score zero."""

    states: int
    rank: int
    q: int
    enumerated_subsets: int
    nonsingular_bases: int
    numerator_terms: int
    primitive_terms: int
    exact_interior_identity_check: bool
    exact_polynomial_identity_check: bool
    common_factor: str
    primitive_scores: list[list[PolynomialTerm]]
    companion_score: list[PolynomialTerm]
    coefficient_note: str


def primitive_degree(
    nuisance: Iterable[Iterable[ExactInput]],
    target: Iterable[ExactInput],
    *,
    max_subsets: int | None = 100_000,
) -> DegreeResult:
    """Solve the finite-domain primitive degree problem by exact enumeration.

    Rows of ``nuisance`` correspond to the declared states. Its columns must be
    independent, span constants, and exclude ``target``. Entries are exact
    rationals. The return value contains q, sparse primitive score polynomials
    and companion coefficients. Unlisted count vectors have score zero. The
    coefficients are for exactly q draws; extra draws can be ignored using a
    rule fixed before sampling. Both polynomial identities and an independent
    exact interior evaluation are checked.

    Enumeration is limited to ``max_subsets`` candidate bases. Raise this limit
    explicitly, or use None to disable it, for larger reference calculations.
    This bounds the number of bases, not the cost of symbolic algebra. Scores
    still need a common deterministic bound before betting.
    """
    design = rational_matrix(nuisance)
    try:
        values = sp.Matrix([rational(value) for value in target])
    except TypeError as exc:
        raise ValueError("target must be a vector") from exc
    n_states, rank = design.shape
    if len(values) != n_states:
        raise ValueError("target must have one value per state")
    if max_subsets is not None and (
        isinstance(max_subsets, bool) or not isinstance(max_subsets, Integral) or max_subsets < 1
    ):
        raise ValueError("max_subsets must be a positive integer or None")
    subset_count = math.comb(n_states, rank)
    if max_subsets is not None and subset_count > max_subsets:
        raise ValueError(
            f"Design requires {subset_count:,} subsets, exceeding max_subsets={max_subsets:,}; "
            "increase max_subsets explicitly if this calculation is intended"
        )
    if (
        design.rank() != rank
        or design.row_join(sp.ones(n_states, 1)).rank() != rank
        or design.row_join(values).rank() == rank
    ):
        raise ValueError("Need full rank, constants in the span, and target outside it.")

    # Cauchy-Binet expansion of the Gram determinant and residual numerators.
    symbols = sp.symbols(f"P0:{n_states}")
    determinant_terms = {}
    residual_terms = [{} for _ in range(n_states)]
    bases = 0
    for subset in combinations(range(n_states), rank):
        basis = design[list(subset), :]
        determinant = basis.det()
        if determinant == 0:
            continue
        bases += 1
        powers = tuple(int(j in subset) for j in range(n_states))
        determinant_terms[powers] = determinant**2
        residual = values - design * basis.inv() * values[list(subset), :]
        for state in range(n_states):
            value = determinant**2 * residual[state]
            if value:
                residual_terms[state][powers] = value
    numerators = [sp.Poly.from_dict(row, symbols, domain=sp.QQ) for row in residual_terms]
    nonzero = [poly for poly in numerators if not poly.is_zero]

    # Remove the common factor to obtain the target-specific primitive degree.
    common = nonzero[0]
    for value in nonzero[1:]:
        common = sp.gcd(common, value)
    primitive = [poly.exquo(common) for poly in numerators]
    companion = sp.Poly.from_dict(determinant_terms, symbols, domain=sp.QQ).exquo(common)
    degree = next(poly.total_degree() for poly in primitive if not poly.is_zero)
    point = {symbol: sp.Rational(1, n_states) for symbol in symbols}
    if companion.as_expr().subs(point) < 0:
        primitive = [-poly for poly in primitive]
        companion, common = -companion, -common

    # Verify the defining projection identities as polynomials, for all P:
    # Psi.T diag(P) V = 0 and V - a*f belongs to the column space of Psi.
    zero = sp.Poly(0, *symbols, domain=sp.QQ)
    orthogonal = all(
        sum(
            (
                design[state, column] * symbols[state] * primitive[state]
                for state in range(n_states)
            ),
            zero,
        ).is_zero
        for column in range(rank)
    )
    in_nuisance_span = all(
        sum(
            (
                vector[state] * (primitive[state] - values[state] * companion)
                for state in range(n_states)
            ),
            zero,
        ).is_zero
        for vector in design.T.nullspace()
    )
    if not (orthogonal and in_nuisance_span):
        raise ArithmeticError("Primitive polynomial projection identity check failed")

    # An exact interior evaluation checks the independent matrix representation.
    weights = [sp.Rational(i + 1, n_states * (n_states + 1) // 2) for i in range(n_states)]
    weighted = sp.diag(*weights)
    residual = values - design * (design.T * weighted * design).inv() * design.T * weighted * values
    at = dict(zip(symbols, weights, strict=True))
    identity_holds = all(
        sp.simplify(poly.as_expr().subs(at) - companion.as_expr().subs(at) * residual[state]) == 0
        for state, poly in enumerate(primitive)
    )
    degrees_match = all(poly.is_zero or poly.total_degree() == degree for poly in primitive)
    if not (identity_holds and degrees_match and companion.total_degree() == degree <= rank):
        raise ArithmeticError("Primitive polynomial identity or degree check failed")

    # Convert polynomial coefficients to scores for multinomial count vectors.
    def serialize(poly: sp.Poly) -> list[PolynomialTerm]:
        data = []
        for powers, coefficient in poly.terms():
            if coefficient == 0:
                continue
            multinomial = math.factorial(degree) // math.prod(math.factorial(c) for c in powers)
            data.append(
                {
                    "counts": list(powers),
                    "polynomial_coefficient": str(coefficient),
                    "score_value": str(coefficient / multinomial),
                }
            )
        return data

    return {
        "states": n_states,
        "rank": rank,
        "q": int(degree),
        "enumerated_subsets": subset_count,
        "nonsingular_bases": bases,
        "numerator_terms": sum(len(poly.terms()) for poly in nonzero),
        "primitive_terms": sum(len(poly.terms()) for poly in primitive if not poly.is_zero),
        "exact_interior_identity_check": True,
        "exact_polynomial_identity_check": True,
        "common_factor": str(common.as_expr()),
        "primitive_scores": [serialize(poly) for poly in primitive],
        "companion_score": serialize(companion),
        "coefficient_note": (
            "Divide listed scores by a common finite absolute bound before using bounded betting factors."
        ),
    }
