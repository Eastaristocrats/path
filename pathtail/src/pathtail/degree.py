"""Primitive polynomial degree and exact count-score coefficients."""

import math
from itertools import combinations

import sympy as sp

from .volume import rational, rational_matrix


def primitive_degree(nuisance, target) -> dict:
    """Solve the finite-domain primitive degree problem by exact enumeration.

    Rows of ``nuisance`` correspond to the declared states. Its columns must be
    independent, span constants, and exclude ``target``. Entries are exact
    rationals. The return value contains q, sparse primitive score polynomials
    and companion coefficients. Enumeration can be expensive for large designs.
    Coefficients still need a common deterministic bound before betting.
    """
    design = rational_matrix(nuisance)
    try:
        values = sp.Matrix([rational(value) for value in target])
    except TypeError as exc:
        raise ValueError("target must be a vector") from exc
    n_states, rank = design.shape
    if len(values) != n_states:
        raise ValueError("target must have one value per state")
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
    def serialize(poly):
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
        "enumerated_subsets": math.comb(n_states, rank),
        "nonsingular_bases": bases,
        "numerator_terms": sum(len(poly.terms()) for poly in nonzero),
        "primitive_terms": sum(len(poly.terms()) for poly in primitive if not poly.is_zero),
        "exact_interior_identity_check": True,
        "common_factor": str(common.as_expr()),
        "primitive_scores": [serialize(poly) for poly in primitive],
        "companion_score": serialize(companion),
        "coefficient_note": (
            "Divide listed scores by a common finite absolute bound before using bounded betting factors."
        ),
    }
