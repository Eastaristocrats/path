"""Count-based pairs for f=x1*x2 and nuisance span(1, x1, x2)."""

import math
from fractions import Fraction
from functools import lru_cache
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .pairs import ScorePair, nonnegative

Method = Literal["monomial", "pooled"]
CHI = np.array([1.0, -1.0, -1.0, 1.0])


def _method(method: str) -> None:
    if method not in ("monomial", "pooled"):
        raise ValueError("method must be 'monomial' or 'pooled'")


def _budget(budget: int) -> int:
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 3:
        raise ValueError("budget must be an integer of at least three")
    # Prevent silent overflow in count accumulation and conversion to float64.
    if budget > 2**31 - 1:
        raise ValueError("budget exceeds the supported integer range")
    return int(budget)


def binary_pair(
    counts: ArrayLike, outcome: ArrayLike, *, method: Method = "monomial"
) -> ScorePair[NDArray[np.float64]]:
    """Compute U and B from counts in state order 00, 01, 10, 11.

    Counts have shape (..., 4), are nonnegative integers and share one fixed
    forecast budget M >= 3. Outcomes are integer state indices broadcastable to
    the batch shape.
    """
    _method(method)
    if np.ma.isMaskedArray(counts) or np.ma.isMaskedArray(outcome):
        raise ValueError("Masked counts and outcomes are not supported")
    c = np.asarray(counts)
    if c.ndim < 1 or c.shape[-1] != 4 or c.size == 0 or c.dtype.kind not in "iu":
        raise ValueError("counts must be a nonempty integer array with last dimension four")
    if np.any(c < 0) or np.any(c > 2**31 - 1):
        raise ValueError("counts must be nonnegative and within the supported integer range")
    totals = c.astype(np.int64).sum(axis=-1)
    m = _budget(int(totals.flat[0]))
    if np.any(totals != m):
        raise ValueError("All batches must share the same forecast budget")
    y = np.asarray(outcome)
    if y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y > 3):
        raise ValueError("outcome must contain integer state indices between zero and three")
    try:
        y = np.broadcast_to(y, c.shape[:-1]).astype(np.intp)
    except ValueError as exc:
        raise ValueError("outcome is not broadcastable to the count batch shape") from exc
    c = c.astype(np.float64)
    a = np.empty(c.shape, dtype=np.float64)
    for state in range(4):
        if method == "monomial":
            a[..., state] = np.prod(c[..., np.arange(4) != state], axis=-1) / (
                m * (m - 1) * (m - 2)
            )
        else:
            pooled = c.copy()
            pooled[..., state] += 1
            a[..., state] = pooled.min(axis=-1) / pooled[..., state]
    u = CHI[y] * np.take_along_axis(a, y[..., None], axis=-1)[..., 0]
    return ScorePair(np.asarray(u), np.asarray(a.sum(axis=-1)))


@lru_cache(maxsize=128)
def _tight_bound(m: int, method: Method, tau: Fraction) -> Fraction:
    denominator = m * (m - 1) * (m - 2)
    maximum = Fraction(0)
    for c0 in range(m + 1):
        for c1 in range(m - c0 + 1):
            for c2 in range(m - c0 - c1 + 1):
                counts = (c0, c1, c2, m - c0 - c1 - c2)
                coefficients = []
                for state in range(4):
                    if method == "monomial":
                        product = math.prod(counts[j] for j in range(4) if j != state)
                        value = Fraction(product, denominator)
                    else:
                        pooled = list(counts)
                        pooled[state] += 1
                        value = Fraction(min(pooled), pooled[state])
                    coefficients.append(value)
                maximum = max(maximum, max(coefficients) + tau * sum(coefficients))
    return maximum


def binary_bound(
    budget: int,
    *,
    method: Method = "monomial",
    tolerance: float = 0.0,
    normalization: Literal["tight", "conservative"] = "tight",
) -> float:
    """Bound |±U - tolerance*B|, using exact enumeration or a conservative formula.

    Tight bounds enumerate at most 100,000 count vectors and are cached. For
    larger budgets choose 'conservative', which uses B <= 4*max_y|U_y|.
    Rational bounds are rounded outward when converted to float64.
    """
    m = _budget(budget)
    _method(method)
    tau = Fraction(str(nonnegative(tolerance, "tolerance")))
    if normalization not in ("tight", "conservative"):
        raise ValueError("normalization must be 'tight' or 'conservative'")
    if method == "monomial":
        q, r = divmod(m, 3)
        maximum = Fraction(q ** (3 - r) * (q + 1) ** r, m * (m - 1) * (m - 2))
    else:
        maximum = Fraction(1)
    if normalization == "conservative" or tau == 0:
        exact = maximum * (1 + 4 * tau)
    else:
        if math.comb(m + 3, 3) > 100_000:
            raise ValueError("Tight enumeration exceeds 100,000 vectors; use 'conservative'")
        exact = _tight_bound(m, method, tau)
    try:
        limit = float(exact)
    except OverflowError as exc:
        raise ValueError("tolerance is too large for finite normalization") from exc
    if Fraction.from_float(limit) < exact:
        limit = math.nextafter(limit, math.inf)
    if not np.isfinite(limit):
        raise ValueError("tolerance is too large for finite normalization")
    return limit
