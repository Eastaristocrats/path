"""Shared pair representation and deterministic score normalization."""

from dataclasses import dataclass
from decimal import Decimal
from numbers import Real

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ScorePair[Value]:
    """Store score U and companion B; their expectation identity is a precondition.

    For a valid construction E[U | history] = c * Delta and E[B | history] = c
    with c >= 0. General algebraic companions need not be nonnegative on each
    realization. This container does not verify a probabilistic model.
    """

    score: Value
    companion: Value


def nonnegative(value: float, name: str) -> float:
    """Validate a finite scalar parameter."""
    scalar = real_array(value, name)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    number = float(scalar)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return number


def real_array(value, name: str) -> NDArray[np.float64]:
    """Convert real numerical inputs without silently discarding imaginary parts."""
    try:
        if np.ma.isMaskedArray(value):
            raise ValueError("Masked inputs are not supported")
        raw = np.asarray(value)
        if raw.dtype.kind == "O":
            for item in raw.flat:
                if isinstance(item, (bool, np.bool_)) or not isinstance(item, (Real, Decimal)):
                    raise ValueError("Object arrays must contain real numerical scalars")
        elif raw.dtype.kind not in "iuf":
            raise ValueError("Expected real numerical values")
        return raw.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numerical values") from exc


def normalize_pair(pair: ScorePair, *, tolerance: float = 0.0, bound: float) -> NDArray[np.float64]:
    """Return (-U - tolerance*B, U - tolerance*B) / bound, shape (..., 2).

    ``bound`` must be fixed before current inference data and bound both signed
    numerators for every possible realization. This function checks only the
    observed range. A numerical overshoot up to 1e-12 is clipped to [-1, 1];
    float64 evaluation is not a certified interval-arithmetic guarantee.
    """
    tau = nonnegative(tolerance, "tolerance")
    limit = nonnegative(bound, "bound")
    if limit == 0:
        raise ValueError("bound must be positive")
    try:
        u, b = np.broadcast_arrays(
            real_array(pair.score, "Score"),
            real_array(pair.companion, "Companion"),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Score and companion must be real, broadcastable arrays") from exc
    if not np.isfinite(u).all() or not np.isfinite(b).all():
        raise ValueError("Score and companion must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        z = (np.stack((-u, u), axis=-1) - tau * b[..., None]) / limit
    if not np.isfinite(z).all() or np.any(np.abs(z) > 1 + 1e-12):
        raise ValueError("The declared bound does not contain the realized scores")
    return np.clip(z, -1.0, 1.0)
