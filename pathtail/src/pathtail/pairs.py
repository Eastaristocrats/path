"""Shared pair representation and deterministic score normalization."""

from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

Value = TypeVar("Value")


@dataclass(frozen=True)
class ScorePair(Generic[Value]):
    """A score U and companion B satisfying a common expectation scale."""

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
    """Return (-U - tolerance*B, U - tolerance*B) / bound."""
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
