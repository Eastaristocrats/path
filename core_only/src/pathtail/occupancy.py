"""Label-only occupancy stopping for partition nuisance spaces."""

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral

from .pairs import ScorePair, nonnegative


@dataclass(frozen=True)
class OccupancyResult:
    """Stopped pair, actual consumed forecast count, and quota completion flag."""

    pair: ScorePair[float]
    draws: int
    success: bool


def _unit_value(value: float) -> float:
    number = nonnegative(value, "target value")
    if number > 1:
        raise ValueError("Target values must lie in [0, 1]")
    return number


def occupancy_pair(
    forecasts: Iterable[tuple[Hashable, float]],
    *,
    labels: Sequence[Hashable],
    outcome_label: Hashable,
    outcome_value: float,
    quota: int = 1,
    cap: int,
) -> OccupancyResult:
    """Stop when every declared cell reaches quota, or at the fixed cap.

    Each forecast is (partition label, bounded target value). Only labels affect
    stopping. On success U=f(Y)-within-cell mean and B=1; at the cap without
    success U=B=0. ``1 + tolerance`` bounds both normalized score numerators.
    A prematurely exhausted stream is an error, not an allowed stopping rule.
    Fix the target, partition, quota and cap before drawing forecasts. The
    residual is relative to the full span of the partition indicators; this
    does not implement stopping for an arbitrary linear nuisance space.
    """
    for value in (quota, cap):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError("quota and cap must be positive integers")
    try:
        cells = tuple(labels)
        counts = dict.fromkeys(cells, 0)
        valid_outcome = outcome_label in counts
    except TypeError as exc:
        raise ValueError("Provide an iterable of hashable partition labels") from exc
    if not cells or len(counts) != len(cells) or not valid_outcome:
        raise ValueError("Declare distinct nonempty labels including the outcome label")
    if cap < quota * len(cells):
        raise ValueError("cap must allow the declared occupancy quota")
    target = _unit_value(outcome_value)
    sums = dict.fromkeys(cells, 0.0)
    try:
        stream = iter(forecasts)
    except TypeError as exc:
        raise ValueError("forecasts must be an iterable of (label, value) pairs") from exc
    for draw in range(1, cap + 1):
        try:
            item = next(stream)
        except StopIteration as exc:
            raise ValueError("Forecast stream ended before the declared stopping rule") from exc
        try:
            label, value = item
        except (TypeError, ValueError) as exc:
            raise ValueError("Each forecast must be a (label, value) pair") from exc
        try:
            known = label in counts
        except TypeError as exc:
            raise ValueError("Forecast partition label must be hashable") from exc
        if not known:
            raise ValueError("Forecast label is outside the declared partition")
        value = _unit_value(value)
        counts[label] += 1
        sums[label] += value
        if all(count >= quota for count in counts.values()):
            mean = sums[outcome_label] / counts[outcome_label]
            return OccupancyResult(ScorePair(target - mean, 1.0), draw, True)
    return OccupancyResult(ScorePair(0.0, 0.0), cap, False)
