"""Online equal mixtures of fixed nonnegative betting factors."""

import math
from collections.abc import Sequence

import numpy as np

from .pairs import ScorePair, nonnegative, normalize_pair, real_array


class MixtureEProcess:
    """Mix two score signs and fixed stakes in log space.

    Each normalized score must have conditional mean at most zero under the
    declared null, given the history before its update. The two-sided pair tests
    |Delta_t| <= tolerance at every update. Stakes and alpha are fixed at
    construction; targets, bounds and any time-varying tolerance must be chosen
    before the current inference data. Range validation alone cannot establish
    these statistical assumptions. Keep one instance for a full audit stream.
    """

    def __init__(
        self,
        stakes: Sequence[float] = (0.05, 0.1, 0.2, 0.4, 0.8, 0.95),
        *,
        alpha: float = 0.05,
    ) -> None:
        alpha = nonnegative(alpha, "alpha")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        values = real_array(stakes, "stakes")
        if (
            values.ndim != 1
            or values.size == 0
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values >= 1)
        ):
            raise ValueError("stakes must be a nonempty finite sequence in [0, 1)")
        self._stakes = values.copy()
        self._capital = np.zeros((2, len(values)))
        self._boundary = -math.log(alpha)
        self._updates = 0
        self._first_crossing: int | None = None

    @property
    def log_e_value(self) -> float:
        """Natural log of the equal mixture over signs and stakes."""
        return float(np.logaddexp.reduce(self._capital.ravel()) - math.log(self._capital.size))

    @property
    def updates(self) -> int:
        """Number of successfully consumed updates, including neutral pairs."""
        return self._updates

    @property
    def first_crossing(self) -> int | None:
        """First one-based update reaching 1/alpha, or None if never reached."""
        return self._first_crossing

    @property
    def crossed(self) -> bool:
        """Whether the process has ever crossed, even if current evidence fell."""
        return self._first_crossing is not None

    def update(self, normalized_scores: Sequence[float]) -> float:
        """Consume one (negative sign, positive sign) update; return log e-value."""
        z = real_array(normalized_scores, "normalized_scores")
        if z.shape != (2,) or not np.isfinite(z).all() or np.any(np.abs(z) > 1):
            raise ValueError("Provide two finite normalized scores in [-1, 1]")
        with np.errstate(over="raise", invalid="raise"):
            capital = self._capital + np.log1p(z[:, None] * self._stakes[None, :])
        self._capital = capital
        self._updates += 1
        value = self.log_e_value
        if self._first_crossing is None and value >= self._boundary:
            self._first_crossing = self._updates
        return value

    def update_pair(self, pair: ScorePair, *, bound: float, tolerance: float = 0.0) -> float:
        """Consume one scalar pair with a bound fixed before its inference data.

        The bound must contain both signed numerators over all possible inputs,
        not just this realization. A failed validation leaves state unchanged.
        """
        return self.update(normalize_pair(pair, tolerance=tolerance, bound=bound))
