import math

import numpy as np
import pytest

from pathtail import MixtureEProcess, ScorePair, normalize_pair


def test_online_log_mixture_matches_direct_products_at_every_prefix():
    stakes = np.array([0.1, 0.4, 0.8])
    process = MixtureEProcess(stakes)
    assert process.log_e_value == pytest.approx(0)
    assert process.updates == 0 and process.first_crossing is None and not process.crossed
    direct = np.ones((2, len(stakes)))
    for step, z in enumerate([[-0.2, 0.1], [0.3, -0.6], [-0.8, 0.5]], start=1):
        direct *= 1 + np.asarray(z)[:, None] * stakes
        assert process.update(z) == pytest.approx(math.log(direct.mean()))
        assert process.updates == step


def test_crossing_is_remembered_after_evidence_falls():
    process = MixtureEProcess([0.9], alpha=0.4)
    for _ in range(2):
        process.update([-1, 1])
        assert process.first_crossing is None
    process.update([-1, 1])
    assert process.first_crossing == 3
    process.update([-1, -1])
    assert process.log_e_value < math.log(1 / 0.4)
    assert process.crossed and process.first_crossing == 3


@pytest.mark.parametrize("stake,alpha", [(0.2, 0.5), (0.5, 0.2), (0.8, 0.05)])
def test_first_crossing_matches_independent_product_calculation(stake, alpha):
    process = MixtureEProcess([stake], alpha=alpha)
    expected = next(
        t for t in range(1, 100) if ((1 + stake) ** t + (1 - stake) ** t) / 2 >= 1 / alpha
    )
    for t in range(1, expected + 2):
        process.update([-1, 1])
        assert process.first_crossing == (None if t < expected else expected)


def test_stakes_are_copied_and_log_arithmetic_stays_finite():
    stakes = np.array([0.5])
    process = MixtureEProcess(stakes)
    stakes[0] = 0
    for _ in range(2000):
        process.update([-1, 1])
    assert np.isfinite(process.log_e_value) and process.log_e_value > 700


def test_pair_update_has_declared_tolerance_signs():
    pair = ScorePair(0.25, 0.5)
    np.testing.assert_allclose(normalize_pair(pair, tolerance=0.1, bound=0.5), [-0.6, 0.4])
    left, right = MixtureEProcess(), MixtureEProcess()
    assert left.update_pair(pair, tolerance=0.1, bound=0.5) == right.update([-0.6, 0.4])


@pytest.mark.parametrize("alpha", [0, 1, -0.1, np.nan, np.inf, True, "0.1", [0.1], 1j, object()])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        MixtureEProcess(alpha=alpha)


@pytest.mark.parametrize("stakes", [[], [[0.1]], [1], [-0.1], [np.nan], [np.inf], [1j], [True]])
def test_invalid_stakes(stakes):
    with pytest.raises(ValueError):
        MixtureEProcess(stakes)


@pytest.mark.parametrize("scores", [[0], [0, 0, 0], [np.nan, 0], [1.01, 0], [-1.01, 0], [1j, 0]])
def test_invalid_update_is_atomic(scores):
    process = MixtureEProcess()
    old = process.log_e_value
    with pytest.raises(ValueError):
        process.update(scores)
    assert process.log_e_value == old and process.updates == 0 and not process.crossed


@pytest.mark.parametrize(
    "pair,tolerance,bound",
    [
        (ScorePair(0, 1), 0, 0),
        (ScorePair(0, 1), -1, 1),
        (ScorePair(np.nan, 1), 0, 1),
        (ScorePair(0, np.inf), 0, 1),
        (ScorePair([1, 2], [1, 2, 3]), 0, 1),
        (ScorePair("bad", 1), 0, 1),
        (ScorePair(2, 1), 0, 1),
        (ScorePair(1e308, 1e308), 1e308, 1),
    ],
)
def test_invalid_pair_inputs(pair, tolerance, bound):
    with pytest.raises(ValueError):
        normalize_pair(pair, tolerance=tolerance, bound=bound)


def test_batched_normalization_and_roundoff_boundary():
    z = normalize_pair(ScorePair([0.2, -0.3], 1), tolerance=0.1, bound=1)
    np.testing.assert_allclose(z, [[-0.3, 0.1], [0.2, -0.4]])
    z = normalize_pair(ScorePair(1 + 1e-14, 0), bound=1)
    np.testing.assert_array_equal(z, [-1, 1])


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([np.complex128(0.2 + 0.7j), 0], dtype=object),
        np.array(["1970-01-01", "1970-01-02"], dtype="datetime64[D]"),
        np.array([0, 1], dtype="timedelta64[D]"),
        np.array([True, False], dtype=object),
        np.array(["0.1", "0.2"], dtype=object),
        np.ma.array([0.1, 0.2], mask=[False, True]),
    ],
)
def test_nonreal_or_missing_inputs_are_rejected_without_state_changes(invalid):
    process = MixtureEProcess()
    with pytest.raises(ValueError):
        process.update(invalid)
    assert process.updates == 0
    with pytest.raises(ValueError):
        normalize_pair(ScorePair(invalid, 0), bound=1)


@pytest.mark.parametrize("value", [np.bool_(True), np.array(True), np.datetime64("1970-01-02")])
def test_scalar_parameters_reject_boolean_and_datetime(value):
    with pytest.raises(ValueError):
        normalize_pair(ScorePair(0, 1), bound=1, tolerance=value)


def test_object_rationals_remain_supported():
    from decimal import Decimal
    from fractions import Fraction

    import sympy as sp

    for value in (Fraction(1, 4), Decimal("0.25"), sp.Rational(1, 4)):
        np.testing.assert_allclose(normalize_pair(ScorePair(value, 1), bound=1), [-0.25, 0.25])
