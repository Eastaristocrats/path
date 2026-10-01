import pytest

from pathtail import occupancy_pair


def apply(stream, **changes):
    options = dict(labels=[0, 1], outcome_label=0, outcome_value=0.7, quota=1, cap=5)
    options.update(changes)
    return occupancy_pair(stream, **options)


def test_stopping_uses_labels_and_does_not_consume_future_draws():
    stream = iter([(0, 0.2), (0, 0.4), (1, 0.9), (0, 0.8)])
    record = apply(stream)
    assert record.success and record.draws == 3
    assert record.pair.companion == 1
    assert record.pair.score == pytest.approx(0.4)
    assert next(stream) == (0, 0.8)
    changed_values = apply([(0, 0.8), (0, 0.9), (1, 0.1)])
    assert changed_values.draws == record.draws


def test_cap_failure_returns_zero_pair():
    record = apply([(0, 0.1)] * 5)
    assert not record.success and record.draws == 5
    assert record.pair.score == record.pair.companion == 0


def test_quota_and_success_exactly_at_cap():
    record = apply([(0, 0.1), (1, 0.2), (0, 0.3), (1, 0.4)], quota=2, cap=4)
    assert record.success and record.draws == 4
    assert record.pair.score == pytest.approx(0.5)


@pytest.mark.parametrize(
    "changes",
    [
        {"quota": 0},
        {"cap": 0},
        {"quota": True},
        {"cap": 2.5},
        {"cap": 1},
        {"labels": []},
        {"labels": [0, 0]},
        {"labels": [[0], [1]]},
        {"outcome_label": []},
        {"outcome_label": 2},
        {"outcome_value": 1.1},
        {"outcome_value": -0.1},
    ],
)
def test_invalid_rule(changes):
    with pytest.raises(ValueError):
        apply([], **changes)


@pytest.mark.parametrize("stream", [[], [(2, 0.2)], [([], 0.2)], [(0, 1.1)], [(0, float("nan"))]])
def test_invalid_or_truncated_stream(stream):
    with pytest.raises(ValueError):
        apply(stream)
