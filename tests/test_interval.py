from copy import deepcopy
import pytest
from terrymok.monitoring import Metrics
from terrymok.interval import interval


def snapshots():
    m = Metrics()
    a = dict(m.snapshot(), model_sha256='model', policy_sha256='policy')
    m.observe_predictions(['a', 'word ' * 21], [True, False])
    b = dict(m.snapshot(), model_sha256='model', policy_sha256='policy')
    a['uptime_seconds'], b['uptime_seconds'] = 1, 3
    return a, b


def test_interval_uses_deltas():
    a, b = snapshots()
    r = interval(a, b)
    assert r['messages_per_second'] == 1
    assert r['review_fraction'] == r['short_fraction'] == 0.5
    assert r['mean_characters'] == 53
    c = deepcopy(b)
    c['uptime_seconds'] = 5
    assert interval(b, c)['review_fraction'] is None
    assert interval(b, c)['mean_characters'] is None


@pytest.mark.parametrize('key', ['instance_id', 'model_sha256', 'policy_sha256'])
def test_identity_changes_need_new_baseline(key):
    a, b = snapshots()
    b[key] = 'different'
    assert interval(a, b)['status'] == 'baseline_required'


def test_reset_and_invalid_counters():
    a, b = snapshots()
    assert Metrics().instance_id != Metrics().instance_id
    assert interval(b, a)['status'] == 'baseline_required'
    a['predictions']['messages'] = 10
    assert interval(a, b)['status'] == 'baseline_required'
    a['predictions']['messages'] = -1
    with pytest.raises(ValueError):
        interval(a, b)
