import pytest
from terrymok.alerts import warnings_for
from terrymok.interval import interval
from terrymok.monitoring import Metrics


def test_synthetic_stream_and_restart():
    m = Metrics()
    def snap(t):
        return dict(m.snapshot(), uptime_seconds=t, model_sha256='fixture', policy_sha256='fixture')
    first = snap(1)
    m.observe_predictions(['word ' * 25] * 100, [False] * 100)
    second = snap(2)
    assert warnings_for(interval(first, second), .4, .7)['warnings'] == []
    m.observe_predictions(['short'] * 100, [True] * 100)
    third = snap(3)
    report = warnings_for(interval(second, third), .4, .7)
    assert [w['signal'] for w in report['warnings']] == ['review', 'short_messages']
    third['instance_id'] = 'restarted'
    assert warnings_for(interval(second, third), .4, .7)['status'] == 'not_evaluated'


def test_support_and_exact_boundary():
    result = dict(status='ok', messages=100, policy_messages=99, review_fraction=1, short_fraction=.7)
    report = warnings_for(result, .4, .7)
    assert report['warnings'] == []
    assert report['checks']['review'] == 'insufficient_samples'
    result['policy_messages'] = 100
    assert len(warnings_for(result, .4, .7)['warnings']) == 1


@pytest.mark.parametrize('limit', [-1, 2, float('nan'), True])
def test_invalid_limits(limit):
    with pytest.raises(ValueError):
        warnings_for({'status': 'baseline_required'}, limit, .7)
