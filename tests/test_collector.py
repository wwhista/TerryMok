from copy import deepcopy
from terrymok.collector import Collector, poll
from terrymok.monitoring import Metrics


def test_streaks_recovery_and_restart():
    m = Metrics()
    c = Collector(.4, .7, min_messages=2, consecutive=2)
    def snapshot(t):
        return dict(m.snapshot(), uptime_seconds=t, model_sha256='m', policy_sha256='p')
    assert c.observe(snapshot(1))['status'] == 'baseline'
    for t in (2, 3):
        m.observe_predictions(['short'] * 2, [True] * 2)
        result = c.observe(snapshot(t))
        assert len(result['active_warnings']) == (0 if t == 2 else 2)
    m.observe_predictions(['word ' * 21] * 2, [False] * 2)
    assert c.observe(snapshot(4))['active_warnings'] == []
    changed = snapshot(5)
    changed['instance_id'] = 'new'
    assert c.observe(changed)['status'] == 'baseline_required'
    assert all(v == 0 for v in c.streaks.values())


def test_fetch_failure_discards_baseline_and_never_leaks_error_text():
    m = Metrics()
    snapshot = dict(m.snapshot(), uptime_seconds=1, model_sha256='m', policy_sha256=None)
    calls = iter([snapshot, OSError('private endpoint'), snapshot])
    def fetch():
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)
    results = list(poll(Collector(.4, .7), fetch, 3, 0, sleep=lambda _: None))
    assert [r['status'] for r in results] == ['baseline', 'collection_error', 'baseline']
    assert 'private endpoint' not in str(results)


def test_insufficient_samples_break_streak():
    m = Metrics()
    c = Collector(.4, .7, min_messages=2, consecutive=2)
    for t, count in [(1, 0), (2, 2), (3, 1), (4, 2)]:
        m.observe_predictions(['short'] * count, [True] * count)
        result = c.observe(dict(m.snapshot(), uptime_seconds=t, model_sha256='m', policy_sha256='p'))
        assert result['active_warnings'] == []
