from concurrent.futures import ThreadPoolExecutor

from terrymok.monitoring import Metrics


def test_concurrent_totals_and_cumulative_histogram():
    metrics = Metrics()
    def record(_):
        metrics.begin()
        metrics.finish(422, 0.02)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record, range(100)))
    result = metrics.snapshot()
    assert result["completed"] == 100
    assert result["in_flight"] == 0
    assert result["status_classes"]["4xx"] == 100
    assert result["duration_buckets_cumulative"]["0.01"] == 0
    assert result["duration_buckets_cumulative"]["0.025"] == 100


def test_prediction_counters_are_atomic_and_do_not_retain_text():
    from concurrent.futures import ThreadPoolExecutor
    import json
    metrics = Metrics()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: metrics.observe_predictions(['private text', 'word ' * 21], [True, False]), range(100)))
    snapshot = metrics.snapshot()['predictions']
    assert snapshot['messages'] == 200
    assert snapshot['short_messages_le_20_words'] == 100
    assert snapshot['review_messages'] == 100
    assert snapshot['review_fraction'] == 0.5
    metrics.observe_predictions(['other'])
    assert metrics.snapshot()['predictions']['policy_messages'] == 200
    assert 'private text' not in json.dumps(metrics.snapshot())
    assert not any(isinstance(value, str) and 'private text' in value for value in vars(metrics).values())
