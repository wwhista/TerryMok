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
