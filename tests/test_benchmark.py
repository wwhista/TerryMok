import pytest

from terrymok.benchmark import percentile, run_benchmark, summarize


def test_nearest_rank_and_failure_accounting():
    assert percentile(list(range(1, 101)), 95) == 95
    report = summarize([(10, True), (20, True), (999, False)], 2)
    assert report["success_latency_ms"] == {"p50": 10, "p95": 20, "p99": 20}
    assert report["failures"] == 1
    assert report["successful_requests_per_second"] == 1


def test_all_failed_requests_have_no_latency_claim():
    report = summarize([(1000, False)], 1)
    assert report["success_latency_ms"]["p95"] is None
    assert report["successes"] == 0


def test_invalid_limits_fail_before_network_access():
    for kwargs in [{"requests": 0}, {"concurrency": 33}, {"port": 0}, {"warmup": -1}]:
        with pytest.raises(ValueError):
            run_benchmark(**kwargs)


def test_mismatched_model_and_network_error_are_counted(monkeypatch):
    import io
    import json
    from terrymok import benchmark

    class Response(io.StringIO):
        status = 200

    calls = iter(["expected", "changed", None])

    def respond(request, **kwargs):
        if isinstance(request, str):
            return Response(json.dumps({"model_sha256": "expected", "labels": ["graphics"]}))
        digest = next(calls)
        if digest is None:
            raise OSError("connection failed")
        return Response(json.dumps({"model_sha256": digest, "predictions": [{"label": "graphics"}]}))

    monkeypatch.setattr(benchmark, "urlopen", respond)
    report = run_benchmark(requests=3, concurrency=1, warmup=0)
    assert report["successes"] == 1
    assert report["failures"] == 2


def test_timed_batch_workload(monkeypatch):
    import io
    import json
    from terrymok import benchmark

    class Response(io.StringIO):
        status = 200

    def respond(request, **kwargs):
        if isinstance(request, str):
            return Response(json.dumps({"model_sha256": "expected", "labels": ["graphics"]}))
        texts = json.loads(request.data)["texts"]
        assert len(texts) == 2
        assert all(len(text) > 50 for text in texts)
        return Response(json.dumps({"model_sha256": "expected", "predictions": [{"label": "graphics"}] * 2}))

    monkeypatch.setattr(benchmark, "urlopen", respond)
    report = run_benchmark(duration=0.02, concurrency=1, warmup=0, batch_size=2, text_repeat=3)
    assert report["requests"] > 0
    assert report["failures"] == 0
    assert report["elapsed_seconds"] >= 0.02
