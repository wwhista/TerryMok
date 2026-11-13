"""Bounded loopback HTTP benchmark; no corpus text or external traffic."""

import argparse
import json
import math
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

MESSAGES = ["My Windows display driver crashes", "How can I upgrade Macintosh memory?",
            "How do I render a three dimensional image?", "My PC motherboard needs new RAM"]


def percentile(values, percent):
    """Nearest-rank percentile, explicitly defined even for small samples."""
    if not values or not 0 < percent <= 100:
        raise ValueError("Nonempty samples and percentile in (0, 100] required")
    return sorted(values)[math.ceil(len(values) * percent / 100) - 1]


def summarize(results, elapsed):
    latencies = [duration for duration, ok in results if ok]
    return {"requests": len(results), "successes": len(latencies),
            "failures": len(results) - len(latencies), "elapsed_seconds": elapsed,
            "successful_requests_per_second": len(latencies) / elapsed,
            "success_latency_ms": {f"p{p}": percentile(latencies, p) if latencies else None
                                   for p in (50, 95, 99)}}


def run_benchmark(port=8000, requests=100, concurrency=4, warmup=5):
    if not (1 <= port <= 65535 and 1 <= requests <= 10000 and 1 <= concurrency <= 32 and 0 <= warmup <= 100):
        raise ValueError("Invalid benchmark limits")
    base = f"http://127.0.0.1:{port}"
    with urlopen(base + "/health", timeout=10) as response:
        health = json.load(response)
    digest = health["model_sha256"]

    def send(index):
        payload = json.dumps({"texts": [MESSAGES[index % len(MESSAGES)]]}).encode()
        request = Request(base + "/predict", data=payload, headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            with urlopen(request, timeout=10) as response:
                body = json.load(response)
                predictions = body.get("predictions", [])
                ok = (response.status == 200 and body.get("model_sha256") == digest
                      and len(predictions) == 1 and predictions[0].get("label") in health["labels"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            ok = False
        return (time.perf_counter() - start) * 1000, ok

    for index in range(warmup):
        if not send(index)[1]:
            raise RuntimeError("Warmup request failed")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        start = time.perf_counter()
        results = list(pool.map(send, range(requests)))
        elapsed = time.perf_counter() - start
    report = summarize(results, elapsed)
    report.update(concurrency=concurrency, warmup_requests=warmup, batch_size=1,
                  model_sha256=digest, python_version=platform.python_version(),
                  platform=platform.system(), machine=platform.machine(),
                  workload="four fixed synthetic English messages; cyclic order",
                  methodology="closed-loop threads; fresh urllib HTTP connection per request; nearest-rank percentiles of successful end-to-end requests; warmup excluded")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    report = run_benchmark(**vars(args))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/api-benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
