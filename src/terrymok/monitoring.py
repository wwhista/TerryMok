"""Bounded per-process counters; never store request text or arbitrary labels."""

import threading
import time


class Metrics:
    bounds = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0)

    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.active = 0
        self.completed = 0
        self.statuses = {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 0}
        self.duration_sum = 0.0
        self.buckets = [0] * len(self.bounds)

    def begin(self):
        with self.lock:
            self.active += 1

    def finish(self, status, seconds):
        with self.lock:
            self.active -= 1
            self.completed += 1
            self.statuses[f"{min(5, max(2, status // 100))}xx"] += 1
            self.duration_sum += seconds
            for index, bound in enumerate(self.bounds):
                self.buckets[index] += int(seconds <= bound)

    def snapshot(self):
        with self.lock:
            return {"uptime_seconds": time.monotonic() - self.started,
                    "in_flight": self.active, "completed": self.completed,
                    "status_classes": dict(self.statuses),
                    "duration_seconds_sum": self.duration_sum,
                    "duration_buckets_cumulative": dict(zip(map(str, self.bounds), self.buckets)),
                    "duration_count": self.completed}
