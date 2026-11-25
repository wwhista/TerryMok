"""Bounded per-process counters; never store request text or arbitrary labels."""

import threading
import time
import uuid


class Metrics:
    bounds = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0)

    def __init__(self):
        self.lock = threading.Lock()
        self.instance_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.active = 0
        self.completed = 0
        self.statuses = {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 0}
        self.duration_sum = 0.0
        self.buckets = [0] * len(self.bounds)
        self.messages = 0
        self.characters = 0
        self.short_messages = 0
        self.policy_messages = 0
        self.review_messages = 0

    def observe_predictions(self, texts, review_flags=None):
        count = len(texts)
        characters = sum(len(text) for text in texts)
        short = sum(len(text.split()) <= 20 for text in texts)
        if review_flags is not None and len(review_flags) != count:
            raise ValueError("Review flags must match messages")
        with self.lock:
            self.messages += count
            self.characters += characters
            self.short_messages += short
            if review_flags is not None:
                self.policy_messages += count
                self.review_messages += sum(review_flags)

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
            return {"instance_id": self.instance_id, "uptime_seconds": time.monotonic() - self.started,
                    "in_flight": self.active, "completed": self.completed,
                    "status_classes": dict(self.statuses),
                    "duration_seconds_sum": self.duration_sum,
                    "duration_buckets_cumulative": dict(zip(map(str, self.bounds), self.buckets)),
                    "duration_count": self.completed,
                    "predictions": {"messages": self.messages,
                        "characters_sum": self.characters,
                        "short_messages_le_20_words": self.short_messages,
                        "policy_messages": self.policy_messages,
                        "review_messages": self.review_messages,
                        "review_fraction": self.review_messages / self.policy_messages if self.policy_messages else None}}
