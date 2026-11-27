"""Bounded localhost metrics polling with consecutive-window warning checks."""
import argparse
import json
import time
from urllib.request import urlopen
from terrymok.alerts import warnings_for
from terrymok.interval import interval, validate_snapshot


class Collector:
    def __init__(self, review_limit, short_limit, min_messages=100, consecutive=3):
        warnings_for({'status': 'baseline_required'}, review_limit, short_limit, min_messages)
        if type(consecutive) is not int or consecutive < 1:
            raise ValueError('Consecutive windows must be a positive integer')
        self.options = review_limit, short_limit, min_messages
        self.consecutive = consecutive
        self.reset()

    def reset(self):
        self.previous = None
        self.streaks = {'review': 0, 'short_messages': 0}

    def observe(self, snapshot):
        # Validate even the first snapshot before keeping it as a baseline.
        validate_snapshot(snapshot)
        if self.previous is None:
            self.previous = snapshot
            return {'status': 'baseline', 'active_warnings': []}
        result = interval(self.previous, snapshot)
        checks = warnings_for(result, *self.options)
        self.previous = snapshot
        for signal in self.streaks:
            self.streaks[signal] = self.streaks[signal] + 1 if checks.get('checks', {}).get(signal) == 'above_limit' else 0
        return {'status': result['status'], 'interval': result, 'checks': checks,
                'streaks': dict(self.streaks),
                'active_warnings': [k for k, v in self.streaks.items() if v >= self.consecutive]}


def poll(collector, fetch, samples, seconds, sleep=time.sleep):
    for index in range(samples):
        try:
            yield collector.observe(fetch())
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            collector.reset()
            yield {'status': 'collection_error', 'error_type': type(error).__name__, 'active_warnings': []}
        if index + 1 < samples:
            sleep(seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--review-limit', type=float, required=True)
    parser.add_argument('--short-limit', type=float, required=True)
    parser.add_argument('--min-messages', type=int, default=100)
    parser.add_argument('--consecutive', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 2 <= args.samples <= 10000 or not .1 <= args.seconds <= 60:
        parser.error('Invalid port, samples (2..10000) or seconds (0.1..60)')
    try:
        collector = Collector(args.review_limit, args.short_limit, args.min_messages, args.consecutive)
    except ValueError as error:
        parser.error(str(error))
    def fetch():
        with urlopen(f'http://127.0.0.1:{args.port}/metrics', timeout=5) as response:
            return json.load(response)
    failed = False
    for result in poll(collector, fetch, args.samples, args.seconds):
        failed |= result['status'] == 'collection_error'
        print(json.dumps(result, allow_nan=False), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
