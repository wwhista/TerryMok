"""Compare two aggregate /metrics snapshots from one application instance."""
import argparse
import json
import math
from pathlib import Path

FIELDS = ('messages', 'characters_sum', 'short_messages_le_20_words',
          'policy_messages', 'review_messages')


def validate_snapshot(snapshot):
    if not isinstance(snapshot, dict):
        raise ValueError('Snapshot must be an object')
    for key in ('instance_id', 'model_sha256', 'policy_sha256'):
        if key not in snapshot:
            raise ValueError('Missing snapshot identity')
        value = snapshot[key]
        if key == 'policy_sha256' and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Snapshot identities must be nonempty strings')
    uptime = snapshot.get('uptime_seconds')
    if type(uptime) not in (int, float) or not math.isfinite(uptime) or uptime < 0:
        raise ValueError('Uptime must be finite and nonnegative')
    counts = snapshot.get('predictions')
    if not isinstance(counts, dict):
        raise ValueError('Missing prediction counters')
    for field in FIELDS:
        if type(counts.get(field)) is not int or counts[field] < 0:
            raise ValueError('Counters must be nonnegative integers')
    if not counts['review_messages'] <= counts['policy_messages'] <= counts['messages'] or counts['short_messages_le_20_words'] > counts['messages']:
        raise ValueError('Inconsistent snapshot counters')
    if snapshot['policy_sha256'] is None and counts['policy_messages']:
        raise ValueError('Policy counters require policy identity')


def interval(before, after):
    validate_snapshot(before)
    validate_snapshot(after)
    for key in ('instance_id', 'model_sha256', 'policy_sha256'):
        if key not in before or key not in after:
            raise ValueError('Missing snapshot identity')
        if before[key] != after[key]:
            return {'status': 'baseline_required', 'reason': key + ' changed'}
    elapsed = after['uptime_seconds'] - before['uptime_seconds']
    if not 0 < elapsed < float('inf'):
        return {'status': 'baseline_required', 'reason': 'non-increasing or invalid uptime'}
    delta = {}
    for field in FIELDS:
        a, b = before['predictions'][field], after['predictions'][field]
        if type(a) is not int or type(b) is not int or min(a, b) < 0:
            raise ValueError('Counters must be nonnegative integers')
        if b < a:
            return {'status': 'baseline_required', 'reason': 'counter decreased'}
        delta[field] = b - a
    n, p = delta['messages'], delta['policy_messages']
    if not 0 <= delta['review_messages'] <= p <= n or delta['short_messages_le_20_words'] > n:
        raise ValueError('Inconsistent counter deltas')
    return {'status': 'ok', 'seconds': elapsed, **delta,
            'messages_per_second': n / elapsed,
            'mean_characters': delta['characters_sum'] / n if n else None,
            'short_fraction': delta['short_messages_le_20_words'] / n if n else None,
            'review_fraction': delta['review_messages'] / p if p else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--review-limit', type=float)
    parser.add_argument('--short-limit', type=float)
    parser.add_argument('--min-messages', type=int, default=100)
    args = parser.parse_args()
    if (args.review_limit is None) != (args.short_limit is None):
        parser.error('Provide both --review-limit and --short-limit')
    result = interval(json.loads(args.before.read_text(encoding='utf-8-sig')),
                      json.loads(args.after.read_text(encoding='utf-8-sig')))
    if args.review_limit is not None:
        from terrymok.alerts import warnings_for
        result['alerts'] = warnings_for(result, args.review_limit, args.short_limit, args.min_messages)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
