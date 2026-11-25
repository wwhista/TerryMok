"""Compare two aggregate /metrics snapshots from one application instance."""
import argparse
import json
from pathlib import Path

FIELDS = ('messages', 'characters_sum', 'short_messages_le_20_words',
          'policy_messages', 'review_messages')


def interval(before, after):
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
    args = parser.parse_args()
    result = interval(json.loads(args.before.read_text(encoding='utf-8-sig')),
                      json.loads(args.after.read_text(encoding='utf-8-sig')))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
