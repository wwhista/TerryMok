"""Explicit operational limits for one interval; not accuracy or drift tests."""
import math


def warnings_for(result, review_limit, short_limit, min_messages=100):
    if type(min_messages) is not int or min_messages < 1:
        raise ValueError('Minimum messages must be a positive integer')
    for limit in (review_limit, short_limit):
        if type(limit) not in (int, float) or not math.isfinite(limit) or not 0 <= limit <= 1:
            raise ValueError('Limits must be finite fractions')
    if result['status'] != 'ok':
        return {'status': 'not_evaluated', 'warnings': [], 'reason': result['status']}
    checks = {}
    warnings = []
    for name, count_key, fraction_key, limit in (
            ('review', 'policy_messages', 'review_fraction', review_limit),
            ('short_messages', 'messages', 'short_fraction', short_limit)):
        count, value = result[count_key], result[fraction_key]
        if count < min_messages or value is None:
            checks[name] = 'insufficient_samples'
        else:
            checks[name] = 'above_limit' if value > limit else 'within_limit'
            if value > limit:
                warnings.append({'signal': name, 'fraction': value, 'limit': limit, 'samples': count})
    return {'status': 'evaluated', 'checks': checks, 'warnings': warnings}
