"""Fixed-policy validation stress checks; no fitting or threshold selection."""
import hashlib
import json
import math
import re
from pathlib import Path

import joblib
import numpy as np
from terrymok.api import routing_scores


def typo(text):
    """Swap the second and third letters of every fifth eligible ASCII word."""
    count = 0
    def swap(match):
        nonlocal count
        count += 1
        word = match.group()
        return word[0] + word[2] + word[1] + word[3:] if count % 5 == 0 else word
    return re.sub(r'\b[A-Za-z]{4,}\b', swap, text)


def stress(model, rows, labels, threshold):
    if not rows or (threshold is not None and
                   (type(threshold) not in (float, int) or not math.isfinite(threshold) or threshold < 0)):
        raise ValueError('Nonempty rows and valid fixed threshold required')
    texts, targets = zip(*rows)
    targets = np.asarray(targets)
    if not np.isin(targets, range(len(labels))).all():
        raise ValueError('Invalid targets')
    profiles = {'original': lambda text: text, 'typos': typo,
                'first_20_words': lambda text: ' '.join(text.split()[:20])}
    results = {}
    original = None
    for name, transform in profiles.items():
        changed = [transform(text) for text in texts]
        predictions, gaps = routing_scores(model, changed, len(labels))
        accepted = np.zeros(len(rows), dtype=bool) if threshold is None else gaps >= threshold
        count = int(accepted.sum())
        correct = predictions == targets
        if original is None:
            original = predictions
        results[name] = {'size': len(rows), 'changed_texts': sum(a != b for a, b in zip(texts, changed)),
                         'accuracy': float(correct.mean()), 'accepted': count,
                         'review': len(rows) - count, 'coverage': count / len(rows),
                         'accepted_errors': int((accepted & ~correct).sum()),
                         'accepted_accuracy': float(correct[accepted].mean()) if count else None,
                         'prediction_changes': int((predictions != original).sum())}
    return results


def main():
    model_path = Path('artifacts/selected-model.joblib')
    data_path = Path('data/prepared.joblib')
    policy_path = Path('reports/selective-validation.json')
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    # Only trusted locally generated artifacts may be deserialized.
    artifact, data = joblib.load(model_path), joblib.load(data_path)
    if (digest(model_path) != policy['model_sha256'] or digest(data_path) != policy['prepared_data_sha256']
            or artifact['labels'] != policy['labels'] or data['labels'] != policy['labels']
            or list(artifact['pipeline'].classes_) != list(range(len(policy['labels'])))):
        raise ValueError('Model, data and policy identities must agree')
    report = {'model_sha256': digest(model_path), 'policy_sha256': digest(policy_path),
              'prepared_data_sha256': digest(data_path), 'threshold': policy['selected']['threshold'],
              'test_evaluated': False, 'profiles': stress(artifact['pipeline'], data['splits']['validation'],
                                                       policy['labels'], policy['selected']['threshold'])}
    Path('reports/robustness.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['profiles'], indent=2))


if __name__ == '__main__':
    main()
