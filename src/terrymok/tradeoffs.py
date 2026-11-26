"""Repeat validation comparison without replacing the serving model or policy."""
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy
import scipy
import sklearn
from terrymok.experiment import select_model


def main():
    path = Path('data/prepared.joblib')
    prepared = joblib.load(path)  # Trusted local output of terrymok.prepare only.
    _, report = select_model(prepared['splits']['train'], prepared['splits']['validation'])
    report.update(labels=prepared['labels'], random_state=42,
        prepared_data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        versions={'python': platform.python_version(), 'sklearn': sklearn.__version__,
                  'numpy': numpy.__version__, 'scipy': scipy.__version__, 'joblib': joblib.__version__},
        platform=platform.system(), machine=platform.machine(),
        protocol='Train-only fitting; first validation prediction warms each estimator; five subsequent full-batch predictions. Uncompressed estimator-only joblib bytes; not RAM usage. Winner uses macro-F1 only.')
    Path('reports').mkdir(exist_ok=True)
    Path('reports/tradeoffs.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in report['candidates']:
        print(row['name'], row['macro_f1'], row['estimator_joblib_bytes'], row['warm_batch_predict_seconds_median'])


if __name__ == '__main__':
    main()
