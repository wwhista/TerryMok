"""Reproduce the already selected configuration on real data inside Linux."""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score

from terrymok.experiment import candidates
from terrymok.prepare import prepare_splits
from terrymok.train import CATEGORIES

options = dict(categories=CATEGORIES, remove=("headers", "footers", "quotes"),
               random_state=42, data_home="/tmp/terrymok-data")
train = fetch_20newsgroups(subset="train", **options)
test = fetch_20newsgroups(subset="test", **options)
splits, preparation = prepare_splits(train.data, train.target, test.data, test.target)
name = "char_wb_svm_C1.0"
model = candidates()[name]
texts, labels = zip(*splits["train"])
model.fit(list(texts), labels)
validation_texts, validation_labels = zip(*splits["validation"])
score = float(f1_score(validation_labels, model.predict(list(validation_texts)), average="macro"))
# No tuning and no renewed test scoring in this release check.
path = Path("/models/selected-model.joblib")
joblib.dump({"pipeline": model, "labels": train.target_names, "candidate": name}, path)
demo_texts = ["hardware processor", "graphics pixels"]
report = {"candidate": name, "validation_macro_f1": score,
          "python": platform.python_version(), "sklearn": sklearn.__version__,
          "platform": platform.system(), "sizes": preparation["sizes"],
          "artifact_bytes": path.stat().st_size,
          "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
          "expected_labels": [train.target_names[int(value)] for value in model.predict(demo_texts)]}
Path("/models/release.json").write_text(json.dumps(report), encoding="utf-8")
