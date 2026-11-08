"""Explicit network-enabled training entry point; never run by unit tests."""

import json
import platform
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from terrymok.model import build_pipeline
from terrymok.prepare import prepare_splits

CATEGORIES = [
    "comp.graphics", "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
]


def main() -> None:
    options = dict(
        categories=CATEGORIES, remove=("headers", "footers", "quotes"),
        random_state=42, data_home="data",
    )
    train = fetch_20newsgroups(subset="train", **options)
    test = fetch_20newsgroups(subset="test", **options)
    splits, preparation = prepare_splits(train.data, train.target, test.data, test.target)
    train.data, train.target = map(list, zip(*splits["train"]))
    test.data, test.target = map(list, zip(*splits["test"]))
    model = build_pipeline()
    model.fit(train.data, train.target)
    predictions = model.predict(test.data)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit([[0]] * len(train.data), train.target)
    baseline_predictions = baseline.predict([[0]] * len(test.data))
    report = {
        "dataset": "20newsgroups", "categories": train.target_names,
        "preparation": preparation,
        "remove": list(options["remove"]), "random_state": 42,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "train_size": len(train.data), "test_size": len(test.data),
        "accuracy": accuracy_score(test.target, predictions),
        "macro_f1": f1_score(test.target, predictions, average="macro"),
        "baseline_macro_f1": f1_score(test.target, baseline_predictions, average="macro"),
        "classification_report": classification_report(
            test.target, predictions, target_names=test.target_names,
            output_dict=True, zero_division=0,
        ),
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump({"pipeline": model, "labels": train.target_names}, "artifacts/model.joblib")
    Path("reports/metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
