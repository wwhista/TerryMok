"""Evaluate a frozen validation-selected model without fitting or tuning."""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_model(model, rows, labels):
    if not rows or not labels or len(set(labels)) != len(labels):
        raise ValueError("Nonempty rows and unique class names are required")
    texts, targets = zip(*rows)
    class_ids = list(range(len(labels)))
    if not set(targets).issubset(class_ids):
        raise ValueError("Target outside the declared class mapping")
    predictions = model.predict(list(texts))
    if len(predictions) != len(targets) or not set(predictions).issubset(class_ids):
        raise ValueError("Invalid model predictions")
    return {
        "test_size": len(rows), "labels": list(labels),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=class_ids, average="macro", zero_division=0)),
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(targets, predictions, labels=class_ids).tolist(),
        "classification_report": classification_report(targets, predictions, labels=class_ids,
                                                        target_names=labels, output_dict=True, zero_division=0),
        "error_count": sum(int(actual != predicted) for actual, predicted in zip(targets, predictions)),
    }


def main():
    model_path = Path("artifacts/selected-model.joblib")
    data_path = Path("data/prepared.joblib")
    # These artifacts must be generated locally by prepare and experiment.
    artifact = joblib.load(model_path)
    prepared = joblib.load(data_path)
    selection = json.loads(Path("reports/experiments.json").read_text(encoding="utf-8"))
    if artifact["labels"] != prepared["labels"] or artifact["candidate"] != selection["winner"]:
        raise ValueError("Model, class mapping and selection report do not match")
    report = evaluate_model(artifact["pipeline"], prepared["splits"]["test"], prepared["labels"])
    report.update(candidate=artifact["candidate"], protocol="Frozen train-only model selected by validation macro-F1; no refit or test-based tuning",
                  python_version=platform.python_version(), sklearn_version=sklearn.__version__,
                  model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  prepared_data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest())
    Path("reports").mkdir(exist_ok=True)
    Path("reports/selected-test-metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
