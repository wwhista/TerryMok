"""Compare fixed candidates on validation only; never score the test set."""

import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def candidates():
    models = {"majority": DummyClassifier(strategy="most_frequent")}
    for strength in (0.5, 1.0, 2.0):
        models[f"word_lr_C{strength}"] = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("classifier", LogisticRegression(C=strength, max_iter=1000, random_state=42)),
        ])
    for analyzer, ngrams in [("word", (1, 2)), ("char_wb", (3, 5))]:
        models[f"{analyzer}_svm_C1.0"] = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, sublinear_tf=True)),
            ("classifier", LinearSVC(C=1.0, random_state=42, max_iter=5000)),
        ])
    return models


def select_model(train, validation, models=None):
    """Ties retain candidate insertion order, independent of noisy timings."""
    if not train or not validation:
        raise ValueError("Training and validation must be nonempty")
    models = candidates() if models is None else models
    if not models:
        raise ValueError("At least one candidate is required")
    train_text, train_y = zip(*train)
    val_text, val_y = zip(*validation)
    rows = []
    winner = None
    best_score = -1.0
    for name, estimator in models.items():
        fitted = clone(estimator)
        start = perf_counter()
        fitted.fit(list(train_text), train_y)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        predictions = fitted.predict(list(val_text))
        predict_seconds = perf_counter() - start
        score = float(f1_score(val_y, predictions, average="macro", zero_division=0))
        rows.append({"name": name, "macro_f1": score,
                     "accuracy": float(accuracy_score(val_y, predictions)),
                     "fit_seconds": fit_seconds, "validation_batch_predict_seconds": predict_seconds,
                     "classification_report": classification_report(val_y, predictions, output_dict=True, zero_division=0)})
        if score > best_score:
            best_score, winner = score, (name, fitted)
    return winner, {"selection_metric": "validation_macro_f1", "winner": winner[0],
                    "train_size": len(train), "validation_size": len(validation),
                    "candidates": rows, "test_evaluated": False}


def main():
    # Only load this project's locally generated, trusted artifact.
    prepared = joblib.load("data/prepared.joblib")
    (name, model), report = select_model(prepared["splits"]["train"], prepared["splits"]["validation"])
    report.update(python_version=platform.python_version(), sklearn_version=sklearn.__version__,
                  random_state=42, labels=prepared["labels"])
    Path("artifacts").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump({"pipeline": model, "labels": prepared["labels"], "candidate": name}, "artifacts/selected-model.joblib")
    report["selected_artifact_bytes"] = Path("artifacts/selected-model.joblib").stat().st_size
    Path("reports/experiments.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"winner": name, "scores": {row["name"]: row["macro_f1"] for row in report["candidates"]}}, indent=2))


if __name__ == "__main__":
    main()
