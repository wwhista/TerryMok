"""Validation-only exploration of abstention using multiclass score gaps."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn


def select_threshold(scores, targets, target_accuracy=0.9, min_accepted=30):
    """Maximize coverage subject to empirical accuracy; keep tied gaps together.

    Scores must have columns in contiguous class-ID order. Gaps are not
    probabilities. No feasible threshold means route every sample to review.
    """
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets)
    if (scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 3
            or not np.isfinite(scores).all()):
        raise ValueError("Finite nonempty multiclass score matrix required")
    if (targets.shape != (len(scores),)
            or not np.isin(targets, np.arange(scores.shape[1])).all()):
        raise ValueError("Targets must match rows and class columns")
    if not 0 < target_accuracy <= 1 or isinstance(min_accepted, bool) or not isinstance(min_accepted, int) or min_accepted < 1:
        raise ValueError("Invalid accuracy target or minimum accepted count")
    ordered = np.sort(scores, axis=1)
    gaps = ordered[:, -1] - ordered[:, -2]
    correct = scores.argmax(axis=1) == targets
    curve = []
    for threshold in np.unique(gaps):
        accepted = gaps >= threshold
        count = int(accepted.sum())
        errors = int((~correct & accepted).sum())
        curve.append({"threshold": float(threshold), "accepted": count,
                      "review": len(scores) - count, "coverage": count / len(scores),
                      "errors": errors, "accuracy": 1 - errors / count})
    feasible = [row for row in curve if row["accepted"] >= min_accepted
                and row["accuracy"] >= target_accuracy]
    selected = feasible[0] if feasible else {
        "threshold": None, "accepted": 0, "review": len(scores),
        "coverage": 0.0, "errors": 0, "accuracy": None}
    accepted = np.zeros(len(scores), dtype=bool) if selected["threshold"] is None else gaps >= selected["threshold"]
    predictions = scores.argmax(axis=1)
    per_class = []
    for class_id in range(scores.shape[1]):
        actual = targets == class_id
        kept = actual & accepted
        routed = (predictions == class_id) & accepted
        support, count, routed_count = int(actual.sum()), int(kept.sum()), int(routed.sum())
        errors = int((kept & ~correct).sum())
        per_class.append({"class_id": class_id, "support": support,
                          "accepted": count, "review": support - count,
                          "coverage": count / support if support else None,
                          "accepted_errors": errors,
                          "accepted_accuracy": 1 - errors / count if count else None,
                          "predicted_accepted": routed_count,
                          "accepted_precision": float((routed & correct).sum()) / routed_count if routed_count else None})
    return {"validation_size": len(scores), "target_accuracy": target_accuracy,
            "min_accepted": min_accepted, "score": "largest minus second-largest decision score",
            "rule": "gap >= threshold; null threshold means review all",
            "selected": selected, "curve": curve, "per_class": per_class, "test_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-accuracy", type=float, default=0.9)
    parser.add_argument("--min-accepted", type=int, default=30)
    args = parser.parse_args()
    model_path = Path("artifacts/selected-model.joblib")
    data_path = Path("data/prepared.joblib")
    # Load only this project's trusted, locally generated artifacts.
    artifact = joblib.load(model_path)
    prepared = joblib.load(data_path)
    model = artifact["pipeline"]
    if (artifact["labels"] != prepared["labels"]
            or list(model.classes_) != list(range(len(artifact["labels"])))):
        raise ValueError("Model and data class mappings must agree")
    texts, targets = zip(*prepared["splits"]["validation"])
    report = select_threshold(model.decision_function(list(texts)), targets,
                              args.target_accuracy, args.min_accepted)
    report.update(model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  prepared_data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  candidate=artifact["candidate"], labels=artifact["labels"],
                  python_version=platform.python_version(), sklearn_version=sklearn.__version__,
                  limitation="Selection-set accuracy, not an independent estimate or guarantee; scores are not probabilities")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/selective-validation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["selected"], indent=2))


if __name__ == "__main__":
    main()
