import json

import pytest

from terrymok.evaluate import evaluate_model


class FrozenModel:
    def fit(self, *args):
        raise AssertionError("Evaluation must never fit")

    def predict(self, texts):
        return [0, 1, 1, 0]


def test_evaluation_counts_errors_and_fixes_confusion_matrix_order():
    rows = [("private message", label) for label in [0, 0, 1, 1]]
    report = evaluate_model(FrozenModel(), rows, ["graphics", "hardware"])
    assert report["accuracy"] == 0.5
    assert report["macro_f1"] == 0.5
    assert report["error_count"] == 2
    assert report["confusion_matrix_rows_true_columns_predicted"] == [[1, 1], [1, 1]]
    assert "private message" not in json.dumps(report)


def test_evaluation_rejects_bad_mapping_and_prediction_count():
    for rows, labels in [([], ["a"]), ([("x", 2)], ["a"]),
                         ([("x", 0)], ["a", "a"]), ([("x", 0)], ["a", "b"])]:
        with pytest.raises(ValueError):
            evaluate_model(FrozenModel(), rows, labels)
