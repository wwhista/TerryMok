import numpy as np
import pytest

from terrymok.selective import select_threshold


def test_maximum_coverage_and_ties_are_not_split():
    scores = [[3, 0, 0], [0, 2, 0], [0, 0, 2], [1, 0, 0]]
    report = select_threshold(scores, [0, 1, 1, 1], 0.9, 1)
    assert report["selected"]["threshold"] == 3
    assert report["selected"]["accepted"] == 1
    assert report["selected"]["coverage"] == 0.25
    assert [r["accepted"] for r in report["curve"]] == [4, 3, 1]
    assert report["test_evaluated"] is False


def test_infeasible_policy_reviews_everything_with_no_fake_accuracy():
    report = select_threshold([[1, 0, 0], [0, 2, 0]], [0, 1], 0.9, 3)
    assert report["selected"] == dict(threshold=None, accepted=0, review=2,
                                      coverage=0.0, errors=0, accuracy=None)


def test_all_correct_accepts_everything_including_zero_gap():
    report = select_threshold([[0, 0, 0], [0, 1, 0]], [0, 1], 1, 1)
    assert report["selected"]["threshold"] == 0
    assert report["selected"]["coverage"] == 1


@pytest.mark.parametrize("scores,targets", [([], []), ([[1, 0]], [0]),
    ([[1, 0, np.nan]], [0]), ([[1, 0, np.inf]], [0]),
    ([[1, 0, 0]], [3]), ([[1, 0, 0]], []), ([[1, 0, 0]], [[0]])])
def test_invalid_scores_or_targets(scores, targets):
    with pytest.raises(ValueError):
        select_threshold(scores, targets)


@pytest.mark.parametrize("accuracy,minimum", [(0, 1), (1.1, 1), (np.nan, 1),
    (0.9, 0), (0.9, 1.5), (0.9, True)])
def test_invalid_constraints(accuracy, minimum):
    with pytest.raises(ValueError):
        select_threshold([[1, 0, 0]], [0], accuracy, minimum)


def test_command_uses_validation_only_and_exports_no_text(tmp_path, monkeypatch):
    import json
    from terrymok import selective

    class FrozenModel:
        classes_ = [0, 1, 2]

        def decision_function(self, texts):
            assert texts == ["private validation message"]
            return [[2, 0, 0]]

    labels = ["a", "b", "c"]
    artifact = {"pipeline": FrozenModel(), "labels": labels, "candidate": "fixture"}
    prepared = {"labels": labels, "splits": {
        "validation": [("private validation message", 0)],
        "train": None, "test": None}}
    monkeypatch.chdir(tmp_path)
    for path in [tmp_path / "artifacts/selected-model.joblib", tmp_path / "data/prepared.joblib"]:
        path.parent.mkdir()
        path.write_bytes(b"fixture")
    monkeypatch.setattr(selective.joblib, "load", lambda path:
                        artifact if path.parent.name == "artifacts" else prepared)
    monkeypatch.setattr("sys.argv", ["selective", "--min-accepted", "1"])
    selective.main()
    content = (tmp_path / "reports/selective-validation.json").read_text()
    report = json.loads(content)
    assert report["selected"]["coverage"] == 1
    assert report["test_evaluated"] is False
    assert "private validation message" not in content
    assert len(report["model_sha256"]) == 64
