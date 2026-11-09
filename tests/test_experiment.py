import pytest
from sklearn.dummy import DummyClassifier

from terrymok.experiment import select_model
from terrymok.model import build_pipeline


def test_selection_uses_validation_and_does_not_fit_its_vocabulary():
    train = [("graphics image pixels", 0), ("graphics image rendering", 0),
             ("hardware memory processor", 1), ("hardware memory motherboard", 1)]
    val = [("graphics pixels heldouttoken", 0), ("hardware processor heldouttoken", 1)]
    (name, model), report = select_model(train, val, {
        "dummy": DummyClassifier(strategy="most_frequent"), "model": build_pipeline(),
    })
    assert name == "model"
    assert report["candidates"][1]["macro_f1"] == 1.0
    assert "heldouttoken" not in model.named_steps["tfidf"].vocabulary_
    assert report["test_evaluated"] is False


def test_ties_are_stable_and_empty_inputs_fail():
    rows = [("a", 0), ("b", 1)]
    (name, _), _ = select_model(rows, rows, {
        "first": DummyClassifier(strategy="most_frequent"),
        "second": DummyClassifier(strategy="most_frequent"),
    })
    assert name == "first"
    with pytest.raises(ValueError):
        select_model([], rows)
    with pytest.raises(ValueError):
        select_model(rows, rows, {})
