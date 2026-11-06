import joblib
import pytest

from terrymok.model import build_pipeline


@pytest.fixture
def model():
    pipeline = build_pipeline()
    pipeline.fit(
        ["graphics image rendering", "image graphics pixels",
         "hardware motherboard memory", "memory hardware processor"],
        ["graphics", "graphics", "hardware", "hardware"],
    )
    return pipeline


def test_predicts_unseen_messages(model):
    assert model.predict(["graphics rendering pixels", "hardware memory processor"]).tolist() == [
        "graphics", "hardware",
    ]


def test_inference_does_not_fit_vocabulary(model):
    vocabulary = dict(model.named_steps["tfidf"].vocabulary_)
    model.predict(["previouslyunseentoken"])
    assert model.named_steps["tfidf"].vocabulary_ == vocabulary
    assert "previouslyunseentoken" not in vocabulary


def test_saved_pipeline_preserves_predictions(model, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    texts = ["graphics rendering", "hardware processor"]
    assert restored.predict(texts).tolist() == model.predict(texts).tolist()
