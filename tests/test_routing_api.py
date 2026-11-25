import hashlib
import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from terrymok.api import create_app


class ScoredModel:
    classes_ = [0, 1, 2]

    def decision_function(self, texts):
        rows = {"accept": [0, 2, 0], "boundary": [0, 0, 1],
                "review": [0.2, 0, 0], "tie": [0, 0, 0],
                "invalid": [np.nan, 0, 0]}
        return [rows.get(text, [1, 0, 0]) for text in texts]


@pytest.fixture
def files(tmp_path):
    model = tmp_path / "model.joblib"
    labels = ["a", "b", "c"]
    joblib.dump({"pipeline": ScoredModel(), "labels": labels}, model)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "labels": labels, "selected": {"threshold": 1},
        "score": "largest minus second-largest decision score",
        "rule": "gap >= threshold; null threshold means review all"}))
    return model, policy


def test_routing_order_boundary_identity_and_single_load(files, monkeypatch):
    model, policy = files
    monkeypatch.setenv("TERRYMOK_POLICY_PATH", str(policy))
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    with TestClient(create_app(model)) as client:
        policy.unlink()
        response = client.post("/predict", json={"texts": ["accept", "boundary", "review", "tie"]})
        assert response.status_code == 200
        body = response.json()
        assert body["policy_sha256"] == client.get("/health").json()["policy_sha256"] == digest
        assert [r["review_required"] for r in body["predictions"]] == [False, False, True, True]
        assert [r["label_id"] for r in body["predictions"]] == [1, 2, 0, 0]
        assert [r["score_gap"] for r in body["predictions"]] == [2, 1, 0.2, 0]


def test_null_threshold_reviews_everything(files):
    model, policy = files
    document = json.loads(policy.read_text())
    document["selected"]["threshold"] = None
    policy.write_text(json.dumps(document))
    with TestClient(create_app(model, policy)) as client:
        result = client.post("/predict", json={"texts": ["accept"]}).json()
        assert result["predictions"][0]["review_required"] is True


@pytest.mark.parametrize("field,value", [("model_sha256", "wrong"), ("labels", ["c", "b", "a"]),
    ("score", "probability"), ("rule", "gap > threshold"),
    ("selected", {}), ("selected", {"threshold": True}),
    ("selected", {"threshold": -1}), ("selected", {"threshold": "1"}),
    ("selected", {"threshold": float("nan")}), ("selected", {"threshold": float("inf")})])
def test_invalid_policy_fails_startup(files, field, value):
    model, policy = files
    document = json.loads(policy.read_text())
    document[field] = value
    policy.write_text(json.dumps(document))
    with pytest.raises((ValueError, KeyError)):
        with TestClient(create_app(model, policy)):
            pass


def test_missing_policy_fails_startup(files):
    model, policy = files
    policy.unlink()
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(model, policy)):
            pass


def test_nonfinite_runtime_scores_fail_closed(files):
    with TestClient(create_app(*files), raise_server_exceptions=False) as client:
        assert client.post("/predict", json={"texts": ["invalid"]}).status_code == 500
        assert client.get("/metrics").json()["status_classes"]["5xx"] == 1


def test_binary_score_shape_rejected():
    from terrymok.api import routing_scores

    class BinaryModel:
        def decision_function(self, texts):
            return [1.0] * len(texts)

    with pytest.raises(ValueError, match="multiclass"):
        routing_scores(BinaryModel(), ["example"], 3)


def test_prediction_metrics_count_messages_not_requests(files):
    with TestClient(create_app(*files), raise_server_exceptions=False) as client:
        assert client.get('/metrics').json()['predictions']['review_fraction'] is None
        client.post('/predict', json={'texts': ['accept', 'review', 'tie']})
        client.post('/predict', json={'texts': ['invalid']})
        client.post('/predict', json={'texts': ['']})
        result = client.get('/metrics').json()
        assert result['policy_sha256'] == client.get('/health').json()['policy_sha256']
        assert result['predictions'] == {'messages': 3, 'characters_sum': 15,
            'short_messages_le_20_words': 3, 'policy_messages': 3,
            'review_messages': 2, 'review_fraction': 2 / 3}
