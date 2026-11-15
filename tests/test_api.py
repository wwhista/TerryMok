import joblib
import pytest
from fastapi.testclient import TestClient

from terrymok.api import create_app
from terrymok.model import build_pipeline


@pytest.fixture
def artifact(tmp_path):
    model = build_pipeline().fit(
        ["graphics image pixels", "image rendering graphics", "hardware memory processor", "processor hardware memory"],
        [0, 0, 1, 1],
    )
    path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": model, "labels": ["graphics", "hardware"]}, path)
    return path


def test_batch_order_health_and_model_identity(artifact):
    with TestClient(create_app(artifact)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        response = client.post("/predict", json={"texts": [" hardware processor ", "graphics image"]})
        assert response.status_code == 200
        assert response.json()["predictions"] == [
            {"label_id": 1, "label": "hardware"}, {"label_id": 0, "label": "graphics"},
        ]
        assert response.json()["model_sha256"] == health.json()["model_sha256"]


@pytest.mark.parametrize("body", [{"texts": []}, {"texts": [" \n"]}, {"texts": [123]},
    {"texts": ["x" * 10001]}, {"texts": ["x"] * 33}, {"texts": ["ok"], "extra": 1}])
def test_invalid_requests_are_rejected(artifact, body):
    with TestClient(create_app(artifact)) as client:
        assert client.post("/predict", json=body).status_code == 422


def test_missing_model_fails_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path / "missing.joblib")):
            pass


def test_mismatched_classes_fail_startup(artifact):
    saved = joblib.load(artifact)
    saved["labels"] = ["only-one-class"]
    joblib.dump(saved, artifact)
    with pytest.raises(ValueError, match="mapping"):
        with TestClient(create_app(artifact)):
            pass


def test_environment_path_and_single_startup_load(artifact, monkeypatch):
    monkeypatch.setenv("TERRYMOK_MODEL_PATH", str(artifact))
    with TestClient(create_app()) as client:
        artifact.unlink()
        for _ in range(2):
            assert client.post("/predict", json={"texts": ["graphics pixels"]}).status_code == 200
        assert client.get("/health").status_code == 200


def test_metrics_count_success_validation_and_server_error(artifact):
    app = create_app(artifact)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/predict", json={"texts": ["graphics pixels"]})
        client.post("/predict", json={"texts": [""]})
        class BrokenModel:
            def predict(self, texts):
                raise RuntimeError("synthetic failure")
        app.state.model = BrokenModel()
        assert client.post("/predict", json={"texts": ["hardware"]}).status_code == 500
        metrics = client.get("/metrics").json()
        assert metrics["completed"] == 3
        assert metrics["status_classes"] == {"2xx": 1, "3xx": 0, "4xx": 1, "5xx": 1}
        assert metrics["in_flight"] == 0
        assert client.get("/metrics").json()["completed"] == 3
