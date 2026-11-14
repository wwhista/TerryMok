"""Build-independent container smoke check using only synthetic fixture data."""

import argparse
import hashlib
from http.client import HTTPException
import json
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def fixture():
    import joblib
    from terrymok.model import build_pipeline

    model = build_pipeline().fit(
        ["graphics image pixels", "image graphics rendering", "hardware memory processor", "processor hardware memory"],
        [0, 0, 1, 1],
    )
    joblib.dump({"pipeline": model, "labels": ["graphics", "hardware"]}, "/models/selected-model.joblib")


def check(image):
    name = "terrymok-smoke-" + uuid.uuid4().hex[:12]

    def docker(*args):
        return subprocess.check_output(["docker", *args], text=True, timeout=120).strip()

    with tempfile.TemporaryDirectory() as temp:
        models = Path(temp).resolve()
        models.chmod(0o755)  # Permit the non-root API to traverse the fixture mount.
        script_dir = Path(__file__).resolve().parent
        docker("run", "--rm", "--user", "0", "-v", f"{models}:/models",
               "-v", f"{script_dir}:/smoke:ro", "--entrypoint", "python", image,
               "/smoke/container_smoke.py", "--fixture")
        digest = hashlib.sha256((models / "selected-model.joblib").read_bytes()).hexdigest()
        try:
            docker("run", "-d", "--name", name, "-p", "127.0.0.1::8000",
                   "-v", f"{models}:/models:ro", image)
            port = docker("port", name, "8000/tcp").rsplit(":", 1)[1]
            base = f"http://127.0.0.1:{port}"
            for _ in range(60):
                try:
                    with urlopen(base + "/health", timeout=2) as response:
                        health = json.load(response)
                    break
                except (URLError, OSError, HTTPException):
                    time.sleep(1)
            else:
                raise RuntimeError("Container did not become ready")
            assert health["model_sha256"] == digest
            assert docker("exec", name, "id", "-u") == "10001"
            payload = json.dumps({"texts": ["hardware processor", "graphics pixels"]}).encode()
            request = Request(base + "/predict", data=payload, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=10) as response:
                result = json.load(response)
            assert [row["label"] for row in result["predictions"]] == ["hardware", "graphics"]
            assert result["model_sha256"] == digest
            invalid = Request(base + "/predict", data=b'{"texts": [" "]}', headers={"Content-Type": "application/json"})
            try:
                urlopen(invalid, timeout=10)
            except HTTPError as error:
                assert error.code == 422
            else:
                raise AssertionError("Blank text was accepted")
            print("Container smoke passed: non-root, read-only model, health, ordered predictions, invalid input")
        finally:
            subprocess.run(["docker", "logs", name], check=False, timeout=30)
            subprocess.run(["docker", "rm", "-f", name], check=False, timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--image", default="terrymok:smoke")
    args = parser.parse_args()
    fixture() if args.fixture else check(args.image)
