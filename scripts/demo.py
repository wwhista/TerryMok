"""Show batch inference and operational metrics from a local TerryMok API."""

import argparse
import json
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()
base = f"http://127.0.0.1:{args.port}"
with urlopen(base + "/health", timeout=10) as response:
    print("Loaded model:", json.load(response)["model_sha256"])
texts = ["My Windows display driver crashes", "How do I upgrade memory on a Macintosh?",
         "How can I render a three dimensional image?"]
request = Request(base + "/predict", data=json.dumps({"texts": texts}).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=10) as response:
    predictions = json.load(response)["predictions"]
for text, prediction in zip(texts, predictions):
    print(f"{text} -> {prediction['label']}")
with urlopen(base + "/metrics", timeout=10) as response:
    print("Operational metrics:", json.dumps(json.load(response), indent=2))
