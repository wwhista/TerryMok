# Five-minute TerryMok demonstration

## Prepare once

Use Python 3.12, install the hash-checked development dependencies from the README, then run `python -m terrymok.prepare` and `python -m terrymok.experiment`. These commands download the corpus and produce a trusted local model. Reuse the existing generated artifact when available. Do not load models obtained from strangers.

## Start and demonstrate

```bash
python -m uvicorn terrymok.api:create_app --factory --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
python scripts/demo.py
python -m terrymok.benchmark --duration 60 --concurrency 4 --batch-size 8 --text-repeat 10
```

The demo prints the model digest, three synthetic messages with their predicted classes, and operational counters. Open `http://127.0.0.1:8000/docs` to try a request interactively. Send `{"texts": [" "]}` to demonstrate HTTP 422 and check `/metrics` for a 4xx increment. For Docker, use the README's read-only model mount instructions instead of the local Uvicorn command.

## What to explain in an interview

1. The original forum data contains duplicates, empty content and ambiguous labels. Explain the preparation policy and remaining near-duplicate/metadata limitations.
2. Six fixed candidates were compared on validation; the chosen character SVM achieved test macro-F1 0.7567 versus 0.7238 for the fixed word logistic regression. Test was not used to tune parameters. Error review happened after evaluation.
3. Inference loads the model once, validates batch input, identifies the artifact by SHA-256, and fails startup on an invalid mapping. The API can run as a non-root container with a read-only model mount.
4. Monitoring counts requests, status classes and cumulative latency buckets without recording text. Explain why these per-process aggregates reset on restart and do not measure model drift or accuracy.
5. Show the real-model Linux validation run and measured workload profiles. Describe the benchmark's workload and machine before quoting latency. Localhost load generation and serving share the runner; this does not establish internet latency or a production SLA.

## Resume wording supported by the project

“Built a technical-message classification service with TF-IDF/LinearSVC, leakage-aware data preparation and validation-based model selection; achieved macro-F1 0.757 on a documented cleaned test split. Implemented a validated FastAPI batch API, per-process operational metrics, non-root Docker deployment and CI tests with hash-locked dependencies.”

Add load-test numbers only together with the exact profile and its published report. Do not claim public production deployment, calibrated confidence, independent human error annotation, drift monitoring, or byte-for-byte Windows-to-Linux artifact transfer. The Linux check retrains the already selected configuration on the real corpus using locked ML dependencies; it does not upload the private local model.
