# Real-model Linux release validation

On November 15, 2026, [GitHub Actions run 43710016964](https://github.com/wwhista/TerryMok/actions/runs/43710016964) built the locked runtime image, reproduced the previously selected character TF-IDF/LinearSVC configuration on the real corpus, and served it in a separate non-root container with a read-only model mount.

The Linux run used Python 3.12.16 and scikit-learn 1.9.1. Validation macro-F1 reproduced **0.8047973**, using the same 1,812 training and 453 validation rows. No new hyperparameter search, validation refit, or test scoring was performed. The generated artifact was **46,556,754 bytes** and its SHA-256 matched the previously evaluated local artifact (`341fd8ce5777e236fa856d34077d023ce8f0c6eaffd54c4fe0b892e6ab9ca811`). This demonstrates equivalent reproduced artifact contents and successful Linux serving; no private Windows model was uploaded.

## Timed workload results

Each profile ran for 60 seconds after five serial warmup requests. The client and single-worker Uvicorn server shared one GitHub-hosted Linux runner. Each request used a fresh localhost HTTP connection; this is closed-loop load, so higher latency reduces the offered rate. Four synthetic messages were cycled. Repeat 10 concatenates each message ten times; it is a workload-length stressor, not realistic natural-language diversity.

| Client threads | Texts/request | Text repeat | Successful requests | Errors | p50 ms | p95 ms | p99 ms | Requests/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 33,042 | 0 | 1.77 | 1.97 | 2.08 | 550.69 |
| 4 | 1 | 1 | 37,096 | 0 | 6.19 | 8.23 | 9.23 | 618.20 |
| 4 | 8 | 10 | 21,978 | 0 | 10.74 | 12.71 | 14.25 | 366.24 |

The measured phases total **92,116 successful requests**, with no network/response-validation failures. Server metrics counted **92,133** completed requests: measured traffic, 15 warmups, one valid smoke request and one deliberately invalid request. Counters reported **0 server errors**, **1 expected 4xx** and **0 in-flight requests** at the end. These counts are asserted by the workflow rather than inferred from the table. The complete [aggregate report](release-validation.json) records durations, hashes, model details and provenance.

## Interpretation and limits

Increasing concurrency raised throughput moderately for short requests while also increasing latency; batching longer inputs reduced requests/s but each request contained eight messages. These profiles do not locate saturation or establish a capacity ceiling. Each profile was run once; there are no confidence intervals, multi-hour soak tests, public network/proxy effects, or CPU/RAM instrumentation. The exact CPU model and memory were not recorded. Do not compare these numbers directly with the earlier Windows smoke run or claim a production SLA.

## Reproduce

Run the manually triggered **Real model release validation** workflow from GitHub Actions, or `gh workflow run release-validation.yml`. It downloads the public corpus, builds a model in the ephemeral runner, and prints a `RELEASE_REPORT=` JSON line. It does not publish dataset files, model artifacts or an image. The regular push/PR CI continues using a small synthetic fixture and does not download the corpus.

For an existing local model, use the [demo guide](demo.md) and the timed benchmark command. Save each report before running another profile. Model hashes identify content; only load artifacts created by trusted code. Python base-image digest/build-tool pinning, public deployment, external metrics collection, alerting and calibrated abstention remain future work.
