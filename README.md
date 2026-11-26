# TerryMok

Reproducible classification of English technical messages with TF-IDF and logistic regression. An ML engineering portfolio project covering data separation, baseline evaluation, offline tests, and CI.

**Status:** initial implementation with offline tests and CI on Python 3.11/3.12. Data preparation is implemented and tested; validation model selection is implemented; the selected model has been evaluated on test; a qualitative review of 20 errors is documented.

## Problem and data

Predict one of four technical discussion categories: computer graphics, Windows, IBM PC hardware, or Mac hardware. This is a prototype of topic routing; it is not a validated customer-support system.

The [20 Newsgroups loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) provides the corpus and its original train/test split. Headers, footers, and quotes are removed to reduce metadata shortcuts. The corpus is historical and English-only; forum classification results do not establish quality on modern support tickets. Review dataset terms before redistributing source messages. Raw data is not included in this repository.

## Quick start

Install Python 3.11 or 3.12 first (3.12 recommended). From the repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --require-hashes --only-binary=:all: -r requirements-dev.lock
.venv\Scripts\python -m pip install --no-deps -e ".[dev,api]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m terrymok.train
```

On Linux/macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements-dev.lock
.venv/bin/python -m pip install --no-deps -e '.[dev,api]'
.venv/bin/python -m pytest -q
.venv/bin/python -m terrymok.train
```

Installation requires internet. Tests use tiny synthetic fixtures and do not download datasets. The training command downloads the corpus into `data/`, cleans the corpus, reserves validation data, and evaluates on the cleaned test split, and writes `artifacts/model.joblib` and `reports/metrics.json`. These generated directories are ignored by Git. Run commands from the repository root.

## Evaluation

Before training, run `.venv\Scripts\python -m terrymok.audit` (Linux/macOS: `.venv/bin/python -m terrymok.audit`). This separate network-enabled command downloads the same four-category corpus and writes aggregate counts to `reports/data_audit.json`. It counts empty messages, duplicate extra rows within each split, and cross-split overlap after Unicode case folding and whitespace normalization. Empty messages do not count as overlap. No raw text or message hashes are exported. The audit is diagnostic: it does not modify either split, detect near-duplicates, check conflicting labels, or remove leakage automatically. Resolve a cleaning protocol before treating future test metrics as a benchmark.

The [first audit snapshot](docs/data-audit.json), run on November 8, 2026, found:

| Split | Rows | Empty messages | Duplicate extra rows |
| --- | ---: | ---: | ---: |
| Train | 2,343 | 68 | 8 |
| Test | 1,560 | 40 | 1 |

Three distinct nonempty normalized messages appear in both splits, covering three training rows and four test rows. This is a data-quality finding, not a model score. The training entry point now applies the preparation protocol below.

The pipeline fits TF-IDF on training messages only. Logistic regression uses a fixed seed of 42 and is compared with a majority-class DummyClassifier. Reports contain accuracy, macro-F1, per-class precision/recall/F1, sample counts, categories, and Python/scikit-learn versions.

The initial baseline has no hyperparameter search. Future tuning must use a validation split inside train, leaving test untouched. Dependency ranges are declared in `pyproject.toml`; runtime and test dependencies now have hash-checked locks; validation selection is available through the separate experiment command.

## Layout

```text
src/terrymok/model.py    Text preprocessing and classifier
src/terrymok/audit.py    Aggregate duplicate and split-overlap diagnostics
src/terrymok/train.py    Dataset download, training, evaluation, export
tests/test_model.py     Offline behavior and serialization tests
.github/workflows/      Tests on Python 3.11 and 3.12
```

GitHub Actions runs tests on pushes and pull requests; it also supports manual runs. CI installs dependencies but does not download the training corpus.

## Next milestones

- Run and document the first reproducible benchmark, inspect duplicates and classification errors.
- Add a validation protocol, experiment comparison, and a model card with measured limitations.
- Add an inference CLI, FastAPI service, container, and latency benchmark.
- Evaluate a confidence threshold for human review using validation data.

Only load trusted local joblib artifacts: deserializing an untrusted model can execute code. No production readiness or achieved quality is claimed at this stage.


## Data preparation protocol

Run `python -m terrymok.prepare` in the project environment. It writes local `data/prepared.joblib` and `reports/preparation.json`. Training applies the same function automatically. The [aggregate preparation report](docs/preparation.json) contains no source messages.

Within each original split, remove blank messages, remove every row in groups with conflicting labels, and keep the first row of each remaining normalized duplicate group. Remove test messages matching any original training message, regardless of label. Training cleanup never uses test labels. Split the cleaned training pool into stratified 80/20 train/validation sets with seed 42. TF-IDF fits only on the final training set. Validation is reserved for future model selection; the current fixed baseline does not use it.

The prepared corpus has 1,812 train, 453 validation, and 1,516 test messages, with zero normalized exact overlap between all pairs. Four conflicting training rows were removed. Near-duplicates are not detected. Removing ambiguous and repeated test examples changes the evaluation population, so future results must be labelled as a cleaned-corpus benchmark and cannot be directly compared with raw 20 Newsgroups scores.

The first fixed-model integration run on the cleaned test set achieved accuracy 0.7243 and macro-F1 0.7238; the majority-class baseline macro-F1 was 0.1011. See [full metrics](docs/baseline-metrics.json). These are initial results from one split, with no hyperparameter search or uncertainty estimate. Future model choices must use validation rather than these test results.

## Validation experiments

Run `python -m terrymok.prepare` followed by `python -m terrymok.experiment` in the project environment. The experiment command loads only the trusted, locally generated `data/prepared.joblib`. It compares a majority baseline, word TF-IDF (1–2 grams) with logistic regression at C=0.5/1/2, word TF-IDF with linear SVM at C=1, and character-within-word TF-IDF (3–5 grams) with linear SVM at C=1. All TF-IDF variants use sublinear term frequency. Estimators use seed 42.

Selection maximizes macro-F1 on the 453-message validation set; exact ties preserve the declared candidate order. Each candidate fits only on the 1,812 training messages, including its vocabulary. The experiment does not evaluate test data or refit on validation. The earlier fixed-baseline test evaluation remains a separate result.

Outputs are `reports/experiments.json` and `artifacts/selected-model.joblib`. The report includes per-class validation metrics, environment versions, fit time, one batch prediction time, and selected artifact size. Timings are single local observations, not a production latency benchmark. The selected model has a separate frozen-model test evaluation and model card; a qualitative review of 20 sampled errors is available in docs/error-review.md. Validation selection can overfit this one split; a higher validation score alone does not establish improvement on unseen data.


The [November 9 validation run](docs/validation-experiments.json) selected character TF-IDF + LinearSVC (C=1): macro-F1 **0.8048**, versus **0.7893** for word TF-IDF + LinearSVC and **0.7697** for the original word logistic regression (C=1), all on the same validation split. These scores must not be compared directly with the earlier test score of 0.7238.


## Frozen-model test evaluation

After prepare and experiment, run `python -m terrymok.evaluate`. This command predicts with the selected local artifact without fitting or selecting parameters. It checks class mappings and the selected candidate name, then saves aggregate metrics, a confusion matrix, and artifact digests to `reports/selected-test-metrics.json`.

The selected character SVM achieved test accuracy **0.7579** and macro-F1 **0.7567** on 1,516 cleaned messages, compared with **0.7238** macro-F1 for the fixed logistic-regression baseline on the same test set. There were 367 mistakes. See the [model card](docs/model-card.md) for protocol, aggregate error analysis, limitations and reproduction, and the [complete test report](docs/selected-test-metrics.json). These results must not be used to tune parameters on test.

## Inference API

Install `python -m pip install -e '.[dev,api]'`. Generate the trusted model with prepare and experiment first, then start from the repository root:

```bash
python -m uvicorn terrymok.api:create_app --factory --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive OpenAPI documentation. `GET /health` reports readiness, class labels and the loaded model SHA-256. `POST /predict` accepts JSON:

```json
{"texts": ["My graphics card driver crashes in Windows", "How do I upgrade memory on my Mac?"]}
```

The response contains `model_sha256` and an ordered `predictions` list, each with `label_id` and `label`. No calibrated confidence is claimed. One request accepts 1–32 nonblank strings, up to 10,000 characters each; invalid input returns HTTP 422. Unknown fields and non-string values are rejected. The service loads the model once per process at startup using FastAPI lifespan. Missing models and inconsistent class mappings fail startup rather than serving a misleading healthy response.

Set `TERRYMOK_MODEL_PATH` to select another trusted local artifact. SHA-256 identifies the loaded file, not its trustworthiness. The service does not persist request bodies. This is a local demo with no authentication, rate limiting, or reverse-proxy body-size limit. It has a short local HTTP benchmark but has not been publicly deployed; request schema limits apply after JSON parsing. Docker deployment, per-process metrics and timed workload profiles are implemented; production rollout and external monitoring remain future work.

## Local HTTP benchmark

Start the API in one terminal, then run `python -m terrymok.benchmark --port 8000 --requests 200 --concurrency 4` in another. The target is restricted to IPv4 loopback. This sends four fixed synthetic English messages cyclically, one per request. Five serial warmup requests are excluded. The closed-loop thread pool issues a new request after completion, with a fresh urllib HTTP connection each time. End-to-end latency includes connection setup, server processing and response parsing; it excludes time queued before a worker starts.

The command saves `reports/api-benchmark.json` and exits nonzero on measured failures. Only HTTP 200 responses with the expected model digest and one recognized label count as successes. Percentiles use nearest rank over successes; failures are counted separately, so always read both. A failed health check or warmup aborts the run.

The [November 13 local snapshot](docs/api-benchmark.json) recorded 200/200 successful requests, p50 **7.98 ms**, p95 **20.75 ms**, p99 **27.69 ms**, and approximately **441 successful requests/s**, using four client threads and one Uvicorn process on Windows/Python 3.12. The measured phase lasted only **0.45 seconds**. This is a short smoke benchmark on one machine, not a sustained capacity, saturation or production SLA claim. The client and server share the host; the workload uses short repeated messages, no public network, no authentication and no proxy. Longer texts, batches, contention and deployment hardware can change results substantially. See the sustained real-model validation below for the subsequent Linux checks.

## Docker deployment

Build with `docker build -t terrymok:local .`. The image contains application code and dependencies only. The allowlisted build context excludes models, datasets, local agent documents, Git history and environment files. The API runs as UID 10001 and has a Docker health check.

Generate the model with prepare/experiment first, then mount the trusted artifact read-only. In PowerShell:

```powershell
$modelDir = (Resolve-Path artifacts).Path
docker run --rm -p 127.0.0.1:8000:8000 --mount "type=bind,source=$modelDir,target=/models,readonly" terrymok:local
```

On Linux/macOS:

```bash
docker run --rm -p 127.0.0.1:8000:8000 --mount "type=bind,source=$(pwd)/artifacts,target=/models,readonly" terrymok:local
```

The default model path is `/models/selected-model.joblib`; use `-e TERRYMOK_MODEL_PATH=/models/your-model.joblib` for another trusted artifact. Model files must be readable by UID 10001. Preserve the training library versions when deploying serialized scikit-learn models. Runtime dependencies are installed from requirements-runtime.lock with hash verification. The Python base-image tag and isolated setuptools build environment are not pinned by digest or lock, so complete image reproducibility is still pending.

CI builds the image and runs `python3 scripts/container_smoke.py`. The check creates a tiny synthetic model inside the same image, starts the API with a read-only model mount and dynamically assigned loopback port, verifies non-root execution, health, model identity, ordered predictions and HTTP 422, and removes its temporary container. It does not download the real corpus or publish an image. Run `docker build -t terrymok:smoke .` before invoking this check locally. The default CI smoke test covers container wiring; the separate real-model workflow below checks a reproduced Linux model and timed workloads. The development machine currently has no running Docker engine; container validation is performed by GitHub Actions.

## Dependency locks

`requirements-runtime.lock` pins the full API/ML dependency graph; `requirements-dev.lock` includes tests and is constrained by the runtime lock. Both include distribution hashes and environment markers for Python 3.11/3.12 and Windows/Linux. CI installs the dev lock in fresh runners; Docker installs the runtime lock, then the local project with `--no-deps`. Use the hash-checked Quick start above for development. Installing directly from pyproject extras without the lock still resolves floating versions.

Regenerate deliberately with uv 0.13.8 (see the [uv locking documentation](https://docs.astral.sh/uv/pip/compile/)):

```bash
uv pip compile pyproject.toml --extra api --universal --python-version 3.11 --generate-hashes -o requirements-runtime.lock
uv pip compile pyproject.toml --extra api --extra dev --universal --python-version 3.11 --generate-hashes -c requirements-runtime.lock -o requirements-dev.lock
```

Existing pins are preserved unless explicitly upgraded. After dependency edits, regenerate both locks, review the diff, run tests and the container smoke check. For a clean local reproduction use a new virtual environment: pip install does not remove unrelated packages already present. Python 3.11 has older NumPy/SciPy pins than 3.12; serialized models should use their training Python/library versions, not just any supported test environment. The original Python 3.12 model's scikit-learn 1.9.1, NumPy 2.5.4, SciPy 1.18.1 and joblib 1.6.0 remain pinned. FastAPI moves from the locally tested 0.162.4 to 0.163.1. Image digest pinning, build-tool pinning and real-model Linux transfer remain separate work.

## Operational monitoring and longer workloads

`GET /metrics` returns JSON aggregates for `/predict`: completed/in-flight requests, 2xx/3xx/4xx/5xx counts, duration sum and cumulative duration buckets in seconds. Counters are thread-safe, bounded, per process and reset on restart. Health/metrics scraping is excluded. Timings cover request handling until response construction, not complete network transfer. The total count acts as the histogram's final unbounded bucket. This is not Prometheus exposition, persistent telemetry, accuracy monitoring or drift detection; no message text is retained. Poll the endpoint with your monitoring system and aggregate multiple workers separately.

For timed load use `python -m terrymok.benchmark --duration 60 --concurrency 4 --batch-size 8 --text-repeat 10`. Nonzero duration replaces the request-count limit; up to 300 seconds, 32 threads, 32 texts per request and 100 text repetitions are allowed. Workers stop launching requests after the deadline; in-flight requests may extend elapsed time. Client p50/p95/p99 cover successful requests only; inspect the failure count too. Reports are overwritten at reports/api-benchmark.json, so save each profile separately if running manually.

See the [demonstration and resume guide](docs/demo.md). The manually triggered **Real model release validation** workflow rebuilds the fixed selected model from real 20 Newsgroups data in Linux, mounts it read-only in a separate API container, then executes three 60-second profiles: (concurrency 1, batch 1, repeat 1), (4, 1, 1), and (4, 8, 10). It checks HTTP results, artifact identity and server counter consistency. Only aggregate results are printed; source data and model files are ephemeral on the runner and are not uploaded. It does not tune on test or re-evaluate test scores, and it does not prove transfer of the existing Windows artifact.

The [completed Linux release validation](docs/release-validation.md) reproduced the original model SHA-256 and validation score, then served **92,116 measured requests without errors** over three 60-second profiles. At concurrency 4, p95 was **8.23 ms** for single short texts and **12.71 ms** for batches of eight repeated longer texts. These are single-run localhost CI measurements, not production guarantees. The [aggregate report](docs/release-validation.json) includes model identity and monitoring-counter reconciliation.

## Validation-only selective routing

Run `python -m terrymok.selective --target-accuracy 0.9 --min-accepted 30`
with the trusted local model and prepared data. This freezes the existing model
and reads only validation rows. It searches the gap between the two largest
multiclass decision scores, accepting `gap >= threshold`. These scores are not
probabilities. Equal gaps stay together. The lowest feasible threshold maximizes
coverage subject to the requested empirical accuracy and minimum accepted count.
If none qualifies, a null threshold means review everything; accuracy is then null.

The [November 16 aggregate snapshot](docs/selective-validation.json) selected a
gap of 0.262699: 351/453 messages accepted (77.48% coverage), 35 accepted errors,
90.03% accepted accuracy, and 102 messages for review. The full threshold curve
is generated locally in `reports/selective-validation.json`; no raw text is exported.
The report binds results to model/data digests and records library versions.

The policy can now be enabled in the API as described below; no human queue exists yet. Validation was already used for model selection,
so this additional tuning produces an optimistic selection-set estimate, not an
independent quality guarantee. The minimum count is a support constraint, not a
statistical confidence bound. Test data is not scored or used to choose the policy.
Before claiming deployment quality, freeze the policy and assess it on a new
independent holdout. Historical English forum messages do not establish ticket
routing accuracy. Next: opt-in API review flags with model-digest checks and
per-class coverage analysis.

## Opt-in review flags in the API

Generate the policy with `python -m terrymok.selective`, then set
`TERRYMOK_POLICY_PATH` before starting Uvicorn. In PowerShell:

```powershell
$env:TERRYMOK_POLICY_PATH = "reports/selective-validation.json"
python -m uvicorn terrymok.api:create_app --factory --host 127.0.0.1 --port 8000
```

On Linux use `TERRYMOK_POLICY_PATH=reports/selective-validation.json python -m uvicorn terrymok.api:create_app --factory`.
For Docker, mount the policy file read-only and set the variable to its container path.
Startup checks the model SHA-256, ordered labels, score definition, threshold rule
and multiclass score shape. Missing or incompatible policies fail startup.
The policy is loaded once; restart to change it. SHA-256 identifies content, not trust.

With a policy enabled, `/predict` adds `review_required` and `score_gap` to each
prediction and `policy_sha256` to the response; `/health` also reports policy identity.
A gap equal to the threshold is accepted; a null threshold marks every result for
review. Scores are not probabilities. Invalid runtime scores return an error.
Without the environment variable, the existing response format remains unchanged.
The label is still a suggestion when review is required: the caller must handle
that flag. This does not create a human queue, store texts, or guarantee accuracy.

## Category-level selective diagnostics

`python -m terrymok.selective` now includes `per_class` in the validation report.
`support`, `accepted`, `review`, `coverage`, `accepted_errors` and
`accepted_accuracy` group messages by their **true** class. `predicted_accepted`
and `accepted_precision` instead group accepted messages by their **predicted**
destination. Empty denominators are null, not perfect scores.

The November 18 run kept the same threshold and overall 351 accepted / 102 review:

| True category | Total | Accepted | Review | Accepted errors |
| --- | ---: | ---: | ---: | ---: |
| Graphics | 114 | 84 | 30 | 4 |
| Windows | 112 | 93 | 19 | 7 |
| PC hardware | 116 | 93 | 23 | 13 |
| Mac hardware | 111 | 81 | 30 | 11 |

Thus the 90.03% aggregate accepted accuracy does not imply 90% in every category:
PC hardware reaches 86.02% and Mac hardware 86.42% on this validation selection set.
See the updated aggregate report in docs/selective-validation.json for route precision.
These are descriptive diagnostics of the selected threshold, not independent quality
estimates; no threshold retuning or test evaluation was performed.

## Fixed-policy robustness checks

Run `python -m terrymok.robustness` after generating the selective policy. It checks
model/data identities and evaluates validation only, without fitting or selecting
thresholds. Profiles are unchanged text, swapping letters two and three in every
fifth ASCII word of at least four letters, and keeping the first 20 whitespace-separated
words. The counter resets per message; there is no randomness. Truncation also
normalizes whitespace and may remove information needed to infer the original label.

The November 21 [aggregate report](docs/robustness.json) records:

| Profile | Accepted | Review | Accepted errors | Accepted accuracy |
| --- | ---: | ---: | ---: | ---: |
| Original | 351 | 102 | 35 | 90.03% |
| Synthetic typos | 343 | 110 | 39 | 88.63% |
| First 20 words | 315 | 138 | 56 | 82.22% |

All profiles contain the same 453 validation messages and use the same frozen
threshold. Prediction changes are 25 for typos and 99 for truncation. More reviews
do not preserve the original accepted accuracy. These deterministic synthetic
stress checks are not independent evaluation, realistic typo-frequency estimates,
or a production drift detector. No test examples or raw messages are exported.

## Prediction telemetry

`GET /metrics` now includes `policy_sha256` (null when disabled) and a `predictions`
object. `messages` counts messages after successful inference, not HTTP requests;
`characters_sum` counts Python string characters after API whitespace stripping,
and `short_messages_le_20_words` counts texts with at most 20 whitespace-separated
words. `policy_messages` counts only messages evaluated with a review policy;
`review_messages` counts their review flags and `review_fraction` is their ratio
(null before any policy predictions). No text or per-message history is retained.

Counters are atomic, per process and reset with a new application instance. Failed
inference and invalid input are excluded; HTTP counters still track those failures.
Prediction completion is measured before response transmission, so these counters
are not proof of client delivery. Scraping does not increase them. Use counter
differences between scrapes for interval statistics, handle resets, and group by
model and policy identity. Multiple workers need separate collection.
These signals help inspect traffic changes; they do not measure accuracy or
establish drift. Automatic alert thresholds and an external collector remain pending.

## Comparing monitoring snapshots

Save two `/metrics` JSON responses from the same worker and run
`python -m terrymok.interval before.json after.json`. The command reports counter
differences, messages per second, mean characters, short-message fraction and
review fraction for that interval. Empty denominators produce null. Use UTF-8 JSON.

Metrics now include a random `instance_id` per application instance. A different
instance, model or policy, decreasing counters, or non-increasing uptime returns
`status: baseline_required`; retain the newer snapshot as the next baseline.
Missing identities or malformed counters fail rather than producing estimates.
Do not combine different workers behind a load balancer. This CLI reads saved
snapshots; it does not poll, persist traffic, send alerts or measure model accuracy.

## Explicit interval warnings

Use `python -m terrymok.interval before.json after.json --review-limit 0.4 --short-limit 0.7 --min-messages 100`
to add an `alerts` object. Both limits must be supplied and must be fractions in
[0, 1]. A warning occurs only when the interval fraction strictly exceeds its
limit. Review checks require at least `min-messages` policy-evaluated messages;
short-text checks require that many total messages. Insufficient samples are
reported per check; restarts or identity changes produce `not_evaluated`.

The example limits are illustrative operator choices, not calibrated statistical
thresholds. The offline synthetic test feeds 100 long accepted messages followed
by 100 short review-marked messages: only the second interval triggers both
warnings. It also checks exact boundaries, inadequate sample counts and restarts.
These flags are printed locally; they do not send notifications, establish drift,
measure accuracy, or replace a persistent external monitoring system. Warning
results retain exit code zero; consumers must inspect the JSON warnings list.

## Local polling collector

Start the API, then run:

```bash
python -m terrymok.collector --review-limit 0.4 --short-limit 0.7 --samples 10 --seconds 5 --min-messages 100 --consecutive 3
```

The collector polls `http://127.0.0.1:8000/metrics` (override port with `--port`)
and emits one JSON line per sample to stdout. The first sample establishes a
baseline; ten samples permit nine evaluated intervals. Poll spacing is a delay
after each fetch, not a fixed-rate schedule. HTTP timeout is five seconds.
Runs are bounded to 2..10000 samples with 0.1..60 seconds between samples.

Each signal becomes active after the configured consecutive eligible intervals
above its limit. Normal values, inadequate support and identity/reset changes
break the streak. Fetch or malformed-snapshot errors emit `collection_error`,
discard baseline and streaks, then retry at the next sample. Error output contains
only exception type. The next successful sample establishes a new baseline.
Active warnings repeat while the condition persists; they are level indicators,
not deduplicated notifications. All state is in memory and resets on collector
restart. For local persistence redirect stdout into the ignored `reports/` directory.
No external messages are sent. There is no daemon installation, dashboard or
accuracy guarantee; limits remain operator choices. Use a single API worker.

## HTTP monitoring integration check

Run `python -m pytest tests/test_collector_http.py -q`. This offline test starts a
real Uvicorn server on a reserved loopback socket, creates a trusted synthetic
three-class scoring fixture and matching policy, and sends four batches of two
messages over HTTP. The collector polls real `/metrics` responses: baseline,
normal traffic, two short/review intervals (both warnings activate), then normal
traffic (warnings clear). Final counters must show eight messages, four reviews
and four prediction requests. The server is shut down in a finally block.

This validates HTTP wiring and monitoring behavior, not model quality or the
collector CLI subprocess. Polling uses a nonzero delay to avoid identical monotonic
clock readings on Windows. No corpus download, persistent server or raw data export
is required. The check runs in the existing Python 3.11/3.12 CI test matrix.

Snapshot validation runs before baseline storage and before identity/reset checks.
It rejects non-object snapshots, missing/empty identities, nonfinite or negative
uptime, boolean/nonnumeric counters, inconsistent cumulative counts (review >
policy messages, policy messages > total, or short messages > total), and policy
counts without a policy identity. Both cumulative totals and interval differences
must be valid. Invalid samples become collection errors and discard collector
state; a subsequent valid sample starts a fresh baseline. Identity strings are
checked structurally, not authenticated. Derived fractions from incoming snapshots
are ignored and recalculated from counters.
