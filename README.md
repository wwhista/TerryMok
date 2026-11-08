# TerryMok

Reproducible classification of English technical messages with TF-IDF and logistic regression. An ML engineering portfolio project covering data separation, baseline evaluation, offline tests, and CI.

**Status:** initial implementation with offline tests and CI on Python 3.11/3.12. Data preparation is implemented and tested; model selection and benchmark analysis are still pending.

## Problem and data

Predict one of four technical discussion categories: computer graphics, Windows, IBM PC hardware, or Mac hardware. This is a prototype of topic routing; it is not a validated customer-support system.

The [20 Newsgroups loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) provides the corpus and its original train/test split. Headers, footers, and quotes are removed to reduce metadata shortcuts. The corpus is historical and English-only; forum classification results do not establish quality on modern support tickets. Review dataset terms before redistributing source messages. Raw data is not included in this repository.

## Quick start

Install Python 3.11 or 3.12 first (3.12 recommended). From the repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m terrymok.train
```

On Linux/macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
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

The initial baseline has no hyperparameter search. Future tuning must use a validation split inside train, leaving test untouched. Dependency ranges are declared in `pyproject.toml`; a fully locked environment and hyperparameter selection are still pending.

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
