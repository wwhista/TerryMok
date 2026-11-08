# TerryMok

Reproducible classification of English technical messages with TF-IDF and logistic regression. An ML engineering portfolio project covering data separation, baseline evaluation, offline tests, and CI.

**Status:** initial implementation with offline tests and CI on Python 3.11/3.12. Full corpus training has not yet been run; no benchmark results are claimed.

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

Installation requires internet. Tests use tiny synthetic fixtures and do not download datasets. The training command downloads the corpus into `data/`, evaluates once on the original test split, and writes `artifacts/model.joblib` and `reports/metrics.json`. These generated directories are ignored by Git. Run commands from the repository root.

## Evaluation

Before training, run `.venv\Scripts\python -m terrymok.audit` (Linux/macOS: `.venv/bin/python -m terrymok.audit`). This separate network-enabled command downloads the same four-category corpus and writes aggregate counts to `reports/data_audit.json`. It counts empty messages, duplicate extra rows within each split, and cross-split overlap after Unicode case folding and whitespace normalization. Empty messages do not count as overlap. No raw text or message hashes are exported. The audit is diagnostic: it does not modify either split, detect near-duplicates, check conflicting labels, or remove leakage automatically. Resolve a cleaning protocol before treating future test metrics as a benchmark.

The [first audit snapshot](docs/data-audit.json), run on November 8, 2026, found:

| Split | Rows | Empty messages | Duplicate extra rows |
| --- | ---: | ---: | ---: |
| Train | 2,343 | 68 | 8 |
| Test | 1,560 | 40 | 1 |

Three distinct nonempty normalized messages appear in both splits, covering three training rows and four test rows. This is a data-quality finding, not a model score. Cleaning and validation separation are the next steps; the current training entry point still uses the original uncleaned splits.

The pipeline fits TF-IDF on training messages only. Logistic regression uses a fixed seed of 42 and is compared with a majority-class DummyClassifier. Reports contain accuracy, macro-F1, per-class precision/recall/F1, sample counts, categories, and Python/scikit-learn versions.

The initial baseline has no hyperparameter search. Future tuning must use a validation split inside train, leaving test untouched. Dependency ranges are declared in `pyproject.toml`; a fully locked environment and a validated cleaning protocol are still pending.

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
