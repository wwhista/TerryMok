# TerryMok selected model

## Intended use and scope

Portfolio prototype for classifying English technical discussion messages into graphics, Windows, IBM PC hardware, and Mac hardware. This historical 20 Newsgroups corpus is not a modern support-ticket dataset. The model is not validated for production routing, other languages, or out-of-domain messages.

## Model and selection

Character-within-word TF-IDF (3–5 grams, sublinear TF) with LinearSVC, C=1, random_state=42. Selected from six fixed candidates by validation macro-F1 (0.8048). Vocabulary and estimator were fitted on 1,812 training messages only; 453 validation messages were used for selection. No refit on validation was performed. SVM scores are not calibrated probabilities.

## Data and evaluation

Headers, footers, and quotations were stripped. Blank messages, within-split conflicting-label groups, duplicate messages, and test overlap with original train were removed. See [preparation report](preparation.json). Normalized exact duplicates are detected; near-duplicates are not. Removing test examples changes the evaluated population, so these results are not directly comparable with unmodified 20 Newsgroups benchmarks.

The frozen selected artifact was evaluated on November 10, 2026, on 1,516 cleaned test messages. Evaluation invokes prediction only and performs no model selection or fitting. The earlier fixed logistic-regression baseline had already been evaluated on this test set; subsequent candidate selection used validation only. The test set should not guide further parameter choices.

| Model | Test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| Fixed word logistic regression | 0.7243 | 0.7238 |
| Validation-selected character SVM | 0.7579 | 0.7567 |

The majority baseline test macro-F1 was 0.1011. The selected model improves macro-F1 by about 0.033 on this split. No confidence interval or statistical-significance claim is provided. Full [selected metrics and confusion matrix](selected-test-metrics.json) and [baseline metrics](baseline-metrics.json) are available.

## Aggregate error analysis

There were 367 mistakes. The largest directed confusions were Windows to graphics (53) and IBM hardware to Mac hardware (53). The reverse Mac-to-IBM confusion contributed 43 more mistakes. IBM hardware had the lowest class F1 (0.7092), while graphics had the highest (0.8115). These counts establish where mistakes occur, not why; semantic explanations require inspecting examples. Manual review of at least 20 errors remains pending.

## Reproduction and artifacts

With the project environment installed, run `python -m terrymok.prepare`, `python -m terrymok.experiment`, then `python -m terrymok.evaluate`. Only load locally generated, trusted joblib artifacts. Evaluation checks the label mapping and selected candidate name and records SHA-256 digests of the model and prepared dataset, plus Python/scikit-learn versions. Digests identify the evaluated artifacts; they do not establish their provenance or replace trust checks. Dependencies are not yet fully locked.

The published report contains aggregates, not messages. Raw corpus, model, and local reports remain ignored by Git. This model has no abstention policy, drift monitoring, fairness assessment, deployment latency validation, or modern-ticket evaluation yet.
