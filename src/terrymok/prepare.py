"""Deterministic cleaning and validation separation before model fitting."""

import json
from collections import defaultdict
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from terrymok.audit import fingerprint, audit_splits


def clean_split(texts, labels):
    if len(texts) != len(labels):
        raise ValueError("Texts and labels must have equal lengths")
    groups = defaultdict(list)
    for text, label in zip(texts, labels):
        key = fingerprint(text)
        if key is not None:
            groups[key].append((text, int(label)))
    kept = []
    conflicts = duplicates = 0
    for rows in groups.values():
        if len({label for _, label in rows}) > 1:
            conflicts += len(rows)
        else:
            kept.append(rows[0])
            duplicates += len(rows) - 1
    return kept, {
        "input_rows": len(texts),
        "empty_rows_removed": len(texts) - sum(map(len, groups.values())),
        "conflicting_rows_removed": conflicts,
        "duplicate_rows_removed": duplicates,
    }


def prepare_splits(train_texts, train_labels, test_texts, test_labels, seed=42):
    # Train cleaning uses only train labels. Test labels never affect training.
    train, train_report = clean_split(train_texts, train_labels)
    test, test_report = clean_split(test_texts, test_labels)
    train_keys = {fingerprint(text) for text in train_texts} - {None}
    independent_test = [row for row in test if fingerprint(row[0]) not in train_keys]
    test_report["overlap_rows_removed"] = len(test) - len(independent_test)
    if not train or not independent_test:
        raise ValueError("Cleaning left an empty training or test split")
    fit, validation = train_test_split(
        train, test_size=0.2, random_state=seed,
        stratify=[label for _, label in train],
    )
    splits = {"train": fit, "validation": validation, "test": independent_test}
    report = {
        "policy": "v1: remove blanks, remove all conflicting groups within each split, keep first duplicate; remove test overlap with original train; stratified 80/20 train split",
        "random_state": seed, "train_cleaning": train_report,
        "test_cleaning": test_report,
        "sizes": {name: len(rows) for name, rows in splits.items()},
        "overlap": {
            f"{left}_{right}": audit_splits(
                [text for text, _ in splits[left]], [text for text, _ in splits[right]],
            )["overlap_unique_messages"]
            for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]
        },
    }
    return splits, report


def main():
    import platform
    import sklearn
    from sklearn.datasets import fetch_20newsgroups
    from terrymok.train import CATEGORIES

    options = dict(categories=CATEGORIES, remove=("headers", "footers", "quotes"),
                   random_state=42, data_home="data")
    train = fetch_20newsgroups(subset="train", **options)
    test = fetch_20newsgroups(subset="test", **options)
    splits, report = prepare_splits(train.data, train.target, test.data, test.target)
    report.update(python_version=platform.python_version(), sklearn_version=sklearn.__version__,
                  categories=train.target_names)
    Path("data").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump({"splits": splits, "labels": train.target_names}, "data/prepared.joblib")
    Path("reports/preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
