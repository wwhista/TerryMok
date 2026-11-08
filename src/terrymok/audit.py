"""Count empty messages and normalized exact duplicates without exporting text."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence


def fingerprint(text: str) -> str | None:
    """Ignore case and repeated whitespace; empty messages have no fingerprint."""
    normalized = " ".join(text.casefold().split())
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def audit_splits(train: Sequence[str], test: Sequence[str]) -> dict:
    train_keys = [fingerprint(text) for text in train]
    test_keys = [fingerprint(text) for text in test]
    train_counts = Counter(key for key in train_keys if key is not None)
    test_counts = Counter(key for key in test_keys if key is not None)
    overlap = train_counts.keys() & test_counts.keys()

    def summarize(keys, counts):
        return {
            "rows": len(keys),
            "empty_rows": keys.count(None),
            "unique_nonempty_messages": len(counts),
            "duplicate_extra_rows": sum(count - 1 for count in counts.values()),
        }

    return {
        "normalization": "Unicode casefold and collapsed whitespace; SHA-256",
        "train": summarize(train_keys, train_counts),
        "test": summarize(test_keys, test_counts),
        "overlap_unique_messages": len(overlap),
        "overlap_train_rows": sum(train_counts[key] for key in overlap),
        "overlap_test_rows": sum(test_counts[key] for key in overlap),
    }


def main() -> None:
    from sklearn.datasets import fetch_20newsgroups

    from terrymok.train import CATEGORIES

    options = dict(categories=CATEGORIES, remove=("headers", "footers", "quotes"),
                   random_state=42, data_home="data")
    train = fetch_20newsgroups(subset="train", **options)
    test = fetch_20newsgroups(subset="test", **options)
    report = {
        "dataset": "20newsgroups", "categories": train.target_names,
        "remove": list(options["remove"]),
        "audit": audit_splits(train.data, test.data),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
