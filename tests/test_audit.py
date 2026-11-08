from terrymok.audit import audit_splits, fingerprint


def test_normalization_preserves_punctuation_and_ignores_case_and_whitespace():
    assert fingerprint(" Hello\n WORLD ") == fingerprint("hello world")
    assert fingerprint("hello!") != fingerprint("hello")
    assert fingerprint(" \t\n") is None


def test_counts_duplicate_rows_and_cross_split_overlap_without_empty_matches():
    report = audit_splits(["Alpha", " ALPHA ", "", "beta"], ["alpha", "alpha", " ", "gamma"])
    assert report["train"] == {
        "rows": 4, "empty_rows": 1, "unique_nonempty_messages": 2, "duplicate_extra_rows": 1,
    }
    assert report["test"] == report["train"]
    assert report["overlap_unique_messages"] == 1
    assert report["overlap_train_rows"] == 2
    assert report["overlap_test_rows"] == 2


def test_empty_splits_and_empty_messages_do_not_create_overlap():
    report = audit_splits([], ["", "\n"])
    assert report["train"]["rows"] == 0
    assert report["test"]["empty_rows"] == 2
    assert report["test"]["duplicate_extra_rows"] == 0
    assert report["overlap_unique_messages"] == 0


def test_report_never_contains_source_text():
    import json

    text = "private source message"
    assert text not in json.dumps(audit_splits([text], [text]))
