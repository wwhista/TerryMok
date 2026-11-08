import pytest

from terrymok.prepare import clean_split, prepare_splits


def fixture_data():
    return [f"message {i}" for i in range(30)], [i % 2 for i in range(30)]


def test_cleaning_removes_all_conflicting_rows_and_preserves_alignment():
    rows, report = clean_split(["", "A", " a ", "b", "B", "c"], [0, 0, 1, 1, 1, 0])
    assert rows == [("b", 1), ("c", 0)]
    assert report["conflicting_rows_removed"] == 2
    assert report["empty_rows_removed"] == 1
    assert report["duplicate_rows_removed"] == 1


def test_splits_are_reproducible_disjoint_and_test_labels_cannot_change_training():
    texts, labels = fixture_data()
    result, report = prepare_splits(texts, labels, ["MESSAGE 0", "new", "NEW", "other"], [1, 0, 0, 1])
    again, _ = prepare_splits(texts, labels, ["MESSAGE 0", "new", "NEW", "other"], [0, 1, 1, 0])
    assert result["train"] == again["train"]
    assert result["validation"] == again["validation"]
    assert result["test"] == [("new", 0), ("other", 1)]
    assert report["sizes"] == {"train": 24, "validation": 6, "test": 2}
    assert all(value == 0 for value in report["overlap"].values())
    assert {label for _, label in result["validation"]} == {0, 1}


def test_invalid_input_and_empty_test_fail_explicitly():
    with pytest.raises(ValueError, match="equal lengths"):
        clean_split(["text"], [])
    texts, labels = fixture_data()
    with pytest.raises(ValueError, match="empty"):
        prepare_splits(texts, labels, ["message 0"], [0])
