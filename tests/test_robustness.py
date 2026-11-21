import pytest
from terrymok.robustness import stress, typo


def test_typo_is_deterministic_and_preserves_other_characters():
    text = 'alpha bravo charlie delta echo! 123'
    assert typo(text) == 'alpha bravo charlie delta ehco! 123'
    assert typo(text) == typo(text)


class Frozen:
    def decision_function(self, texts):
        return [[2, 0, 0] if len(text.split()) > 20 else [0, 0.1, 0] for text in texts]


def test_fixed_threshold_exposes_truncation_and_review():
    report = stress(Frozen(), [('word ' * 21, 0)], ['a', 'b', 'c'], 1)
    assert report['original']['accepted_accuracy'] == 1
    assert report['first_20_words']['prediction_changes'] == 1
    assert report['first_20_words']['review'] == 1
    assert report['first_20_words']['accepted_accuracy'] is None
    assert report['first_20_words']['accuracy'] == 0
    assert stress(Frozen(), [('word', 0)], ['a', 'b', 'c'], None)['original']['accepted'] == 0


@pytest.mark.parametrize('threshold', [-1, float('nan'), True])
def test_bad_threshold(threshold):
    with pytest.raises(ValueError):
        stress(Frozen(), [('word', 0)], ['a', 'b', 'c'], threshold)
