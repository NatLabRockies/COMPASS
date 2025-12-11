"""Test ngram utilities"""

from pathlib import Path

import pytest

from compass.utilities.ngrams import (
    _check_word,
    _filtered_words,
    sentence_ngram_containment,
    convert_text_to_sentence_ngrams,
)


def test_check_word_filters_common_terms_and_punctuation():
    """Test `_check_word` rejects stop words and punctuation"""

    assert not _check_word("the")
    assert not _check_word(",")
    assert _check_word("solar")


def test_filtered_words_removes_noise_tokens():
    """Test `_filtered_words` only returns significant tokens"""

    sentence = "The solar arrays, and storage!"
    assert _filtered_words(sentence) == [
        "solar",
        "arrays",
        "storage",
        "!",
    ]


def test_convert_text_to_sentence_ngrams_multiple_sentences():
    """Test `convert_text_to_sentence_ngrams` builds ngrams per sentence"""

    text = "The solar arrays store energy. Solar storage thrives."
    assert convert_text_to_sentence_ngrams(text, 2) == [
        ("solar", "arrays"),
        ("arrays", "store"),
        ("store", "energy"),
        ("solar", "storage"),
        ("storage", "thrives"),
    ]


def test_sentence_ngram_containment_computes_fraction():
    """Test `sentence_ngram_containment` returns containment ratio"""

    original = "Solar arrays store energy. Batteries support solar arrays."
    test_text = "Solar arrays store energy. Solar arrays fail."
    result = sentence_ngram_containment(original, test_text, 2)
    assert result == pytest.approx(0.8)


def test_sentence_ngram_containment_handles_empty_test_text():
    """Test containment logic handles empty-or-stopword sentences"""

    assert sentence_ngram_containment("", "The and is", 2) == 0.0


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
