import pytest

from atlas import (
    HeuristicTokenizer,
    compress_trace,
    estimate_tokens,
    get_default_tokenizer,
    rollout,
    set_default_tokenizer,
)

from .helpers import Walker, make_walk


def test_heuristic_tokenizer_matches_estimate():
    tk = HeuristicTokenizer(chars_per_token=4.0)
    assert tk.count("abcd" * 10) == estimate_tokens("abcd" * 10)
    assert tk.count("") == 1


def test_estimate_tokens_accepts_explicit_tokenizer():
    class WordTokenizer:
        def count(self, text):
            return max(1, len(text.split()))

    assert estimate_tokens("one two three", tokenizer=WordTokenizer()) == 3


def test_default_tokenizer_is_used_by_compress_trace():
    class WordTokenizer:
        def count(self, text):
            return max(1, len(text.split()))

    original = get_default_tokenizer()
    try:
        set_default_tokenizer(WordTokenizer())
        trace = rollout(Walker(), make_walk(goal=5))
        c = compress_trace(trace)  # no explicit tokenizer -> uses default
        # Word count is much smaller than the char/4 estimate; just assert the
        # backend actually swapped (counts differ from the heuristic).
        assert c.compressed_tokens == len(c.text.split())
    finally:
        set_default_tokenizer(original)


def test_compress_trace_explicit_tokenizer_overrides_default():
    class ConstTokenizer:
        def count(self, text):
            return 7

    trace = rollout(Walker(), make_walk(goal=3))
    c = compress_trace(trace, tokenizer=ConstTokenizer())
    assert c.raw_tokens == 7 and c.compressed_tokens == 7


def test_tiktoken_backend_if_available():
    pytest.importorskip("tiktoken")
    from atlas import TiktokenTokenizer

    tk = TiktokenTokenizer()
    assert tk.count("hello world") >= 1
