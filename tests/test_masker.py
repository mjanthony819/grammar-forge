"""Unit tests for the token masker: the sampler's only grammar gate.

At every decoding step engine.py asks TokenMasker which vocabulary tokens
the parser could still consume, and forces every other logit to -inf. A
mask that is too wide lets invalid JSON through; too narrow (or a missing
EOS) stalls generation. The Earley engine, the JSON automaton and their
equivalence are all tested, but nothing exercised the masker itself.

Run: python tests/test_masker.py
"""
import sys

sys.path.insert(0, ".")
from gce.earley import EarleyParser
from gce.grammar import json_grammar
from gce.masker import TokenMasker

EOS = 99

# A small vocab with the shapes a real BPE vocab contains: structural
# chars, keywords whole and in pieces, digits, whitespace runs, and junk.
VOCAB = [
    "{", "}", "[", "]", '"', ":", ",",      # 0-6 structural
    "0", "1", "12", "-",                     # 7-10 number pieces
    "true", "false", "null",                 # 11-13 keywords
    "n", "ul", "ull", "t", "rue",            # 14-18 keyword pieces
    " ", "  ", "\n", "\t",                   # 19-22 whitespace
    "x", "e", "+",                           # 23-25 junk that never fits
    "",                                      # 26 empty surface
]


def fresh():
    return EarleyParser(json_grammar())


def feed(parser, text):
    for ch in text:
        assert parser.advance(ch), f"parser rejected {text!r} at {ch!r}"


def expected(parser, banned=frozenset()):
    """Ground truth: ask the parser directly about every token."""
    out = set()
    for tid, text in enumerate(VOCAB):
        if text and tid not in banned and parser.can_consume(text):
            out.add(tid)
    if parser.accepts():
        out.add(EOS)
    return frozenset(out)


# 1. Masker agrees with the parser at the start state.
p = fresh()
m = TokenMasker(VOCAB, EOS)
assert m.allowed_ids(p) == expected(p), "start-state mask disagrees with parser"

# 2. First-step survivors are exactly the docstring's set: tokens starting
# with '{', '[', a quote, a digit, '-', 't', 'f', 'n', or whitespace.
start_allowed = {VOCAB[t] for t in m.allowed_ids(p)}
for tok in ["{", "[", '"', "0", "1", "12", "-", "true", "false", "null", "n", "t", " ", "  ", "\n", "\t"]:
    assert tok in start_allowed, f"{tok!r} should be allowed at start"
for tok in ["}", "]", ":", ",", "x", "e", "+"]:
    assert tok not in start_allowed, f"{tok!r} should be rejected at start"
assert EOS not in m.allowed_ids(p), "EOS allowed before the document is complete"

# 3. Mid-terminal continuation: after consuming 'n', the pieces of 'null'
# are allowed and everything unrelated is not.
p = fresh()
feed(p, "n")
mid = {VOCAB[t] for t in m.allowed_ids(p)}
assert "ul" in mid and "ull" in mid, "keyword pieces must continue a partial token"
assert "n" not in mid and "true" not in mid and "{" not in mid

# 4. Once the document is complete, EOS is allowed; when no continuation
# remains, EOS is alone. 'true' ends with no possible continuation, while
# a bare number can still grow ('1' -> '12'), so there EOS competes with
# digit tokens and both must be offered.
p = fresh()
feed(p, "true")
assert p.accepts()
assert m.allowed_ids(p) == frozenset({EOS}), "EOS must be alone after a keyword-terminated document"
p = fresh()
feed(p, "1")
assert p.accepts()
num_mask = {VOCAB[t] for t in m.allowed_ids(p) if t != EOS}
assert EOS in m.allowed_ids(p), "EOS missing on a completed number"
assert "0" in num_mask and "12" in num_mask, "a completed number must still be extendable"

# 5. Banned ids are excluded even when the parser would consume them, and
# the empty-surface token is never offered.
p = fresh()
m2 = TokenMasker(VOCAB, EOS, banned_ids={VOCAB.index("{")})
masked = {VOCAB[t] for t in m2.allowed_ids(p)}
assert "{" not in masked and "[" in masked, "banned id leaked into the mask"
assert VOCAB.index("") not in m.allowed_ids(p), "empty token offered"

# 6. Cache: one entry per distinct parser-state signature, and repeat calls
# return the identical frozenset.
p1, p2 = fresh(), fresh()
m3 = TokenMasker(VOCAB, EOS)
a = m3.allowed_ids(p1)
assert m3.allowed_ids(p2) is a, "same state must hit the cache"
assert m3.cache_size == 1
feed(p1, "{")
m3.allowed_ids(p1)
assert m3.cache_size == 2, "advanced state must cost a new cache entry"

print("masker OK: parser agreement + first-step set + mid-terminal + EOS-only + bans + cache")
