"""Token masker: turns parser state into the set of allowed token ids.

At each decoding step the sampler asks the masker which vocabulary tokens
may come next. A token is allowed iff the parser could consume the token's
entire surface text from its current state (the token may end mid-terminal,
e.g. inside a string or in the middle of `null`). When the parser accepts
the document as complete, EOS is the single allowed token, so generation
stops cleanly instead of trailing off.

Results are cached by parser-state signature, so each distinct grammar
state pays the vocab scan once per process.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple


class TokenMasker:
    def __init__(
        self,
        vocab_texts: List[str],
        eos_id: int,
        banned_ids: Optional[Set[int]] = None,
    ):
        self.vocab_texts = vocab_texts
        self.eos_id = eos_id
        self.banned_ids = set(banned_ids or ())
        self._cache: Dict[Tuple, frozenset] = {}

    def allowed_ids(self, parser) -> frozenset:
        sig = parser.signature()
        hit = self._cache.get(sig)
        if hit is not None:
            return hit
        allowed: Set[int] = set()
        for tid, text in enumerate(self.vocab_texts):
            if not text or tid in self.banned_ids:
                continue
            if parser.can_consume(text):
                allowed.add(tid)
        if parser.accepts():
            allowed.add(self.eos_id)
        out = frozenset(allowed)
        self._cache[sig] = out
        return out

    @property
    def cache_size(self) -> int:
        return len(self._cache)
