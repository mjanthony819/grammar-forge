"""Compiled JSON pushdown automaton: the fast production recognizer.

This is a hand-compiled form of the character-level JSON CFG in
grammar.py. Each CFG derivation state maps to a small stack of frames;
every frame is a tuple, so the whole machine state is a hashable tuple.
That gives us the two properties the constrained sampler needs:

1. O(1) work per character (no Earley set operations in the hot path).
2. A cheap state signature (the stack tuple itself), so token-mask results
   can be cached per state. All positions inside a string, for example,
   share one signature regardless of the string's content.

The test suite (tests/test_equivalence.py) property-checks that this
machine accepts exactly the same language as the reference Earley parser
over the CFG, and offers exactly the same next characters at every prefix
of valid and mutated documents.

Frame vocabulary (each frame is a tuple, top of stack = last element):
  ('value',)          expecting any JSON value
  ('obj_first',)      after '{': a key string or '}'
  ('obj_key',)        after ',' in an object: a key string
  ('colon',)          expecting ':' after a key
  ('after_pair',)     after a pair: ',' or '}'
  ('arr_first',)      after '[': a value or ']'
  ('after_elem',)     after an element: ',' or ']'
  ('str',)            inside a string
  ('esc',)            after a backslash
  ('uni', n)          n hex digits still needed
  ('lit', rest)       remaining characters of true/false/null
  ('num', sub)        number sub-state: start int0 zero digits frac0 fracd
                      exp0 exps expd
"""
from __future__ import annotations

from typing import List, Tuple

WS = " \t\n\r"
DIGITS = "0123456789"
DIGITS19 = "123456789"
HEXS = "0123456789abcdefABCDEF"
ESC_SIMPLES = '"\\/bfnrt'

_NUM_COMPLETABLE = {"zero", "digits", "fracd", "expd"}


class JSONAutomaton:
    """Incremental JSON recognizer. feed() one char at a time."""

    __slots__ = ("stack",)

    def __init__(self):
        self.stack: List[Tuple] = [("value",)]

    # -- helpers ----------------------------------------------------------
    def clone(self) -> "JSONAutomaton":
        other = JSONAutomaton.__new__(JSONAutomaton)
        other.stack = list(self.stack)
        return other

    def signature(self) -> Tuple:
        return tuple(self.stack)

    def _settle(self) -> None:
        """A child construct just completed: pop every 'value' frame whose
        value is now finished."""
        while self.stack and self.stack[-1][0] == "value":
            self.stack.pop()

    def accepts(self) -> bool:
        """True iff the consumed input is a complete JSON document.

        A bare number at top level is complete once it is in a sub-state
        that is allowed to end (so '42' and '-1.5e3' are accepted, while
        '1.' and '1e' are not).
        """
        if not self.stack:
            return True
        if len(self.stack) == 2 and self.stack[0] == ("value",):
            top = self.stack[-1]
            return top[0] == "num" and top[1] in _NUM_COMPLETABLE
        return False

    # -- main entry -------------------------------------------------------
    def feed(self, ch: str) -> bool:
        """Consume one character. Returns False if it is not allowed."""
        while True:
            if not self.stack:
                return False  # document already complete
            top = self.stack[-1]
            kind = top[0]

            if kind == "value":
                if ch in WS:
                    return True
                if ch == "{":
                    self.stack.append(("obj_first",))
                    return True
                if ch == "[":
                    self.stack.append(("arr_first",))
                    return True
                if ch == '"':
                    self.stack.append(("str",))
                    return True
                if ch == "t":
                    self.stack.append(("lit", "rue"))
                    return True
                if ch == "f":
                    self.stack.append(("lit", "alse"))
                    return True
                if ch == "n":
                    self.stack.append(("lit", "ull"))
                    return True
                if ch == "-" or ch in DIGITS:
                    self.stack.append(("num", "start"))
                    continue  # let the number frame process this char
                return False

            if kind == "obj_first":
                if ch in WS:
                    return True
                if ch == '"':
                    self.stack[-1] = ("colon",)
                    self.stack.append(("str",))
                    return True
                if ch == "}":
                    self.stack.pop()
                    self._settle()
                    return True
                return False

            if kind == "obj_key":
                if ch in WS:
                    return True
                if ch == '"':
                    self.stack[-1] = ("colon",)
                    self.stack.append(("str",))
                    return True
                return False

            if kind == "colon":
                if ch in WS:
                    return True
                if ch == ":":
                    self.stack[-1] = ("after_pair",)
                    self.stack.append(("value",))
                    return True
                return False

            if kind == "after_pair":
                if ch in WS:
                    return True
                if ch == ",":
                    self.stack[-1] = ("obj_key",)
                    return True
                if ch == "}":
                    self.stack.pop()
                    self._settle()
                    return True
                return False

            if kind == "arr_first":
                if ch in WS:
                    return True
                if ch == "]":
                    self.stack.pop()
                    self._settle()
                    return True
                # any value start: become after_elem and push a value frame
                self.stack[-1] = ("after_elem",)
                self.stack.append(("value",))
                continue

            if kind == "after_elem":
                if ch in WS:
                    return True
                if ch == ",":
                    # stay after_elem underneath: when the new value
                    # completes we must be back here, not in arr_first
                    self.stack.append(("value",))
                    return True
                if ch == "]":
                    self.stack.pop()
                    self._settle()
                    return True
                return False

            if kind == "str":
                if ch == '"':
                    self.stack.pop()
                    self._settle()
                    return True
                if ch == "\\":
                    self.stack.append(("esc",))
                    return True
                if ord(ch) >= 0x20:
                    return True
                return False

            if kind == "esc":
                if ch in ESC_SIMPLES:
                    self.stack.pop()
                    return True
                if ch == "u":
                    self.stack[-1] = ("uni", 4)
                    return True
                return False

            if kind == "uni":
                if ch in HEXS:
                    if top[1] > 1:
                        self.stack[-1] = ("uni", top[1] - 1)
                    else:
                        self.stack.pop()
                    return True
                return False

            if kind == "lit":
                rest = top[1]
                if ch == rest[0]:
                    if len(rest) > 1:
                        self.stack[-1] = ("lit", rest[1:])
                    else:
                        self.stack.pop()
                        self._settle()
                    return True
                return False

            if kind == "num":
                sub = top[1]
                if sub == "start":
                    if ch == "-":
                        self.stack[-1] = ("num", "int0")
                        return True
                    if ch == "0":
                        self.stack[-1] = ("num", "zero")
                        return True
                    if ch in DIGITS19:
                        self.stack[-1] = ("num", "digits")
                        return True
                    return False
                if sub == "int0":
                    if ch == "0":
                        self.stack[-1] = ("num", "zero")
                        return True
                    if ch in DIGITS19:
                        self.stack[-1] = ("num", "digits")
                        return True
                    return False
                if sub == "zero" or sub == "digits":
                    if sub == "digits" and ch in DIGITS:
                        return True
                    if ch == ".":
                        self.stack[-1] = ("num", "frac0")
                        return True
                    if ch in "eE":
                        self.stack[-1] = ("num", "exp0")
                        return True
                    # number complete; do not consume the delimiter
                    self.stack.pop()
                    self._settle()
                    continue
                if sub == "frac0":
                    if ch in DIGITS:
                        self.stack[-1] = ("num", "fracd")
                        return True
                    return False
                if sub == "fracd":
                    if ch in DIGITS:
                        return True
                    if ch in "eE":
                        self.stack[-1] = ("num", "exp0")
                        return True
                    self.stack.pop()
                    self._settle()
                    continue
                if sub == "exp0":
                    if ch in DIGITS:
                        self.stack[-1] = ("num", "expd")
                        return True
                    if ch in "+-":
                        self.stack[-1] = ("num", "exps")
                        return True
                    return False
                if sub == "exps":
                    if ch in DIGITS:
                        self.stack[-1] = ("num", "expd")
                        return True
                    return False
                if sub == "expd":
                    if ch in DIGITS:
                        return True
                    self.stack.pop()
                    self._settle()
                    continue
                return False

            return False

    def can_consume(self, text: str) -> bool:
        """Could this machine consume the whole token string, without
        mutating its state?"""
        sim = self.clone()
        for ch in text:
            if not sim.feed(ch):
                return False
        return True
