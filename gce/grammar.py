"""Context-free grammar definitions.

A grammar is a list of rules ``A -> alpha`` where ``alpha`` is a sequence of
symbols. A symbol is either a nonterminal (the name of another rule) or a
Terminal that matches exactly one input character. Multi-character keywords
such as ``true`` are written as one single-character terminal per letter.

Because every terminal matches exactly one character, an incremental parser
over the grammar can consume a multi-character token one character at a
time. That is exactly what token-level grammar-constrained decoding needs:
at every decoding step we ask "which vocabulary tokens could the parser
still consume?", and mask out everything else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union


@dataclass(frozen=True)
class Terminal:
    """A single-character terminal symbol of a CFG."""

    name: str
    predicate: Callable[[str], bool]

    def matches(self, ch: str) -> bool:
        return self.predicate(ch)


def L(ch: str) -> Terminal:
    """Literal-character terminal."""
    assert len(ch) == 1, "L() takes exactly one character"
    return Terminal(repr(ch), lambda c, ch=ch: c == ch)


# --- Character-class terminals used by the JSON grammar -------------------
DIGIT = Terminal("DIGIT", lambda c: c in "0123456789")
DIGIT19 = Terminal("DIGIT19", lambda c: c in "123456789")
HEX = Terminal("HEX", lambda c: c in "0123456789abcdefABCDEF")
WS_CHAR = Terminal("WS_CHAR", lambda c: c in " \t\n\r")
NORMAL = Terminal("NORMAL", lambda c: c not in '"\\' and ord(c) >= 0x20)
ESC_SIMPLE = Terminal("ESC_SIMPLE", lambda c: c in '"\\/bfnrt')
EXP_MARK = Terminal("EXP_MARK", lambda c: c in "eE")
EXP_SIGN = Terminal("EXP_SIGN", lambda c: c in "+-")

Symbol = Union[str, Terminal]


class Grammar:
    """A context-free grammar: start symbol plus a list of rules."""

    def __init__(self, start: str):
        self.start = start
        self.rules: List[Tuple[str, Tuple[Symbol, ...]]] = []
        self.by_lhs: Dict[str, List[int]] = {}
        self._nullable = None

    def add(self, lhs: str, *alternatives: List[Symbol]) -> "Grammar":
        for alt in alternatives:
            idx = len(self.rules)
            self.rules.append((lhs, tuple(alt)))
            self.by_lhs.setdefault(lhs, []).append(idx)
        self._nullable = None
        return self

    @property
    def nullable(self):
        """Nonterminals that can derive the empty string (fixed point)."""
        if self._nullable is None:
            nullable = set()
            changed = True
            while changed:
                changed = False
                for lhs, rhs in self.rules:
                    if lhs not in nullable and all(
                        isinstance(s, str) and s in nullable for s in rhs
                    ):
                        nullable.add(lhs)
                        changed = True
            self._nullable = nullable
        return self._nullable

    def check(self) -> None:
        """Sanity check: every referenced nonterminal is defined."""
        for lhs, rhs in self.rules:
            for sym in rhs:
                if isinstance(sym, str):
                    assert sym in self.by_lhs, f"undefined nonterminal: {sym}"
        assert self.start in self.by_lhs, "start symbol undefined"


def json_grammar() -> Grammar:
    """The JSON grammar (RFC 8259) as a character-level CFG.

    document -> ws value          (no trailing content: generation stops
                                   the moment the value is complete)
    """
    g = Grammar("document")

    g.add("document", ["ws", "value"])
    g.add("ws", [], [WS_CHAR, "ws"])

    g.add(
        "value",
        ["object"],
        ["array"],
        ["string"],
        ["number"],
        ["lit_true"],
        ["lit_false"],
        ["lit_null"],
    )
    g.add("lit_true", [L("t"), L("r"), L("u"), L("e")])
    g.add("lit_false", [L("f"), L("a"), L("l"), L("s"), L("e")])
    g.add("lit_null", [L("n"), L("u"), L("l"), L("l")])

    # object -> '{' ws '}' | '{' ws members ws '}'
    g.add("object", [L("{"), "ws", L("}")], [L("{"), "ws", "members", "ws", L("}")])
    g.add("members", ["pair"], ["pair", "ws", L(","), "ws", "members"])
    g.add("pair", ["string", "ws", L(":"), "ws", "value"])

    # array -> '[' ws ']' | '[' ws elements ws ']'
    g.add("array", [L("["), "ws", L("]")], [L("["), "ws", "elements", "ws", L("]")])
    g.add("elements", ["value"], ["value", "ws", L(","), "ws", "elements"])

    # string -> '"' s_chars '"'
    g.add("string", [L('"'), "s_chars", L('"')])
    g.add("s_chars", [], ["s_char", "s_chars"])
    g.add("s_char", [NORMAL], [L("\\"), "escape"])
    g.add("escape", [ESC_SIMPLE], [L("u"), HEX, HEX, HEX, HEX])

    # number -> sign int [frac] [exp]   (RFC 8259: no leading zeros,
    # frac and exp need at least one digit)
    g.add(
        "number",
        ["sign", "int", "frac", "exp"],
        ["sign", "int", "frac"],
        ["sign", "int", "exp"],
        ["sign", "int"],
    )
    g.add("sign", [], [L("-")])
    g.add("int", [L("0")], [DIGIT19], [DIGIT19, "digits"])
    g.add("digits", [DIGIT], [DIGIT, "digits"])
    g.add("frac", [L("."), DIGIT], [L("."), DIGIT, "digits"])
    g.add("exp", [EXP_MARK, "exp_sign", DIGIT], [EXP_MARK, "exp_sign", DIGIT, "digits"])
    g.add("exp_sign", [], [EXP_SIGN])

    g.check()
    return g


def arithmetic_grammar() -> Grammar:
    """A small expression grammar, used to prove the engine is generic.

    expr -> expr '+' term | expr '*' term | '(' expr ')' | num
    (written in non-left-recursive form plus left recursion on purpose:
    Earley handles left recursion fine, recursive descent would not)
    """
    g = Grammar("expr")
    g.add("expr", ["expr", L("+"), "term"], ["term"])
    g.add("term", ["term", L("*"), "factor"], ["factor"])
    g.add("factor", [L("("), "expr", L(")")], ["num"])
    g.add("num", [DIGIT], [DIGIT, "num"])
    g.check()
    return g
