"""Incremental Earley parser over a character-level CFG.

This is the reference recognition engine. It works for ANY context-free
grammar (including left-recursive and ambiguous ones), which is what makes
the project about grammars rather than about JSON. The compiled JSON
pushdown automaton in json_automaton.py is the fast production path; the
test suite property-checks that the two engines accept exactly the same
language and offer exactly the same next characters at every prefix.

Earley items are (rule_index, dot, origin). The parser is incremental: you
feed one character at a time with advance(), which is the operation the
constrained sampler needs when it simulates a candidate token's characters.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, FrozenSet, List, Set, Tuple

from .grammar import Grammar, Terminal

Item = Tuple[int, int, int]  # (rule_idx, dot, origin)


class EarleyParser:
    def __init__(self, grammar: Grammar):
        self.g = grammar
        self.chart: List[Set[Item]] = [set()]
        for r in grammar.by_lhs[grammar.start]:
            self.chart[0].add((r, 0, 0))
        self._close_into(self.chart[0], 0, lambda i: self.chart[i])

    # -- core -------------------------------------------------------------
    def _close_into(self, col: Set[Item], idx: int, col_at: Callable[[int], Set[Item]]) -> None:
        """Earley predict/complete closure of `col` (chart column `idx`)."""
        # Fixpoint passes over the column. A completed item can advance
        # items in its origin column; when origin == idx that is the column
        # itself, so a simple one-shot worklist can miss pairs. Columns are
        # tiny, so re-scanning to a fixpoint is cheap and obviously correct.
        g = self.g
        nullable = g.nullable
        changed = True
        while changed:
            changed = False
            for r, d, o in list(col):
                lhs, rhs = g.rules[r]
                if d == len(rhs):
                    # complete: advance items in the origin column expecting lhs
                    for r2, d2, o2 in list(col_at(o)):
                        lhs2, rhs2 = g.rules[r2]
                        if d2 < len(rhs2) and rhs2[d2] == lhs:
                            it = (r2, d2 + 1, o2)
                            if it not in col:
                                col.add(it)
                                changed = True
                else:
                    sym = rhs[d]
                    if isinstance(sym, str):
                        # predict
                        for rb in g.by_lhs[sym]:
                            it = (rb, 0, idx)
                            if it not in col:
                                col.add(it)
                                changed = True
                        # nullable completion (Aycock-Horspool style)
                        if sym in nullable:
                            it = (r, d + 1, o)
                            if it not in col:
                                col.add(it)
                                changed = True

    def _scan(self, col: Set[Item], ch: str) -> Set[Item]:
        out = set()
        for r, d, o in col:
            lhs, rhs = self.g.rules[r]
            if d < len(rhs):
                sym = rhs[d]
                if isinstance(sym, Terminal) and sym.matches(ch):
                    out.add((r, d + 1, o))
        return out

    # -- incremental interface -------------------------------------------
    def advance(self, ch: str) -> bool:
        """Consume one character. Returns False if the char is not allowed."""
        nxt = self._scan(self.chart[-1], ch)
        if not nxt:
            return False
        self.chart.append(nxt)
        idx = len(self.chart) - 1
        self._close_into(nxt, idx, lambda i: self.chart[i])
        return True

    def accepts(self) -> bool:
        """True iff the input consumed so far is a complete sentence."""
        for r, d, o in self.chart[-1]:
            lhs, rhs = self.g.rules[r]
            if lhs == self.g.start and d == len(rhs) and o == 0:
                return True
        return False

    def expected_terminals(self) -> Set[Terminal]:
        out = set()
        for r, d, o in self.chart[-1]:
            lhs, rhs = self.g.rules[r]
            if d < len(rhs) and isinstance(rhs[d], Terminal):
                out.add(rhs[d])
        return out

    def allows_char(self, ch: str) -> bool:
        return any(t.matches(ch) for t in self.expected_terminals())

    def signature(self) -> FrozenSet[Item]:
        return frozenset(self.chart[-1])

    # -- token simulation (no mutation) -----------------------------------
    def can_consume(self, text: str) -> bool:
        """Could the parser consume this whole token string from its
        current state, without mutating the parser?"""
        base = self.chart
        extra: List[Set[Item]] = []

        def col_at(i: int) -> Set[Item]:
            return base[i] if i < len(base) else extra[i - len(base)]

        cur = base[-1]
        for ch in text:
            nxt = self._scan(cur, ch)
            if not nxt:
                return False
            idx = len(base) + len(extra)
            extra.append(nxt)
            self._close_into(nxt, idx, col_at)
            cur = nxt
        return True
