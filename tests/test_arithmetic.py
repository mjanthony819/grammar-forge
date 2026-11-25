"""Unit tests for the arithmetic grammar: proves the Earley engine is generic.

The JSON automaton only speaks JSON; the claim of the project is that the
Earley parser recognizes ANY context-free grammar, including left-recursive
ones that make naive recursive descent loop forever. grammar.py ships the
arithmetic grammar as that proof, but nothing exercised it until now.

Run: python tests/test_arithmetic.py
"""
import random
import sys

sys.path.insert(0, ".")
from gce.earley import EarleyParser
from gce.grammar import arithmetic_grammar

# expr -> expr '+' term | term ; term -> term '*' factor | factor ;
# factor -> '(' expr ')' | num ; num -> DIGIT+
VALID = [
    "1", "42", "007", "1+2", "1*2", "1+2*3", "1*2+3", "(1+2)*3",
    "((7))", "1*(2+3)*4", "123+456*(7+8)",
    "1+2+3",  # left recursion: recursive descent would never finish
    "1*2*3", "9*(1*(2*(3+4)))",
]
INVALID = [
    "", "+1", "1+", "1*", "(1", "1)", "()", "1++2", "1**2", "((1)",
    "(1))", "1+2*", "1 2", "a", "1a", "1.5", "1+*2", "(1+2",
]


def accepts(s: str) -> bool:
    p = EarleyParser(arithmetic_grammar())
    for c in s:
        if not p.advance(c):
            return False
    return p.accepts()


def random_expr(depth: int, rng: random.Random) -> str:
    if depth <= 0 or rng.random() < 0.3:
        return str(rng.randint(0, 10**rng.randint(1, 3)))
    if rng.random() < 0.25:
        return "(" + random_expr(depth - 1, rng) + ")"
    op = rng.choice("+*")
    return random_expr(depth - 1, rng) + op + random_expr(depth - 1, rng)


def main() -> None:
    bad = 0
    for s in VALID:
        if not accepts(s):
            bad += 1
            print("FAIL valid", repr(s))
    for s in INVALID:
        if accepts(s):
            bad += 1
            print("FAIL invalid", repr(s))

    # next-char probes at partial input: what may follow must match the grammar
    probes = [
        ("", "5", True), ("", "(", True), ("", "+", False), ("", ")", False),
        ("1+", "5", True), ("1+", "(", True), ("1+", "+", False), ("1+", ")", False),
        ("(1+2", ")", True), ("(1+2", "*", True), ("(1+2", "5", True),  # "1+25" is still one number
        ("1*2", "+", True), ("1*2", "*", True), ("1*2", "(", False),
        ("12", "3", True), ("12", ".", False),
    ]
    for prefix, ch, expected in probes:
        p = EarleyParser(arithmetic_grammar())
        for c in prefix:
            p.advance(c)
        if p.allows_char(ch) != expected:
            bad += 1
            print("FAIL probe", repr(prefix), repr(ch), "expected", expected)

    # 200 seeded well-formed expressions must all be recognized
    rng = random.Random(2024)
    for _ in range(200):
        s = random_expr(4, rng)
        if not accepts(s):
            bad += 1
            print("FAIL generated", repr(s))

    assert bad == 0, f"{bad} failures"
    print(f"arithmetic grammar OK: {len(VALID)} valid + {len(INVALID)} invalid "
          f"+ {len(probes)} next-char probes + 200 generated expressions")


if __name__ == "__main__":
    main()
