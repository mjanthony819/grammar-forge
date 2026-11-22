"""Property tests: the compiled JSON automaton and the reference Earley
parser over the CFG must recognize exactly the same language.

For every test string we check, at EVERY prefix:
  - the set of next characters each engine allows (over a probe alphabet)
  - whether the engine accepts the prefix as a complete document
"""
import json
import random
import sys

sys.path.insert(0, ".")
from gce.earley import EarleyParser
from gce.grammar import json_grammar
from gce.json_automaton import JSONAutomaton

PROBE_ALPHABET = sorted(set(
    "0123456789" + "19" + "abcdefABCDEFxyz" + "eEu" + "tfn" + "rugs"
    + "{}[]\":,.-+\\/" + " \t\n\r" + "b" + "\né\x01\x1f"
))
G = json_grammar()


def check_string(s: str) -> None:
    earley = EarleyParser(G)
    pda = JSONAutomaton()
    for i, ch in enumerate(s):
        # next-char agreement at this prefix
        for probe in PROBE_ALPHABET:
            e_ok = earley.allows_char(probe)
            p_ok = pda.clone().feed(probe)
            assert e_ok == p_ok, (
                f"next-char mismatch at prefix {s[:i]!r} probe {probe!r}: "
                f"earley={e_ok} pda={p_ok}"
            )
        assert earley.accepts() == pda.accepts(), (
            f"accept mismatch at prefix {s[:i]!r}: "
            f"earley={earley.accepts()} pda={pda.accepts()}"
        )
        e_ok = earley.advance(ch)
        p_ok = pda.feed(ch)
        assert e_ok == p_ok, (
            f"feed mismatch on {ch!r} at {s[:i]!r}: earley={e_ok} pda={p_ok}"
        )
        if not e_ok:
            return
    assert earley.accepts() == pda.accepts(), (
        f"final accept mismatch on {s!r}: earley={earley.accepts()} pda={pda.accepts()}"
    )


def random_json_value(depth, rng):
    if depth <= 0:
        choice = rng.randint(0, 3)
    else:
        choice = rng.randint(0, 5)
    if choice == 0:
        return rng.randint(-1000, 1000) * rng.choice([1, 0.5, 1e-3])
    if choice == 1:
        return rng.choice([True, False, None])
    if choice == 2:
        return "".join(rng.choice("ab \"\\cdé\n") for _ in range(rng.randint(0, 6)))
    if choice == 3:
        return rng.choice([0, -1, 3.14, 1e10, -2.5e-3])
    if choice == 4:
        return [random_json_value(depth - 1, rng) for _ in range(rng.randint(0, 4))]
    return {f"k{j}": random_json_value(depth - 1, rng) for j in range(rng.randint(0, 4))}


def mutate(s: str, rng) -> str:
    if not s:
        return s
    pos = rng.randrange(len(s))
    op = rng.randint(0, 2)
    if op == 0:
        return s[:pos] + rng.choice(PROBE_ALPHABET) + s[pos:]
    if op == 1:
        return s[:pos] + s[pos + 1:]
    return s[:pos] + rng.choice(PROBE_ALPHABET) + s[pos + 1:]


def main() -> None:
    rng = random.Random(12345)
    fixed = [
        '{"a": [1, 2.5e3, true, null]}', "[]", "{}", "[[]]", '{"a":{"b":{"c":[]}}}',
        '"\\u00Af\\n"', "-0.5", "1e+5", "0", '"simple"', ' [ 1 , 2 ] ',
        "42", "-0", '{"":""}', "3.14159", '"tab\\tchar"',
    ]
    n = 0
    for s in fixed:
        check_string(s)
        n += 1
    for _ in range(60):
        doc = json.dumps(random_json_value(3, rng), ensure_ascii=False)
        check_string(doc)
        n += 1
        for _ in range(3):
            m = mutate(doc, rng)
            check_string(m)
            n += 1
    print(f"equivalence OK: {n} strings, every prefix checked on {len(PROBE_ALPHABET)} probe chars")


if __name__ == "__main__":
    main()
