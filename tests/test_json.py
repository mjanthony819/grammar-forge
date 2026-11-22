"""Unit tests for both recognizers. Run: python tests/test_json.py"""
import sys

sys.path.insert(0, ".")
from gce.earley import EarleyParser
from gce.grammar import json_grammar
from gce.json_automaton import JSONAutomaton

VALID = [
    '{"a": [1, 2.5e3, true, null]}', "[]", "{}", "[[]]", "42", "-0.5",
    '"a\\u00Af"', "{ }", "[ ]", '{"a" :1}', "1e+5", '"esc\\\\q"',
    '{"n": [1, [2, [3, {"x": null}]]]}',
]
INVALID = [
    '{"a": }', "[1,2,]", "01", "1.", "1e", "tru", "truE", '{"a":1,}',
    '"unterminated', "[1]extra", "1 2", "{'a':1}", "{a:1}", '"bad\\x"',
    "--1", "+1", ".5", "1e+", '["a",]',
]


def main():
    bad = 0
    for make in (lambda: JSONAutomaton(), lambda: EarleyParser(json_grammar())):
        for s in VALID:
            eng = make()
            ok = all(eng.feed(c) if hasattr(eng, "feed") else eng.advance(c) for c in s)
            ok = ok and eng.accepts()
            if not ok:
                bad += 1
                print("FAIL valid", type(eng).__name__, repr(s))
        for s in INVALID:
            eng = make()
            alive = True
            for c in s:
                alive = eng.feed(c) if hasattr(eng, "feed") else eng.advance(c)
                if not alive:
                    break
            if alive and eng.accepts():
                bad += 1
                print("FAIL invalid", type(eng).__name__, repr(s))
    assert bad == 0, f"{bad} failures"
    print(f"unit tests OK: {len(VALID)} valid + {len(INVALID)} invalid x 2 engines")


if __name__ == "__main__":
    main()
