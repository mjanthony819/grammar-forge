"""Demo CLI: generate JSON with or without grammar constraints.

Examples:
    python demo.py --task person
    python demo.py --task order --mode free
    python demo.py --prompt "Write a JSON object for a book: title, author, year"
    python demo.py --task person --engine earley   # reference CFG engine
"""
import argparse
import json
import sys

from gce.engine import GrammarConstrainedDecoder
from gce.model import OnnxLLM, ensure_model
from gce.schemas import TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=[t[0] for t in TASKS], default="person")
    ap.add_argument("--prompt", type=str, default=None, help="custom prompt (overrides --task)")
    ap.add_argument("--mode", choices=["constrained", "free"], default="constrained")
    ap.add_argument("--engine", choices=["pda", "earley"], default="pda")
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.prompt:
        prompt, schema = args.prompt, None
    else:
        name, schema, desc = next(t for t in TASKS if t[0] == args.task)
        prompt = f"Write a JSON object for: {desc}"

    llm = OnnxLLM(ensure_model())
    dec = GrammarConstrainedDecoder(llm, backend=args.engine)
    r = dec.generate(prompt, constrained=(args.mode == "constrained"),
                     max_new_tokens=args.max_new_tokens,
                     temperature=args.temperature, seed=args.seed)
    print("--- output ---")
    print(r.text)
    print("--- stats ---")
    print(f"mode={args.mode} engine={args.engine} tokens={r.n_tokens} "
          f"seconds={r.seconds:.1f} tok/s={r.tokens_per_second:.1f} stopped_by={r.stopped_by}")
    try:
        parsed = json.loads(r.text)
        print("strict JSON valid: True")
        if schema is not None:
            try:
                schema.model_validate(parsed)
                print("schema valid: True")
            except Exception as exc:
                print(f"schema valid: False ({exc.__class__.__name__})")
    except Exception as exc:
        print(f"strict JSON valid: False ({exc})")


if __name__ == "__main__":
    main()
