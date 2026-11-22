"""Benchmark: constrained vs unconstrained JSON generation.

For every task we generate twice with the same seed and sampling settings:
once free-running, once grammar-constrained. We then measure strict JSON
validity (the raw model output must parse, no fence-stripping), pydantic
schema conformance, and generation speed.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gce.engine import GrammarConstrainedDecoder
from gce.model import OnnxLLM, ensure_model
from gce.schemas import TASKS

PROMPT = "Write a JSON object for: {desc}"


def run(tasks, max_new_tokens, temperature, top_p, out_prefix):
    llm = OnnxLLM(ensure_model())
    dec = GrammarConstrainedDecoder(llm)
    results = []
    for i, (name, schema, desc) in enumerate(tasks):
        row = {"task": name}
        for mode, constrained in (("free", False), ("constrained", True)):
            r = dec.generate(
                PROMPT.format(desc=desc),
                constrained=constrained,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=1000 + i,
            )
            try:
                parsed = json.loads(r.text)
                json_valid = True
            except Exception:
                parsed, json_valid = None, False
            try:
                schema.model_validate(parsed) if json_valid else (_ for _ in ()).throw(ValueError("no json"))
                schema_valid = True
            except Exception:
                schema_valid = False
            row[mode] = {
                "text": r.text,
                "n_tokens": r.n_tokens,
                "seconds": round(r.seconds, 2),
                "tok_per_s": round(r.tokens_per_second, 1),
                "stopped_by": r.stopped_by,
                "json_valid": json_valid,
                "schema_valid": schema_valid,
            }
            print(f"[{name}/{mode}] valid={json_valid} schema={schema_valid} "
                  f"tok={r.n_tokens} {r.tokens_per_second:.1f} tok/s stop={r.stopped_by}", flush=True)
        results.append(row)
        with open(f"{out_prefix}.partial.json", "w") as fh:
            json.dump(results, fh, indent=2)
    with open(f"{out_prefix}.json", "w") as fh:
        json.dump({"settings": {"max_new_tokens": max_new_tokens, "temperature": temperature,
                                "top_p": top_p, "model": "SmolLM2-360M-Instruct-ONNX q4"},
                   "results": results}, fh, indent=2)
    n = len(results)
    for mode in ("free", "constrained"):
        jv = sum(r[mode]["json_valid"] for r in results)
        sv = sum(r[mode]["schema_valid"] for r in results)
        tps = sum(r[mode]["tok_per_s"] for r in results) / n
        print(f"{mode}: json {jv}/{n} schema {sv}/{n} avg {tps:.1f} tok/s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--only", type=str, default=None, help="comma-separated task names")
    ap.add_argument("--out", type=str, default="results/benchmark")
    args = ap.parse_args()
    tasks = TASKS if not args.only else [t for t in TASKS if t[0] in args.only.split(",")]
    run(tasks, args.max_new_tokens, args.temperature, args.top_p, args.out)
