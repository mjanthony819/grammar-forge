# grammar-forge

A grammar-constrained decoding engine for small local LLMs. It forces a
model to emit 100% syntactically valid JSON at the token level, using a
context-free grammar and a custom sampling loop. No API keys, no GPU, no
paid services: a quantized 360M-parameter model runs on plain CPU.

## The idea

Small models are bad at structure. Asked for JSON, they eventually emit
placeholder tokens like `<input>`, trailing commas, unquoted keys or prose
around the object, and any downstream parser dies. Prompting harder does
not fix it; it only makes failures rarer.

Instead of asking nicely, grammar-forge makes invalid output impossible.
At every decoding step, an incremental parser for the JSON grammar tracks
what the partial output can legally continue with. Every vocabulary token
that would break the grammar is masked to minus-infinity logit before
sampling, so its probability is exactly zero. The model is still free to
choose any valid continuation; it just cannot choose an invalid one.

## Measured results (this repo, CPU, 12 tasks, temperature 1.0)

| Mode | Strict JSON valid | Schema valid | Speed |
| --- | --- | --- | --- |
| Free-running | 8/12 (67%) | 3/12 | 17.8 tok/s |
| Grammar-constrained | 12/12 (100%) | 3/12 | 16.6 tok/s |

Strict validity means the raw model output parses with `json.loads`, no
fence-stripping or cleanup. The constraint guarantees syntax, not
semantics: schema conformance is a separate layer (see "Limitations" in
the manual). The grammar mask costs about 7% decoding speed.

## Layout

```
gce/
  grammar.py         the JSON grammar as a character-level CFG
  earley.py          incremental Earley parser: works for ANY CFG (reference engine)
  json_automaton.py  the CFG hand-compiled to a pushdown automaton (fast engine)
  masker.py          parser state -> set of allowed token ids (cached)
  model.py           ONNX Runtime + KV-cache inference for a small quantized LLM
  engine.py          the custom sampling loop that ties it all together
  schemas.py         pydantic schemas for the benchmark
benchmark.py         constrained vs free: validity, schema, speed
demo.py              try one generation from the command line
tests/               unit tests + Earley-vs-automaton equivalence property test
```

## Two engines, one language

`earley.py` implements a general Earley recognizer over the CFG as
written; it handles any context-free grammar (there is an arithmetic
grammar in `grammar.py` to prove it). `json_automaton.py` is the same
grammar compiled by hand into a pushdown automaton whose state is a small
hashable tuple, which makes token-mask caching effective.
`tests/test_equivalence.py` property-checks that both engines accept the
same language and allow the same next character at every prefix of 256
valid and mutated documents.

## Run it

```
pip install -r requirements.txt
python download_model.py        # one-time, ~390 MB, free from Hugging Face
python demo.py --task person
python demo.py --task order --mode free     # watch it break without the mask
python benchmark.py             # reproduces the table above
python tests/test_json.py       # unit tests
python tests/test_equivalence.py
```

## How the mask works

1. Sample loop asks the parser for its current state.
2. The masker simulates every vocabulary token's characters through a
   clone of the parser; survivors are the allowed set (cached per state).
3. All other logits become -inf; temperature + top-p sampling runs on the
   allowed set only.
4. When the parser accepts the document, EOS is the only legal token, so
   generation stops cleanly.

Built with AI assistance.
