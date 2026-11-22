"""Generate the grammar-forge interview-prep manual (.docx)."""
from docx import Document
from docx.shared import Pt

doc = Document()
doc.add_heading("grammar-forge: Grammar-Constrained Decoding Engine", 0)
doc.add_paragraph("Project manual and interview prep. Every number in this document comes from the actual benchmark run shipped in results/.")

def h(t, l=1): doc.add_heading(t, l)
def p(t): doc.add_paragraph(t)
def b(items):
    for i in items: doc.add_paragraph(i, style="List Bullet")
def n(items):
    for i in items: doc.add_paragraph(i, style="List Number")
def qa(q, a):
    para = doc.add_paragraph(); para.add_run("Q: " + q).bold = True
    doc.add_paragraph("A: " + a)

h("1. The one-line pitch")
p("grammar-forge makes a small local language model incapable of emitting invalid JSON: at every token it asks a context-free-grammar parser which vocabulary tokens are still legal, masks every other token's logit to minus infinity, and samples only from what survives. Measured result on 12 tasks: strict JSON validity went from 8/12 free-running to 12/12 constrained, at a cost of about 7% decoding speed, all on a free CPU-only stack.")

h("2. Input, workflow, output")
b([
 "Input: a natural-language prompt describing the JSON object you want (e.g. \"Write a JSON object for a product: title, price, in_stock, tags\").",
 "Workflow: a quantized 360M-parameter instruct model (SmolLM2, ONNX, CPU) generates token by token; before each sampling step, an incremental parser for the JSON context-free grammar computes the set of vocabulary tokens the partial output can legally continue with, and the sampler zeroes out everything else.",
 "Output: a JSON document guaranteed to parse with json.loads, plus stats (tokens, seconds, tok/s, why generation stopped).",
 "USP in plain words: prompt engineering asks the model to behave; this makes misbehaving impossible. The guarantee is mathematical, not statistical: an invalid token has probability exactly zero, so validity no longer depends on model size, prompt wording, temperature or luck.",
])

h("3. Why this problem matters")
p("Agents and pipelines consume model output with parsers, not eyes. A single trailing comma, an unquoted key, a markdown fence or a placeholder like <input> breaks the whole chain. Industry answers to this are Outlines, lm-format-enforcer, llama.cpp's GBNF grammars, OpenAI structured outputs, and JSON mode. This project rebuilds the core mechanism from scratch so every part of it is understood, measurable, and runs free on a laptop CPU.")

h("4. Architecture and modules")
b([
 "gce/grammar.py - the JSON grammar (RFC 8259) written as a character-level context-free grammar: start symbol, production rules, single-character terminals. Also a small arithmetic grammar used to prove generality.",
 "gce/earley.py - the reference recognition engine: an incremental Earley parser. It works for ANY CFG, including left-recursive and ambiguous grammars. One character is consumed per advance() call.",
 "gce/json_automaton.py - the production engine: the same grammar hand-compiled into a pushdown automaton. State is a small tuple stack, so it is hashable and O(1) per character.",
 "gce/masker.py - the bridge to the model: given a parser state, simulate every vocabulary token's characters through a clone of the parser; the survivors are the allowed token set. Cached per parser-state signature.",
 "gce/model.py - inference: ONNX Runtime session, KV cache management, temperature + top-p (nucleus) sampling with an optional allowed-id mask.",
 "gce/engine.py - the custom sampling loop that ties parser, masker and model together.",
 "gce/schemas.py + benchmark.py - 12 pydantic-typed tasks; each is generated free-running and constrained with the same seed and settings.",
 "tests/ - unit tests for both engines plus a property test that proves the automaton and the Earley parser recognize exactly the same language.",
])

h("5. The concepts, explained")

h("5.1 Context-free grammars", 2)
p("A CFG is a set of rewrite rules A -> alpha, where A is a nonterminal and alpha mixes nonterminals and terminals. JSON is almost context-free: objects and arrays nest, which a regular expression cannot count, but a grammar handles naturally. We write terminals as single characters (multi-character keywords like true become four one-character rules), because the decoder needs to consume text one character at a time when simulating multi-character tokens.")

h("5.2 Earley parsing (the reference engine)", 2)
p("Earley's algorithm is a chart parser that recognizes any CFG in O(n^3) worst case, O(n) for unambiguous grammars. It keeps chart columns, one per input position; each column holds items (rule, dot, origin) meaning \"rule matched up to the dot, starting at origin\". Three operations close each column: predict (expand expected nonterminals), scan (consume the next character), complete (a finished rule advances items waiting for it in its origin column). Our parser is incremental: feed a character, get a new column; the current column is the complete parser state. Nullable rules (ws can be empty) are handled with Aycock-Horspool-style nullable completion, and the closure runs to a fixpoint because a completed item can advance items in its own column.")

h("5.3 The compiled pushdown automaton (the fast engine)", 2)
p("Earley columns contain origins, so the state changes at every character inside a string, which destroys mask caching. The fix: compile the grammar into a pushdown automaton whose frames carry no positions. Frames like ('value',), ('after_elem',), ('str',), ('num','frac0') form a tuple stack. Every position inside any string anywhere has the same signature, so one cache entry serves them all. tests/test_equivalence.py proves the two engines agree on the accepted language and on the allowed next character at every prefix of 256 valid and mutated documents over a 50-character probe alphabet.")

h("5.4 Logit masking: the math", 2)
p("The model emits a logit vector z over 49,152 vocabulary tokens. Normally softmax(z/T) gives the sampling distribution. With masking we build an allowed set A from the parser and set z_i = -inf for every i not in A before softmax. Since exp(-inf) = 0, every invalid token gets probability exactly 0, and the renormalized distribution over A keeps the model's relative preferences among valid choices. This is the whole trick: we never touch the model, its weights, or its distribution over legal continuations.")

h("5.5 Temperature and nucleus (top-p) sampling", 2)
p("Temperature T divides the logits: T > 1 flattens the distribution (more variety, more mistakes), T -> 0 approaches greedy. Top-p keeps the smallest set of tokens whose cumulative probability reaches p and drops the tail. The benchmark uses T = 1.0, top_p = 0.95: deliberately hostile settings where small models frequently break JSON.")

h("5.6 KV cache", 2)
p("Without caching, each new token would re-run attention over the whole sequence, O(n^2) total. The ONNX model exposes past_key_values inputs and present outputs (32 layers, 5 KV heads, head dim 64); we feed each step's cache into the next, so each token costs one forward pass over one position.")

h("6. Stage-by-stage: one constrained generation")
n([
 "The prompt is wrapped in the model's ChatML template and tokenized (about 60 tokens).",
 "Prefill runs the whole prompt once and returns the first logit vector plus the KV cache.",
 "The parser starts in its initial state (expecting whitespace, then any JSON value).",
 "The masker scans the 49,152-token vocabulary: a token survives if the parser could consume its entire surface text. First-step survivors are tokens starting with '{', '[', a quote, a digit, '-', 't', 'f', 'n', or whitespace.",
 "Invalid logits go to -inf; temperature + top-p sampling picks one token; its text is fed into the real parser, advancing the state.",
 "Repeat. Inside a string, almost every token survives (free text); after '\"name\":' the closing quote, comma and brace structure is again forced.",
 "When the parser accepts the document as complete, EOS is the only allowed token and the loop stops: output ends the moment the JSON ends. No trailing prose is possible.",
])

h("7. Benchmark (real numbers, this repo, CPU, 2 threads)")
tbl = doc.add_table(rows=1, cols=6)
tbl.style = "Light Grid Accent 1"
hdr = tbl.rows[0].cells
for i, t in enumerate(["Task", "Free JSON", "Free schema", "Constrained JSON", "Constrained schema", "tok/s free vs constr."]):
    hdr[i].text = t
import json as _json
_data = _json.load(open("results/benchmark.json"))["results"]
for r in _data:
    f, c = r["free"], r["constrained"]
    row = tbl.add_row().cells
    row[0].text = r["task"]
    row[1].text = "valid" if f["json_valid"] else "INVALID"
    row[2].text = "valid" if f["schema_valid"] else "no"
    row[3].text = "valid" if c["json_valid"] else "INVALID"
    row[4].text = "valid" if c["schema_valid"] else "no"
    row[5].text = f"{f['tok_per_s']} vs {c['tok_per_s']}"
p("")
b([
 "Strict JSON validity: 8/12 free-running, 12/12 constrained. The free failures are real small-model behavior: placeholder tokens like <input> and <number> inside objects, broken nested arrays, and unterminated structures.",
 "Speed: 17.8 tok/s free vs 16.6 tok/s constrained, about 7% overhead for the mask (with per-state caching).",
 "Schema validity: 3/12 in BOTH modes. This is the honest limit of a general JSON grammar: it guarantees the output parses, not that it has the fields you wanted. Both modes sometimes wrap the object in an extra key (e.g. {\"address\": {...}}).",
])

h("8. Design decisions (and why)")
n([
 "Character-level terminals instead of token-level grammar: BPE tokens are arbitrary strings ('true' can be one token or four), so the grammar must consume text one character at a time and let tokens end mid-terminal. This makes the engine tokenizer-independent.",
 "Two engines instead of one: a general Earley parser proves the CFG itself is right and handles any grammar; a hand-compiled pushdown automaton is the fast production path. The property test proving they agree is what makes the fast path trustworthy.",
 "Hashable automaton state for cacheable masks: scanning 49k tokens per step is affordable only because parser states repeat; Earley origins break this, the position-free automaton restores it.",
 "Stop at acceptance: the grammar is document -> ws value with no trailing content, so the instant the value completes, EOS is forced. This eliminates trailing-prose failures by construction.",
 "ONNX Runtime instead of PyTorch: a 386 MB q4 model and a ~50 MB runtime, no multi-GB torch install, still 15-18 tok/s on 2 CPU cores. Keeps the project free and laptop-friendly.",
 "Same seed and settings for both benchmark modes: the only variable is the mask, so the validity delta is attributable to the constraint, not to luck.",
 "ban special tokens: tokens like <|im_start|> are excluded from the mask everywhere, so markup can never leak into a JSON string as literal text.",
 "Fixpoint closure in Earley: a completed item can advance items in its own column (self-origin), which a one-shot worklist misses; looping to a fixpoint is cheap on tiny columns and obviously correct.",
])

h("9. Honest limitations")
b([
 "Syntax is guaranteed, semantics are not. Schema conformance (right keys, right types) needs a schema-compiled grammar: generate grammar rules from a JSON Schema/pydantic model so keys, order-independence and types are constrained too. This is the documented next step (Outlines and llama.cpp do versions of it).",
 "Mask cost scales with vocabulary size on first visit of each parser state; the cache hides it in practice, but a cold cache on a huge vocab costs a one-time scan.",
 "The automaton is hand-compiled: a new grammar needs a new automaton (or the slower Earley path, which the engine already supports via --engine earley).",
 "A weak model constrained to JSON can still produce nonsense values (Ender T. as an author name). Grammar constrains form, not truth.",
 "Constrained generation can take more tokens than free generation when the model would have given up early (the recipe task: 196 vs 119 tokens).",
])

h("10. Interview Q&A")
qa("What did you build?", "A grammar-constrained decoding engine: a custom token sampling loop where a context-free-grammar parser decides, at every step, which vocabulary tokens are legal continuations, and all other logits are forced to -inf. It turns JSON validity from a hope into a guarantee.")
qa("Why not just prompt the model to output JSON?", "Prompting changes probabilities, not possibilities. Measured: at temperature 1.0 the same model produced strictly valid JSON in only 8 of 12 tasks; with the mask it was 12 of 12, at about 7% speed cost.")
qa("Why is JSON a context-free language problem and not a regex problem?", "Nested objects and arrays need balanced-bracket counting, which finite automata cannot do; a stack (context-free grammar / pushdown automaton) can.")
qa("Explain Earley's algorithm in one breath.", "A chart parser that tracks partial rule matches per input position using three operations: predict, scan, complete. It recognizes any CFG, which is why it is our reference engine.")
qa("If Earley works for any grammar, why build the automaton?", "Speed and caching. Earley items carry origin positions, so parser state changes at every character and the 49k-token mask scan cannot be cached. The compiled automaton's state is a position-free tuple stack, so identical states (e.g. anywhere inside any string) share one cached mask.")
qa("How do you know the fast engine is correct?", "Property testing against the reference: 256 valid and mutated documents, at every prefix, over a 50-character probe alphabet, both engines agree on acceptance and on the allowed next character.")
qa("Walk me through the masking math.", "Take the logit vector, set logits of disallowed token ids to -inf, then softmax. exp(-inf)=0, so invalid tokens have exactly zero probability; the distribution over valid tokens is just the renormalized original.")
qa("Does masking distort the model?", "It restricts the support of the distribution, not the shape over what remains. Any valid continuation keeps its relative probability.")
qa("What happens inside a string?", "The grammar's string rule allows almost any character, so the mask opens up and the model writes free text; the mask only bites on the closing quote, escape rules, and control characters.")
qa("How do multi-character tokens interact with a character-level grammar?", "We simulate the token's whole surface text through a clone of the parser; the token is allowed iff every character is consumed legally. Tokens may end mid-terminal, e.g. 'nu' on the way to 'null'.")
qa("How does generation stop?", "The top-level rule is document -> ws value with no trailing content. The moment the parser accepts, EOS is the only allowed token, so output ends exactly when the JSON does.")
qa("What is nucleus (top-p) sampling?", "Keep the smallest set of tokens covering p of the probability mass and sample from it. It adapts to distribution shape, unlike top-k.")
qa("Why temperature 1.0 in the benchmark?", "To make the baseline honest. Greedy or cold sampling makes small models look better than they are in production settings, where variety is wanted and failures show up.")
qa("What is the KV cache and why does it matter here?", "It stores each layer's attention keys and values so each new token costs one forward pass over one position instead of reprocessing the whole sequence; without it, 200-token generations on CPU would be minutes each.")
qa("Why ONNX Runtime instead of PyTorch?", "A 386 MB quantized model and a tiny runtime, no GPU and no multi-GB dependencies: the whole project stays free and CPU-only, which was a hard requirement.")
qa("What does quantization cost here?", "The q4 model is about 4x smaller than fp16; quality for structure-following is still fine, and the mask removes the syntax errors quantization might otherwise introduce.")
qa("Where does the 7% overhead come from?", "Per-step mask lookups; with caching they are dict hits plus a set construction. The uncached path, simulating the vocabulary, runs once per distinct parser state.")
qa("How would you extend this to enforce a pydantic schema?", "Compile the schema into grammar rules: fix the key set, constrain each value position by the field's type grammar (integer, ISO date, enum), and handle key order either by fixing an order or by tracking remaining keys in the automaton state.")
qa("How is this different from OpenAI's JSON mode?", "JSON mode is trained/sampling-level encouragement and still fails sometimes; structured outputs use constrained decoding like this project. We implement the mechanism, not the vibe.")
qa("What are the failure modes left?", "Semantic ones: wrong fields, nonsense values, wrapper objects. Syntactic failure is eliminated by construction.")
qa("Could the mask ever block every token?", "Only if the grammar is wrong for the generated prefix; the engine asserts the sampled token is always consumable, and unit tests cover the dead-end cases (e.g. after '1e').")
qa("Why does the parser allow whitespace but not require it?", "Because JSON allows it everywhere between tokens; making it optional keeps the grammar minimal and lets the model emit compact or pretty-printed output.")
qa("How does this relate to llama.cpp GBNF or Outlines?", "Same family: grammar-driven constrained decoding. GBNF compiles a grammar to a state machine; Outlines compiles regex/JSON Schema to an FSM. This project shows the mechanism end-to-end at a readable scale, with a general Earley engine as the correctness anchor.")
qa("What would you do with a much bigger vocabulary or model?", "Precompile the mask offline per parser state, intersect tokenizer-prefix tries with the automaton, or move masking into the sampling kernel; the interface stays the same.")
qa("Is the constraint reversible if the model wants to be creative?", "The mask only forbids syntactically invalid continuations; all creativity inside valid JSON is untouched. Inside string values the mask is nearly wide open.")
qa("What did you measure and how?", "12 pydantic-typed tasks, identical seeds and sampling settings, free vs constrained. Metrics: strict json.loads validity, pydantic schema validity, tokens/second. All raw outputs are in results/benchmark.json.")
qa("What is the proudest engineering detail?", "The equivalence property test: the fast hand-written automaton is only trustworthy because a general, obviously-correct Earley parser pins the language it must recognize.")
qa("How would this serve trillion-parameter-scale systems differently?", "The mask is model-agnostic and adds constant per-step work; at scale it sits in the serving layer (vLLM-style logits processors) and the parser state becomes part of each request's decode state.")

h("11. Glossary / cheat sheet")
b([
 "CFG: context-free grammar; rules A -> alpha; powers nested structure.",
 "Terminal / nonterminal: literal character vs rule reference.",
 "Earley item: (rule, dot, origin); a partial match.",
 "Predict / scan / complete: Earley's three closure operations.",
 "Pushdown automaton (PDA): finite control + stack; recognizes exactly the context-free languages.",
 "Logit: model's raw pre-softmax score per token.",
 "Masking: set disallowed logits to -inf before softmax so their probability is exactly 0.",
 "Temperature: logit divisor controlling randomness.",
 "Top-p (nucleus): sample from the smallest token set covering mass p.",
 "KV cache: stored attention keys/values making decoding O(1) per token.",
 "Strict validity: raw output parses as JSON with zero cleanup.",
 "Schema conformance: parsed JSON also matches a pydantic model (not guaranteed by a general JSON grammar).",
])
doc.save("grammar-forge-manual.docx")
print("manual written")
