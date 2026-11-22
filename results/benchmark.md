# Benchmark results

Model: SmolLM2-360M-Instruct (q4 ONNX), CPU only (2 threads).
Settings: temperature 1.0, top_p 0.95, max 220 new tokens, same seed per task for both modes.
Strict JSON valid = raw output parses with json.loads (no cleanup). Schema valid = pydantic model validation.

| Task | Free: JSON | Free: schema | Constrained: JSON | Constrained: schema | Free tok/s | Constr. tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| person | yes | yes | yes | yes | 18.8 | 14.7 |
| product | yes | yes | yes | yes | 17.6 | 17.5 |
| address | yes | NO | yes | NO | 17.6 | 15.6 |
| event | NO | NO | yes | NO | 17.3 | 15.5 |
| recipe | yes | NO | yes | NO | 17.2 | 16.2 |
| book | NO | NO | yes | NO | 17.7 | 17.5 |
| review | yes | NO | yes | NO | 17.5 | 17.2 |
| order | NO | NO | yes | NO | 17.8 | 16.7 |
| profile | yes | NO | yes | NO | 17.5 | 17.0 |
| geo | yes | yes | yes | yes | 18.4 | 17.4 |
| playlist | yes | NO | yes | NO | 18.0 | 17.1 |
| invoice | NO | NO | yes | NO | 17.6 | 17.3 |

**Totals:** free 8/12 JSON valid, 3/12 schema valid, 17.8 tok/s avg; constrained 12/12 JSON valid, 3/12 schema valid, 16.6 tok/s avg.

Note: the grammar guarantees syntax, not semantics. Schema validity is unchanged because both modes rely on the model to pick the right fields; constraining keys/types is the documented extension.