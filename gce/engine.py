"""The constrained decoding engine: a custom sampling loop with grammar
masking at every step.

Two recognizer backends:

- "pda":    the compiled JSON pushdown automaton (fast; production path)
- "earley": the general Earley parser over the CFG (works for ANY grammar,
             slower; reference path and proof of generality)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .earley import EarleyParser
from .grammar import json_grammar
from .json_automaton import JSONAutomaton
from .masker import TokenMasker
from .model import OnnxLLM, sample_next

CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
DEFAULT_SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


@dataclass
class GenerationResult:
    text: str
    token_ids: List[int]
    n_tokens: int
    seconds: float
    constrained: bool
    stopped_by: str  # "grammar-complete" | "eos" | "length"
    mask_cache_size: int = 0
    per_token_ms: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.n_tokens / self.seconds if self.seconds else 0.0


class GrammarConstrainedDecoder:
    def __init__(self, llm: OnnxLLM, backend: str = "pda"):
        self.llm = llm
        self.backend = backend
        # Never let the model emit markup/control special tokens as text.
        banned = {
            i
            for i, t in enumerate(llm.vocab_texts)
            if t.startswith("<|") or t == ""
        }
        self.masker = TokenMasker(llm.vocab_texts, llm.eos_token_id, banned)

    def _new_parser(self):
        if self.backend == "earley":
            return EarleyParser(json_grammar())
        return JSONAutomaton()

    def generate(
        self,
        user_prompt: str,
        constrained: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = 0,
        system: str = DEFAULT_SYSTEM,
    ) -> GenerationResult:
        llm = self.llm
        rng = np.random.default_rng(seed)
        prompt = CHAT_TEMPLATE.format(system=system, user=user_prompt)
        prompt_ids = llm.encode(prompt)

        parser = self._new_parser() if constrained else None
        logits, past, past_len = llm.prefill(prompt_ids)

        out_ids: List[int] = []
        stopped_by = "length"
        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            allowed = None
            if parser is not None:
                if parser.accepts():
                    stopped_by = "grammar-complete"
                    break
                allowed = set(self.masker.allowed_ids(parser))
                if not allowed:
                    stopped_by = "grammar-dead-end"
                    break
            next_id = sample_next(
                logits, rng, temperature=temperature, top_p=top_p, allowed=allowed
            )
            if next_id == llm.eos_token_id:
                stopped_by = "eos"
                break
            out_ids.append(next_id)
            if parser is not None:
                text = llm.vocab_texts[next_id]
                ok = parser.can_consume(text)
                assert ok, "masker allowed a token the parser rejects"
                # advance the real parser (the masker simulated on a clone)
                if isinstance(parser, JSONAutomaton):
                    for ch in text:
                        parser.feed(ch)
                else:
                    for ch in text:
                        parser.advance(ch)
            logits, past, past_len = llm.step(next_id, past, past_len)
        seconds = time.perf_counter() - t0
        text = llm.decode(out_ids)
        return GenerationResult(
            text=text,
            token_ids=out_ids,
            n_tokens=len(out_ids),
            seconds=seconds,
            constrained=constrained,
            stopped_by=stopped_by,
            mask_cache_size=self.masker.cache_size,
            per_token_ms=(seconds / len(out_ids) * 1000) if out_ids else 0.0,
        )
