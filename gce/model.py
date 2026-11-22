"""Small local LLM inference over ONNX Runtime (CPU only, no PyTorch).

Uses a quantized decoder-only transformer (default: SmolLM2-360M-Instruct,
q4) with an explicit KV cache. Everything runs on the free tier of
Hugging Face Hub and plain CPU, which keeps the whole project free.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_REPO = "onnx-community/SmolLM2-360M-Instruct-ONNX"
DEFAULT_VARIANT = "onnx/model_q4.onnx"


def ensure_model(repo: str = DEFAULT_REPO, variant: str = DEFAULT_VARIANT) -> str:
    """Download (or reuse the cached copy of) the ONNX model files.

    Returns the snapshot directory containing the model and tokenizer.
    """
    from huggingface_hub import hf_hub_download

    files = [
        variant,
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        "generation_config.json",
    ]
    snapshot_dir = None
    for f in files:
        path = hf_hub_download(repo, f)
        if f == variant:
            snapshot_dir = os.path.dirname(os.path.dirname(path))
    return snapshot_dir


class OnnxLLM:
    """Decoder-only transformer with KV cache, running on onnxruntime."""

    def __init__(self, snapshot_dir: str, variant: str = DEFAULT_VARIANT, n_threads: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = os.path.join(snapshot_dir, variant)
        opts = ort.SessionOptions()
        if n_threads:
            opts.intra_op_num_threads = n_threads
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(os.path.join(snapshot_dir, "tokenizer.json"))
        with open(os.path.join(snapshot_dir, "config.json")) as fh:
            cfg = json.load(fh)
        self.vocab_size = cfg["vocab_size"]
        self.eos_token_id = cfg.get("eos_token_id", 2)
        self.n_layers = cfg["num_hidden_layers"]
        self.n_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]

        # Surface text of every vocabulary token (what the token would add
        # to the output). The masker simulates these through the parser.
        self.vocab_texts: List[str] = [
            self.tokenizer.decode([i], skip_special_tokens=False)
            for i in range(self.vocab_size)
        ]

    # -- tokenization ------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    # -- inference ----------------------------------------------------------
    def _empty_past(self) -> Dict[str, np.ndarray]:
        return {
            f"past_key_values.{i}.{kv}": np.zeros(
                (1, self.n_kv_heads, 0, self.head_dim), dtype=np.float32
            )
            for i in range(self.n_layers)
            for kv in ("key", "value")
        }

    def _run(
        self, ids: List[int], past: Dict[str, np.ndarray], past_len: int
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        seq = len(ids)
        inputs = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, past_len + seq), dtype=np.int64),
            "position_ids": np.arange(past_len, past_len + seq, dtype=np.int64)[None, :],
        }
        inputs.update(past)
        outputs = self.session.run(None, inputs)
        logits = outputs[0][0, -1]  # last position: (vocab,)
        new_past = {}
        for i in range(self.n_layers):
            new_past[f"past_key_values.{i}.key"] = outputs[1 + 2 * i]
            new_past[f"past_key_values.{i}.value"] = outputs[2 + 2 * i]
        return logits, new_past

    def prefill(self, ids: List[int]) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
        logits, past = self._run(ids, self._empty_past(), 0)
        return logits, past, len(ids)

    def step(
        self, last_id: int, past: Dict[str, np.ndarray], past_len: int
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
        logits, new_past = self._run([last_id], past, past_len)
        return logits, new_past, past_len + 1


def sample_next(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 0.8,
    top_p: float = 0.95,
    allowed: Optional[set] = None,
) -> int:
    """Sample one token id from logits with temperature + nucleus (top-p)
    filtering, restricted to `allowed` ids when given.

    This is the core of the project: `allowed` is computed by the grammar
    masker, and every id outside it is forced to -inf before softmax, so an
    invalid continuation has exactly zero probability.
    """
    logits = logits.astype(np.float64)
    if allowed is not None:
        mask = np.full(logits.shape, -np.inf)
        idx = np.fromiter(allowed, dtype=np.int64)
        mask[idx] = logits[idx]
        logits = mask
    if temperature <= 0:  # greedy
        return int(np.argmax(logits))
    logits = logits / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        keep = cum <= top_p
        keep[0] = True  # always keep the most likely token
        probs = probs[order] * keep
        if probs.sum() == 0:
            probs = np.ones_like(probs)
        probs /= probs.sum()
        return int(order[rng.choice(len(probs), p=probs)])
    return int(rng.choice(len(probs), p=probs))
