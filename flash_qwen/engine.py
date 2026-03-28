"""Flash inference engine for Qwen3.5-35B-A3B."""

import json
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.qwen3_5 import TextModelArgs
from flash_qwen._native import FlashExpertManager
from flash_qwen.model import Model


class SimpleTokenizer:
    def __init__(self, tokenizer_path: str):
        from tokenizers import Tokenizer as HFTokenizer
        self.tokenizer = HFTokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))
        template_file = Path(tokenizer_path) / "chat_template.jinja"
        self.chat_template = template_file.read_text() if template_file.exists() else None
        config_file = Path(tokenizer_path) / "config.json"
        if config_file.exists():
            cfg = json.loads(config_file.read_text())
            eos = cfg.get("eos_token_id", [248044])
            self.eos_token_id = eos if isinstance(eos, list) else [eos]
        else:
            self.eos_token_id = [248044]

    def __call__(self, text, return_tensors=None):
        ids = self.tokenizer.encode(text).ids
        if return_tensors == "np":
            import numpy as np
            return {"input_ids": np.array([ids])}
        return {"input_ids": [ids]}

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if self.chat_template is None:
            parts = []
            for msg in messages:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)
        from jinja2 import Template
        rendered = Template(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt,
        )
        return rendered


class FlashInferenceEngine:
    def __init__(self, split_model_path: str, cache_size_mb: int = 6144,
                 original_model_path: Optional[str] = None,
                 kv_bits: Optional[int] = None,
                 warm_experts: Optional[str] = None,
                 use_mmap: bool = False):
        split_path = Path(split_model_path)

        with open(split_path / "config.json") as f:
            config = json.load(f)

        text_config = config.get("text_config", config)
        quant_config = config.get("quantization", {})

        args = TextModelArgs.from_dict(text_config)

        # Initialize expert manager (Rust)
        expert_dir = str(split_path / "experts")
        backend = "mmap" if use_mmap else "pread"
        print(f"Initializing expert manager: cache={cache_size_mb}MB, backend={backend}")
        expert_manager = FlashExpertManager(expert_dir, cache_size_mb, use_mmap=use_mmap)
        self.expert_manager = expert_manager

        # Build model
        print("Building model...")
        self.model = Model(args, expert_manager)

        # Quantize model structure before loading weights
        quant_bits = quant_config.get("bits", 8)
        quant_group_size = quant_config.get("group_size", 32)

        def quant_predicate(path, module):
            if not isinstance(module, (nn.Linear, nn.Embedding)):
                return False
            # Check per-layer quant config from model config
            for cfg_key, cfg_val in quant_config.items():
                if isinstance(cfg_val, dict) and path.endswith(cfg_key.split("language_model.")[-1]):
                    return {"group_size": cfg_val.get("group_size", quant_group_size),
                            "bits": cfg_val.get("bits", quant_bits)}
            return {"group_size": quant_group_size, "bits": quant_bits}

        nn.quantize(self.model, quant_group_size, quant_bits, class_predicate=quant_predicate)

        # Load resident weights (keep language_model. prefix!)
        resident_path = split_path / "resident" / "resident.safetensors"
        print(f"Loading resident weights from {resident_path}...")
        weights = mx.load(str(resident_path))

        # Weights are already sanitized - add language_model. prefix to match model structure
        prefixed = {}
        for k, v in weights.items():
            prefixed[f"language_model.{k}"] = v

        self.model.load_weights(list(prefixed.items()), strict=False)

        resident_size = sum(v.nbytes for v in weights.values())
        print(f"Resident weights loaded: {resident_size / (1024**3):.2f} GB")

        self.kv_bits = kv_bits
        if kv_bits is not None:
            print(f"TurboQuant KV cache: {kv_bits}-bit")

        # Load warm set if available
        warm_path = None
        if warm_experts:
            warm_path = Path(warm_experts)
        else:
            auto_path = split_path / "warm_experts.json"
            if auto_path.exists():
                warm_path = auto_path

        if warm_path and warm_path.exists():
            warm = json.loads(warm_path.read_text())
            experts = [(l, e) for l, e in warm["experts"]]
            print(f"Loading warm set: {len(experts)} experts...")
            count, nbytes = expert_manager.preload_warm_set(experts)
            locked = expert_manager.mlock_cache()
            print(f"Warm set: {count} experts ({nbytes / (1024**3):.1f} GB), "
                  f"mlocked {locked / (1024**3):.1f} GB, "
                  f"coverage: {warm.get('coverage', 0):.0%}")

        # Load tokenizer
        tokenizer_path = original_model_path or str(split_path)
        self.tokenizer = SimpleTokenizer(tokenizer_path)
        print("Engine ready.")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
                 top_p: float = 0.9, verbose: bool = True) -> str:
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = mx.array(inputs["input_ids"])
        cache = self.model.make_cache(kv_bits=self.kv_bits)

        if verbose:
            print(f"Prefilling {input_ids.shape[-1]} tokens...")
        t0 = time.time()
        logits = self.model(input_ids, cache=cache)
        mx.eval(logits)
        prefill_time = time.time() - t0
        if verbose:
            print(f"Prefill: {prefill_time:.2f}s ({input_ids.shape[-1] / prefill_time:.1f} tok/s)")

        next_token = self._sample(logits[:, -1, :], temperature, top_p)
        mx.eval(next_token)

        generated = [next_token.item()]
        eos_ids = set(self.tokenizer.eos_token_id)

        t_start = time.time()
        tokens_generated = 0
        for _ in range(max_tokens - 1):
            logits = self.model(next_token.reshape(1, 1), cache=cache)
            next_token = self._sample(logits[:, -1, :], temperature, top_p)
            mx.eval(next_token)

            tok_id = next_token.item()
            if tok_id in eos_ids:
                break
            generated.append(tok_id)
            tokens_generated += 1

            if verbose and tokens_generated % 10 == 0:
                elapsed = time.time() - t_start
                _, _, hit_rate = self.expert_manager.cache_stats()
                print(f"  {tokens_generated} tokens, {tokens_generated / elapsed:.1f} tok/s, cache hit: {hit_rate:.1%}", end="\r")

        elapsed = time.time() - t_start
        if verbose:
            hits, misses, hit_rate = self.expert_manager.cache_stats()
            print(f"\nGeneration: {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated / elapsed:.1f} tok/s)")
            print(f"Expert cache: {hits} hits, {misses} misses ({hit_rate:.1%} hit rate)")
            print(f"Expert cache size: {self.expert_manager.cache_bytes() / (1024**2):.0f} MB")
            kv_bytes = sum(c.nbytes for c in cache if hasattr(c, 'nbytes'))
            if kv_bytes > 0:
                label = f"TQ-{self.kv_bits}b" if self.kv_bits else "FP16"
                print(f"KV cache ({label}): {kv_bytes / (1024**2):.1f} MB")

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _sample(self, logits, temperature, top_p):
        if temperature == 0:
            return mx.argmax(logits, axis=-1)
        logits = logits / temperature
        if top_p < 1.0:
            sorted_indices = mx.argsort(-logits, axis=-1)
            sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
            probs = mx.softmax(sorted_logits, axis=-1)
            cumulative = mx.cumsum(probs, axis=-1)
            mask = cumulative - probs > top_p
            sorted_logits = mx.where(mask, mx.array(float("-inf")), sorted_logits)
            inv_indices = mx.argsort(sorted_indices, axis=-1)
            logits = mx.take_along_axis(sorted_logits, inv_indices, axis=-1)
        probs = mx.softmax(logits, axis=-1)
        return mx.random.categorical(mx.log(probs + 1e-10))
