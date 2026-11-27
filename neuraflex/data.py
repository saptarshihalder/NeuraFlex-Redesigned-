"""Token streams shared by train.py and the Hive swarm workers."""
from __future__ import annotations

import hashlib
import os

import torch


def shard_of(worker_id: str, n_shards: int) -> int:
    return int(hashlib.md5(worker_id.encode()).hexdigest(), 16) % n_shards


def token_stream(data: str, tokenizer: str | None = None, data_config: str | None = None,
                 text_field: str = "text", shard: int = 0, n_shards: int = 1, skip: int = 0):
    """Infinite stream of token ids. Local file or HF dataset; byte-level if no tokenizer.
    `shard/n_shards` give each worker a different slice; `skip` resumes after a restart."""
    if tokenizer is None:
        raw = open(data, "rb").read()
        pos = (len(raw) * shard // max(1, n_shards) + skip) % len(raw)
        while True:
            yield from raw[pos:]
            yield from raw[:pos]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    eos = tok.eos_token_id
    while True:
        if os.path.exists(data):
            lines = open(data, encoding="utf-8").read().splitlines()
            start = len(lines) * shard // max(1, n_shards)
            texts = iter(lines[start:] + lines[:start])
        else:
            from datasets import load_dataset
            ds = load_dataset(data, data_config, split="train", streaming=True)
            if n_shards > 1:
                ds = ds.shard(num_shards=n_shards, index=shard)
            if skip:
                ds = ds.skip(skip)
            texts = (ex[text_field] for ex in ds)
        for t in texts:
            yield from tok(t, add_special_tokens=False)["input_ids"]
            yield eos


def batches(stream, batch_size: int, seq_len: int, device="cpu"):
    n = batch_size * (seq_len + 1)
    while True:
        buf = torch.tensor([next(stream) for _ in range(n)], dtype=torch.long)
        buf = buf.view(batch_size, seq_len + 1).to(device, non_blocking=True)
        yield buf[:, :-1], buf[:, 1:]
