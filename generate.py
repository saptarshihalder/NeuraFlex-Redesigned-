"""Sample from a NeuraFlex-Ouroboros checkpoint.

    python generate.py --ckpt checkpoints/step_1000.pt --prompt "The universe" \
        --loops 8 --exit-threshold 0.02
More loops = more latent "thinking" per token; --exit-threshold lets easy tokens stop early.
"""
import argparse
import os

import torch

from neuraflex import NeuraFlex, NeuraFlexConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None, help="defaults to config.json next to the checkpoint")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer; omit for byte-level models")
    ap.add_argument("--prompt", default="NeuraFlex is")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--loops", type=int, default=None)
    ap.add_argument("--exit-threshold", type=float, default=None)
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(os.path.dirname(args.ckpt), "config.json")
    cfg = NeuraFlexConfig.from_json(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuraFlex(cfg)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd.get("model", sd))
    model.to(device).eval()

    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        ids = torch.tensor([tok(args.prompt)["input_ids"]], device=device)
        decode = lambda t: tok.decode(t)
        eos = tok.eos_token_id
    else:
        ids = torch.tensor([list(args.prompt.encode())], device=device)
        decode = lambda t: bytes(t).decode("utf-8", errors="replace")
        eos = None

    out, loops = model.generate(ids, args.max_new_tokens, args.temperature, args.top_p,
                                n_loops=args.loops, exit_threshold=args.exit_threshold, eos_id=eos)
    print(decode(out[0].tolist()))
    print(f"\n[avg loops/token: {sum(loops) / len(loops):.2f}]")


if __name__ == "__main__":
    main()
