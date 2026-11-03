"""Turn scripts/toolformer_train.py's PEFT adapter into one mlx_lm.server loads with --adapter-path.

PEFT keeps each LoRA pair as lora_A (rank x in) and lora_B (out x rank) under base_model.model.model.layers.N...;
mlx-lm's LoRALinear computes y + scale * (x @ lora_a) @ lora_b, with lora_a (in x rank) and lora_b (rank x out)
under model.layers.N..., and scale is PEFT's alpha / rank. So each matrix is transposed and renamed, and
adapter_config.json says which layers and projections carry one.

    .venv/bin/python scripts/toolformer_mlx.py                       # .wbc/toolformer/adapter-peft -> adapter-mlx
    mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --adapter-path .wbc/toolformer/adapter-mlx ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PEFT_KEY = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([AB])\.weight")


def convert(peft_dir: Path, out_dir: Path) -> dict:
    """Write adapters.safetensors and adapter_config.json for mlx-lm into out_dir; returns the config."""
    from safetensors.numpy import load_file, save_file

    peft = json.loads((peft_dir / "adapter_config.json").read_text())
    weights = load_file(str(peft_dir / "adapter_model.safetensors"))
    out, layers, keys = {}, set(), set()
    for name, w in weights.items():
        m = PEFT_KEY.fullmatch(name)
        if m is None:
            raise ValueError(f"not a LoRA weight this converter knows: {name}")
        layer, block, proj, ab = m.groups()
        layers.add(int(layer))
        keys.add(f"{block}.{proj}")
        out[f"model.layers.{layer}.{block}.{proj}.lora_{ab.lower()}"] = w.T.copy()
    if layers != set(range(max(layers) + 1)):
        raise ValueError("mlx-lm puts adapters on the last num_layers layers, so every layer from 0 needs one")
    config = {"fine_tune_type": "lora", "num_layers": len(layers),
              "lora_parameters": {"rank": peft["r"], "scale": peft["lora_alpha"] / peft["r"], "dropout": 0.0,
                                  "keys": sorted(keys)}}
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "adapters.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=1))
    return config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--peft", type=Path, default=Path(".wbc/toolformer/adapter-peft"))
    ap.add_argument("--out", type=Path, default=Path(".wbc/toolformer/adapter-mlx"))
    args = ap.parse_args()
    print(json.dumps(convert(args.peft, args.out), indent=1), f"\nin {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
