"""Fine-tune the fast path's Llama 3.2 3B on Modal, on scripts/toolformer_data.py's labels, as a LoRA adapter.

LoRA on every layer's q_proj and v_proj (rank 8, alpha 16), the loss on the answer only, a few epochs on one A10G.
The base is Llama 3.2 3B Instruct from unsloth's ungated copy, the same weights as Meta's, so no Hugging Face token
is needed. The adapter comes back to .wbc/toolformer/adapter-peft; scripts/toolformer_mlx.py turns it into one
mlx_lm.server loads with --adapter-path, on top of the 4-bit model the fast path runs.

    .venv/bin/modal setup                                           # once: logs this Mac in to Modal, in the browser
    .venv/bin/modal run scripts/toolformer_train.py                 # PAID on Modal: an A10G for a few minutes, under $1
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

BASE = "unsloth/Llama-3.2-3B-Instruct"
KEYS = ["q_proj", "v_proj"]
RANK, ALPHA = 8, 16
app = modal.App("wbc-toolformer")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.5.1", "transformers==4.46.3", "peft==0.13.2", "accelerate==1.1.1", "safetensors==0.4.5")


@app.function(gpu="A10G", image=image, timeout=3600)
def train(train_rows: list, valid_rows: list, epochs: int = 3, lr: float = 2e-4) -> dict:
    import random

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.05, target_modules=KEYS,
                                             task_type="CAUSAL_LM"))

    def encode(row):
        """The whole chat as tokens, with every token before the answer masked out of the loss."""
        full = tok.apply_chat_template(row["messages"], tokenize=True)
        prompt = tok.apply_chat_template(row["messages"][:-1], tokenize=True, add_generation_prompt=True)
        labels = [-100] * len(prompt) + full[len(prompt):]
        return torch.tensor([full], device="cuda"), torch.tensor([labels], device="cuda")

    def answer(row) -> dict:
        prompt = tok.apply_chat_template(row["messages"][:-1], add_generation_prompt=True, return_tensors="pt")
        out = model.generate(prompt.to("cuda"), max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, prompt.shape[1]:], skip_special_tokens=True)
        try:
            return json.loads(text[text.index("{"):text.index("}") + 1])
        except ValueError:
            return {}

    def tools_right(rows) -> float:
        model.eval()
        with torch.no_grad():
            right = sum(answer(r).get("tool") == json.loads(r["messages"][-1]["content"])["tool"] for r in rows)
        model.train()
        return right / max(1, len(rows))

    before = tools_right(valid_rows)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    accumulate, losses = 8, []
    model.train()
    for epoch in range(epochs):
        rows = train_rows[:]
        random.Random(epoch).shuffle(rows)
        for i, row in enumerate(rows):
            ids, labels = encode(row)
            loss = model(input_ids=ids, labels=labels).loss / accumulate
            loss.backward()
            losses.append(loss.item() * accumulate)
            if (i + 1) % accumulate == 0 or i == len(rows) - 1:
                opt.step()
                opt.zero_grad()
        print(f"epoch {epoch + 1}: mean loss {sum(losses[-len(rows):]) / len(rows):.4f}", flush=True)
    after = tools_right(valid_rows)
    model.save_pretrained("/tmp/adapter")
    from safetensors.torch import load_file, save_file
    weights = Path("/tmp/adapter/adapter_model.safetensors")  # in float16, which numpy reads where bf16 isn't
    save_file({k: v.to(torch.float16).contiguous() for k, v in load_file(weights).items()}, weights)
    files = {p.name: p.read_bytes() for p in Path("/tmp/adapter").iterdir() if p.is_file()}
    return {"files": files, "valid_tool_right_before": before, "valid_tool_right_after": after,
            "train_examples": len(train_rows), "valid_examples": len(valid_rows)}


@app.local_entrypoint()
def main(data: str = ".wbc/toolformer", epochs: int = 3):
    folder = Path(data)
    rows = {name: [json.loads(l) for l in (folder / f"{name}.jsonl").read_text().splitlines() if l.strip()]
            for name in ("train", "valid")}
    result = train.remote(rows["train"], rows["valid"], epochs=epochs)
    out = folder / "adapter-peft"
    out.mkdir(parents=True, exist_ok=True)
    for name, data_bytes in result.pop("files").items():
        (out / name).write_bytes(data_bytes)
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), f"\nadapter in {out}")
