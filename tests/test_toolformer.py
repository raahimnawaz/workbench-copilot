"""The Toolformer-style fine-tune's offline parts: the labeler's input, the filter, and the adapter conversion."""
import json
import sys

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from wbc.agent import fast_arm
from wbc.agent.budget import Budget
from wbc.agent.tools import Bench, build_toolbox
from wbc.config import ROOT, load_yaml

sys.path.insert(0, str(ROOT / "scripts"))
import toolformer_data  # noqa: E402
import toolformer_mlx  # noqa: E402


def test_the_labeler_is_asked_what_the_fast_path_asks_llama():
    asked = fast_arm.asked("put it on the tray", "screwdriver", ["The screwdriver goes in the \"cup\"."])
    assert asked.startswith("Bench state:\n- Notebook: (1) The screwdriver goes in the")
    assert "- The arm is holding: the screwdriver, which it picked up." in asked
    assert asked.endswith('What the user said: "put it on the tray"')
    row = toolformer_data.example("put it on the tray", "screwdriver", {"tool": "put_down", "place": "tray"})
    system, user, answer = (m["content"] for m in row["messages"])
    assert system == fast_arm.SYSTEM and "holding: the screwdriver" in user
    assert json.loads(answer) == {"tool": "put_down", "thing": None, "place": "tray"}


def test_a_label_is_kept_only_when_its_call_works_on_the_recorded_bench():
    bench = Bench(ROOT, load_yaml("models.yaml"), None, Budget(10, {}), sim=True, say=lambda s: None,
                  robot_cfg=load_yaml("robot.yaml"))
    tb = build_toolbox(bench)
    runs = lambda label, holding=None: toolformer_data.runs(tb, bench, label, holding)[0]  # noqa: E731
    assert runs({"tool": "move_thing", "thing": "screwdriver", "place": "cup"})
    assert not runs({"tool": "put_down", "place": "tray"}), "nothing held to put down"
    assert runs({"tool": "put_down", "place": "tray"}, "screwdriver")
    assert not runs({"tool": "pick_up", "thing": "hammer"}), "not on the desk"
    assert runs({"tool": "hold"}) and runs({"tool": "not_mine"}), "no call to run: the kind check decides"
    assert toolformer_data.kind("put_down") == toolformer_data.kind("move_thing") != toolformer_data.kind("not_mine")


def test_lines_drops_numbering_quotes_and_blanks():
    assert toolformer_data.lines('1. "Copilot, grab the cup"\n\n- put it on the tray\nok') == [
        "Copilot, grab the cup", "put it on the tray"]


def fake_peft(folder, layers=28, rank=8, alpha=16):
    """A PEFT adapter shaped like Llama 3.2 3B's: q_proj 3072 -> 3072, v_proj 3072 -> 1024."""
    rng = np.random.default_rng(0)
    weights = {}
    for i in range(layers):
        for proj, out in (("q_proj", 3072), ("v_proj", 1024)):
            base = f"base_model.model.model.layers.{i}.self_attn.{proj}"
            weights[f"{base}.lora_A.weight"] = rng.standard_normal((rank, 3072)).astype(np.float16)
            weights[f"{base}.lora_B.weight"] = rng.standard_normal((out, rank)).astype(np.float16)
    folder.mkdir(parents=True)
    save_file(weights, str(folder / "adapter_model.safetensors"))
    (folder / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": alpha}))
    return weights


def test_the_peft_adapter_becomes_mlx_lms_transposed_and_renamed(tmp_path):
    peft = fake_peft(tmp_path / "peft")
    config = toolformer_mlx.convert(tmp_path / "peft", tmp_path / "mlx")
    assert config == {"fine_tune_type": "lora", "num_layers": 28,
                      "lora_parameters": {"rank": 8, "scale": 2.0, "dropout": 0.0,
                                          "keys": ["self_attn.q_proj", "self_attn.v_proj"]}}
    mlx = load_file(str(tmp_path / "mlx" / "adapters.safetensors"))
    a, b = mlx["model.layers.5.self_attn.v_proj.lora_a"], mlx["model.layers.5.self_attn.v_proj.lora_b"]
    assert a.shape == (3072, 8) and b.shape == (8, 1024)
    x = np.ones((1, 3072), np.float32)
    A = peft["base_model.model.model.layers.5.self_attn.v_proj.lora_A.weight"].astype(np.float32)
    B = peft["base_model.model.model.layers.5.self_attn.v_proj.lora_B.weight"].astype(np.float32)
    assert x @ a.astype(np.float32) @ b.astype(np.float32) == pytest.approx(x @ A.T @ B.T, rel=1e-3), \
        "the same update, PEFT's B(A x) as mlx's (x a) b"


def test_a_peft_adapter_missing_a_layer_is_refused(tmp_path):
    fake_peft(tmp_path / "peft")
    weights = load_file(str(tmp_path / "peft" / "adapter_model.safetensors"))
    save_file({k: v for k, v in weights.items() if ".layers.0." not in k}, str(tmp_path / "peft" / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="every layer from 0"):
        toolformer_mlx.convert(tmp_path / "peft", tmp_path / "mlx")
