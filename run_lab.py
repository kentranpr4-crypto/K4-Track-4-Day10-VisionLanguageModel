"""End-to-end VM runner for Lab Day 10.

Examples:
  python run_lab.py --experiment e0 --data data --output work/e0.jsonl
  python run_lab.py --experiment e1 --data data --output work/e1.jsonl --max-steps 300

The script deliberately keeps dataset preparation and scoring in scripts/ unchanged.
It uses the same 0-1000 coordinate contract as scripts.score.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # Keep data/label helpers importable before VM dependencies are installed.
    torch = None
from PIL import Image

from scripts.labcommon import read_jsonl, write_jsonl

PROMPT = (
    "Detect every road-surface defect in this image. Return only a valid JSON list "
    "with objects of the form {\\\"label\\\":\\\"pothole|crack|manhole\\\", "
    "\\\"bbox_2d\\\":[x1,y1,x2,y2]}. Coordinates must be integers on the 0-1000 "
    "scale (relative to image width and height). Return [] when there is no defect."
)
CLASSES = {"pothole", "crack", "manhole"}


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for run_lab.py; install requirements-vm.txt first")


def yolo_to_box(label: str, yolo: list[float], scale: int = 1000) -> dict[str, Any]:
    xc, yc, w, h = yolo
    vals = ((xc - w / 2) * scale, (yc - h / 2) * scale,
            (xc + w / 2) * scale, (yc + h / 2) * scale)
    box = [max(0, min(scale, int(round(v)))) for v in vals]
    box[2] = max(box[0] + 1, box[2])
    box[3] = max(box[1] + 1, box[3])
    box[2] = min(scale, box[2])
    box[3] = min(scale, box[3])
    return {"label": label, "bbox_2d": box}


def objects_to_answer(objects: list[dict[str, Any]], scale: int = 1000) -> str:
    boxes = [yolo_to_box(o["label"], o["yolo"], scale) for o in objects]
    boxes.sort(key=lambda x: (x["bbox_2d"][1], x["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def seed_everything(seed: int) -> None:
    _require_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_classes():
    _require_torch()
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration as Qwen3VLForConditionalGeneration
    return AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration


def load_model(model_id: str, training: bool = False):
    _require_torch()
    AutoProcessor, BitsAndBytesConfig, Model = _model_classes()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        q = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"],
        )
        kwargs["quantization_config"] = q
        kwargs["device_map"] = {"": 0}
    model = Model.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    return model, processor


def _messages(image: Image.Image, answer: str | None = None):
    user = {"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": PROMPT}]}
    if answer is None:
        return [user]
    return [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}]


def generate_predictions(model, processor, rows, data_root: Path, max_new_tokens=256,
                         limit: int | None = None):
    device = next(model.parameters()).device
    out = []
    model.eval()
    with torch.inference_mode():
        for row in rows[:limit]:
            image = Image.open(data_root / row["image_path"]).convert("RGB")
            enc = processor.apply_chat_template(
                _messages(image), tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            enc = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}
            generated = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            prompt_len = enc["input_ids"].shape[1]
            raw = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0]
            out.append({"image_id": row["image_id"], "raw": raw.strip()})
    return out


def _pad(values, pad, side="right"):
    n = max(x.shape[-1] for x in values)
    out = values[0].new_full((len(values), n), pad)
    for i, x in enumerate(values):
        if side == "left":
            out[i, -x.shape[-1]:] = x
        else:
            out[i, :x.shape[-1]] = x
    return out


def encode_example(processor, image, answer: str):
    prompt = processor.apply_chat_template(_messages(image), tokenize=True,
                                           add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt")
    full = processor.apply_chat_template(_messages(image, answer), tokenize=True,
                                         add_generation_prompt=False,
                                         return_dict=True, return_tensors="pt")
    n = prompt["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][0, :n], prompt["input_ids"][0]):
        raise RuntimeError("chat template is not prefix-stable; cannot mask prompt loss")
    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    full["labels"] = labels
    return full


def collate(batch, processor):
    tokenizer = processor.tokenizer
    keys = ("input_ids", "attention_mask", "labels", "mm_token_type_ids")
    result = {}
    for key in keys:
        vals = [x[key] for x in batch if key in x]
        if not vals:
            continue
        if key == "labels":
            pad = -100
        elif key == "mm_token_type_ids":
            # 0 is the text modality. Padding with pad_token_id (151643)
            # makes Qwen3-VL look up a nonexistent modality grid key.
            pad = 0
        else:
            pad = tokenizer.pad_token_id or 0
        result[key] = _pad(vals, pad, getattr(tokenizer, "padding_side", "right"))
    for key in ("pixel_values", "image_grid_thw"):
        vals = [x[key] for x in batch if key in x]
        if vals:
            result[key] = torch.cat(vals, dim=0)
    return result


def apply_setup(model, setup: str, rank=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    # PEFT 0.21 treats a list as module-name suffixes. This is more portable
    # across Qwen3-VL wrappers than relying on a full-name regular expression.
    targets = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
    if setup == "e3":
        targets += ["qkv", "proj", "linear_fc1", "linear_fc2"]
    peft = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha,
                    lora_dropout=dropout, target_modules=targets,
                    task_type="CAUSAL_LM", bias="none"))
    # Keep trainable LoRA weights in FP32 so GradScaler can safely handle the
    # optimizer together with the FP32 projector parameters.
    for name, p in peft.named_parameters():
        if p.requires_grad and "lora_" in name:
            p.data = p.data.float()
    if setup in ("e2", "e3"):
        for name, p in peft.named_parameters():
            if "visual.merger" in name or "visual.deepstack_merger_list" in name:
                p.data = p.data.float()
                p.requires_grad = True
    return peft


def _projector_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if "visual.merger" in n or "visual.deepstack_merger_list" in n}


def train(model, processor, rows, data_root: Path, setup: str, output_dir: Path,
          max_steps=300, batch_size=2, grad_accum=4, lr=2e-4, seed=42):
    _require_torch()
    seed_everything(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters; check LoRA target module names")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0)
    device = next(model.parameters()).device
    # Autocast may still produce FP16 gradients for wrapped LoRA modules even
    # when their storage dtype is FP32, so GradScaler.unscale_ is unsupported
    # by this PyTorch build. Keep autocast but use direct backward/step.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    peak = 0
    model.train()
    for step in range(max_steps):
        opt.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(grad_accum):
            batch_rows = [rows[((step * grad_accum + _) * batch_size + i) % len(rows)]
                          for i in range(batch_size)]
            encoded = []
            for row in batch_rows:
                image = Image.open(data_root / row["image_path"]).convert("RGB")
                encoded.append(encode_example(processor, image, objects_to_answer(row["objects"])))
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in collate(encoded, processor).items()}
            # Projector training can overflow FP16 on Qwen3-VL. Ada GPUs have
            # native BF16, which keeps the same mixed-precision speed with a
            # much wider exponent range.
            amp_dtype = torch.float16
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                loss = model(**batch).loss / grad_accum
            if not math.isfinite(float(loss.detach())):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_value += float(loss.detach())
        if scaler.is_enabled():
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
        if (step + 1) % 25 == 0:
            print(f"{setup} step {step + 1}/{max_steps} loss={loss_value:.4f}")
    model.save_pretrained(output_dir / "adapter")
    if setup in ("e2", "e3"):
        torch.save(_projector_state(model), output_dir / "projector.pt")
    stats = {"trainable_params": sum(p.numel() for p in trainable),
             "peak_vram_gb": peak, "train_minutes": (time.monotonic() - started) / 60}
    (output_dir / "run.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", choices=["e0", "e1", "e2", "e3", "e4"], required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--output", required=True, help="prediction jsonl for e0, or final jsonl for e1-e3")
    ap.add_argument("--run-dir", default="work/run")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16,
                    help="LoRA rank; use 4 and 64 for the E4 rank ablation")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    data = Path(args.data)
    split = read_jsonl(data / "splits" / "test.jsonl")
    model, processor = load_model(args.model_id, training=args.experiment != "e0")
    if args.experiment == "e0":
        rows = generate_predictions(model, processor, split, data, limit=args.limit)
        write_jsonl(args.output, rows)
        return
    setup = "e3" if args.experiment == "e4" else args.experiment
    model = apply_setup(model, setup, rank=args.rank, alpha=2 * args.rank)
    train_rows = read_jsonl(data / "splits" / "train.jsonl")
    stats = train(model, processor, train_rows, data, setup, Path(args.run_dir),
                  max_steps=args.max_steps or 300, batch_size=args.batch_size,
                  grad_accum=args.grad_accum)
    rows = generate_predictions(model, processor, split, data, limit=args.limit)
    write_jsonl(args.output, rows)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
