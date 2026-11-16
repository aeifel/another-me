"""Training, evaluation and serving on Modal (PyTorch + PEFT).

Run through scripts/train.py, which loads .env and checks the token:

    --action smoke      CPU only, a few seconds: image, config, volume and secret all work
    --action download   CPU only: put the base model on the Modal volume
    --action train      GPU: train the adapter (add --max-steps 3 for a cheap dry run)
    --action compare    GPU: tuned and base replies on held-out questions
    --action deploy     publish the Twin model service that the demo page talks to

The base model and every adapter live on the Modal volume, adapters under adapters/<run-name>.

Cost guard: every function has a timeout. The longest, 45 minutes of L4 time, is about $0.85.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
# This module is imported twice: where you launch it and inside the Modal container.
# The container has no repo checkout, so the config is copied into the image.
_LOCAL_CFG = ROOT / "configs" / "default.toml"
_REMOTE_CFG = Path("/cfg/default.toml")
with (_LOCAL_CFG if _LOCAL_CFG.exists() else _REMOTE_CFG).open("rb") as _f:
    CFG = tomllib.load(_f)

# One GPU type, or a comma-separated fallback list such as "L4,A10" to shorten queueing.
_gpus = [g.strip() for g in os.environ.get("ANOTHER_ME_GPU", "L4").split(",") if g.strip()]
GPU = _gpus[0] if len(_gpus) == 1 else _gpus
VOL = "/vol"

app = modal.App(CFG["modal"]["app_name"])
volume = modal.Volume.from_name(CFG["modal"]["volume"], create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=4.51", "peft", "accelerate", "huggingface_hub")
    .env({"HF_HOME": f"{VOL}/hf",  # cache the base model on the volume between runs
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
if _LOCAL_CFG.exists():
    image = image.add_local_file(str(_LOCAL_CFG), str(_REMOTE_CFG))

# Optional Hugging Face token, taken from the launching environment (.env via scripts/train.py).
_hf_token = os.environ.get("HF_TOKEN")
secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})] if _hf_token else []

cpu_fn = dict(image=image, volumes={VOL: volume}, secrets=secrets)
gpu_fn = dict(
    **cpu_fn,
    gpu=GPU,
    cpu=CFG["modal"]["cpu_cores"],
    memory=CFG["modal"]["memory_gib"] * 1024,
)


@app.function(**cpu_fn, timeout=5 * 60)
def smoke() -> dict:
    """Cheap end-to-end check of everything except the GPU and the model."""
    import platform
    import time

    import peft
    import torch
    import transformers

    marker = Path(VOL) / "smoke.txt"
    marker.write_text(str(time.time()))
    volume.commit()
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),  # plain str: TorchVersion cannot be unpickled without torch
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "config_model": CFG["model"]["hf_id"],
        "config_read_from": str(_REMOTE_CFG if _REMOTE_CFG.exists() else _LOCAL_CFG),
        "volume_writable": marker.exists(),
        "hf_token_set": bool(os.environ.get("HF_TOKEN")),
        "model_cached": (Path(VOL) / "hf").exists(),
    }


@app.function(**cpu_fn, timeout=30 * 60)
def download_model(hf_id: str) -> dict:
    """Download the base model onto the volume without paying for a GPU."""
    import time

    from huggingface_hub import snapshot_download

    start = time.time()
    path = snapshot_download(hf_id)
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    volume.commit()
    return {"path": path, "gib": round(size / 2**30, 2), "seconds": round(time.time() - start, 1)}


def _render(tok, messages: list[dict], add_generation_prompt: bool = False) -> str:
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=add_generation_prompt)


def _load_base(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"]["hf_id"])
    try:  # transformers 5 renamed torch_dtype to dtype
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["hf_id"], dtype=torch.bfloat16, device_map="cuda")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["hf_id"], torch_dtype=torch.bfloat16, device_map="cuda")
    print(f"weights: {next(model.parameters()).dtype}, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB on GPU after load")
    return tok, model


def _load_tuned(cfg: dict, run_name: str):
    """Base model with the adapter attached. `disable_adapter()` gives the base behaviour."""
    from peft import PeftModel

    tok, base = _load_base(cfg)
    model = PeftModel.from_pretrained(base, f"{VOL}/adapters/{run_name}")
    model.eval()
    return tok, model


def _reply(tok, model, messages: list[dict], arm: str, max_tokens: int, temperature: float,
           seed: int) -> str:
    """One reply. arm "tuned" uses the adapter, "base" switches it off."""
    import torch

    text = _render(tok, messages, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")

    def run() -> str:
        torch.manual_seed(seed)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                                 temperature=temperature, top_p=0.9)
        return tok.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    if arm == "base":
        with model.disable_adapter():
            return run()
    return run()


@app.function(**gpu_fn, timeout=45 * 60)
def train(train_rows: list[dict], valid_rows: list[dict], cfg: dict, run_name: str,
          max_steps: int = 0) -> dict:
    import math
    import time

    import torch
    from peft import LoraConfig, get_peft_model

    t, lora = cfg["train"], cfg["lora"]
    torch.manual_seed(t["seed"])
    load_start = time.time()
    tok, model = _load_base(cfg)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["rank"],
            lora_alpha=int(lora["rank"] * lora["scale"]),
            lora_dropout=lora["dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    load_seconds = time.time() - load_start
    print(f"loaded in {load_seconds:.0f}s; trainable {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    def encode(row: dict) -> tuple[list[int], list[int]]:
        """Token ids, plus labels with everything before the reply masked out."""
        msgs = row["messages"]
        full_text = _render(tok, msgs)
        prefix_text = _render(tok, msgs[:-1], add_generation_prompt=True)
        ids = tok(full_text, add_special_tokens=False)["input_ids"][: t["max_seq_length"]]
        n = 0
        if t["mask_prompt"] and full_text.startswith(prefix_text):
            n = min(len(tok(prefix_text, add_special_tokens=False)["input_ids"]), len(ids))
        return ids, [-100] * n + ids[n:]

    def batches(rows: list[dict], size: int):
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        for i in range(0, len(rows), size):
            enc = [encode(r) for r in rows[i : i + size]]
            width = max(len(ids) for ids, _ in enc)
            ids = torch.tensor([e + [pad] * (width - len(e)) for e, _ in enc])
            labels = torch.tensor([l + [-100] * (width - len(l)) for _, l in enc])
            mask = torch.tensor([[1] * len(e) + [0] * (width - len(e)) for e, _ in enc])
            yield ids.cuda(), mask.cuda(), labels.cuda()

    eval_seconds = 0.0

    def eval_loss() -> float | None:
        nonlocal eval_seconds
        if not valid_rows:
            return None
        started = time.time()
        model.eval()
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for ids, mask, labels in batches(valid_rows, 4):
                losses.append(model(input_ids=ids, attention_mask=mask, labels=labels).loss.item())
        model.train()
        eval_seconds += time.time() - started
        return sum(losses) / len(losses)

    # Sanity check on the first example: the masked labels must cover only the reply.
    ids0, labels0 = encode(train_rows[0])
    reply_tokens = sum(1 for x in labels0 if x != -100)
    print(f"first example: {len(ids0)} tokens, {reply_tokens} of them trained on")
    print("trained-on text:", repr(tok.decode([x for x in labels0 if x != -100]))[:200])

    # Micro-batches with gradient accumulation keep long examples within a 24 GB GPU.
    batch_size, accumulate = t["batch_size"], t["grad_accumulation"]
    steps_per_epoch = math.ceil(math.ceil(len(train_rows) / batch_size) / accumulate)
    eval_every = t.get("eval_every_steps", 0)
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                              lr=t["learning_rate"])
    out = f"{VOL}/adapters/{run_name}"
    history = [{"epoch": 0, "steps": 0, "valid_loss": eval_loss()}]
    print(history[0])
    best = {"epoch": None, "steps": None, "valid_loss": None}
    tokens_seen, steps, start = 0, 0, time.time()
    running: list[float] = []

    def checkpoint() -> None:
        """Measure validation loss and keep the adapter if it is the best so far.

        Later checkpoints can memorise the training answers and get worse on unseen questions.
        """
        nonlocal best
        entry = {"epoch": round(steps / steps_per_epoch, 2), "steps": steps,
                 "train_loss": sum(running) / len(running), "valid_loss": eval_loss()}
        running.clear()
        print(entry)
        history.append(entry)
        loss_now = entry["valid_loss"]
        if loss_now is None or best["valid_loss"] is None or loss_now < best["valid_loss"]:
            best = {k: entry[k] for k in ("epoch", "steps", "valid_loss")}
            model.save_pretrained(out)
            print(f"saved adapter at step {steps}")

    def step() -> bool:
        """Apply the accumulated gradients. Returns True when training should stop."""
        nonlocal steps
        optim.step()
        optim.zero_grad()
        steps += 1
        if max_steps and steps >= max_steps:
            return True
        if eval_every and steps % eval_every == 0:
            checkpoint()
        return False

    model.train()
    stop = False
    for epoch in range(1, t["epochs"] + 1):
        g = torch.Generator().manual_seed(t["seed"] + epoch)
        order = torch.randperm(len(train_rows), generator=g).tolist()
        micro = 0
        for ids, mask, labels in batches([train_rows[i] for i in order], batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            (loss / accumulate).backward()
            running.append(loss.item())
            tokens_seen += int(mask.sum())
            micro += 1
            if micro % accumulate == 0 and step():
                stop = True
                break
        if micro % accumulate and not stop:  # flush the last partial accumulation
            stop = step()
        if running:  # end of the epoch, unless a checkpoint just happened
            checkpoint()
        if stop:
            break
    seconds = time.time() - start - eval_seconds
    volume.commit()
    final = best["valid_loss"]
    return {
        "run_name": run_name,
        "gpu_name": torch.cuda.get_device_name(0),  # what Modal actually assigned
        "model": cfg["model"]["hf_id"],
        "train_pairs": len(train_rows),
        "valid_pairs": len(valid_rows),
        "epochs_run": round(steps / steps_per_epoch, 2),
        "best_epoch": best["epoch"],
        "best_step": best["steps"],
        "steps": steps,
        "batch_size": batch_size,
        "grad_accumulation": accumulate,
        "learning_rate": t["learning_rate"],
        "trainable_params": trainable,
        "trainable_percent": round(100 * trainable / total, 3),
        "model_load_seconds": round(load_seconds, 1),
        "train_seconds": round(seconds, 1),
        "tokens_seen": tokens_seen,
        "tokens_per_second": round(tokens_seen / seconds, 1),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "valid_loss": round(final, 4) if final is not None else None,
        "valid_perplexity": round(math.exp(final), 2) if final is not None else None,
        "history": history,
        "adapter_path": out,
    }


@app.function(**gpu_fn, timeout=30 * 60)
def generate(rows: list[dict], cfg: dict, run_name: str, arms: list[str],
             temperatures: list[float]) -> dict:
    """Reply to every prompt, once per arm and temperature, on one loaded model.

    Each row is {"messages": [...]} ending with the user turn. A row may carry
    "base_messages": what the base arm gets instead (the same turn without profile notes).
    Returns {"<temperature>": [{"tuned": ..., "base": ...}, ...]}.
    """
    tok, model = _load_tuned(cfg, run_name)
    max_tokens = cfg["generate"]["max_tokens"]
    out = {}
    for temperature in temperatures:
        out[str(temperature)] = [
            {arm: _reply(tok, model,
                         row.get("base_messages", row["messages"]) if arm == "base" else row["messages"],
                         arm, max_tokens, temperature,
                         cfg["train"]["seed"] + i) for arm in arms}
            for i, row in enumerate(rows)
        ]
    return out


@app.cls(**gpu_fn, scaledown_window=CFG["serve"]["scaledown_seconds"], timeout=10 * 60,
         max_containers=1)
class Twin:
    """The model service behind the demo page: base model plus adapter, loaded once.

    The container stays up for [serve] scaledown_seconds after the last request, so only
    the first request after a quiet period pays the cold start. max_containers=1 caps cost.
    """

    @modal.enter()
    def load(self):
        import time

        start = time.time()
        self.adapter = CFG["serve"]["adapter"]
        self.tok, self.model = _load_tuned(CFG, self.adapter)
        self.load_seconds = round(time.time() - start, 1)

    @modal.method()
    def ping(self) -> dict:
        return {"ready": True, "adapter": self.adapter, "load_seconds": self.load_seconds}

    @modal.method()
    def reply(self, messages: list[dict], arms: list[str], max_tokens: int = 256,
              seed: int = 0) -> dict:
        """Answer once per arm, one after the other, on the same loaded model."""
        import time

        start = time.time()
        temperature = CFG["generate"]["temperature"]
        replies = {arm: _reply(self.tok, self.model, messages, arm, max_tokens, temperature, seed)
                   for arm in arms}
        return {"replies": replies, "adapter": self.adapter,
                "seconds": round(time.time() - start, 2)}


def _words(text: str) -> int:
    return len(text.split())


@app.local_entrypoint()
def main(action: str = "train", run_name: str = "v1", max_steps: int = 0, limit: int = 0,
         split: str = "test", temperatures: str = ""):
    import datetime
    import json

    def rows(name: str) -> list[dict]:
        p = ROOT / CFG["data"]["processed"] / name
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def save(name: str, result) -> None:
        out = ROOT / "runs" / name
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"saved runs/{name}")

    if action == "smoke":
        print(json.dumps(smoke.remote(), indent=2))
    elif action == "download":
        print(json.dumps(download_model.remote(CFG["model"]["hf_id"]), indent=2))
    elif action == "train":
        train_rows = rows("train.jsonl")
        if not train_rows:
            raise SystemExit("datasets/processed/train.jsonl not found. Run scripts/prepare.py first.")
        if limit:
            train_rows = train_rows[:limit]
        result = train.remote(train_rows, rows("valid.jsonl"), CFG, run_name, max_steps)
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
        for entry in result["history"]:
            print(entry)
        save(f"{run_name}.modal.json", result)
    elif action == "compare":
        held = rows(f"{split}.jsonl")
        if not held:
            raise SystemExit(f"datasets/processed/{split}.jsonl not found. Run scripts/prepare.py first.")
        if limit:
            held = held[:limit]

        def without_notes(messages: list[dict]) -> list[dict]:
            # The system prompt is "<instruction>\n\nWhat you know about <name>: ..."
            # (another_me/schema.py). The base model gets the instruction only.
            system = messages[0]["content"].split("\n\nWhat you know about ")[0]
            return [{"role": "system", "content": system}] + messages[1:]

        prompts = [{"messages": r["messages"][:-1], "base_messages": without_notes(r["messages"][:-1])}
                   for r in held]
        real = [r["messages"][-1]["content"] for r in held]
        work = ["friend" not in r["messages"][0]["content"] for r in held]

        def mean_words(texts: list[str], want_work: bool) -> float:
            picked = [_words(t) for t, w in zip(texts, work) if w == want_work]
            return round(sum(picked) / len(picked), 1) if picked else 0.0

        if temperatures:
            # Sweep: tuned replies only, to choose a temperature on the validation split.
            temps = [float(x) for x in temperatures.split(",")]
            outputs = generate.remote(prompts, CFG, run_name, ["tuned"], temps)
            print(f"\naverage reply length in words, {split} split, run {run_name}")
            print(f"{'':12s} {'work':>6s} {'casual':>7s}")
            print(f"{'real':12s} {mean_words(real, True):>6} {mean_words(real, False):>7}")
            for temp in temps:
                tuned = [o["tuned"] for o in outputs[str(temp)]]
                print(f"{'tuned @' + str(temp):12s} {mean_words(tuned, True):>6} {mean_words(tuned, False):>7}")
            save(f"{run_name}.sweep.{split}.json",
                 {"split": split, "prompts": [p["messages"][-1]["content"] for p in prompts],
                  "real": real, "outputs": outputs})
            return

        temp = CFG["generate"]["temperature"]
        outputs = generate.remote(prompts, CFG, run_name, ["tuned", "base"], [temp])[str(temp)]
        tuned, base = [o["tuned"] for o in outputs], [o["base"] for o in outputs]
        summary = {
            "questions": len(held),
            "temperature": temp,
            "average_words": {
                arm: {"all": round(sum(map(_words, texts)) / len(texts), 1),
                      "work": mean_words(texts, True), "casual": mean_words(texts, False)}
                for arm, texts in (("real", real), ("tuned", tuned), ("base", base))
            },
        }
        print(json.dumps(summary, indent=2))
        result = {
            "sample": False,
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": CFG["model"]["hf_id"],
            "adapter": run_name,
            "split": split,
            "summary": summary,
            "arms": ["your answer", "tuned", "base"],
            "rows": [
                {
                    "register": "professional" if w else "casual",
                    "prompt": p["messages"][-1]["content"],
                    "outputs": {"your answer": {"text": r}, "tuned": {"text": t}, "base": {"text": b}},
                }
                for p, r, t, b, w in zip(prompts, real, tuned, base, work)
            ],
        }
        save(f"{run_name}.heldout.json", result)
        if split == "test":  # the page shows the held-out test questions
            out = ROOT / "web" / "data" / "comparison.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            print("wrote web/data/comparison.json")
    else:
        raise SystemExit(f"unknown action '{action}'")
