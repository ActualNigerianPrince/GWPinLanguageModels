"""
Course training script (simplified from nanoGPT).

Focus:
- Train a small GPT-style model from scratch on a tiny dataset.
- Staff version includes CodeCarbon tracking to validate expected outputs.
"""

import os
import time
import pickle
from dataclasses import asdict

import numpy as np
import torch

from model import GPTConfig, GPT

# --------- CodeCarbon (staff validation) ----------
from codecarbon import EmissionsTracker
# -----------------------------------------------

# -----------------------------------------------------------------------------
# Experiment configuration

# I/O
OUT_DIR = "out"
DATA_DIR = os.path.join("data")
EVAL_INTERVAL = 200
EVAL_ITERS = 50
LOG_INTERVAL = 50
SAVE_CHECKPOINT = True

# Model (main tunables)
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1
BIAS = True

# Training (main tunables)
SEED = 1337

# Use "auto" to pick cuda if available, else cpu (recommended for staff testing)
DEVICE = "auto"
DTYPE = "float32"
BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_ITERS = 2000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# -----------------------------------------------------------------------------

def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)

def get_batch(split: str, data_dir: str, block_size: int, batch_size: int, device: str):
    bin_path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])

    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model: GPT, data_dir: str, block_size: int, batch_size: int, device: str, eval_iters: int):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        split_losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch(split, data_dir, block_size, batch_size, device)
            _, loss = model(x, y)
            split_losses[k] = loss
        losses[split] = split_losses.mean().item()
    model.train()
    return losses

def save_checkpoint(out_dir: str, model: GPT, optimizer: torch.optim.Optimizer, iter_num: int, config: dict):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "iter_num": iter_num,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = resolve_device(DEVICE)
    print(f"[train] Using device: {device}")
    set_seed(SEED)

    meta = load_meta(DATA_DIR)
    vocab_size = meta["vocab_size"] if meta and "vocab_size" in meta else 50304

    cfg = GPTConfig(
        block_size=BLOCK_SIZE,
        vocab_size=vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=BIAS,
    )

    model = GPT(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # ---- CodeCarbon tracker (training) ----
    # output_dir keeps artifacts local; project_name helps identify runs
    tracker = EmissionsTracker(
        project_name="slm-training",
        output_dir=OUT_DIR,
        output_file="codecarbon_training.csv",
        log_level="error",
    )
    tracker.start()
    # --------------------------------------

    t0 = time.time()
    try:
        for it in range(MAX_ITERS + 1):
            # periodic evaluation
            if it % EVAL_INTERVAL == 0:
                losses = estimate_loss(model, DATA_DIR, BLOCK_SIZE, BATCH_SIZE, device, EVAL_ITERS)
                dt = time.time() - t0
                print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | elapsed {dt:.1f}s")

                if SAVE_CHECKPOINT and it > 0:
                    config_dump = {
                        "data_dir": DATA_DIR,
                        "train": {
                            "batch_size": BATCH_SIZE,
                            "block_size": BLOCK_SIZE,
                            "max_iters": MAX_ITERS,
                            "learning_rate": LEARNING_RATE,
                            "weight_decay": WEIGHT_DECAY,
                            "grad_clip": GRAD_CLIP,
                            "dtype": DTYPE,
                            "device": device,
                        },
                        "model": asdict(cfg),
                    }
                    save_checkpoint(OUT_DIR, model, optimizer, it, config_dump)

            # training step
            x, y = get_batch("train", DATA_DIR, BLOCK_SIZE, BATCH_SIZE, device)
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if GRAD_CLIP and GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            if it % LOG_INTERVAL == 0:
                print(f"iter {it:5d} | loss {loss.item():.4f}")

    finally:
        # ---- Stop CodeCarbon and print summary ----
        emissions_kg = tracker.stop()  # returns kgCO2eq
        print(f"[train] CodeCarbon emissions (kgCO2eq): {emissions_kg}")
        # -------------------------------------------

    print("Training completed.")

if __name__ == "__main__":
    main()
