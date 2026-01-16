"""
Prompting / inference script for the trained SLM.
Staff version includes CodeCarbon tracking.
"""

import os
import pickle
import torch

from model import GPT, GPTConfig

from codecarbon import EmissionsTracker

OUT_DIR = "out"
DATA_DIR = "data"
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")

DEVICE = "auto"   # "cpu", "cuda", or "auto"
DTYPE = "float32" # keep simple
PROMPT = "ROMEO:"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.9
TOP_K = 50

def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        return pickle.load(f)

def encode(meta, s: str):
    stoi = meta["stoi"]
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(meta, ids):
    itos = meta["itos"]
    return "".join([itos[int(i)] for i in ids])

@torch.no_grad()
def main():
    device = resolve_device(DEVICE)
    print(f"[prompt] Using device: {device}")

    meta = load_meta(DATA_DIR)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    cfg_dict = ckpt["config"]["model"]
    cfg = GPTConfig(**cfg_dict)

    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- CodeCarbon tracker (inference) ----
    tracker = EmissionsTracker(
        project_name="slm-inference",
        output_dir=OUT_DIR,
        output_file="codecarbon_inference.csv",
        log_level="error",
    )
    tracker.start()
    # ---------------------------------------

    try:
        idx = encode(meta, PROMPT).unsqueeze(0).to(device)  # (1, T)
        out = model.generate(
            idx,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )
        text = decode(meta, out[0].cpu().tolist())
        print("\n--- Generated ---\n")
        print(text)
        print("\n-----------------\n")
    finally:
        emissions_kg = tracker.stop()
        print(f"[prompt] CodeCarbon emissions (kgCO2eq): {emissions_kg}")

if __name__ == "__main__":
    main()
