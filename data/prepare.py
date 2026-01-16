"""
Prepare the Tiny Shakespeare dataset for character-level language modeling.

It will create (in the same folder):
  data/input.txt  
  data/train.bin
  data/val.bin
  data/meta.pkl
"""

import os
import pickle
import numpy as np
from urllib.request import urlopen

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(DATA_DIR, "input.txt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

TRAIN_FRACTION = 0.9  # 90% train / 10% val


def download_if_missing():
    if os.path.exists(INPUT_FILE):
        return
    print("input.txt not found. Downloading Tiny Shakespeare...")
    with urlopen(DATA_URL) as r:
        text = r.read().decode("utf-8")
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Downloaded to {INPUT_FILE}")


def main():
    download_if_missing()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # unique characters 
    chars = sorted(set(data))
    vocab_size = len(chars)
    print("vocab size:", vocab_size)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str):
        return [stoi[c] for c in s]

    # split
    n = len(data)
    n_train = int(n * TRAIN_FRACTION)
    train_data = data[:n_train]
    val_data = data[n_train:]

    # encode
    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has   {len(val_ids):,} tokens")

    # write binaries
    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    # write meta
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
        "train_fraction": TRAIN_FRACTION,
        "dataset": "tiny_shakespeare_char",
        "source_url": DATA_URL,
    }
    meta_path = os.path.join(DATA_DIR, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print("Done.")
    print(f"Wrote: {train_path}, {val_path}, {meta_path}")


if __name__ == "__main__":
    main()
