"""Check for identical images shared between train/valid/test, and split sizes.

Usage: python check_data.py [--data Data]
"""
import argparse
import hashlib
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def hashes(split_dir):
    out = {}
    for f in split_dir.rglob("*"):
        if f.suffix.lower() in EXT:
            out.setdefault(hashlib.md5(f.read_bytes()).hexdigest(), []).append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data")
    data = Path(ap.parse_args().data)
    h = {s: hashes(data / s) for s in ("train", "valid", "test")}
    for s, d in h.items():
        n = sum(len(v) for v in d.values())
        print(f"{s}: {n} files, {len(d)} unique, {n - len(d)} duplicates inside split")
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        shared = set(h[a]) & set(h[b])
        print(f"{a} & {b}: {len(shared)} identical images shared")
        for k in list(shared)[:3]:
            print("   e.g.", h[a][k][0].name, "<->", h[b][k][0].name)


if __name__ == "__main__":
    main()