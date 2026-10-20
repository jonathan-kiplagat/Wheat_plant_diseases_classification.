"""Build a leak-free split from the Kaggle data.

Pools train/valid/test, removes exact duplicate files (by content hash),
then re-splits each class 70/15/15 into Data_clean/{train,valid,test}.
Files are hard-linked when possible (no extra disk space), else copied.

Usage: python make_clean_split.py [--data Data] [--out Data_clean]
Then:  python train.py --data Data_clean
       python evaluate.py --data Data_clean
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def norm(name):
    n = name.lower().replace("_", " ").strip()
    return re.sub(r"(\s+(train|valid|val|test))+$", "", n).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data")
    ap.add_argument("--out", default="Data_clean")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)

    class_names = json.loads(Path("models/class_names.json").read_text())
    by_norm = {norm(c): c for c in class_names}

    bad = [str(p) for sp in ("train", "valid", "test")
           for p in (data / sp).iterdir()
           if p.is_dir() and norm(p.name) not in by_norm]
    if bad:
        raise SystemExit("Unrecognised class folders:\n  " + "\n  ".join(bad))

    seen, per_class, dropped = set(), {c: [] for c in class_names}, 0
    for split in ("train", "valid", "test"):
        for cdir in sorted(p for p in (data / split).iterdir() if p.is_dir()):
            cls = by_norm.get(norm(cdir.name))
            if cls is None:
                raise SystemExit(f"Unrecognised class folder: {cdir}")
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() not in EXT:
                    continue
                h = hashlib.md5(f.read_bytes()).hexdigest()
                if h in seen:
                    dropped += 1
                    continue
                seen.add(h)
                per_class[cls].append(f)
    print(f"Kept {len(seen)} unique images, dropped {dropped} duplicates")

    rng = random.Random(args.seed)
    if out.exists():
        shutil.rmtree(out)
    counts = {"train": 0, "valid": 0, "test": 0}
    for cls, files in per_class.items():
        rng.shuffle(files)
        n = len(files)
        cut = {"train": files[:int(n * .7)],
               "valid": files[int(n * .7):int(n * .85)],
               "test": files[int(n * .85):]}
        for split, fs in cut.items():
            d = out / split / cls
            d.mkdir(parents=True)
            for i, f in enumerate(fs):
                dst = d / f"{i:05d}{f.suffix.lower()}"
                try:
                    os.link(f, dst)
                except OSError:
                    shutil.copy2(f, dst)
            counts[split] += len(fs)
    print("Split sizes:", counts)


if __name__ == "__main__":
    main()