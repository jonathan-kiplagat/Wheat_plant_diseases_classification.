"""Identify the wheat disease in a photo.

Usage: python predict.py path/to/photo.jpg [--top 3]
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

IMG_SIZE = (256, 256)
HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Wheat disease classifier")
    ap.add_argument("image", help="path to a wheat photo (jpg/png)")
    ap.add_argument("--model", default=str(HERE / "models" / "wheat_cnn.keras"))
    ap.add_argument("--top", type=int, default=3, help="number of predictions to show")
    args = ap.parse_args()

    if not Path(args.image).is_file():
        sys.exit(f"Image not found: {args.image}")
    if not Path(args.model).is_file():
        sys.exit(f"Model not found: {args.model}. Run train.py first.")

    class_names = json.loads((HERE / "models" / "class_names.json").read_text())
    model = tf.keras.models.load_model(args.model)

    img = tf.keras.utils.load_img(args.image, target_size=IMG_SIZE)
    x = np.expand_dims(tf.keras.utils.img_to_array(img), 0)  # rescaling is inside the model
    probs = model.predict(x, verbose=0)[0]

    print(f"\nPrediction: {class_names[int(np.argmax(probs))]} "
          f"({probs.max():.1%} confidence)\n")
    for i in np.argsort(probs)[::-1][:args.top]:
        print(f"  {class_names[i]:<22} {probs[i]:.1%}")


if __name__ == "__main__":
    main()
