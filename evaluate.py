"""Evaluate the saved model on Data/test.

Usage: python evaluate.py [--data Data]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (256, 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data")
    ap.add_argument("--model", default="models/wheat_cnn.keras")
    args = ap.parse_args()

    class_names = json.loads(Path("models/class_names.json").read_text())
    model = tf.keras.models.load_model(args.model)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        Path(args.data) / "test", image_size=IMG_SIZE, shuffle=False)

    # Test folders are named like "aphid_test"; the model uses "Aphid".
    # Match them by normalised name and remap the labels accordingly.
    def norm(name):
        n = name.lower().replace("_", " ").strip()
        return n[:-5].strip() if n.endswith(" test") else n

    model_idx = {norm(n): i for i, n in enumerate(class_names)}
    missing = [n for n in test_ds.class_names if norm(n) not in model_idx]
    if missing:
        raise SystemExit(f"Test folders with no matching class: {missing}")
    remap = np.array([model_idx[norm(n)] for n in test_ds.class_names])

    y_true = remap[np.concatenate([y.numpy() for _, y in test_ds])]
    y_pred = np.argmax(model.predict(test_ds), axis=1)

    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=60, ha="right")
    plt.savefig("models/confusion_matrix.png", dpi=120, bbox_inches="tight")
    print("Saved models/confusion_matrix.png")


if __name__ == "__main__":
    main()