"""Train the simple CNN on the wheat disease dataset.

Usage: python train.py [--data Data] [--epochs 50]
Expects Data/train and Data/valid (one sub-folder per class).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

IMG_SIZE = (256, 256)


def build_model(num_classes: int, augment: bool = False,
                dropout: float = 0.0) -> tf.keras.Model:
    # By default this is the same simple CNN as the original notebook.
    # Augmentation / dropout are optional (--augment, --dropout 0.3): in a
    # first run they made this tiny network underfit, so they are off by default.
    aug = [layers.RandomFlip("horizontal"), layers.RandomRotation(0.05)] if augment else []
    drop = [layers.Dropout(dropout)] if dropout > 0 else []
    return Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        *aug,
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        *drop,
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--augment", action="store_true", help="light flip/rotate augmentation")
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()
    data = Path(args.data)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data / "train", seed=123, image_size=IMG_SIZE, batch_size=args.batch_size)
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        data / "valid", seed=123, image_size=IMG_SIZE, batch_size=args.batch_size)
    class_names = train_ds.class_names

    Path("models").mkdir(exist_ok=True)
    Path("models/class_names.json").write_text(json.dumps(class_names, indent=2))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(len(class_names), args.augment, args.dropout)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("models/wheat_cnn.keras",
                                           monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                         restore_best_weights=True),
    ]
    history = model.fit(train_ds, validation_data=valid_ds,
                        epochs=args.epochs, callbacks=callbacks)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, k in zip(ax, ["accuracy", "loss"]):
        a.plot(history.history[k], label="train")
        a.plot(history.history["val_" + k], label="validation")
        a.set_title(k)
        a.set_xlabel("epoch")
        a.legend()
    fig.savefig("models/training_curves.png", dpi=120, bbox_inches="tight")
    print("Saved best model to models/wheat_cnn.keras")


if __name__ == "__main__":
    main()