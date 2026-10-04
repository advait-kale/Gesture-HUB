"""Train the IMU gesture 1D-CNN and export it for the ESP32.

Pipeline:
  1. Load CSVs -> fixed-length [N, T, 6] windows.
  2. Stratified k-fold CV for an honest accuracy estimate (small dataset).
  3. Train a final model on a train/val split with heavy augmentation.
  4. Evaluate on a held-out test split; print report + confusion matrix.
  5. Export int8 TFLite + C sources under out/.

Usage:
  .venv/Scripts/python.exe train.py
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.gesturehub import augment, data, export
from src.gesturehub.model import build_cnn

SEED = 42


def set_seeds(seed: int = SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=25, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5
        ),
    ]


def class_weights(y: np.ndarray) -> dict[int, float]:
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}


def apply_class_aliases(X, y, class_names, alias_specs):
    """Optionally merge/rename folder labels, e.g. swirl=twirl."""
    aliases = {}
    for spec in alias_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --class-alias {spec!r}; use SOURCE=TARGET")
        source, target = (part.strip() for part in spec.split("=", 1))
        if not source or not target:
            raise ValueError(f"Invalid --class-alias {spec!r}; source and target must be non-empty")
        aliases[source] = target

    mapped_names = [aliases.get(name, name) for name in class_names]
    final_names = sorted(set(mapped_names))
    final_index = {name: i for i, name in enumerate(final_names)}
    old_to_new = {i: final_index[name] for i, name in enumerate(mapped_names)}
    y = np.asarray([old_to_new[int(label)] for label in y], dtype=np.int64)
    if aliases:
        applied = {source: target for source, target in aliases.items() if source in class_names}
        if applied:
            print(f"Class aliases: {applied}")
    return X, y, final_names


def describe_dataset(y: np.ndarray, class_names: list[str]) -> np.ndarray:
    """Print discovered labels and return their sample counts."""
    counts = np.bincount(y, minlength=len(class_names))
    print(f"Loaded {len(y)} samples, {len(class_names)} classes: {class_names}")
    print("Samples per class: " + ", ".join(
        f"{name}={int(count)}" for name, count in zip(class_names, counts)
    ))
    if len(class_names) < 2:
        raise ValueError("Training requires at least two class folders with CSV examples.")
    if np.any(counts < 2):
        sparse = [f"{name}={int(count)}" for name, count in zip(class_names, counts) if count < 2]
        raise ValueError("Each class needs at least two CSV examples for a stratified split: " + ", ".join(sparse))
    return counts


def train_one(X_tr, y_tr, X_val, y_val, n_classes, epochs, n_aug):
    """Train a fresh model on one split. Standardization + augmentation are fit
    on the train portion only, then applied to val."""
    mean, std = data.channel_stats(X_tr)

    Xa, ya = augment.expand(X_tr, y_tr, n_aug=n_aug, seed=SEED)
    Xa = data.standardize(Xa, mean, std)
    Xv = data.standardize(X_val, mean, std)

    model = build_cnn(X_tr.shape[1], X_tr.shape[2], n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        Xa, ya,
        validation_data=(Xv, y_val),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weights(ya),
        callbacks=make_callbacks(),
        verbose=0,
    )
    return model, mean, std


def cross_validate(X, y, n_classes, folds, epochs, n_aug):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    accs = []
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        Xtr2, Xval, ytr2, yval = train_test_split(
            Xtr, ytr, test_size=0.2, stratify=ytr, random_state=SEED
        )
        model, mean, std = train_one(Xtr2, ytr2, Xval, yval, n_classes, epochs, n_aug)
        Xte_s = data.standardize(Xte, mean, std)
        acc = model.evaluate(Xte_s, yte, verbose=0)[1]
        accs.append(acc)
        print(f"  fold {k}/{folds}: acc={acc:.3f}")
        tf.keras.backend.clear_session()
    accs = np.array(accs)
    print(f"CV accuracy: {accs.mean():.3f} +/- {accs.std():.3f}")
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/dataset")
    ap.add_argument("--out", default="out")
    ap.add_argument("--target-len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-aug", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument(
        "--class-alias",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="Optionally rename/merge a folder label; repeat as needed (example: swirl=twirl)",
    )
    args = ap.parse_args()

    set_seeds()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, class_names = data.load_dataset(args.dataset, target_len=args.target_len)
    if len(X) == 0:
        raise ValueError(f"No usable CSV examples found under {args.dataset!r}.")
    X, y, class_names = apply_class_aliases(X, y, class_names, args.class_alias)
    n_classes = len(class_names)
    counts = describe_dataset(y, class_names)
    print(f"Window: {X.shape[1]} steps x {X.shape[2]} channels")

    if not args.skip_cv:
        folds = min(args.folds, int(counts.min()))
        if folds < 2:
            raise ValueError("Cross-validation needs at least two samples in every class.")
        if folds != args.folds:
            print(f"Reducing CV folds from {args.folds} to {folds} to fit the smallest class.")
        print("\n=== Stratified k-fold CV ===")
        cross_validate(X, y, n_classes, folds, args.epochs, args.n_aug)

    print("\n=== Final model (train/val/test split) ===")
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=SEED
    )
    model, mean, std = train_one(X_tr, y_tr, X_val, y_val, n_classes, args.epochs, args.n_aug)

    X_test_s = data.standardize(X_test, mean, std)
    y_pred = model.predict(X_test_s, verbose=0).argmax(axis=1)
    print("\nTest set report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    model.save(out_dir / "gesture_model.keras")

    print("\n=== Export int8 TFLite + C sources ===")
    rep = data.standardize(X_tr, mean, std)
    tflite_bytes = export.to_int8_tflite(model, rep)
    (out_dir / "gesture_model.tflite").write_bytes(tflite_bytes)
    export.write_c_model(tflite_bytes, out_dir)
    in_scale, in_zp = export.input_quant_params(tflite_bytes)
    export.write_params_header(
        out_dir, X.shape[1], X.shape[2], class_names, mean, std, in_scale, in_zp
    )
    print(f"tflite size: {len(tflite_bytes)} bytes")
    print(f"wrote: {sorted(p.name for p in out_dir.iterdir())}")


if __name__ == "__main__":
    main()
