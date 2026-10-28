"""Train a QCNN and select the best validation checkpoint."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import numpy as np
import jax
import jax.numpy as jnp
import optax

from data import DATASETS, training_loaders
from evaluate import score
from model import DEPTHS, build_model, build_optimizer, initialize, nll, parameter_count, save_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--qconv2-layers", type=int, choices=DEPTHS, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="One epoch on 16 training and 16 validation images, batch 8; not a paper result")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("Seed must be non-negative")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output directory must be empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(dataset=args.dataset, seed=args.seed, qconv1_layers=8,
                    qconv2_layers=args.qconv2_layers, qlinear_layers=8,
                    learning_rate=DATASETS[args.dataset]["learning_rate"],
                    epochs=1 if args.smoke else 20, batch=8 if args.smoke else 64,
                    n_train=16 if args.smoke else 55000, n_val=16 if args.smoke else 5000,
                    smoke=args.smoke, quantum_parameters=parameter_count(args.qconv2_layers),
                    classical_trainable_parameters=0)
    (args.out_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    train_loader, val_loader, train_ids, val_ids = training_loaders(
        args.dataset, args.data_dir, args.seed, settings["batch"], settings["n_train"], settings["n_val"])
    np.savez_compressed(args.out_dir / "split_indices.npz", train=train_ids, validation=val_ids)
    forward = build_model(args.qconv2_layers)
    params = initialize(args.seed, args.qconv2_layers)
    optimizer = build_optimizer(len(train_loader), settings["epochs"], settings["learning_rate"])
    state = optimizer.init(params)

    @jax.jit
    def step(params, state, x, y):
        def objective(p):
            probabilities = forward(p, x)
            return nll(probabilities, y), probabilities
        (loss, probabilities), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, state = optimizer.update(gradients, state, params)
        return optax.apply_updates(params, updates), state, loss, probabilities

    best_accuracy = -1.0
    rows = []
    start = time.perf_counter()
    for epoch in range(1, settings["epochs"] + 1):
        total_loss, correct, seen = 0.0, 0, 0
        for images, target in train_loader:
            x = jnp.asarray(images.numpy(), dtype=jnp.float64)
            y = jnp.asarray(target.numpy(), dtype=jnp.int32)
            params, state, loss, probabilities = step(params, state, x, y)
            p = np.asarray(probabilities)
            if not np.isfinite(float(loss)) or not np.isfinite(p).all():
                raise FloatingPointError(f"Non-finite training output at epoch {epoch}")
            total_loss += float(loss) * len(target)
            correct += int(np.sum(p.argmax(1) == target.numpy()))
            seen += len(target)
        val_loss, val_accuracy, _, _ = score(forward, params, val_loader)
        row = dict(epoch=epoch, train_loss=total_loss / seen, train_accuracy=100 * correct / seen,
                   validation_loss=val_loss, validation_accuracy=val_accuracy, training_samples=seen)
        rows.append(row)
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch, best_params = val_accuracy, epoch, params
        print(json.dumps(row), flush=True)
    metadata = dict(settings, selected_epoch=best_epoch, validation_accuracy=best_accuracy)
    save_model(args.out_dir / "best_model.npz", best_params, metadata)
    with (args.out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(metadata, seconds=time.perf_counter() - start, test_evaluated=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Selected epoch {best_epoch}; validation accuracy {best_accuracy:.4f}%. Test separately with evaluate.py.", flush=True)


if __name__ == "__main__":
    main()
