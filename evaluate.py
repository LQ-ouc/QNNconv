"""Evaluate a selected model on the official test set (or its validation split)."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import numpy as np
import jax.numpy as jnp
from torch.utils.data import DataLoader, Subset

from data import image_dataset, split_indices
from model import build_model, load_model


def score(forward, params, loader):
    outputs, labels = [], []
    for images, target in loader:
        probabilities = np.asarray(forward(params, jnp.asarray(images.numpy(), dtype=jnp.float64)))
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities.sum(axis=1) <= 0):
            raise FloatingPointError("Invalid model probabilities")
        outputs.append(probabilities)
        labels.append(target.numpy())
    if not outputs:
        raise ValueError("Evaluation dataset is empty")
    probabilities = np.concatenate(outputs)
    labels = np.concatenate(labels)
    loss = float(-np.log(probabilities[np.arange(len(labels)), labels] + 1e-9).mean())
    accuracy = float(100 * np.mean(probabilities.argmax(1) == labels))
    return loss, accuracy, probabilities, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output directory must be empty")
    params, metadata = load_model(args.model)
    if metadata["smoke"] and args.split == "test":
        parser.error("A smoke model is for validation only; train the full configuration before official testing")
    dataset = image_dataset(metadata["dataset"], args.data_dir, training=args.split == "validation")
    if args.split == "validation":
        _, ids = split_indices(dataset.targets, metadata["seed"], metadata["n_train"], metadata["n_val"])
    else:
        ids = list(range(len(dataset)))
    loader = DataLoader(Subset(dataset, ids), batch_size=metadata["batch"], shuffle=False, num_workers=0)
    forward = build_model(metadata["qconv2_layers"])
    loss, accuracy, probabilities, labels = score(forward, params, loader)
    predicted = probabilities.argmax(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "predictions.npz", probabilities=probabilities,
                        labels=labels, predictions=predicted, sample_ids=ids)
    summary = dict(dataset=metadata["dataset"], seed=metadata["seed"], split=args.split,
                   n_samples=len(labels), accuracy_percent=accuracy, nll=loss,
                   qconv2_layers=metadata["qconv2_layers"], selected_epoch=metadata["selected_epoch"])
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
