"""MNIST and Fashion-MNIST preprocessing and stratified splits."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATASETS = {
    "mnist": dict(mean=.5, std=.5, learning_rate=.03),
    "fashionmnist": dict(mean=.2860, std=.3530, learning_rate=.05),
}


def image_dataset(name, data_dir, training=True):
    settings = DATASETS[name]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([settings["mean"]], [settings["std"]]),
    ])
    cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
    return cls(str(data_dir), train=training, download=True, transform=transform)


def _proportional_counts(total, capacities):
    capacities = np.asarray(capacities, dtype=np.int64)
    if total < 0 or total > int(capacities.sum()):
        raise ValueError("Invalid split size")
    if total == 0:
        return np.zeros_like(capacities)
    raw = total * capacities.astype(np.float64) / float(capacities.sum())
    counts = np.minimum(np.floor(raw).astype(np.int64), capacities)
    remainder = total - int(counts.sum())
    order = np.argsort(-(raw - counts))
    while remainder:
        for class_id in order:
            if counts[class_id] < capacities[class_id]:
                counts[class_id] += 1
                remainder -= 1
                if remainder == 0:
                    break
    return counts


def split_indices(labels, seed, n_train=55000, n_val=5000):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= 10)):
        raise ValueError("Expected one-dimensional class labels in [0, 9]")
    if n_train <= 0 or n_val <= 0 or n_train + n_val > len(labels):
        raise ValueError("Invalid training/validation sizes")
    selected = _proportional_counts(n_train + n_val, np.bincount(labels, minlength=10))
    train_counts = _proportional_counts(n_train, selected)
    val_counts = selected - train_counts
    train, val = [], []
    for class_id in range(10):
        indices = np.flatnonzero(labels == class_id)
        np.random.default_rng(seed + class_id).shuffle(indices)
        nt, nv = train_counts[class_id], val_counts[class_id]
        train.extend(indices[:nt].tolist())
        val.extend(indices[nt:nt + nv].tolist())
    random.Random(seed).shuffle(train)
    random.Random(seed + 1000).shuffle(val)
    return train, val


def training_loaders(name, data_dir, seed, batch=64, n_train=55000, n_val=5000):
    dataset = image_dataset(name, data_dir, training=True)
    train, val = split_indices(dataset.targets, seed, n_train, n_val)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(Subset(dataset, train), batch_size=batch, shuffle=True,
                              drop_last=True, num_workers=0, generator=generator)
    val_loader = DataLoader(Subset(dataset, val), batch_size=batch, shuffle=False,
                            drop_last=False, num_workers=0)
    return train_loader, val_loader, train, val
