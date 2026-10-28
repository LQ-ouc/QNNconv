"""Measurement-mediated quantum convolutional neural network."""
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml

DEPTHS = (1, 2, 3, 5, 7, 10, 13, 15, 17, 20, 22)


def parameter_count(depth=7):
    return 240 + 192 * depth


def initialize(seed, depth=7):
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}")
    # Fixed PRNG partition for reproducible initialization.
    keys = jax.random.split(jax.random.PRNGKey(seed), 12)
    shapes = ((1, 8, 4, 3), (16, depth, 4, 3), (8, 6, 3))
    return {name: 0.01 * jax.random.normal(key, shape, dtype=jnp.float64)
            for name, key, shape in zip(("q1", "q2", "ql"), keys[:3], shapes)}


def extract_patches(x, padding):
    """Row-major patch positions, then row-major entries within each patch."""
    b, c, h, w = x.shape
    x = jnp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    rows = (h + 2 * padding - 4) // 2 + 1
    cols = (w + 2 * padding - 4) // 2 + 1
    ri = jnp.arange(rows)[:, None] * 2 + jnp.arange(4)
    ci = jnp.arange(cols)[:, None] * 2 + jnp.arange(4)
    patches = x[:, :, ri[:, :, None, None], ci[None, None, :, :]]
    return patches.transpose(0, 2, 4, 1, 3, 5).reshape(b, rows * cols, c * 16)


def max_pool(x):
    return jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, (1, 1, 2, 2), (1, 1, 2, 2), "VALID")


def norm_log(x):
    """Channel normalization followed by the log ratio to a uniform reference."""
    b, c, h, w = x.shape
    grouped = x.reshape(b, 1, 16, h, w)
    mass = grouped.sum(axis=2, keepdims=True)
    normalized = grouped / (mass + 1e-6)
    y = jnp.log(normalized + 1e-6) - np.log(1.0 / 16)
    return jnp.where(jnp.isfinite(y), y, 0.0).reshape(b, c, h, w)


def _quantum_convolution(n_qubits, layers, padding):
    wires = list(range(n_qubits))
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface="jax", diff_method="backprop")
    def circuit(values, weights):
        qml.AmplitudeEmbedding(values, wires=wires, normalize=True, pad_with=0.0)
        qml.StronglyEntanglingLayers(weights, wires=wires[-4:])
        return qml.probs(wires=wires)

    batched = jax.jit(jax.vmap(circuit, in_axes=(0, 0)))

    def apply(weights, x):
        b, c, _, _ = x.shape
        patches = extract_patches(x, padding)
        positions = patches.shape[1]
        values = patches.reshape(b, positions, c, 16).transpose(0, 2, 1, 3)
        values = values.reshape(b * c, positions * 16)
        expanded = jnp.broadcast_to(weights[None], (b, c, layers, 4, 3))
        probs = batched(values, expanded.reshape(b * c, layers, 4, 3))
        probs = probs[:, :positions * 16].reshape(b, c, positions, 16)
        responses = probs.transpose(0, 1, 3, 2).sum(axis=1)
        side = int(positions ** .5)
        return responses.reshape(b, 16, side, side)

    return apply


def _quantum_linear():
    wires = list(range(6))
    device = qml.device("default.qubit", wires=6)

    @qml.qnode(device, interface="jax", diff_method="backprop")
    def circuit(features, weights):
        features = features / (jnp.linalg.norm(features) + 1e-9)
        qml.AmplitudeEmbedding(features, wires=wires, normalize=False, pad_with=0.0)
        qml.StronglyEntanglingLayers(weights, wires=wires)
        return qml.probs(wires=wires)

    batched = jax.jit(jax.vmap(circuit, in_axes=(0, None)))

    def apply(weights, features):
        p = batched(features, weights)[:, :10]
        return p / (p.sum(axis=-1, keepdims=True) + 1e-9)

    return apply


def build_model(depth=7):
    """Build the classification function."""
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}")
    q1 = _quantum_convolution(12, 8, 3)
    q2 = _quantum_convolution(8, depth, 1)
    ql = _quantum_linear()

    def forward(params, x):
        raw1 = q1(params["q1"], x)
        log1 = norm_log(raw1)
        pool1 = max_pool(log1)
        raw2 = q2(params["q2"], pool1)
        log2 = norm_log(raw2)
        pool2 = max_pool(log2)
        flat = pool2.reshape(x.shape[0], 64)
        return ql(params["ql"], flat)

    return jax.jit(forward)


def nll(probabilities, labels):
    return -jnp.log(probabilities[jnp.arange(labels.shape[0]), labels] + 1e-9).mean()


def build_optimizer(steps_per_epoch, epochs, learning_rate):
    schedule = optax.cosine_decay_schedule(
        learning_rate, decay_steps=max(1, steps_per_epoch * epochs))
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))


def save_model(path, params, metadata):
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in params.items()},
                        metadata=np.array(json.dumps(metadata)))


def load_model(path):
    with np.load(Path(path), allow_pickle=False) as saved:
        if set(saved.files) != {"q1", "q2", "ql", "metadata"}:
            raise ValueError("Unexpected model arrays")
        metadata = json.loads(str(saved["metadata"]))
        required = {"dataset", "seed", "qconv2_layers", "batch", "n_train", "n_val",
                    "smoke", "selected_epoch"}
        if not isinstance(metadata, dict) or not required.issubset(metadata):
            raise ValueError("Incomplete model metadata")
        depth = metadata["qconv2_layers"]
        if type(depth) is not int or depth not in DEPTHS:
            raise ValueError("Invalid convolution depth in model metadata")
        if metadata["dataset"] not in ("mnist", "fashionmnist"):
            raise ValueError("Invalid dataset in model metadata")
        for key in ("batch", "n_train", "n_val", "selected_epoch"):
            if type(metadata[key]) is not int or metadata[key] <= 0:
                raise ValueError(f"Invalid model metadata: {key}")
        if type(metadata["seed"]) is not int or metadata["seed"] < 0:
            raise ValueError("Invalid model seed")
        if type(metadata["smoke"]) is not bool or metadata["n_train"] + metadata["n_val"] > 60000:
            raise ValueError("Invalid training configuration in model metadata")
        expected = dict(q1=(1, 8, 4, 3), q2=(16, depth, 4, 3), ql=(8, 6, 3))
        params = {}
        for key, shape in expected.items():
            value = saved[key]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid parameter array: {key}")
            params[key] = jnp.asarray(value, dtype=jnp.float64)
    return params, metadata
