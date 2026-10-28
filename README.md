# Quantum neural networks are inherently convolutional

Training and evaluation code for the image-classification experiments in *Quantum neural networks are inherently convolutional*.

The baseline uses two quantum convolutional layers and a quantum linear layer, with depths (8, 7, 8) and 1,584 trainable quantum parameters. There are no trainable classical parameters. Pretrained weights are not included.

## Files

| File | Purpose |
|---|---|
| `model.py` | Quantum layers, inference, initialization and model files |
| `data.py` | Dataset preprocessing and stratified splits |
| `train.py` | Training and validation-based model selection |
| `evaluate.py` | Test or validation evaluation |

## Installation

Python 3.10 is recommended. The CPU reference was tested on Linux/WSL2 with the versions in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

GPU execution requires a compatible JAX 0.6.2 CUDA installation. The original experiments used an NVIDIA RTX 6000 Ada Generation. Training and evaluation scripts disable GPU memory preallocation and default to a 0.45 memory fraction; existing environment settings take precedence.

## Training and evaluation

```bash
python train.py --dataset mnist --seed 42 --data-dir ./data --out-dir ./outputs/mnist42
python evaluate.py --model ./outputs/mnist42/best_model.npz --data-dir ./data --out-dir ./outputs/mnist42_test
```

For Fashion-MNIST, use `--dataset fashionmnist` and separate output directories. Repeat with seeds 43 and 44 for three-seed results. Each run uses 55,000 training images, 5,000 validation images, 20 epochs and batches of 64. Learning rates are 0.03 for MNIST and 0.05 for Fashion-MNIST, with Adam, cosine decay and global gradient clipping at 1.0.

The earliest checkpoint attaining the highest validation accuracy is saved as `best_model.npz`. Run test evaluation once on that selected checkpoint. Training never loads the official test dataset. Outputs include the split indices, epoch metrics and model-selection summary. Training drops the final incomplete batch (859 updates, 54,976 images per epoch); validation and test evaluation use all images. Training metrics are measured before each update, and validation metrics at the end of the epoch.

Evaluation saves class probabilities, labels and predictions in `predictions.npz`, and accuracy and negative log-likelihood in `summary.json`.

For the parameter-dependence experiments, set `--qconv2-layers` to one of `1, 2, 3, 5, 7, 10, 13, 15, 17, 20, 22`. The other two depths remain eight, and the parameter count is `240 + 192 * depth`. Report the mean and sample standard deviation (`ddof=1`) across seeds 42, 43 and 44. Use an empty output directory for each run.

A small installation check is available:

```bash
python train.py --dataset mnist --smoke --data-dir ./data --out-dir ./outputs/smoke
python evaluate.py --model ./outputs/smoke/best_model.npz --split validation --data-dir ./data --out-dir ./outputs/smoke_val
```

This uses 16 training and 16 validation images, one epoch and batches of eight. It takes about 20 seconds on the local CPU with data already downloaded and does not reproduce the paper's accuracy.

## Model details

Pixel intensities in [0,1] are standardized using `(mean, std) = (0.5, 0.5)` for MNIST and `(0.2860, 0.3530)` for Fashion-MNIST, before zero padding. Overlapping 4-by-4 patches are extracted with stride two. Both patch positions and entries within each patch use row-major ordering. The concatenated vector is globally normalized for amplitude encoding, separately for each input channel.

Both quantum convolutions act on four content qubits using `StronglyEntanglingLayers`. The first uses 12 qubits; the second uses 16 independently parameterized eight-qubit circuits, with corresponding output channels added across circuits. At each spatial position, both stages apply:

```text
p_tilde = p / (sum_channels(p) + 1e-6)
y = log(p_tilde + 1e-6) - log(1/16)
```

A 2-by-2 max-pooling follows each transformation. The resulting 64 features enter a six-qubit quantum linear layer. Its first ten probabilities are divided by their sum plus 1e-9, and the loss is `-mean(log(p_y + 1e-9))`. Evaluation preserves these values without a second normalization.

PennyLane's default template uses RZ-RY-RZ rotations and CNOT connections whose range cycles from one to the number of active qubits minus one. Probabilities are evaluated analytically using `default.qubit`, JAX backpropagation and float64 computation.

## Data and reproducibility

Datasets are downloaded through torchvision from the original [MNIST](https://yann.lecun.org/exdb/mnist/) and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) distributions. Full datasets are not redistributed.

The archived runs used torch 2.11.0 and torchvision 0.26.0 for data loading. Local verification used torch 2.6.0 and torchvision 0.21.0; PennyLane 0.42.3, JAX 0.6.2 and Optax 0.2.8 are common to both.

Local checks cover agreement with the experimental implementation and short end-to-end runs. The full 66-model training sweep and a fresh-environment installation have not been rerun for this release. Hardware and software differences can change training trajectories despite identical seeds.
