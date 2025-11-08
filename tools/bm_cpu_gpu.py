#!/usr/bin/env python3
"""
Simple standalone benchmark that trains a binary logistic regression model
on both CPU (NumPy) and GPU (CuPy) using roughly 1 GB of synthetic data.

The script generates a random dataset, runs a fixed number of gradient-descent
epochs on the CPU, then mirrors the run on the GPU.  Timing for each phase is
printed so you can compare throughput.

Usage (defaults shown):
    python tools/benchmarks/logreg_cpu_gpu.py \
        --samples 1200000 \
        --features 128 \
        --epochs 5 \
        --lr 0.1

The default settings allocate ~1.2M × 128 float32 matrix (~614 MB) plus labels
(~5 MB).  Combined with intermediate buffers the working set is just under 1 GB.

Requirements:
    - NumPy
    - CuPy (matching your CUDA toolkit) for the GPU run

Tip: run once to warm up the GPU so the timing reflects steady-state kernels.
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore
except ImportError:  # pragma: no cover - GPU dependency optional
    cp = None  # sentinel so we can raise a helpful error later


def make_dataset(
    samples: int,
    features: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic binary classification data set.

    Features are standard-normal float32 values.  Labels are determined by a
    random, fixed weight vector with additive noise so the task is non-trivial.
    """
    rng = np.random.default_rng(seed)
    weights = rng.standard_normal(features).astype(np.float32)
    X = rng.standard_normal((samples, features), dtype=np.float32)

    logits = X @ weights + rng.normal(0.0, 0.5, size=samples).astype(np.float32)
    y = (logits > 0).astype(np.float32)  # shape (samples,)

    # reshape labels for easier matrix math later
    y = y.reshape(samples, 1)
    return X, y


def sigmoid(x):
    """Numerically-stable sigmoid that works with NumPy or CuPy arrays."""
    return 1.0 / (1.0 + np.exp(-x)) if isinstance(x, np.ndarray) else 1.0 / (1.0 + cp.exp(-x))


def train_logreg(
    X,
    y,
    epochs: int,
    lr: float,
    xp,
) -> Tuple[np.ndarray, float]:
    """
    Run batch gradient descent logistic regression in the provided array module.

    Parameters
    ----------
    X, y : xp.ndarray
        Feature matrix (n × d) and binary labels (n × 1).
    epochs : int
        Number of gradient descent passes.
    lr : float
        Learning rate.
    xp : module
        Either numpy or cupy.
    """
    n_samples, n_features = X.shape
    w = xp.zeros((n_features, 1), dtype=X.dtype)

    t0 = time.perf_counter()
    for epoch in range(epochs):
        z = X @ w  # (n × 1)
        preds = 1.0 / (1.0 + xp.exp(-z))
        grad = (X.T @ (preds - y)) / n_samples
        w -= lr * grad

        if epoch == epochs - 1:
            # compute loss for reference
            eps = xp.asarray(1e-7, dtype=X.dtype)
            loss = -(y * xp.log(preds + eps) + (1 - y) * xp.log(1 - preds + eps)).mean()
        else:
            loss = xp.nan
    elapsed = time.perf_counter() - t0
    return w, float(loss), elapsed


def evaluate_accuracy(X, y, w, xp) -> float:
    preds = 1.0 / (1.0 + xp.exp(-(X @ w)))
    pred_classes = (preds > 0.5).astype(xp.float32)
    accuracy = (pred_classes == y).mean()
    return float(accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU vs GPU logistic regression benchmark.")
    parser.add_argument("--samples", type=int, default=1_200_000, help="Number of samples (~1e6 for ~0.6GB).")
    parser.add_argument("--features", type=int, default=128, help="Number of float features per sample.")
    parser.add_argument("--epochs", type=int, default=5, help="Gradient descent epochs.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip the GPU run (useful for CPU-only validation).",
    )
    args = parser.parse_args()

    print(f"Generating data: samples={args.samples:,}, features={args.features}, dtype=float32 …")
    X_cpu, y_cpu = make_dataset(args.samples, args.features, seed=args.seed)
    dataset_size_bytes = X_cpu.nbytes + y_cpu.nbytes
    print(f"Synthetic dataset size ≈ {dataset_size_bytes / (1024**3):.2f} GiB")

    # CPU baseline
    print("\n=== CPU (NumPy) training ===")
    _, cpu_loss, cpu_time = train_logreg(X_cpu, y_cpu, args.epochs, args.lr, np)
    cpu_acc = evaluate_accuracy(X_cpu, y_cpu, _, np)
    print(f"CPU time: {cpu_time:.2f}s over {args.epochs} epochs — loss={cpu_loss:.4f}, accuracy={cpu_acc:.4f}")

    if args.skip_gpu:
        print("\nGPU run skipped (per --skip-gpu).")
        return

    if cp is None:
        raise RuntimeError("CuPy is not installed. Install cupy-cudaXY matching your CUDA toolkit to enable GPU run.")

    print("\nTransferring data to GPU …")
    X_gpu = cp.asarray(X_cpu, dtype=cp.float32)
    y_gpu = cp.asarray(y_cpu, dtype=cp.float32)

    cp.cuda.Device().synchronize()
    print("=== GPU (CuPy) training ===")
    _, gpu_loss, gpu_time = train_logreg(X_gpu, y_gpu, args.epochs, args.lr, cp)
    cp.cuda.Device().synchronize()
    gpu_acc = evaluate_accuracy(X_gpu, y_gpu, _, cp)
    print(f"GPU time: {gpu_time:.2f}s over {args.epochs} epochs — loss={gpu_loss:.4f}, accuracy={gpu_acc:.4f}")

    speedup = cpu_time / gpu_time if gpu_time > 0 else float("nan")
    print(f"\nSpeedup (CPU/GPU): {speedup:.2f}×")


if __name__ == "__main__":
    main()

