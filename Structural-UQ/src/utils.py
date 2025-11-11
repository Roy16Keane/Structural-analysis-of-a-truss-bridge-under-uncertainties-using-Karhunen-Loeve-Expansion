# utils.py
# Utility functions for plotting, error computation, and timing.
# Author: Roykeane Nandabi Syangu
# License: MIT

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
import time


@contextmanager
def timer(label: str):
    """
    Timing context manager for benchmarking.
    """
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[TIMER] {label}: {dt:.3f} s")


def set_plot_style():
    """
    Apply consistent plot style for reports.
    """
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "font.size": 11,
        "figure.dpi": 120,
        "axes.labelweight": "bold",
    })


def plot_covariance_matrix(C: np.ndarray, title: str = "Covariance Matrix"):
    """
    Plots a covariance matrix with colorbar.
    """
    plt.imshow(C, cmap="viridis")
    plt.title(title)
    plt.colorbar(label="Covariance")
    plt.tight_layout()
    plt.show()


def relative_error(x_ref: np.ndarray, x_est: np.ndarray) -> float:
    """
    Computes relative L2 norm error between reference and estimate.
    """
    return np.linalg.norm(x_est - x_ref) / np.linalg.norm(x_ref)


def covariance_error(C_ref: np.ndarray, C_est: np.ndarray) -> float:
    """
    Frobenius norm error between covariance matrices.
    """
    return np.linalg.norm(C_est - C_ref, ord="fro") / np.linalg.norm(C_ref, ord="fro")
