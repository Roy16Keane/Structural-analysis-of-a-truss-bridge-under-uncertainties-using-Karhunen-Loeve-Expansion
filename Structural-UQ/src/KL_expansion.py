# kl_expansion.py
# Implements Karhunen–Loève (KL) expansion for stochastic response fields.
# Author: Roykeane Nandabi Syangu
# License: MIT

from __future__ import annotations
import numpy as np
from typing import Tuple


def kl_decompose(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs eigen-decomposition of a covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix (n x n), symmetric and positive semi-definite.

    Returns
    -------
    eigenvalues : np.ndarray
        Sorted eigenvalues (descending).
    eigenvectors : np.ndarray
        Corresponding eigenvectors (columns), orthonormal.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]  # descending
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    return eigvals, eigvecs


def kl_truncate(eigvals: np.ndarray,
                eigvecs: np.ndarray,
                energy_ratio: float = 0.95) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Truncates eigenmodes based on target cumulative energy ratio.

    Parameters
    ----------
    eigvals : np.ndarray
        Eigenvalues sorted descending.
    eigvecs : np.ndarray
        Corresponding eigenvectors.
    energy_ratio : float, default=0.95
        Fraction of total variance to retain.

    Returns
    -------
    eigvals_r, eigvecs_r, r : tuple
        Truncated eigenvalues, eigenvectors, and retained mode count.
    """
    total = np.sum(eigvals)
    cum = np.cumsum(eigvals)
    r = np.searchsorted(cum / total, energy_ratio) + 1
    return eigvals[:r], eigvecs[:, :r], r


def kl_sample(eigvals: np.ndarray,
              eigvecs: np.ndarray,
              n_samples: int = 1) -> np.ndarray:
    """
    Generates random field realizations from truncated KL expansion.

    u ≈ Φ Λ^{1/2} ξ,  where ξ ~ N(0, I)

    Parameters
    ----------
    eigvals : np.ndarray
        Retained eigenvalues.
    eigvecs : np.ndarray
        Retained eigenvectors.
    n_samples : int
        Number of samples to generate.

    Returns
    -------
    np.ndarray
        Random field samples (n_dof x n_samples).
    """
    r = len(eigvals)
    xi = np.random.randn(r, n_samples)
    u_samples = eigvecs @ np.diag(np.sqrt(eigvals)) @ xi
    return u_samples


def reconstruct_from_kl(eigvals: np.ndarray,
                        eigvecs: np.ndarray,
                        coeffs: np.ndarray) -> np.ndarray:
    """
    Reconstructs the field from KL coefficients.

    Parameters
    ----------
    eigvals : np.ndarray
        Retained eigenvalues.
    eigvecs : np.ndarray
        Retained eigenvectors.
    coeffs : np.ndarray
        Coefficient matrix (r x n_samples).

    Returns
    -------
    np.ndarray
        Reconstructed field (n_dof x n_samples).
    """
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ coeffs
