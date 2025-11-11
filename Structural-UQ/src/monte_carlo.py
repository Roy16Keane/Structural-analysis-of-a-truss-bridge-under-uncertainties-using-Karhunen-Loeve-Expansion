# monte_carlo.py
# Monte Carlo sampling for uncertainty propagation in truss FEM.
# Author: Roykeane Nandabi Syangu
# License: MIT

from __future__ import annotations
import numpy as np
from tqdm import tqdm
from typing import Callable, Dict


def monte_carlo_solver(
    n_samples: int,
    f_generator: Callable[[], np.ndarray],
    fem_solve: Callable[[np.ndarray], np.ndarray],
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """
    Monte Carlo propagation of load uncertainty through the FEM model.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo realizations.
    f_generator : callable
        Function returning a random force vector f.
    fem_solve : callable
        Function accepting f and returning displacement vector u.
    show_progress : bool
        Show progress bar (default True).

    Returns
    -------
    dict
        {'mean': μ_u, 'cov': C_u, 'samples': all_u}
    """
    u_list = []
    iterator = range(n_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Monte Carlo Sampling", unit="sample")

    for _ in iterator:
        f = f_generator()
        u = fem_solve(f)
        u_list.append(u)

    U = np.stack(u_list, axis=1)
    mean_u = np.mean(U, axis=1)
    cov_u = np.cov(U)

    return {"mean": mean_u, "cov": cov_u, "samples": U}
