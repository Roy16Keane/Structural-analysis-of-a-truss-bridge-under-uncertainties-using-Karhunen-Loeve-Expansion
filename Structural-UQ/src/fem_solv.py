# fem_solver.py
# Structural FEM utilities for a 2D truss with 2 DOF per node (Ux, Uy)
# - Global stiffness assembly
# - Boundary condition application
# - Linear solve Ku = f
# - Simple force generators for nodes (zero-based indexing)
#
# Author: Roykeane  Syangu (MSc Project)
# License: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import numpy as np


# ----------------------------- Data Structures ----------------------------- #

@dataclass(frozen=True)
class TrussElement:
    """
    A 2D truss element defined by material, geometry, orientation, and connectivity.

    Attributes
    ----------
    E : float
        Young's modulus (consistent units with A and L).
    A : float
        Cross-sectional area.
    L : float
        Element length.
    angle_deg : float
        Element axis angle in degrees (0° is +x; CCW positive).
    nodes : Tuple[int, int]
        (start_node, end_node), zero-based node indices.
        Each node has 2 DOFs: Ux = 2*node, Uy = 2*node + 1.
    """
    E: float
    A: float
    L: float
    angle_deg: float
    nodes: Tuple[int, int]


# ----------------------------- Core FEM Routines --------------------------- #

def _local_truss_core(angle_deg: float) -> np.ndarray:
    """
    Returns the 4x4 direction-cosine core for a 2D truss element in local/global form.

    The element stiffness contribution is:
        k_glob = (E * A / L) * A_sel @ k_core @ A_sel.T
    where k_core depends only on direction cosines.

    Parameters
    ----------
    angle_deg : float
        Element orientation in degrees.

    Returns
    -------
    np.ndarray
        (4, 4) matrix.
    """
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)

    # Core 4x4 matrix (no EA/L scaling)
    k = np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s],
    ], dtype=float)
    return k


def _selection_matrix(nodes: Tuple[int, int], n_dofs: int) -> np.ndarray:
    """
    Builds a (n_dofs x 4) selection/mapping matrix that places a 4x4 local
    element matrix into the global K.

    DOF ordering per node: [Ux, Uy] => indices [2*n, 2*n + 1]

    Parameters
    ----------
    nodes : Tuple[int, int]
        (n_i, n_j) node indices, zero-based.
    n_dofs : int
        Total number of global DOFs.

    Returns
    -------
    np.ndarray
        (n_dofs, 4) selection matrix.
    """
    n_i, n_j = nodes
    dof_i = (2 * n_i, 2 * n_i + 1)
    dof_j = (2 * n_j, 2 * n_j + 1)

    A = np.zeros((n_dofs, 4), dtype=float)
    A[dof_i[0], 0] = 1.0  # Ux_i
    A[dof_i[1], 1] = 1.0  # Uy_i
    A[dof_j[0], 2] = 1.0  # Ux_j
    A[dof_j[1], 3] = 1.0  # Uy_j
    return A


def assemble_global_stiffness(elements: Sequence[TrussElement], n_dofs: int) -> np.ndarray:
    """
    Assembles the global stiffness matrix K for a 2D truss.

    Parameters
    ----------
    elements : Sequence[TrussElement]
        List of truss elements.
    n_dofs : int
        Total number of DOFs in the model (2 * n_nodes).

    Returns
    -------
    np.ndarray
        (n_dofs, n_dofs) global stiffness matrix.
    """
    K = np.zeros((n_dofs, n_dofs), dtype=float)
    for e in elements:
        k_core = _local_truss_core(e.angle_deg)                 # (4,4)
        A_sel = _selection_matrix(e.nodes, n_dofs)              # (n_dofs,4)
        scale = (e.E * e.A) / e.L
        K += scale * (A_sel @ k_core @ A_sel.T)                 # (n_dofs,n_dofs)
    return K


def apply_boundary_conditions(K: np.ndarray,
                              f: np.ndarray,
                              fixed_dofs: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies essential (Dirichlet) boundary conditions by row/column zeroing and diagonal = 1.

    Parameters
    ----------
    K : np.ndarray
        Global stiffness matrix (n_dofs, n_dofs).
    f : np.ndarray
        Global force vector (n_dofs,).
    fixed_dofs : Iterable[int]
        DOF indices to fix (e.g., supports).

    Returns
    -------
    K_bc, f_bc : Tuple[np.ndarray, np.ndarray]
        Modified K and f with boundary conditions applied.

    Notes
    -----
    - This function does NOT mutate the inputs. It returns new arrays.
    """
    K_bc = K.copy()
    f_bc = f.copy()
    for dof in fixed_dofs:
        K_bc[dof, :] = 0.0
        K_bc[:, dof] = 0.0
        K_bc[dof, dof] = 1.0
        f_bc[dof] = 0.0
    return K_bc, f_bc


def solve_displacements(K: np.ndarray,
                        f: np.ndarray,
                        fixed_dofs: Iterable[int]) -> np.ndarray:
    """
    Solves Ku = f with boundary conditions.

    Parameters
    ----------
    K : np.ndarray
        Global stiffness matrix (n_dofs, n_dofs).
    f : np.ndarray
        Global force vector (n_dofs,).
    fixed_dofs : Iterable[int]
        Fixed DOF indices.

    Returns
    -------
    np.ndarray
        Displacement vector u (n_dofs,).
    """
    K_bc, f_bc = apply_boundary_conditions(K, f, fixed_dofs)
    u = np.linalg.solve(K_bc, f_bc)
    return u


# ----------------------------- Force Generators ---------------------------- #

def force_vector(n_dofs: int,
                 loads: Sequence[Tuple[int, float]]) -> np.ndarray:
    """
    Creates a global force vector with point loads applied at DOF indices.

    Parameters
    ----------
    n_dofs : int
        Total number of DOFs.
    loads : Sequence[Tuple[int, float]]
        Each tuple is (dof_index, magnitude). Downward force is negative on Uy.

    Returns
    -------
    np.ndarray
        Force vector f (n_dofs,).
    """
    f = np.zeros(n_dofs, dtype=float)
    for dof, mag in loads:
        f[dof] += mag
    return f


def force_random_nodes_y(n_dofs: int,
                         node_indices: Sequence[int],
                         mean: float,
                         std: float) -> np.ndarray:
    """
    Generates a random downward force on the Y DOF of given nodes (Gaussian, i.i.d.).

    Parameters
    ----------
    n_dofs : int
        Total DOFs.
    node_indices : Sequence[int]
        Zero-based node indices where Uy will be loaded.
    mean : float
        Mean load (negative for downward).
    std : float
        Standard deviation.

    Returns
    -------
    np.ndarray
        Force vector f (n_dofs,).
    """
    f = np.zeros(n_dofs, dtype=float)
    for n in node_indices:
        dof_y = 2 * n + 1
        f[dof_y] = np.random.normal(loc=mean, scale=std)
    return f


def force_uniform_ramp_y(n_dofs: int,
                         node_indices: Sequence[int],
                         f_min: float,
                         f_max: float,
                         step: float) -> List[np.ndarray]:
    """
    Generates a list of force vectors with uniformly increasing Y-loads on the given nodes.

    Parameters
    ----------
    n_dofs : int
        Total DOFs.
    node_indices : Sequence[int]
        Nodes to load (Uy DOF).
    f_min : float
        Starting load (e.g., 0).
    f_max : float
        Ending load (e.g., -4.023085e6).
    step : float
        Increment size.

    Returns
    -------
    List[np.ndarray]
        A list of force vectors.
    """
    forces = []
    vals = np.arange(f_min, f_max + step, step)
    for val in vals:
        f = np.zeros(n_dofs, dtype=float)
        for n in node_indices:
            f[2 * n + 1] = val
        forces.append(f)
    return forces


# --------------------------- Convenience Builders -------------------------- #

def build_default_bridge() -> Tuple[List[TrussElement], int, List[int]]:
    """
    Constructs the Warren truss example consistent with your notebook defaults.

    Returns
    -------
    elements : List[TrussElement]
        The 11 truss elements.
    n_dofs : int
        Total DOFs (14).
    fixed_dofs : List[int]
        Typical fixed DOFs (e.g., node 0: Ux,Uy; node 6: Ux) -> [0, 1, 13]

    Notes
    -----
    - Angles (deg): [60, 0, 120, 60, 0, 0, 120, 60, 0, 0, 120]
    - Connectivity (0-based): [[0,1], [0,2], [2,1], [2,3], [1,3], [2,4], [4,3], [4,5], [3,5], [4,6], [6,5]]
    - E=200000 (MPa), A=11300 (mm^2), L=4000 (mm) for all elements.
    """
    E_list = [200000.0] * 11     # MPa (N/mm^2)
    A_list = [11300.0] * 11      # mm^2
    L_list = [4000.0] * 11       # mm
    angle_list = [60, 0, 120, 60, 0, 0, 120, 60, 0, 0, 120]
    conn = [(0, 1), (0, 2), (2, 1), (2, 3), (1, 3),
            (2, 4), (4, 3), (4, 5), (3, 5), (4, 6), (6, 5)]

    elements = [
        TrussElement(E=E_list[i], A=A_list[i], L=L_list[i],
                     angle_deg=angle_list[i], nodes=conn[i])
        for i in range(11)
    ]

    n_nodes = 7
    n_dofs = 2 * n_nodes
    fixed_dofs = [0, 1, 13]  # Node 0: Ux,Uy; Node 6: Uy (example consistent with notebook)
    return elements, n_dofs, fixed_dofs


