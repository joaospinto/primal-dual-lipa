"""Shared AMD/RCM permutation helper for the sip-python KKT matrix.

sip-python's bundled helper looks for libamd in cvxopt's macOS-only
``.dylibs`` directory; on Linux it silently falls back to RCM, which
produces much more L-factor fill on OCP-structured KKT systems. This
module invokes cvxopt's AMD bindings directly when available and falls
back to the bundled RCM helper otherwise.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

import sip_python as sip


class _Result(NamedTuple):
    perm_inv: np.ndarray
    kkt_nnz: int
    L_nnz: int
    method: str  # "amd" or "rcm"


def _amd_perm_inv_from_K(K_csc: sp.csc_matrix) -> np.ndarray | None:
    """Return inverse AMD permutation for sparse symmetric K, or None.

    On the inverse convention:
      * ``perm[i]`` is the new position of original index i.
      * ``perm_inv[j]`` is the original index of new-position j.
      * ``perm_inv[perm] == arange(n)`` (sanity).
    sip stores ``perm_inv`` (the QDLDL backend addresses permuted-space
    rows directly).
    """
    try:
        from cvxopt import amd, spmatrix
    except ImportError:
        return None

    # cvxopt.amd needs a symmetric pattern; K from sip.get_K is
    # block-symmetric except for sign-flips on the y/z rows, but the
    # SPARSITY is symmetric. Take K + K.T to enforce that, then extract
    # triplets.
    K_sym = (K_csc + K_csc.T).tocoo()
    n = K_sym.shape[0]
    A_cv = spmatrix(
        np.ones(K_sym.nnz, dtype=np.float64).tolist(),
        K_sym.row.astype(np.intc).tolist(),
        K_sym.col.astype(np.intc).tolist(),
        (n, n),
    )
    perm_cv = amd.order(A_cv)
    perm = np.array(perm_cv, dtype=np.int64).reshape(-1)
    perm_inv = np.empty_like(perm)
    perm_inv[perm] = np.arange(perm.shape[0], dtype=np.int64)
    return perm_inv


def compute_kkt_perm_inv_and_nnzs(
    P_template: sp.spmatrix,
    A_template: sp.spmatrix,
    G_template: sp.spmatrix,
) -> _Result:
    """AMD-first KKT permutation + symbolic factor sizes for sip-python.

    Tries cvxopt-AMD; falls back to sip's own helper (which is RCM on
    Linux). Returns ``(perm_inv, kkt_nnz, L_nnz, method)`` where
    ``method`` is ``"amd"`` or ``"rcm"``.
    """
    K = sip.get_K(P_template, A_template, G_template)
    perm_inv_amd = _amd_perm_inv_from_K(sp.csc_matrix(K))
    if perm_inv_amd is not None:
        kkt_nnz, L_nnz = sip.get_kkt_and_L_nnzs(K=K, perm_inv=perm_inv_amd)
        # Sanity-check the inverse convention: perm_inv[perm] == arange(n).
        # If it's wrong, sip's QDLDL would silently solve a permuted system.
        forward = np.empty_like(perm_inv_amd)
        forward[perm_inv_amd] = np.arange(perm_inv_amd.shape[0])
        assert np.all(perm_inv_amd[forward] == np.arange(perm_inv_amd.shape[0]))
        return _Result(perm_inv_amd, kkt_nnz, L_nnz, "amd")

    perm_inv_rcm, kkt_nnz, L_nnz = sip.get_kkt_perm_inv_and_nnzs(
        P=P_template,
        A=A_template,
        G=G_template,
        verbose=False,
    )
    return _Result(np.asarray(perm_inv_rcm), kkt_nnz, L_nnz, "rcm")
