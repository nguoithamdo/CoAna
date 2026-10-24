"""Periodic-boundary geometry for CoAnaPure (pure NumPy, no MDAnalysis).

This module reimplements the minimum-image-convention (MIC) helper that Stage A
takes from :func:`MDAnalysis.lib.distances.minimize_vectors`.  The only box
convention used anywhere in the toolkit is the MDAnalysis 6-vector
``[lx, ly, lz, alpha, beta, gamma]`` (lengths in angstrom, angles in degrees),
so that is the single input format supported here.

Two paths are provided:

* orthorhombic (all angles == 90 deg) -- an exact, vectorised wrap that matches
  MDAnalysis bit-for-bit for orthorhombic boxes;
* triclinic (any non-90 angle) -- a brute-force search over the 27 nearest
  periodic images, which reproduces MDAnalysis' ``_minimize_vectors_triclinic``
  for the physically relevant (not extremely skewed) case.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "box_is_orthorhombic",
    "triclinic_vectors",
    "minimize_vectors",
    "wrap_positions",
]


def _as_box6(box) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (6,):
        raise ValueError(f"box must be a 6-vector [lx,ly,lz,alpha,beta,gamma]; got shape {box.shape}")
    return box


def box_is_orthorhombic(box, *, atol: float = 1e-4) -> bool:
    """Return True when all three box angles are 90 degrees."""
    box = _as_box6(box)
    return bool(np.all(np.abs(box[3:] - 90.0) <= atol))


def triclinic_vectors(box) -> np.ndarray:
    """Convert an MDAnalysis 6-vector box to a 3x3 matrix of box vectors (rows).

    Uses the same convention as :func:`MDAnalysis.lib.mdamath.triclinic_vectors`:
    the first box vector lies along x, the second in the xy-plane, and the box
    matrix is lower-left/upper-triangular in the row-vector layout returned here.
    """
    box = _as_box6(box)
    lx, ly, lz, alpha, beta, gamma = box
    mat = np.zeros((3, 3), dtype=np.float64)
    if lx <= 0 or ly <= 0 or lz <= 0:
        return mat

    if abs(alpha - 90.0) < 1e-6 and abs(beta - 90.0) < 1e-6 and abs(gamma - 90.0) < 1e-6:
        mat[0, 0] = lx
        mat[1, 1] = ly
        mat[2, 2] = lz
        return mat

    ca = np.cos(np.deg2rad(alpha))
    cb = np.cos(np.deg2rad(beta))
    cg = np.cos(np.deg2rad(gamma))
    sg = np.sin(np.deg2rad(gamma))

    mat[0, 0] = lx
    mat[1, 0] = ly * cg
    mat[1, 1] = ly * sg
    mat[2, 0] = lz * cb
    mat[2, 1] = lz * (ca - cb * cg) / sg
    mat[2, 2] = np.sqrt(max(lz * lz - mat[2, 0] ** 2 - mat[2, 1] ** 2, 0.0))
    # Clean tiny numerical noise so orthogonal-ish boxes stay clean.
    mat[np.abs(mat) < 1e-10] = 0.0
    return mat


def box6_from_vectors(matrix: np.ndarray) -> "np.ndarray | None":
    """Convert a 3x3 box-vector matrix (rows = v1, v2, v3) to an MDAnalysis
    6-vector ``[lx, ly, lz, alpha, beta, gamma]`` (degrees).

    This is the inverse of :func:`triclinic_vectors` and matches
    ``MDAnalysis.lib.mdamath.triclinic_box``.  Returns ``None`` for a
    degenerate (zero-volume) box, the convention GROMACS uses for "no box".
    """
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    lengths = np.linalg.norm(m, axis=1)
    if np.any(lengths <= 0):
        return None

    def angle(a: np.ndarray, b: np.ndarray) -> float:
        cosv = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))

    alpha = angle(m[1], m[2])
    beta = angle(m[0], m[2])
    gamma = angle(m[0], m[1])
    return np.array([lengths[0], lengths[1], lengths[2], alpha, beta, gamma], dtype=np.float64)


def _minimize_orthorhombic(vectors: np.ndarray, box6: np.ndarray) -> np.ndarray:
    lengths = box6[:3]
    out = np.asarray(vectors, dtype=np.float64).copy()
    # Only wrap along axes with a positive box length.
    good = lengths > 0
    out[:, good] -= lengths[good] * np.round(out[:, good] / lengths[good])
    return out


def _minimize_triclinic(vectors: np.ndarray, box6: np.ndarray) -> np.ndarray:
    """Brute-force minimum image for a general triclinic box.

    For each input vector we first reduce it into the primary cell in fractional
    coordinates, then test all 27 combinations of +-1 lattice shifts and keep the
    representative with the smallest Euclidean norm.  This matches MDAnalysis'
    triclinic minimum-image result for standard simulation boxes.
    """
    mat = triclinic_vectors(box6)  # rows are box vectors
    inv = np.linalg.inv(mat)
    vecs = np.asarray(vectors, dtype=np.float64)

    # Fractional coords: frac @ mat = cart  ->  frac = cart @ inv
    frac = vecs @ inv
    frac -= np.round(frac)  # reduce into [-0.5, 0.5)

    shifts = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=np.float64,
    )  # (27, 3)

    best = None
    best_d2 = None
    for shift in shifts:
        cand_frac = frac + shift
        cand = cand_frac @ mat  # (N, 3)
        d2 = np.einsum("ij,ij->i", cand, cand)
        if best is None:
            best = cand
            best_d2 = d2
        else:
            take = d2 < best_d2
            if take.any():
                best[take] = cand[take]
                best_d2[take] = d2[take]
    return best


def minimize_vectors(vectors, box) -> np.ndarray:
    """Return the minimum-image representation of ``vectors`` under ``box``.

    Parameters
    ----------
    vectors : array_like, shape (N, 3) or (3,)
        Displacement vectors to minimise.
    box : array_like, shape (6,)
        ``[lx, ly, lz, alpha, beta, gamma]`` (angstrom, degrees).

    Returns
    -------
    numpy.ndarray, shape (N, 3), dtype float32
        Minimum-image vectors.  ``float32`` mirrors the dtype MDAnalysis returns
        so downstream residue-centre-of-geometry maths is byte-comparable.
    """
    box6 = _as_box6(box)
    vecs = np.asarray(vectors, dtype=np.float64)
    single = vecs.ndim == 1
    if single:
        vecs = vecs[None, :]
    if vecs.shape[1] != 3:
        raise ValueError("vectors must have shape (N, 3)")

    if box_is_orthorhombic(box6):
        out = _minimize_orthorhombic(vecs, box6)
    else:
        out = _minimize_triclinic(vecs, box6)

    out = out.astype(np.float32, copy=False)
    return out[0] if single else out


def wrap_positions(positions: np.ndarray, box) -> np.ndarray:
    """Wrap Cartesian positions into the primary unit cell.

    Used by the neighbour search to place every point in ``[0, L)`` before
    building the cell grid.  Supports orthorhombic and triclinic boxes.
    """
    box6 = _as_box6(box)
    pos = np.asarray(positions, dtype=np.float64)
    if box_is_orthorhombic(box6):
        lengths = box6[:3]
        out = pos.copy()
        good = lengths > 0
        out[:, good] -= lengths[good] * np.floor(out[:, good] / lengths[good])
        return out
    mat = triclinic_vectors(box6)
    inv = np.linalg.inv(mat)
    frac = pos @ inv
    frac -= np.floor(frac)
    return frac @ mat
