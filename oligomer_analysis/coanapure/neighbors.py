# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Neighbour search for CoAnaPure (pure NumPy, no MDAnalysis).

Reimplements the single call the toolkit makes into MDAnalysis::

    capped_distance(pos, pos, max_cutoff=cutoff, box=box, return_distances=False)

Stage A feeds residue centre-of-geometry points and wants every pair of points
within ``max_cutoff`` under the minimum-image convention.  Downstream code keeps
only ``i < j`` pairs, so this function returns each unordered pair exactly once
as ``[i, j]`` with ``i < j`` -- the resulting contact set is identical to
MDAnalysis after its ``i < j`` filter.

The implementation is a periodic linked-cell (cell list) search:

* points are wrapped into the primary cell;
* the box is gridded into cells of side >= cutoff;
* each point is compared only with points in its own and 26 neighbouring cells,
  using minimum-image distances.

For a box that is smaller than one cutoff along an axis, or a missing/degenerate
box, it falls back to a direct all-pairs computation so results stay correct.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np

from .pbc import box_is_orthorhombic, minimize_vectors, wrap_positions

try:  # strongly recommended acceleration; falls back to pure cell lists when absent
    from scipy.spatial import cKDTree as _cKDTree
except Exception:  # pragma: no cover - scipy is a declared dependency
    _cKDTree = None

__all__ = ["capped_distance", "self_capped_pairs", "HAVE_SCIPY"]

HAVE_SCIPY = _cKDTree is not None

if not HAVE_SCIPY:  # pragma: no cover - exercised via monkeypatched routing tests
    warnings.warn(
        "CoAnaPure: scipy is not installed. The neighbour search will use the "
        "pure-NumPy cell lists, which are several times slower than the scipy "
        "cKDTree path. Install scipy to restore full Stage A speed.",
        RuntimeWarning,
        stacklevel=2,
    )


def _kdtree_pairs(
    pos: np.ndarray, cutoff: float, box6: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast neighbour search via scipy cKDTree (orthorhombic PBC or free)."""
    if box6 is None:
        tree = _cKDTree(pos)
    else:
        lengths = box6[:3]
        wrapped = wrap_positions(pos, box6)
        # Guard against a point landing exactly on the upper edge after wrap.
        wrapped = np.minimum(wrapped, np.nextafter(lengths, 0.0))
        tree = _cKDTree(wrapped, boxsize=lengths)
    pairs = tree.query_pairs(r=float(cutoff), output_type="ndarray")
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    pairs = pairs.astype(np.int64, copy=False)
    # Distances in float64 for the returned pairs (minimum image).
    diff = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    if box6 is not None:
        diff = minimize_vectors(diff, box6).astype(np.float64, copy=False)
    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    return pairs, dist


def _all_pairs_mic(
    pos: np.ndarray, cutoff: float, box: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Direct O(N^2) upper-triangle search; correct for any box, small N."""
    n = pos.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    diff = pos[iu] - pos[ju]
    if box is not None:
        diff = minimize_vectors(diff, box).astype(np.float64, copy=False)
    d = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    keep = d <= cutoff
    pairs = np.stack([iu[keep], ju[keep]], axis=1).astype(np.int64, copy=False)
    return pairs, d[keep]


# Half-neighbourhood: the 13 lexicographically-positive offsets.  Visiting only
# these (plus the cell itself) means every unordered cell pair is examined
# exactly once, so no pair deduplication is needed in the non-periodic search.
_HALF_NEIGH = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
]


def _ragged_gather(starts: np.ndarray, counts: np.ndarray):
    """Expand per-source ranges ``[starts, starts+counts)`` into flat index pairs.

    Returns ``(rep, tgt)`` where ``rep[k]`` is the source row that produced flat
    element ``k`` and ``tgt[k]`` walks that row's range.  Vectorised equivalent of
    ``for r, (s, c) in enumerate(zip(starts, counts)): yield from ((r, s+t) ...)``.
    """
    total = int(counts.sum())
    if total == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e
    rep = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    cum = np.concatenate(([0], np.cumsum(counts)[:-1]))
    within = np.arange(total, dtype=np.int64) - np.repeat(cum, counts)
    return rep, np.repeat(starts, counts) + within


def _cell_list_pairs_nopbc(
    pos: np.ndarray, cutoff: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Linked-cell search without periodicity (grid over the bounding box).

    O(N) replacement for the all-pairs fallback when there is no box: the box-less
    case previously fell straight to O(N^2), which made a scipy-less run of a
    typical box-less Martini trajectory ~400x slower than the scipy path.  The
    search is fully vectorised: points are bucketed by cell once, and each of the
    13 half-neighbourhood offsets is resolved with two ``searchsorted`` calls plus
    a ragged gather, so no per-cell Python loop is executed.
    """
    origin = pos.min(axis=0)
    extent = pos.max(axis=0) - origin
    # Cell side >= cutoff along every axis (>= 1 cell per axis).
    ncell = np.maximum(np.floor(extent / cutoff).astype(np.int64), 1)
    cell_size = np.where(extent > 0, extent / ncell, 1.0)

    cidx = np.floor((pos - origin) / cell_size).astype(np.int64)
    cidx = np.clip(cidx, 0, ncell - 1)

    ny, nz = int(ncell[1]), int(ncell[2])
    flat = (cidx[:, 0] * ny + cidx[:, 1]) * nz + cidx[:, 2]

    order = np.argsort(flat, kind="stable")
    fs = flat[order]  # sorted flat cell id per point

    cutoff2 = cutoff * cutoff
    i_all, j_all, d_all = [], [], []

    def keep_pairs(ii: np.ndarray, jj: np.ndarray) -> None:
        diff = pos[ii] - pos[jj]
        d2 = np.einsum("ij,ij->i", diff, diff)
        within = d2 <= cutoff2
        if within.any():
            i_all.append(np.minimum(ii[within], jj[within]))
            j_all.append(np.maximum(ii[within], jj[within]))
            d_all.append(np.sqrt(d2[within]))

    # Same-cell pairs: each sorted point pairs with the later points of its cell.
    p = np.arange(fs.shape[0], dtype=np.int64)
    seg_end = np.searchsorted(fs, fs, side="right")
    rep, tgt = _ragged_gather(p + 1, seg_end - p - 1)
    if rep.size:
        keep_pairs(order[rep], order[tgt])

    # Cross-cell pairs: 13 forward offsets, each unordered cell pair visited once.
    for dx, dy, dz in _HALF_NEIGH:
        tci = cidx + (dx, dy, dz)
        valid = np.all((tci >= 0) & (tci < ncell), axis=1)
        if not valid.any():
            continue
        src = np.flatnonzero(valid)
        tflat = (tci[src, 0] * ny + tci[src, 1]) * nz + tci[src, 2]
        starts = np.searchsorted(fs, tflat, side="left")
        counts = np.searchsorted(fs, tflat, side="right") - starts
        rep, tgt = _ragged_gather(starts, counts)
        if rep.size:
            keep_pairs(src[rep], order[tgt])

    if not i_all:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    pairs = np.stack(
        [np.concatenate(i_all), np.concatenate(j_all)], axis=1
    ).astype(np.int64, copy=False)
    return pairs, np.concatenate(d_all)


def self_capped_pairs(
    positions: np.ndarray,
    cutoff: float,
    box: Optional[np.ndarray] = None,
    *,
    return_distances: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Return unique ``i < j`` pairs of ``positions`` within ``cutoff``.

    Parameters
    ----------
    positions : ndarray (N, 3)
    cutoff : float
        Inclusive distance cutoff (angstrom).
    box : ndarray (6,) or None
        ``[lx, ly, lz, alpha, beta, gamma]``.  ``None`` -> no periodicity.
    return_distances : bool
        When True, also return the pair distances.
    """
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    if n < 2:
        empty = np.empty((0, 2), dtype=np.int64)
        return (empty, np.empty(0, dtype=np.float64)) if return_distances else empty

    cutoff = float(cutoff)

    box6 = None
    if box is not None:
        box6 = np.asarray(box, dtype=np.float64)

    ortho = box6 is not None and box_is_orthorhombic(box6) and np.all(box6[:3] > 0)

    # Fastest path: scipy cKDTree for the free or orthorhombic-periodic case.
    if _cKDTree is not None and (box6 is None or ortho):
        pairs, dist = _kdtree_pairs(pos, cutoff, box6 if ortho else None)
        return (pairs, dist) if return_distances else pairs

    # No box, no scipy: non-periodic cell list keeps the search O(N).
    if box6 is None and n > 350:
        pairs, dist = _cell_list_pairs_nopbc(pos, cutoff)
        return (pairs, dist) if return_distances else pairs

    # Periodic pure cell list, valid for an orthorhombic box with >= 3 cells/axis.
    if ortho:
        lengths = box6[:3]
        ncell = np.floor(lengths / cutoff).astype(np.int64)
        if np.all(ncell >= 3):
            pairs, dist = _cell_list_pairs(pos, cutoff, box6)
            return (pairs, dist) if return_distances else pairs

    # Fallback: direct all-pairs (triclinic, tiny boxes/N, or no scipy + <3 cells).
    pairs, dist = _all_pairs_mic(pos, cutoff, box6)
    return (pairs, dist) if return_distances else pairs


def _cell_list_pairs(
    pos: np.ndarray, cutoff: float, box6: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    lengths = box6[:3]
    wrapped = wrap_positions(pos, box6)  # into [0, L)

    ncell = np.floor(lengths / cutoff).astype(np.int64)
    ncell = np.maximum(ncell, 1)
    cell_size = lengths / ncell

    # Cell index per point along each axis.
    cidx = np.floor(wrapped / cell_size).astype(np.int64)
    cidx = np.clip(cidx, 0, ncell - 1)  # guard against == L rounding

    nx, ny, nz = (int(ncell[0]), int(ncell[1]), int(ncell[2]))
    flat = (cidx[:, 0] * ny + cidx[:, 1]) * nz + cidx[:, 2]

    # Bucket points by cell.
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    # Start offset of each occupied cell in the sorted order.
    boundaries = np.concatenate(([0], np.flatnonzero(np.diff(flat_sorted)) + 1, [len(flat_sorted)]))
    cell_of = {}
    for b in range(len(boundaries) - 1):
        s, e = boundaries[b], boundaries[b + 1]
        cell_of[int(flat_sorted[s])] = order[s:e]

    cutoff2 = cutoff * cutoff
    neigh = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    i_all = []
    j_all = []
    d_all = []

    occupied = np.unique(flat)
    for cell in occupied:
        ci = cell // (ny * nz)
        rem = cell % (ny * nz)
        cj = rem // nz
        ck = rem % nz
        here = cell_of[int(cell)]

        # Gather candidate points from this cell + 26 neighbours (dedup cells).
        seen_cells = set()
        cand_list = []
        for dx, dy, dz in neigh:
            ni = (ci + dx) % nx
            nj = (cj + dy) % ny
            nk = (ck + dz) % nz
            ncflat = int((ni * ny + nj) * nz + nk)
            if ncflat in seen_cells:
                continue
            seen_cells.add(ncflat)
            got = cell_of.get(ncflat)
            if got is not None:
                cand_list.append(got)
        if not cand_list:
            continue
        cand = np.concatenate(cand_list)

        # All (here x cand) combinations, then keep unique i<j.
        ii = np.repeat(here, cand.shape[0])
        jj = np.tile(cand, here.shape[0])
        mask = ii < jj
        ii = ii[mask]
        jj = jj[mask]
        if ii.size == 0:
            continue
        diff = minimize_vectors(pos[ii] - pos[jj], box6).astype(np.float64, copy=False)
        d2 = np.einsum("ij,ij->i", diff, diff)
        within = d2 <= cutoff2
        if not within.any():
            continue
        i_all.append(ii[within])
        j_all.append(jj[within])
        d_all.append(np.sqrt(d2[within]))

    if not i_all:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)

    i_cat = np.concatenate(i_all)
    j_cat = np.concatenate(j_all)
    d_cat = np.concatenate(d_all)

    # A pair can be discovered from more than one neighbour cell when the grid is
    # small; deduplicate on the (i, j) key.
    key = i_cat.astype(np.int64) * pos.shape[0] + j_cat.astype(np.int64)
    _, uniq = np.unique(key, return_index=True)
    pairs = np.stack([i_cat[uniq], j_cat[uniq]], axis=1).astype(np.int64, copy=False)
    return pairs, d_cat[uniq]


def capped_distance(
    reference: np.ndarray,
    configuration: np.ndarray,
    max_cutoff: float,
    box: Optional[np.ndarray] = None,
    *,
    min_cutoff: Optional[float] = None,
    return_distances: bool = False,
):
    """MDAnalysis-compatible ``capped_distance`` for the self-search case.

    Only the self-search path (``reference is configuration`` or equal arrays)
    used by Stage A is optimised; a general reference/configuration pair falls
    back to a correct all-pairs computation.  Pairs are returned as ``i < j``
    (self and mirrored duplicates removed), which is what the caller keeps.
    """
    ref = np.asarray(reference, dtype=np.float64)
    conf = np.asarray(configuration, dtype=np.float64)

    is_self = reference is configuration or (
        ref.shape == conf.shape and np.array_equal(ref, conf)
    )

    if is_self:
        pairs, dist = self_capped_pairs(ref, max_cutoff, box, return_distances=True)
    else:
        # General case: brute force across the two sets under MIC.
        diff = ref[:, None, :] - conf[None, :, :]
        flat = diff.reshape(-1, 3)
        if box is not None:
            flat = minimize_vectors(flat, box).astype(np.float64, copy=False)
        d = np.sqrt(np.einsum("ij,ij->i", flat, flat)).reshape(ref.shape[0], conf.shape[0])
        mask = d <= float(max_cutoff)
        ii, jj = np.nonzero(mask)
        pairs = np.stack([ii, jj], axis=1).astype(np.int64, copy=False)
        dist = d[ii, jj]

    if min_cutoff is not None and pairs.shape[0]:
        keep = dist > float(min_cutoff)
        pairs = pairs[keep]
        dist = dist[keep]

    if return_distances:
        return pairs, dist
    return pairs
