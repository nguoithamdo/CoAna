# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

from collections import deque
from typing import FrozenSet, Iterable, List, Sequence

import numpy as np

try:  # SciPy is a core dependency; the pure-Python BFS below is the fallback.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components as _csgraph_cc
except Exception:  # pragma: no cover - exercised only without SciPy
    csr_matrix = None
    _csgraph_cc = None


def edge_mask_from_cmap(cmap: np.ndarray, inter_thresh: int, *, include_self: bool = False) -> np.ndarray:
    """Return a boolean adjacency mask using the project-wide edge rule.

    An edge exists exactly when ``cmap[i, j] >= inter_thresh``.
    """
    if inter_thresh < 1:
        raise ValueError(f"inter_thresh must be >= 1, got {inter_thresh}")

    adj = np.asarray(cmap) >= int(inter_thresh)
    adj = adj.astype(bool, copy=False)
    np.fill_diagonal(adj, include_self)
    return adj


def _clusters_bfs(adj: np.ndarray) -> List[FrozenSet[int]]:
    """Reference implementation: breadth-first search over a dense mask."""
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    clusters: List[FrozenSet[int]] = []

    for node in range(n):
        if visited[node]:
            continue

        queue = deque([node])
        visited[node] = True
        component: List[int] = []

        while queue:
            v = queue.popleft()
            component.append(v)
            for nbr in np.flatnonzero(adj[v]):
                if not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)

        clusters.append(frozenset(component))

    return clusters


def clusters_from_cmap(cmap: np.ndarray, inter_thresh: int) -> List[FrozenSet[int]]:
    """Connected components of the segment graph induced by ``cmap``.

    Components are returned in ascending order of their smallest member, which
    is the order the original breadth-first implementation produced.  Lineage
    tracking calls this once per frame -- tens of thousands of times for a full
    campaign -- so the component labelling runs in SciPy's compiled csgraph
    routine when SciPy is available, and falls back to the pure-Python search
    otherwise.  Both paths return identical results.
    """
    adj = edge_mask_from_cmap(cmap, inter_thresh, include_self=True)
    n = adj.shape[0]
    if n == 0:
        return []

    if _csgraph_cc is None:  # pragma: no cover - exercised only without SciPy
        return _clusters_bfs(adj)

    graph = csr_matrix(adj)
    n_comp, labels = _csgraph_cc(graph, directed=False, return_labels=True)

    # csgraph labels components in order of first appearance when scanning nodes
    # ascending, which already matches "ascending smallest member". Grouping via
    # argsort keeps that order without a Python loop over nodes.
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
    groups = np.split(order, boundaries)
    clusters = [frozenset(int(x) for x in grp) for grp in groups]
    if len(clusters) != n_comp:  # pragma: no cover - defensive
        raise RuntimeError("connected-component grouping lost a component")
    return clusters


def filter_clusters_by_size(clusters: Sequence[FrozenSet[int]], min_size: int = 2) -> List[FrozenSet[int]]:
    """Return only clusters with size >= ``min_size``.

    Use ``min_size=2`` to exclude monomers from whole-system cluster statistics.
    """
    return [clu for clu in clusters if len(clu) >= int(min_size)]


def build_cluster_transition_matrix(
    frames_clusters: Sequence[Sequence[FrozenSet[int]]],
    *,
    min_size: int = 1,
) -> np.ndarray:
    """Global size-transition count matrix across all clusters in consecutive frames.

    Parameters
    ----------
    frames_clusters
        Cluster list for each frame.
    min_size
        Minimum cluster size to include. Use ``min_size=2`` to exclude monomers.
    """
    filtered_frames = [filter_clusters_by_size(frame, min_size=min_size) for frame in frames_clusters]
    max_size = max((len(clu) for frame in filtered_frames for clu in frame), default=0)
    T = np.zeros((max_size + 1, max_size + 1), dtype=int)

    for prev, nxt in zip(filtered_frames[:-1], filtered_frames[1:]):
        for clu_prev in prev:
            prev_size = len(clu_prev)
            best_next = None
            best_overlap = -1
            for clu_next in nxt:
                overlap = len(clu_prev & clu_next)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_next = clu_next
            next_size = len(best_next) if best_next is not None else 0
            T[prev_size, next_size] += 1

    return T


def intra_cluster_contacts(cluster: FrozenSet[int], cmap: np.ndarray) -> int:
    """Sum of inter-segment residue contacts within a cluster."""
    idx = np.array(sorted(cluster), dtype=int)
    if len(idx) < 2:
        return 0
    sub = cmap[np.ix_(idx, idx)]
    return int(np.triu(sub, 1).sum())


def pathway_transition_matrix(sizes: Iterable[int]) -> np.ndarray:
    """Transition count matrix for one pathway's size series."""
    sizes = list(int(x) for x in sizes)
    max_size = max(sizes, default=0)
    T = np.zeros((max_size + 1, max_size + 1), dtype=int)
    for a, b in zip(sizes[:-1], sizes[1:]):
        T[a, b] += 1
    return T
