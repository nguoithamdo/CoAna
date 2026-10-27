# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .graph_tools import edge_mask_from_cmap
from .models import ContactCache, Pathway

# MDAnalysis is optional for cache-only analysis. Import it lazily only when
# MIC geometry is explicitly requested.
_MINIMIZE_VECTORS = None
_MINIMIZE_VECTORS_TRIED = False

def _get_minimize_vectors():
    """The minimum-image function, from MDAnalysis or from CoAnaPure.

    This used to return None whenever MDAnalysis was absent, which is the
    package's normal configuration. The consequences were silent and serious:
    :func:`unwrap_centers_mic` then applied no correction at all and returned
    its wrapped input unchanged, so every MSD was computed on wrapped
    coordinates and gained a box-sized jump each time a monomer crossed a
    boundary; and ``compactness_timeseries(mode="rg")`` returned all-NaN.

    CoAnaPure ships its own ``minimize_vectors`` with the same signature and
    the same MDAnalysis box convention, handling orthorhombic and triclinic
    boxes -- it is what Stage A already uses for every contact. Preferring
    MDAnalysis when present keeps results bit-identical to the original
    pipeline; falling back to CoAnaPure keeps the dependency policy true, which
    is that nothing in the package *requires* MDAnalysis.
    """
    global _MINIMIZE_VECTORS, _MINIMIZE_VECTORS_TRIED
    if not _MINIMIZE_VECTORS_TRIED:
        try:
            from MDAnalysis.lib.distances import minimize_vectors as _mv
            _MINIMIZE_VECTORS = _mv
        except Exception:
            try:
                from .coanapure.pbc import minimize_vectors as _mv
                _MINIMIZE_VECTORS = _mv
            except Exception:
                _MINIMIZE_VECTORS = None
        _MINIMIZE_VECTORS_TRIED = True
    return _MINIMIZE_VECTORS


@dataclass(slots=True)
class FreeEnergyProfile:
    sizes: np.ndarray
    counts: np.ndarray
    F_over_kBT: np.ndarray
    F_joule: Optional[np.ndarray]
    #: Standard error on each F/kT from the bin count alone, 1/sqrt(count).
    #: A bin holding one frame carries an error of 1 kT, which is the same size
    #: as the features these profiles are usually read for.
    F_stderr: Optional[np.ndarray] = None
    #: Autocorrelation time of the size trace, in frames, when it was supplied.
    #: Consecutive frames are not independent samples, so the honest error is
    #: ``sqrt(2 * tau / count)`` rather than ``1/sqrt(count)``.
    autocorr_time: Optional[float] = None


@dataclass(slots=True)
class SmallWorldResult:
    S: float
    C: float
    L: float
    Cr: float
    Lr: float
    n: int
    m: int


def compute_free_energy_from_sizes(
    sizes: Iterable[int],
    *,
    shift_min_to_zero: bool = True,
    temperature_K: Optional[float] = None,
    min_count: int = 1,
    autocorr_aware: bool = True,
) -> FreeEnergyProfile:
    """Estimate free energy from the empirical size histogram.

    ``F/kT = -ln P`` where ``P`` is the fraction of observations at each size.

    Read the error bars before reading the curve.  The vertical extent of this
    profile is exactly ``ln(c_max / c_min)``: with a single pathway's sizes,
    ``c_min`` is almost always 1, so the range is ``ln(c_max)`` and is set by how
    long the pathway lingered at its most-visited size rather than by any
    thermodynamics.  On the reference 128-mer run, between 12% and 58% of bins
    rest on a single frame.  ``F_stderr`` reports ``1/sqrt(count)``, or
    ``sqrt(2*tau/count)`` when ``autocorr_aware`` and the sizes are correlated,
    which for a one-frame bin is 1 kT.

    ``min_count`` drops sizes observed fewer times than that, which is the
    honest way to avoid plotting a barrier that is one frame of noise.

    Two further cautions.  A single pathway's size histogram is a residence-time
    trace, not an equilibrium ensemble, so ``-ln P`` is only a free energy to the
    extent that the sampling is equilibrium-distributed -- for a monotonically
    growing aggregate it is not.  And pooling sizes across many pathways gives
    far more counts but mixes distinct events.
    """
    counts_dict = Counter(int(x) for x in sizes)
    if not counts_dict:
        raise ValueError("sizes is empty")

    counts_dict = dict(sorted(counts_dict.items()))
    size_vals = np.array(list(counts_dict.keys()), dtype=int)
    counts = np.array(list(counts_dict.values()), dtype=int)

    # The probability must be normalised over everything observed, before any
    # min_count filtering -- dropping rare sizes and then renormalising would
    # move every remaining F by a constant and misstate the populations.
    prob_all = counts / counts.sum()

    if min_count > 1:
        keep = counts >= int(min_count)
        if not keep.any():
            raise ValueError(
                f"no size was observed at least min_count={min_count} times; "
                f"the largest count is {int(counts.max())}"
            )
        size_vals, counts, prob_all = size_vals[keep], counts[keep], prob_all[keep]

    F_over_kBT = -np.log(prob_all)
    if shift_min_to_zero:
        F_over_kBT = F_over_kBT - F_over_kBT.min()

    tau = None
    if autocorr_aware:
        seq = np.asarray(list(sizes), dtype=float) if not isinstance(sizes, np.ndarray) else sizes.astype(float)
        if seq.size >= 4:
            from .convergence import integrated_autocorrelation_time

            tau = float(integrated_autocorrelation_time(seq))

    # 1/sqrt(c) for independent samples; a bin of c correlated frames holds
    # about c / (2 tau) independent ones.
    if tau is not None and tau > 0.5:
        eff = np.maximum(1.0, counts / (2.0 * tau))
        F_stderr = 1.0 / np.sqrt(eff)
    else:
        F_stderr = 1.0 / np.sqrt(counts.astype(float))

    F_joule = None
    if temperature_K is not None:
        kB = 1.380_649e-23
        F_joule = F_over_kBT * kB * float(temperature_K)

    return FreeEnergyProfile(
        sizes=size_vals,
        counts=counts,
        F_over_kBT=F_over_kBT,
        F_joule=F_joule,
        F_stderr=F_stderr,
        autocorr_time=tau,
    )


def free_energy_to_dataframe(profile: FreeEnergyProfile) -> pd.DataFrame:
    data = {
        "size": profile.sizes,
        "count": profile.counts,
        "F_over_kBT": profile.F_over_kBT,
    }
    if profile.F_stderr is not None:
        data["F_stderr"] = profile.F_stderr
    if profile.F_joule is not None:
        data["F_joule"] = profile.F_joule
    return pd.DataFrame(data)


def compactness_timeseries(
    cache: ContactCache,
    pathway: Pathway,
    inter_thresh: int,
    *,
    mode: str = "density",
    rg_use_mic: bool = True,
) -> np.ndarray:
    """Compute a contact-based or geometric compactness trace for one pathway.

    ``mode="rg"`` is the radius of gyration of the member centres. With
    ``rg_use_mic`` the members are first gathered into a common periodic image,
    so an oligomer straddling a boundary is measured as the compact object it
    is. That gathering is exact only while the oligomer is smaller than half the
    box in each dimension -- the minimum-image convention's own precondition --
    so an aggregate that spans the box needs a larger box, not a better
    estimator.
    """
    mode = mode.lower()
    s = int(pathway.start_idx)
    dur = int(pathway.duration)
    y = np.zeros(dur, dtype=float)

    for t in range(dur):
        members = sorted(list(pathway.sets[t]))
        n = len(members)

        if mode in {"density", "degree", "weighted"}:
            if n < 2:
                y[t] = 0.0
                continue
            sub = cache.cmaps[s + t][np.ix_(members, members)]

            if mode == "weighted":
                denom = n * (n - 1) / 2
                y[t] = float(np.triu(sub, 1).sum() / denom)
            else:
                A = edge_mask_from_cmap(sub, inter_thresh, include_self=False)
                E = float(np.triu(A, 1).sum())
                if mode == "density":
                    y[t] = float(2.0 * E / (n * (n - 1)))
                else:
                    y[t] = float(2.0 * E / n)
        elif mode == "rg":
            minimize_vectors = _get_minimize_vectors()
            if minimize_vectors is None:
                return np.full(dur, np.nan, dtype=float)
            if n < 2:
                y[t] = 0.0
                continue
            centers = np.asarray(cache.seg_centers[s + t, members, :], dtype=float)
            # Read the box only when it is going to be used. Testing the box
            # first broke `rg_use_mic=False` on a cache carrying `boxes=None`,
            # which worked before and raised TypeError afterwards.
            box6 = None
            use_mic = False
            if rg_use_mic and getattr(cache, "boxes", None) is not None:
                box6 = np.asarray(cache.boxes[s + t], dtype=float)
                use_mic = box6.shape == (6,) and bool(np.all(box6[:3] > 0))
            if use_mic:
                # Gather the members into ONE periodic image before taking the
                # centroid. Averaging the wrapped coordinates and then
                # minimum-imaging the vectors to that average does not work: for
                # a cluster astride a boundary the average sits in the middle of
                # the box, the members are all ~L/2 away from it, and each of
                # those vectors is already its own minimum image -- so the
                # correction is a no-op and Rg comes out as roughly L/2. A
                # 2 A dimer straddling a 100 A box measured 49.0 instead of 1.0.
                #
                # The image is propagated along the cluster's own contact graph,
                # so the answer does not depend on member order. Referencing
                # every member to members[0] instead is order-dependent once the
                # cluster spans half the box: see gather_cluster_images.
                sub = np.asarray(cache.cmaps[s + t])[np.ix_(members, members)]
                adjacency = sub >= inter_thresh
                local = gather_cluster_images(centers, adjacency, box6)
            else:
                local = centers
            vec = local - local.mean(axis=0, keepdims=True)
            y[t] = float(np.sqrt((vec * vec).sum(axis=1).mean()))
        else:
            raise ValueError("mode must be one of: density, degree, weighted, rg")

    return y


def gather_cluster_images(
    centers: np.ndarray,
    adjacency: Optional[np.ndarray] = None,
    box6: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Put every member of one cluster into a single periodic image.

    Returns coordinates in a common image, so a centroid, an Rg or a span taken
    over them is meaningful even when the cluster straddles a boundary.

    The image is propagated along ``adjacency`` -- the cluster's own contact
    graph -- by breadth-first search: each member is placed next to a neighbour
    that is already placed, using the minimum image of the displacement between
    them. Because every step is between members that are genuinely in contact,
    each displacement is far shorter than half the box and its minimum image is
    unambiguous. The result does not depend on which member is visited first.

    Referencing every member to one arbitrary member instead is only correct
    while the whole cluster fits inside half the box. Beyond that it is not
    merely approximate but *order-dependent*: for collinear members at
    x = 5, 35, 65 in a 100 A box, referencing member 0 yields Rg = 28.674 and
    referencing the middle member 24.495, with the contact-graph answer being
    the latter.

    Without an ``adjacency`` the function falls back to that single-reference
    gather, which is why callers should pass the contact map when they have one.
    A cluster that wraps right around the box has no consistent image at all;
    no gather can fix that, and it needs a larger box.
    """
    c = np.asarray(centers, dtype=float)
    n = c.shape[0]
    if n == 0:
        return c.copy()
    if box6 is None:
        return c.copy()

    minimize_vectors = _get_minimize_vectors()
    if minimize_vectors is None:
        return c.copy()

    box6 = np.asarray(box6, dtype=float)

    def mic(vecs):
        return np.asarray(minimize_vectors(np.atleast_2d(vecs), box6), dtype=float)

    if adjacency is None:
        return c[0] + mic(c - c[0])

    adj = np.asarray(adjacency, dtype=bool)
    out = np.empty_like(c)
    placed = np.zeros(n, dtype=bool)

    # One BFS per connected component, so a member set that is not actually
    # one cluster still gets a defined answer rather than silently keeping its
    # wrapped coordinates.
    for root in range(n):
        if placed[root]:
            continue
        out[root] = c[root]
        placed[root] = True
        queue = [root]
        while queue:
            i = queue.pop(0)
            nbrs = np.flatnonzero(adj[i] & ~placed)
            if nbrs.size:
                d = mic(c[nbrs] - c[i])
                out[nbrs] = out[i] + d
                placed[nbrs] = True
                queue.extend(nbrs.tolist())

    return out


def unwrap_centers_mic(centers: np.ndarray, boxes: Optional[np.ndarray] = None) -> np.ndarray:
    """Unwrap a time series of points with MIC displacements between consecutive frames.

    Each consecutive displacement is reduced to its minimum image before being
    accumulated, so a point crossing a periodic boundary continues smoothly
    instead of jumping by a box length. ``boxes`` must be one MDAnalysis
    6-vector ``[lx, ly, lz, alpha, beta, gamma]`` per frame; without it, or for
    a frame whose box is degenerate, that frame's displacement is accumulated
    as given and the result is only correct if nothing wrapped.

    To unwrap a group that moves together, unwrap the members and then average
    them -- never average first. See :func:`msd_pathway_peak_members`.
    """
    c = np.asarray(centers, dtype=float)
    squeeze_back = False
    if c.ndim == 2:
        c = c[:, None, :]
        squeeze_back = True

    T, _, _ = c.shape
    out = np.empty_like(c)
    out[0] = c[0]

    for t in range(1, T):
        d = c[t] - c[t - 1]
        if boxes is not None:
            minimize_vectors = _get_minimize_vectors()
            if minimize_vectors is not None:
                box = np.asarray(boxes[t], dtype=float)
                if box.shape == (6,) and np.all(box[:3] > 0):
                    d = minimize_vectors(d, box)
        out[t] = out[t - 1] + d

    return out[:, 0, :] if squeeze_back else out


def msd_from_positions(r: np.ndarray, max_lag: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean square displacement from unwrapped positions."""
    r = np.asarray(r, dtype=float)
    T = r.shape[0]
    if max_lag is None or max_lag > T - 1:
        max_lag = T - 1

    msd = np.zeros(max_lag + 1, dtype=float)
    for tau in range(1, max_lag + 1):
        dr = r[tau:] - r[:-tau]
        msd[tau] = np.mean(np.sum(dr * dr, axis=1))
    lags = np.arange(max_lag + 1, dtype=int)
    return lags, msd


def msd_pathway_peak_members(cache: ContactCache, pathway: Pathway, max_lag: int = 150) -> dict:
    """Pathway-local MSD for peak members and for the peak-fixed oligomer center."""
    s = int(pathway.start_idx)
    dur = int(pathway.duration)
    if dur <= 1:
        raise ValueError("Pathway duration too short for MSD; need at least 2 frames")

    sizes = pathway.sizes
    t_peak = next(i for i, z in enumerate(sizes) if z == max(sizes))
    peak_members = sorted(list(pathway.sets[t_peak]))
    max_lag = min(int(max_lag), dur - 1)

    boxes_slice = cache.boxes[s : s + dur] if getattr(cache, "boxes", None) is not None else None

    centers = cache.seg_centers[s : s + dur, peak_members, :]
    centers_unwrapped = unwrap_centers_mic(centers, boxes=boxes_slice)

    member_curves = []
    for k in range(len(peak_members)):
        lags, msd_k = msd_from_positions(centers_unwrapped[:, k, :], max_lag=max_lag)
        member_curves.append(msd_k)

    msd_members = np.vstack(member_curves)
    msd_mean = msd_members.mean(axis=0)

    # Average the ALREADY-UNWRAPPED member positions. Averaging the wrapped
    # ones first and unwrapping the result is wrong whenever the oligomer
    # straddles a periodic boundary: the mean of coordinates sitting at both
    # ends of the box lands in the middle of it, a point the oligomer never
    # occupies, and that fictitious centre jumps by order L as members wrap.
    # On a rigid dimer drifting 2 A/frame astride the boundary, where the true
    # centre-of-mass MSD is exactly 4*tau^2, the old order gave 125 at lag 1
    # against a true 4, and 144 at lag 19 against a true 1444.
    center_unwrapped = centers_unwrapped.mean(axis=1)
    _, msd_oligo = msd_from_positions(center_unwrapped, max_lag=max_lag)

    return {
        "lags": lags,
        "peak_members": peak_members,
        "msd_members": msd_members,
        "msd_mean": msd_mean,
        "msd_oligo": msd_oligo,
        "max_lag": max_lag,
    }


def build_contact_graph_snapshot(members: Sequence[int], cmap: np.ndarray, inter_thresh: int):
    """Build an undirected segment graph for one snapshot."""
    import networkx as nx

    members = sorted(list(members))
    G = nx.Graph()
    G.add_nodes_from(members)
    for i in range(len(members) - 1):
        for j in range(i + 1, len(members)):
            a = members[i]
            b = members[j]
            if cmap[a, b] >= inter_thresh:
                G.add_edge(a, b)
    return G


def largest_connected_component(G):
    """Copy of the largest connected component subgraph."""
    import networkx as nx

    if G.number_of_nodes() == 0:
        return G.copy()
    if nx.is_connected(G):
        return G.copy()
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()


def small_worldness_S(G, n_rand: int = 30, seed: int = 0) -> SmallWorldResult:
    """Compute S = (C/Cr) / (L/Lr) using an ER ensemble matched by (n, m)."""
    import networkx as nx

    Gcc = largest_connected_component(G)
    n = Gcc.number_of_nodes()
    m = Gcc.number_of_edges()

    if n < 2:
        return SmallWorldResult(np.nan, np.nan, np.nan, np.nan, np.nan, n, m)

    C = float(nx.average_clustering(Gcc))
    try:
        L = float(nx.average_shortest_path_length(Gcc))
    except Exception:
        L = np.nan

    rng = np.random.default_rng(seed)
    Cr_list, Lr_list = [], []
    for _ in range(n_rand):
        Gr = nx.gnm_random_graph(n, m, seed=int(rng.integers(0, 2**31 - 1)))
        Grcc = largest_connected_component(Gr)
        if Grcc.number_of_nodes() < 2:
            continue
        Cr_list.append(nx.average_clustering(Grcc))
        try:
            Lr_list.append(nx.average_shortest_path_length(Grcc))
        except Exception:
            pass

    if not Cr_list or not Lr_list or not np.isfinite(L):
        return SmallWorldResult(np.nan, C, L, np.nan, np.nan, n, m)

    Cr = float(np.mean(Cr_list))
    Lr = float(np.mean(Lr_list))
    if Cr <= 0 or L <= 0 or Lr <= 0:
        S = np.nan
    else:
        S = float((C / Cr) / (L / Lr))
    return SmallWorldResult(S, C, L, Cr, Lr, n, m)
