# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests for the ``follow`` modes of the classic pathway tracker.

Background: the legacy continuation can
follow the wrong fragment after a split (10 -> 8+2 recording the 2);
``follow="main_fragment"`` must follow the 8, and ``follow="legacy"`` (the
default) must keep the historical behavior bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from oligomer_analysis import ContactCache, analyze_from_cache


def make_cache(frame_specs, nseg):
    """Synthetic cache whose per-frame clusters are exactly ``frame_specs``.

    Each spec lists the multi-member clusters (as member iterables); members
    of a cluster are chained with contacts so they form one connected
    component.  Unlisted monomers become singleton clusters automatically.
    """
    n = len(frame_specs)
    cmaps = np.zeros((n, nseg, nseg), dtype=np.int32)
    for f, clusters in enumerate(frame_specs):
        for clu in clusters:
            ms = sorted(clu)
            for a, b in zip(ms, ms[1:]):
                cmaps[f, a, b] = cmaps[f, b, a] = 1
    zeros = np.zeros(n, dtype=np.int32)
    return ContactCache(
        frames=np.arange(n, dtype=np.int32),
        cmaps=cmaps,
        nintra=zeros.copy(), ninter=zeros.copy(), ntotal=zeros.copy(),
        n_free=zeros.copy(), n_bound=zeros.copy(),
        seg_centers=np.zeros((n, nseg, 3), dtype=np.float32),
        boxes=np.zeros((n, 6), dtype=np.float32),
        meta={},
    )


def run(cache, follow=None, **overrides):
    kw = dict(inter_thresh=1, size_thresh=3, continue_min_size=1,
              overlap_min=1, shrink_tolerance=3, dur_store_min=1)
    kw.update(overrides)
    if follow is not None:
        kw["follow"] = follow
    return analyze_from_cache(cache, **kw)["finished"]


# The report's canonical case: 10 -> (2 + 8) -> 10.  The 2-fragment contains
# the lowest member index, so the legacy cluster-centric pass visits it first.
SPLIT_SPEC = [
    [range(10)],                 # F0: one 10-mer
    [{0, 1}, range(2, 10)],      # F1: splits into 2 + 8
    [range(10)],                 # F2: re-merges
]


def test_legacy_follows_small_fragment():
    finished = run(make_cache(SPLIT_SPEC, 10), follow="legacy")
    by_start = sorted(finished, key=lambda p: p.start_idx)
    # The original pathway is dragged onto the 2-fragment and dies there;
    # the abandoned 8-fragment is reborn as a *new* label that inherits the
    # re-merged 10-mer.  This test documents the known-wrong behavior so any
    # accidental change to the legacy path is caught.
    assert by_start[0].sizes == [10, 2]
    assert by_start[1].sizes == [8, 10]


def test_main_fragment_follows_main_branch():
    finished = run(make_cache(SPLIT_SPEC, 10), follow="main_fragment")
    assert len(finished) == 1, "no label proliferation at the split"
    assert finished[0].sizes == [10, 8, 10]
    assert finished[0].sets[1] == frozenset(range(2, 10))


def test_default_is_legacy():
    cache = make_cache(SPLIT_SPEC, 10)
    default = run(cache)
    legacy = run(cache, follow="legacy")
    assert [(p.label, p.start_idx, p.sizes, p.sets) for p in default] == \
           [(p.label, p.start_idx, p.sizes, p.sets) for p in legacy]


def test_subset_bonus_steal_fixed():
    # The report's mechanism 3: a degenerate pathway shrunk to one member is a
    # full subset of the merged cluster (score 100001), while the big pathway
    # -- which lost a single member in the same step -- scores only its raw
    # overlap (7).  Legacy hands the merged cluster to the monomer pathway and
    # drags the big pathway onto a stray monomer; main_fragment awards it by
    # overlap.
    spec = [
        [range(8), {8, 9, 10}],       # F0: P (8-mer) and D (3-mer)
        [range(8), {8}],              # F1: D shrinks to the monomer {8}
        [{0, 1, 2, 3, 4, 5, 6, 8}],   # F2: {8} merges into P while member 7 splits off
    ]
    cache = make_cache(spec, 11)

    legacy = {p.sets[0]: p for p in run(cache, follow="legacy")}
    assert legacy[frozenset({8, 9, 10})].sizes == [3, 1, 8], \
        "legacy: the monomer pathway steals the merged 8-mer via the subset bonus"
    assert legacy[frozenset(range(8))].sizes == [8, 8, 1], \
        "legacy: the big pathway crashes onto the stray monomer {7}"

    fixed = {p.sets[0]: p for p in run(cache, follow="main_fragment")}
    p_fixed = fixed[frozenset(range(8))]
    assert p_fixed.sizes == [8, 8, 8], "main_fragment: the big pathway keeps the merged cluster"
    assert frozenset({8, 9, 10}) not in fixed or fixed[frozenset({8, 9, 10})].sizes == [3, 1], \
        "main_fragment: the degenerate pathway does not steal"


def test_no_wrong_branch_under_main_fragment():
    # Randomized splits/merges: under main_fragment, a pathway either follows
    # its max-(overlap, size) cluster, or that cluster was claimed by another
    # pathway with at least as much overlap (a genuine merge contention).
    # A pathway may never lose its main fragment to nothing or to a weaker
    # claimant -- that was the legacy bug.
    rng = np.random.RandomState(7)
    nseg, nframes = 24, 30
    specs = []
    for _ in range(nframes):
        members = list(range(nseg))
        rng.shuffle(members)
        clusters, i = [], 0
        while i < nseg:
            k = int(rng.randint(1, 9))
            clusters.append(set(members[i:i + k]))
            i += k
        specs.append([c for c in clusters if len(c) >= 2])
    cache = make_cache(specs, nseg)
    finished = run(cache, follow="main_fragment")

    from oligomer_analysis.graph_tools import clusters_from_cmap
    frames_clusters = [clusters_from_cmap(cm, 1) for cm in cache.cmaps]

    claims = {}      # (frame_idx, cluster) -> claimant's overlap with its prev set
    for p in finished:
        for t in range(1, len(p.sets)):
            fi = p.start_idx + t
            claims[(fi, p.sets[t])] = len(p.sets[t - 1] & p.sets[t])

    for p in finished:
        for t in range(len(p.sets) - 1):
            prev, rec = p.sets[t], p.sets[t + 1]
            fi = p.start_idx + t + 1
            b_ov, b_sz, b_clu = max(
                ((len(prev & c), len(c), c) for c in frames_clusters[fi]
                 if len(prev & c) > 0),
                key=lambda x: x[:2],
            )
            if (len(prev & rec), len(rec)) == (b_ov, b_sz):
                continue                       # followed (a tie of) the main branch
            assert claims.get((fi, b_clu), -1) >= b_ov, (
                f"{p.label} step {t}: lost its main fragment "
                f"(ov={b_ov}, size={b_sz}) without a stronger claimant"
            )


def test_invalid_follow_rejected():
    with pytest.raises(ValueError, match="follow"):
        run(make_cache(SPLIT_SPEC, 10), follow="banana")