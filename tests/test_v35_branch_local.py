# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""3.5.0: guest roles judged on the guest's own assembly, and directed fates.

Up to 3.4.0 the mechanism classifier judged commitment on the whole split/merge
lineage, so a guest on a small assembly that never grew inherited "passenger" or
"catalytic" from a different branch, even from a merge that happened after the
guest had left.  Every scenario here has a hand-worked answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from oligomer_analysis import (
    SpeciesMap,
    build_lineages_from_cache,
    directed_fate_summary,
    directed_fates,
    residence_events,
)
from oligomer_analysis.models import TargetStateConfig

from synthetic import cache_from_cmaps

N_HOST = 10            # chains 0..9 are host A
GUEST = 10             # chain 10 is guest B
N_FRAMES = 41
META = {"species": [{"name": "A", "n_atoms": 4, "count": N_HOST},
                    {"name": "B", "n_atoms": 4, "count": 1}]}
TARGET = TargetStateConfig(min_size=5, min_duration=1)


def _link(cm, f, ids):
    for i in ids:
        for j in ids:
            if i != j:
                cm[f, i, j] = 5


def _cache(frames_to_clusters):
    cm = np.zeros((N_FRAMES, N_HOST + 1, N_HOST + 1), np.int32)
    for f in range(N_FRAMES):
        for cl in frames_to_clusters(f):
            _link(cm, f, cl)
    return cache_from_cmaps(cm, meta=META)


def _events(cache, *, mode="branch", commit_size=5, commit_duration=None, target=TARGET):
    smap = SpeciesMap.from_cache(cache)
    res = build_lineages_from_cache(cache, inter_thresh=1, cluster_min_size=2,
                                    target_config=target, species_map=smap)
    return residence_events(res, guest_species="B", commit_size=commit_size,
                            commit_duration=commit_duration, mode=mode)


def _small_guest_cluster(merge_later, big_grows_at):
    """A guest on a 2-host cluster that never grows, beside a big assembly.

    The guest sits on {6, 7} for frames 5-20.  The big assembly reaches 6 host
    chains at ``big_grows_at``.  With ``merge_later`` the small cluster's hosts
    merge into the big one at frame 30, ten frames after the guest left.
    """
    def clusters(f):
        big = [0, 1, 2] + ([3, 4, 5] if f >= big_grows_at else [])
        if merge_later and f >= 30:
            return [big + [6, 7]]
        if 5 <= f <= 20:
            return [big, [6, 7, GUEST]]
        return [big, [6, 7]]
    return _cache(clusters)


@pytest.mark.parametrize("merge_later", [True, False])
@pytest.mark.parametrize("big_grows_at", [0, 10])
def test_guest_on_small_branch_is_a_bystander(merge_later, big_grows_at):
    events = _events(_small_guest_cluster(merge_later, big_grows_at))
    assert len(events) == 1
    e = events[0]
    assert (e.start_idx, e.end_idx, e.host_at_entry, e.host_max_during) == (5, 20, 2, 2)
    assert not e.committed
    assert e.classification == "bystander"


def test_lineage_mode_reproduces_the_340_mislabels():
    """The legacy rule, kept for comparison, still shows the defect."""
    already = _events(_small_guest_cluster(True, 0), mode="lineage")[0]
    during = _events(_small_guest_cluster(True, 10), mode="lineage")[0]
    assert already.classification == "passenger"
    assert during.classification == "catalytic"


def test_genuine_passenger_is_still_a_passenger():
    """Joining an assembly that already holds 6 host chains is recruitment."""
    def clusters(f):
        big = [0, 1, 2, 3, 4, 5]
        return [big + [GUEST]] if 15 <= f <= 25 else [big]
    e = _events(_cache(clusters))[0]
    assert e.committed and e.joined_after_commit and not e.present_at_commit
    assert e.classification == "passenger"


def test_guest_merged_into_committed_assembly_mid_visit_is_a_passenger():
    """Riding a small cluster into an already-committed assembly is not commitment."""
    def clusters(f):
        big = [0, 1, 2, 3, 4, 5]
        if 5 <= f < 12:
            return [big, [6, GUEST]]
        if 12 <= f <= 20:
            return [big + [6, GUEST]]
        return [big, [6, 7]]
    e = _events(_cache(clusters))[0]
    assert (e.start_idx, e.end_idx) == (5, 20)
    assert e.committed and e.joined_after_commit
    assert e.classification == "passenger"


def test_one_frame_touch_is_not_commitment():
    """Reaching the commit size for one frame must not count when duration > 1."""
    def clusters(f):
        hosts = [0, 1, 2, 3, 4] if f == 10 else [0, 1, 2, 3]
        return [hosts + [GUEST]] if 5 <= f <= 20 else [hosts]
    flicker = _events(_cache(clusters), commit_duration=3)[0]
    assert not flicker.committed
    assert flicker.classification == "promoting"

    def sustained(f):
        hosts = [0, 1, 2, 3, 4] if 10 <= f <= 14 else [0, 1, 2, 3]
        return [hosts + [GUEST]] if 5 <= f <= 20 else [hosts]
    held = _events(_cache(sustained), commit_duration=3)[0]
    assert held.committed and held.present_at_commit and held.commit_idx == 10


# ---------------------------------------------------------------------------
# Directed fates
# ---------------------------------------------------------------------------


def _fates(clusters, target=TargetStateConfig(min_size=5, min_duration=3)):
    cache = _cache(clusters)
    smap = SpeciesMap.from_cache(cache)
    res = build_lineages_from_cache(cache, inter_thresh=1, cluster_min_size=2,
                                    target_config=target, species_map=smap)
    return res, directed_fates(res).set_index("start_idx")


def test_directed_fates_follow_time_forward():
    """Three births: one dissolves, one merges into the product, one is unresolved.

    * {6, 7} is born at frame 2 and dissolves at frame 8 -> off-pathway.
    * {8, 9} is born at frame 4 and merges into the big assembly at 20 -> on.
    * the big assembly {0..5} exists from frame 0 and is a target -> on.
    * {6, 10} is born at frame 30 and is still alive at the end -> uncertain.
    """
    def clusters(f):
        big = [0, 1, 2, 3, 4, 5] + ([8, 9] if f >= 20 else [])
        out = [big]
        if 2 <= f <= 7:
            out.append([6, 7])
        if 4 <= f < 20:
            out.append([8, 9])
        if f >= 30:
            out.append([6, GUEST])
        return out
    _res, df = _fates(clusters)
    assert df.loc[2, "fate"] == "off-pathway"
    assert df.loc[4, "fate"] == "on-pathway"
    assert df.loc[0, "fate"] == "on-pathway"
    assert df.loc[30, "fate"] == "uncertain"
    assert bool(df.loc[30, "is_hetero_at_birth"]) and not bool(df.loc[2, "is_hetero_at_birth"])

    summ = directed_fate_summary(df.reset_index()).set_index("group")
    assert int(summ.loc["all", "n"]) == 4
    assert int(summ.loc["all", "on"]) == 2
    assert int(summ.loc["mixed at birth", "uncertain"]) == 1
    # min_span drops the short-lived birth
    short = directed_fate_summary(df.reset_index(), min_span=10).set_index("group")
    assert int(short.loc["all", "off"]) == 0


def test_fragment_shed_by_target_is_not_a_birth():
    """A piece that splits off the product has a parent, so it is not counted.

    The undirected lineage puts it in the product's lineage (and so calls it
    on-pathway); the directed table only scores nucleation events.
    """
    def clusters(f):
        if f < 15:
            return [[0, 1, 2, 3, 4, 5, 6, 7]]
        if f < 25:
            return [[0, 1, 2, 3, 4, 5], [6, 7]]
        return [[0, 1, 2, 3, 4, 5]]
    _res, df = _fates(clusters)
    assert list(df.index) == [0]
    assert df.loc[0, "fate"] == "on-pathway"
