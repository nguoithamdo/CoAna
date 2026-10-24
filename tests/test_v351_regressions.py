"""3.5.1: defects in the 3.5.0 branch-local mechanism and directed fates.

Found by an independent review and
reproduced on synthetic caches before any fix was written.  Each scenario pairs a
case with a control that differs only in something the guest's label must not
depend on, and the expected labels are worked out by hand in the docstrings.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from oligomer_analysis import (
    SpeciesMap,
    build_lineages_from_cache,
    directed_fates,
    filter_lineages,
    residence_events,
)
from oligomer_analysis.cache_io import save_contact_cache
from oligomer_analysis.models import TargetStateConfig

from synthetic import cache_from_cmaps

N_FRAMES = 41


def _cache(n_host, n_guest, clusters, frames=None):
    n = n_host + n_guest
    cm = np.zeros((N_FRAMES, n, n), np.int32)
    for f in range(N_FRAMES):
        for cl in clusters(f):
            for i in cl:
                for j in cl:
                    if i != j:
                        cm[f, i, j] = 5
    meta = {"species": [{"name": "A", "n_atoms": 4, "count": n_host},
                        {"name": "B", "n_atoms": 4, "count": n_guest}]}
    return cache_from_cmaps(cm, meta=meta, frames=frames)


def _result(cache, target=5):
    smap = SpeciesMap.from_cache(cache)
    return build_lineages_from_cache(
        cache, inter_thresh=1, cluster_min_size=2,
        target_config=TargetStateConfig(min_size=target, min_duration=1),
        species_map=smap)


def _events(result, frames=None):
    return residence_events(result, guest_species="B", commit_size=5,
                            commit_duration=3, mode="branch", frames=frames)


def _labels(events, chain):
    return [(e.start_idx, e.end_idx, e.classification) for e in events if e.chain == chain]


# ---------------------------------------------------------------------------
# 1. A split sibling's commitment must not leak into the guest's branch.
# ---------------------------------------------------------------------------

def _split(sibling_stays_big):
    """Ten hosts at f0 split into A = {0..4} and B = {5..9} at f1.

    The guest (chain 10) sits on A for f1-f20; A drops to four hosts at f2.
    Along the guest's own assembly the run of >= 5 hosts is f0-f1, two frames,
    shorter than commit_duration = 3, so the guest never meets a committed
    assembly and A never grows: bystander, whatever sibling B does.
    """
    G = 10

    def cl(f):
        if f == 0:
            return [list(range(10))]
        a = [0, 1, 2, 3, 4] if f == 1 else [0, 1, 2, 3]
        a = a + ([G] if f <= 20 else [])
        b = [5, 6, 7, 8, 9] if (sibling_stays_big or f == 1) else [5, 6, 7, 8]
        return [a, b]
    return _cache(10, 1, cl)


def test_sibling_commitment_does_not_leak_across_a_split():
    for stays in (True, False):
        ev = _events(_result(_split(stays)))
        assert _labels(ev, 10) == [(1, 20, "bystander")], f"sibling stays big={stays}"


# ---------------------------------------------------------------------------
# 2. Commitment is not forgotten when the assembly dips below threshold.
# ---------------------------------------------------------------------------

def _dip(dip):
    """Hosts {0..4} hold five chains from f0; the guest joins at f4.

    With ``dip`` the assembly holds only {0..3} at f3-f6.  It committed at f0
    (five hosts for f0-f2 >= commit_duration), so a guest arriving at f4 joins an
    already-committed assembly either way: passenger.  3.5.0 restarted the run
    after the dip and credited the guest with the recommitment (catalytic).
    """
    G = 10

    def cl(f):
        hosts = [0, 1, 2, 3] if (dip and 3 <= f <= 6) else [0, 1, 2, 3, 4]
        return [hosts + ([G] if 4 <= f <= 20 else [])]
    return _cache(10, 1, cl)


def test_commitment_survives_a_temporary_dip():
    for dip in (True, False):
        ev = _events(_result(_dip(dip)))
        assert _labels(ev, 10) == [(4, 20, "passenger")], f"dip={dip}"


# ---------------------------------------------------------------------------
# 3. A hop between host-disjoint assemblies is a new visit, companion or not.
# ---------------------------------------------------------------------------

def _hop(companion):
    """Guest G1 = 10 sits on {0, 1} for f0-f9, then on the committed {5..9}.

    The two assemblies share no host chain, so these are two visits: a bystander
    visit to the dimer and a passenger visit to the committed assembly.  A second
    guest (G2 = 11) making the same hop must not glue them into one visit.
    """
    G1, G2 = 10, 11

    def cl(f):
        big = [5, 6, 7, 8, 9]
        extra = [G2] if companion else []
        if f < 10:
            return [[0, 1, G1] + extra, big]
        return [[0, 1], big + [G1] + extra]
    return _cache(10, 2, cl)


def test_hop_carried_by_a_second_guest_is_two_visits():
    for companion in (True, False):
        ev = _events(_result(_hop(companion)))
        assert _labels(ev, 10) == [(0, 9, "bystander"), (10, 40, "passenger")], \
            f"companion={companion}"


# ---------------------------------------------------------------------------
# 4. Directed ancestry follows real overlaps, not the event's Cartesian product.
# ---------------------------------------------------------------------------

def _ancestry():
    """P1 = {0..3} and P2 = {4, 5} become C1 = {0, 1} and C2 = {2..5} at f10.

    P1 feeds both children, P2 feeds only C2.  C1 grows to the target (five
    chains) at f20; C2 stays at four chains to the end.  P2 therefore never
    reaches a target: alive at the end without one, so ``uncertain``.  3.5.0
    listed C1 among P2's children and called P2 on-pathway.
    """
    def cl(f):
        if f < 10:
            return [[0, 1, 2, 3], [4, 5]]
        if f < 20:
            return [[0, 1], [2, 3, 4, 5]]
        return [[0, 1, 6, 7, 8], [2, 3, 4, 5]]
    return _cache(10, 1, cl)


def test_directed_edges_follow_actual_overlaps():
    res = _result(_ancestry())
    segs = {s.segment_id: s for s in res.segments}
    roots = {frozenset(s.sets[0]): s for s in res.segments if s.start_idx == 0}
    p1, p2 = roots[frozenset({0, 1, 2, 3})], roots[frozenset({4, 5})]
    kids = lambda s: sorted(sorted(segs[c].sets[0]) for c in s.child_ids)
    assert kids(p1) == [[0, 1], [2, 3, 4, 5]]
    assert kids(p2) == [[2, 3, 4, 5]]
    fates = directed_fates(res).set_index("root_id")["fate"]
    assert fates[p1.segment_id] == "on-pathway"
    assert fates[p2.segment_id] == "uncertain"


def test_child_parent_ids_follow_actual_overlaps():
    res = _result(_ancestry())
    segs = {s.segment_id: s for s in res.segments}
    c1 = next(s for s in res.segments if s.start_idx == 10 and set(s.sets[0]) == {0, 1})
    assert sorted(sorted(segs[p].sets[0]) for p in c1.parent_ids) == [[0, 1, 2, 3]]


# ---------------------------------------------------------------------------
# 5. The persistence filter must reach the mechanism classifier.
# ---------------------------------------------------------------------------

def _short_and_long():
    """Guest 10 touches host dimer {0, 1} at f2-f3 only (a two-frame lineage),
    then sits on the long-lived committed assembly {5..9} from f10 to the end."""
    G = 10

    def cl(f):
        out = [[5, 6, 7, 8, 9] + ([G] if f >= 10 else [])]
        if 2 <= f <= 3:
            out.append([0, 1, G])
        return out
    return _cache(10, 1, cl)


def test_filtered_result_drops_visits_to_filtered_lineages():
    res = _result(_short_and_long())
    assert [e.start_idx for e in _events(res) if e.chain == 10] == [2, 10]
    kept = filter_lineages(res, min_duration=10)
    assert _labels(_events(kept), 10) == [(10, 40, "passenger")]


def test_cli_mechanism_applies_lineage_min_duration(tmp_path):
    path = tmp_path / "c.npz"
    save_contact_cache(_short_and_long(), path, save_csv_stats=False)
    out = tmp_path / "ev.csv"
    proc = subprocess.run(
        [sys.executable, "-m", "oligomer_analysis.cli", "mechanism", str(path),
         "--guest", "B", "--min-size", "5", "--min-duration", "3",
         "--lineage-min-duration", "10", "--out", str(out)],
        cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    df = pd.read_csv(out)
    assert list(df["start_frame"]) == [10]


# ---------------------------------------------------------------------------
# 6. commit_frame is a frame number, not an index.
# ---------------------------------------------------------------------------

def test_commit_frame_is_mapped_through_cache_frames():
    frames = [100 + 10 * i for i in range(N_FRAMES)]
    cache = _cache(10, 1, lambda f: [[0, 1, 2, 3, 4] + ([10] if 4 <= f <= 20 else [])],
                   frames=frames)
    (ev,) = _events(_result(cache), frames=frames)
    d = ev.to_dict()
    assert d["commit_idx"] == 0
    assert d["commit_frame"] == 100
    assert d["start_frame"] == 140
