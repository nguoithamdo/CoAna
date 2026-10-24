"""3.6.0 merge acceptance: FIX-02 ``pre-commitment`` against 3.5.1's branch-local rework.

The audit's FIX-02 was written against 3.4.0, where commitment was judged on the
whole split/merge lineage. 3.5.0 moved the default to the guest's own branch
(``mode="branch"``) and added ``commit_duration``. The merge applied textually
clean, but that is not evidence the label means the same thing on the new path.
These tests drive the label end to end -- trajectory cache, lineage build,
``residence_events`` in both modes, ``mechanism_summary``, the CLI -- with the
expected labels worked out by hand, in the style of test_v351_regressions.py.

Scenario (hosts 0..9 are species A, the guest 10 is species B; commit_size=5,
commit_duration=3, growth_min=1):

    f0-f4    assembly {0,1} + guest             2 hosts
    f5-f9    assembly {0,1,2} + guest           3 hosts   <- grew during the visit
    f10-f14  assembly {0,1,2}, guest free       3 hosts   <- guest has left
    f15-f40  assembly {0,1,2,3,4}, guest free   5 hosts   <- commits at f15

The guest's single visit is f0..f9. It ends before the assembly commits, so it
is neither present at commitment nor a late joiner, and the host gained one
chain while it was there. Expected label: ``pre-commitment``.

Control (``grows_during_visit=False``): the host stays at 2 chains for the
whole visit and only grows after the guest leaves. Same timing, no growth
attributable to the visit. Expected label: ``bystander``.

The assembly never splits or merges, so its branch and its lineage coincide:
both modes must agree.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from oligomer_analysis import (
    SpeciesMap,
    build_lineages_from_cache,
    mechanism_summary,
    residence_events,
)
from oligomer_analysis.cache_io import save_contact_cache
from oligomer_analysis.mechanism import residence_dataframe
from oligomer_analysis.models import TargetStateConfig

from synthetic import cache_from_cmaps

N_FRAMES = 41
GUEST = 10


def _scenario(grows_during_visit: bool):
    n = 11

    def hosts(f):
        if f < 5:
            return [0, 1]
        if f < 10:
            return [0, 1, 2] if grows_during_visit else [0, 1]
        if f < 15:
            return [0, 1, 2]
        return [0, 1, 2, 3, 4]

    cm = np.zeros((N_FRAMES, n, n), np.int32)
    for f in range(N_FRAMES):
        members = hosts(f) + ([GUEST] if f < 10 else [])
        for i in members:
            for j in members:
                if i != j:
                    cm[f, i, j] = 5
    meta = {"species": [{"name": "A", "n_atoms": 4, "count": 10},
                        {"name": "B", "n_atoms": 4, "count": 1}]}
    return cache_from_cmaps(cm, meta=meta)


def _events(cache, mode):
    result = build_lineages_from_cache(
        cache, inter_thresh=1, cluster_min_size=2,
        target_config=TargetStateConfig(min_size=5, min_duration=1),
        species_map=SpeciesMap.from_cache(cache))
    return residence_events(result, guest_species="B", commit_size=5,
                            commit_duration=3, growth_min=1, mode=mode)


def _guest(events):
    return [(e.start_idx, e.end_idx, e.classification) for e in events if e.chain == GUEST]


# OPEN-01, decided B+: keep 3.5.0's branch rule for the
# label, fix the reasons that asserted what the visit-scoped rule cannot see,
# and record the later commitment as data for an ensemble test.

def test_branch_mode_keeps_its_rule_and_its_reason_is_now_true():
    e = [x for x in _events(_scenario(True), "branch") if x.chain == GUEST]
    assert [(x.start_idx, x.end_idx, x.classification) for x in e] == [(0, 9, "promoting")]
    assert "never reached" not in e[0].reason
    assert "attributed to the visit" in e[0].reason


def test_lineage_mode_gives_pre_commitment():
    assert _guest(_events(_scenario(True), "lineage")) == [(0, 9, "pre-commitment")]


@pytest.mark.parametrize("mode", ["branch", "lineage"])
@pytest.mark.parametrize("grows", [True, False])
def test_both_modes_agree_on_the_later_commitment_data(mode, grows):
    """No split or merge, so branch and lineage are one object: whatever the
    labels, the data must agree. The assembly commits at f15; the guest left
    at f9."""
    e = [x for x in _events(_scenario(grows), mode) if x.chain == GUEST][0]
    assert (e.later_commit_idx, e.departure_to_commit_lag) == (15, 6), mode


def test_the_walk_stops_when_the_assembly_is_absorbed():
    """The guest's assembly {0,1,2} is swallowed at f12 by a bigger one,
    {3,4,5,6}. Neither has committed (4 hosts < 5); the MERGED cluster, 7 hosts,
    commits at f12 -- after the guest left, so the "starts after departure"
    check alone would accept it. Only the identity rule can refuse it: the
    merged cluster's main line is the bigger assembly's, so its commitment is
    not the guest's assembly's (the merge case 3.5.0's branch rule excludes)."""
    n = 11

    def members(f):
        small = ([0, 1] if f < 5 else [0, 1, 2]) + ([GUEST] if f < 10 else [])
        big = [3, 4, 5, 6]
        return [small + big] if f >= 12 else [small, big]

    cm = np.zeros((N_FRAMES, n, n), np.int32)
    for f in range(N_FRAMES):
        for cl in members(f):
            for i in cl:
                for j in cl:
                    if i != j:
                        cm[f, i, j] = 5
    meta = {"species": [{"name": "A", "n_atoms": 4, "count": 10},
                        {"name": "B", "n_atoms": 4, "count": 1}]}
    e = [x for x in _events(cache_from_cmaps(cm, meta=meta), "branch") if x.chain == GUEST]
    assert e and e[0].later_commit_idx is None, e[0].later_commit_idx


def test_a_visit_already_credited_with_commitment_has_no_later_commit():
    """If the guest is still there when the assembly commits, the commitment is
    the visit's own -- there is nothing 'later' to record."""
    n = 11
    cm = np.zeros((N_FRAMES, n, n), np.int32)
    for f in range(N_FRAMES):
        hosts = [0, 1] if f < 15 else [0, 1, 2, 3, 4]
        mem = hosts + [GUEST]
        for i in mem:
            for j in mem:
                if i != j:
                    cm[f, i, j] = 5
    meta = {"species": [{"name": "A", "n_atoms": 4, "count": 10},
                        {"name": "B", "n_atoms": 4, "count": 1}]}
    for mode in ("branch", "lineage"):
        e = [x for x in _events(cache_from_cmaps(cm, meta=meta), mode) if x.chain == GUEST][0]
        assert e.committed and e.later_commit_idx is None, mode


@pytest.mark.parametrize("mode", ["branch", "lineage"])
def test_the_no_growth_control_stays_a_bystander_in_both_modes(mode):
    assert _guest(_events(_scenario(False), mode)) == [(0, 9, "bystander")], mode


def test_pre_commitment_survives_every_consumer():
    # lineage mode: the label FIX-02 defines is produced there (see OPEN-01)
    """Summary and dataframe must carry the label, and shares must still sum to 1."""
    ev = _events(_scenario(True), "lineage")
    df = residence_dataframe(ev)
    assert "pre-commitment" in set(df["classification"])
    summ = mechanism_summary(ev).set_index("classification")
    assert "pre-commitment" in summ.index
    assert np.isclose(summ["visit_share"].sum(), 1.0)


def test_pre_commitment_reaches_the_cli(tmp_path):
    # the CLI defaults to mode=branch, so the lineage rule is requested explicitly
    path = tmp_path / "pc.npz"
    out = tmp_path / "events.csv"
    save_contact_cache(_scenario(True), str(path), save_csv_stats=False)
    proc = subprocess.run(
        [sys.executable, "-m", "oligomer_analysis.cli", "mechanism", str(path),
         "--guest", "B", "--min-size", "5", "--min-duration", "3",
         "--mode", "lineage", "--out", str(out)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    df = pd.read_csv(out)
    assert "pre-commitment" in set(df["classification"]), proc.stdout
    for col in ("later_commit_idx", "later_commit_frame", "departure_to_commit_lag"):
        assert col in df.columns, col
