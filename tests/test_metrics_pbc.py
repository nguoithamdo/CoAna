# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Periodic-boundary handling in the MSD / unwrapping path.

Every expected value here is analytic. The test system is a rigid dimer in a
cubic 100 A box drifting at a constant +2 A per frame along x, placed so that it
straddles the periodic boundary: the true centre-of-mass MSD is exactly
``(2*tau)**2``, and each member's MSD is the same, so any other answer is a
periodic-image artefact rather than motion.
"""

from __future__ import annotations

import numpy as np
import pytest

from oligomer_analysis.metrics import (
    _get_minimize_vectors,
    compactness_timeseries,
    msd_from_positions,
    msd_pathway_peak_members,
    unwrap_centers_mic,
)
from oligomer_analysis.models import ContactCache, Pathway

L = 100.0
T = 20
DRIFT = 2.0


def _straddling_dimer():
    """Wrapped coordinates of the drifting dimer, plus its true unwrapped path."""
    true = np.zeros((T, 2, 3))
    for t in range(T):
        true[t, 0] = [99.0 + DRIFT * t, 50.0, 50.0]
        true[t, 1] = [101.0 + DRIFT * t, 50.0, 50.0]
    wrapped = true.copy()
    wrapped[..., 0] %= L
    boxes = np.tile(np.array([L, L, L, 90.0, 90.0, 90.0]), (T, 1))
    return wrapped, true, boxes


def _cache(wrapped, boxes):
    cmaps = np.zeros((T, 2, 2), dtype=np.int32)
    cmaps[:, 0, 1] = cmaps[:, 1, 0] = 5          # the dimer is always in contact
    return ContactCache(
        frames=np.arange(T), cmaps=cmaps,
        nintra=np.zeros(T), ninter=np.zeros(T), ntotal=np.zeros(T),
        n_free=np.zeros(T), n_bound=np.full(T, 2),
        seg_centers=wrapped, boxes=boxes, meta={},
    )


def test_a_minimum_image_function_is_always_available():
    """It used to be None without MDAnalysis, which silently turned
    unwrap_centers_mic into a no-op and made Rg all-NaN. CoAnaPure ships its
    own, so there is no configuration in which this is unavailable."""
    assert _get_minimize_vectors() is not None


def test_unwrapping_follows_a_point_across_the_boundary():
    """The wrapped x jumps 99 -> 1; the unwrapped one must continue 99 -> 101."""
    wrapped, true, boxes = _straddling_dimer()
    unw = unwrap_centers_mic(wrapped[:, 0, :].copy(), boxes=boxes)

    assert not np.allclose(unw, wrapped[:, 0, :]), (
        "unwrapping returned its input unchanged, so no correction was applied")
    # displacements must be the true constant drift, never a box-sized jump
    step = np.diff(unw[:, 0])
    assert np.allclose(step, DRIFT), step
    # and the path must match the truth up to the starting image
    assert np.allclose(unw - unw[0], true[:, 0, :] - true[0, 0, :])


def test_member_msd_matches_the_analytic_drift():
    wrapped, _true, boxes = _straddling_dimer()
    unw = unwrap_centers_mic(wrapped[:, 0, :].copy(), boxes=boxes)
    lags, msd = msd_from_positions(unw, max_lag=T - 1)
    assert np.allclose(msd, (DRIFT * lags) ** 2)


def test_oligomer_centre_msd_is_not_corrupted_by_the_boundary():
    """The centre must come from unwrapped members, not from averaging wrapped
    ones. Averaging first put the 'centre' mid-box -- a point the dimer never
    occupies -- giving MSD 125 at lag 1 where the truth is 4."""
    wrapped, _true, boxes = _straddling_dimer()
    cache = _cache(wrapped, boxes)
    pathway = Pathway(
        label="p0", start_idx=0, start_frame=0, duration=T,
        sizes=[2] * T, sets=[frozenset({0, 1})] * T,
    )
    out = msd_pathway_peak_members(cache, pathway, max_lag=T - 1)
    lags = out["lags"]
    expected = (DRIFT * lags) ** 2

    assert np.allclose(out["msd_oligo"], expected), (
        f"oligomer MSD {out['msd_oligo'][:4]} != analytic {expected[:4]}")
    # a rigid dimer: every member moves exactly as the centre does
    assert np.allclose(out["msd_mean"], expected)
    assert np.allclose(out["msd_members"][0], expected)


def test_radius_of_gyration_is_available_without_mdanalysis():
    """Rg returned an all-NaN trace whenever MDAnalysis was missing."""
    wrapped, _true, boxes = _straddling_dimer()
    cache = _cache(wrapped, boxes)
    pathway = Pathway(
        label="p0", start_idx=0, start_frame=0, duration=T,
        sizes=[2] * T, sets=[frozenset({0, 1})] * T,
    )
    rg = compactness_timeseries(cache, pathway, inter_thresh=1, mode="rg")
    assert np.all(np.isfinite(rg)), "Rg trace is NaN, so no PBC function was found"
    # two points 2 A apart about their midpoint: Rg = 1.0 in every frame
    assert np.allclose(rg, 1.0), rg


@pytest.mark.parametrize("mode", ["density", "degree"])
def test_contact_compactness_is_unaffected_by_the_pbc_change(mode):
    """A regression guard: the contact-based modes never used the box."""
    wrapped, _true, boxes = _straddling_dimer()
    cache = _cache(wrapped, boxes)
    pathway = Pathway(
        label="p0", start_idx=0, start_frame=0, duration=T,
        sizes=[2] * T, sets=[frozenset({0, 1})] * T,
    )
    y = compactness_timeseries(cache, pathway, inter_thresh=1, mode=mode)
    # one edge between two nodes: density = 2E/(n(n-1)) = 1, degree = 2E/n = 1
    assert np.allclose(y, 1.0), y


# ---------------------------------------------------------------------------
# the free-energy profile's uncertainty
# ---------------------------------------------------------------------------

def test_free_energy_range_is_bounded_by_counting_statistics():
    """The profile's vertical extent is exactly ln(c_max/c_min), so with any
    size seen only once the range is ln(c_max) -- set by dwell time, not by
    thermodynamics. Pinning the identity makes that impossible to forget."""
    from oligomer_analysis.metrics import compute_free_energy_from_sizes

    # 9 frames at size 68, then one frame each at several other sizes
    seq = [68] * 9 + [57, 59, 63, 67, 71, 74]
    prof = compute_free_energy_from_sizes(seq)
    rng = prof.F_over_kBT.max() - prof.F_over_kBT.min()
    assert rng == pytest.approx(np.log(prof.counts.max() / prof.counts.min()))
    assert rng == pytest.approx(np.log(9.0))


def test_a_one_frame_bin_carries_a_one_kT_error_bar():
    """1/sqrt(count) for an uncorrelated trace; that is the same magnitude as
    the features these profiles get read for."""
    from oligomer_analysis.metrics import compute_free_energy_from_sizes

    rng = np.random.default_rng(0)
    seq = rng.integers(1, 40, size=400)          # fast-moving, so tau ~ 0.5
    prof = compute_free_energy_from_sizes(seq)
    assert prof.F_stderr is not None
    single = prof.counts == 1
    if single.any():
        assert np.allclose(prof.F_stderr[single], 1.0)
    assert np.allclose(prof.F_stderr, 1.0 / np.sqrt(prof.counts), rtol=0.5)


def test_a_sticky_trace_gets_a_wider_error_bar_than_a_fast_one():
    """Correlated frames are not independent samples."""
    from oligomer_analysis.metrics import compute_free_energy_from_sizes

    fast = np.tile([10, 20, 30, 40], 50)                  # tau ~ 0.5
    sticky = np.repeat([10, 20, 30, 40], 50)              # one long dwell each
    p_fast = compute_free_energy_from_sizes(fast)
    p_sticky = compute_free_energy_from_sizes(sticky)
    assert p_sticky.autocorr_time > p_fast.autocorr_time
    assert p_sticky.F_stderr.mean() > p_fast.F_stderr.mean()


def test_min_count_drops_bins_that_cannot_support_an_estimate():
    from oligomer_analysis.metrics import compute_free_energy_from_sizes

    seq = [68] * 9 + [57, 59, 63]
    full = compute_free_energy_from_sizes(seq)
    cut = compute_free_energy_from_sizes(seq, min_count=5)
    assert len(cut.sizes) < len(full.sizes)
    assert set(cut.sizes) == {68}
    # probabilities stay normalised over everything observed, so the surviving
    # F values are not shifted by the filtering
    assert cut.F_over_kBT[0] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="min_count"):
        compute_free_energy_from_sizes(seq, min_count=100)


# ---------------------------------------------------------------------------
# found by an external review of the fixes above
# ---------------------------------------------------------------------------

def test_the_cluster_gather_does_not_depend_on_member_order():
    """Referencing every member to members[0] is order-dependent once the
    cluster spans half the box: collinear members at x = 5, 35, 65 in a 100 A
    box gave Rg 28.674 from member 0 and 24.495 from the middle one. Propagating
    along the contact graph gives one answer."""
    from oligomer_analysis.metrics import gather_cluster_images

    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
    pts = np.array([[5.0, 0, 0], [35.0, 0, 0], [65.0, 0, 0]])
    chain = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)

    answers = set()
    for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2], [2, 0, 1]):
        order = np.array(order)
        g = gather_cluster_images(pts[order], chain[np.ix_(order, order)], box)
        rg = float(np.sqrt(((g - g.mean(axis=0)) ** 2).sum(axis=1).mean()))
        answers.add(round(rg, 6))
    assert len(answers) == 1, f"order-dependent: {sorted(answers)}"
    assert answers.pop() == pytest.approx(np.sqrt(600.0), abs=1e-6)


def test_the_gather_handles_a_disconnected_member_set():
    """Two separate pairs must each be gathered, not left wrapped."""
    from oligomer_analysis.metrics import gather_cluster_images

    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
    pts = np.array([[99.0, 0, 0], [1.0, 0, 0], [49.0, 0, 0], [51.0, 0, 0]])
    adj = np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                    [0, 0, 0, 1], [0, 0, 1, 0]], dtype=bool)
    g = gather_cluster_images(pts, adj, box)
    assert abs(g[1, 0] - g[0, 0]) == pytest.approx(2.0)   # across the boundary
    assert abs(g[3, 0] - g[2, 0]) == pytest.approx(2.0)   # already together


def test_rg_without_mic_works_when_the_cache_has_no_boxes():
    """Reading the box before testing rg_use_mic broke a path that used to work."""
    wrapped, _true, _boxes = _straddling_dimer()
    cmaps = np.zeros((T, 2, 2), dtype=np.int32)
    cmaps[:, 0, 1] = cmaps[:, 1, 0] = 5
    cache = ContactCache(
        frames=np.arange(T), cmaps=cmaps,
        nintra=np.zeros(T), ninter=np.zeros(T), ntotal=np.zeros(T),
        n_free=np.zeros(T), n_bound=np.full(T, 2),
        seg_centers=wrapped, boxes=None, meta={},
    )
    pathway = Pathway(label="p0", start_idx=0, start_frame=0, duration=T,
                      sizes=[2] * T, sets=[frozenset({0, 1})] * T)
    y = compactness_timeseries(cache, pathway, inter_thresh=1, mode="rg",
                               rg_use_mic=False)
    assert np.all(np.isfinite(y))
