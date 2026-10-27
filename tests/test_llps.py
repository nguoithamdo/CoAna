# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""The dense/dilute phase observable.

The regression these guard against: ``dense_phase_fraction_timeseries`` used to
return exactly 1.000 for every frame of every trajectory, so the LLPS figure
showed a flat line whatever the system did. Three independent causes, each of
which on its own is enough to destroy the signal:

* the raw count grid has no dynamic range -- at any resolution fine enough to
  locate a condensate almost every occupied cell holds exactly one chain, so a
  quantile of those counts is 1 and ``density >= 1`` means "occupied";
* the cut was calibrated on the distribution over *cells*, but applied to
  *chains* -- chains fill a tiny fraction of the volume, so every cell holding
  one is already in the top few percent;
* the threshold, and the grid itself, were re-fitted to each frame, so both
  moved with the data and densification was normalised away.
"""

from __future__ import annotations

import numpy as np
import pytest

from oligomer_analysis.llps import (
    compute_density_grid_for_frame,
    dense_phase_fraction_timeseries,
    density_at_centers,
)
from oligomer_analysis.models import ContactCache

N, L = 128, 200.0


def _cache(centers):
    t = centers.shape[0]
    return ContactCache(
        frames=np.arange(t), cmaps=np.zeros((t, N, N), dtype=np.int32),
        nintra=np.zeros(t), ninter=np.zeros(t), ntotal=np.zeros(t),
        n_free=np.zeros(t), n_bound=np.zeros(t),
        seg_centers=centers, boxes=np.zeros((t, 6)), meta={},
    )


def _collapsing(n_frames=12, seed=0):
    """Chains interpolating from filling the box to one tight blob.

    The dense-phase fraction must rise. Anything flat is an artefact.
    """
    rng = np.random.default_rng(seed)
    uniform = rng.random((N, 3)) * L
    blob = np.array([100.0, 100.0, 100.0])
    centers = np.empty((n_frames, N, 3))
    for t in range(n_frames):
        f = t / (n_frames - 1)
        tight = blob + rng.normal(scale=8.0, size=(N, 3))
        centers[t] = (1 - f) * uniform + f * tight
    truth = np.array([
        float(np.mean(np.linalg.norm(centers[t] - blob, axis=1) < 30.0))
        for t in range(n_frames)
    ])
    return centers, truth


@pytest.fixture(scope="module")
def growth_lineages_llps():
    """A collapsing system plus its lineage decomposition.

    The chains are given contacts whenever their centres are close, so the
    lineage tracker sees a real assembly rather than an empty contact map.
    """
    from oligomer_analysis import TargetStateConfig, build_lineages_from_cache

    centers, _truth = _collapsing(n_frames=10, seed=3)
    t = centers.shape[0]
    cmaps = np.zeros((t, N, N), dtype=np.int32)
    for f in range(t):
        d = np.linalg.norm(centers[f][:, None, :] - centers[f][None, :, :], axis=-1)
        near = (d < 25.0) & ~np.eye(N, dtype=bool)
        cmaps[f][near] = 3
    cache = ContactCache(
        frames=np.arange(t), cmaps=cmaps,
        nintra=np.zeros(t), ninter=np.zeros(t), ntotal=np.zeros(t),
        n_free=np.zeros(t), n_bound=np.zeros(t),
        seg_centers=centers, boxes=np.zeros((t, 6)), meta={},
    )
    result = build_lineages_from_cache(
        cache, inter_thresh=1, cluster_min_size=2,
        target_config=TargetStateConfig(min_size=5, min_duration=2,
                                        min_contribution=0.7),
    )
    return cache, result


def test_dense_fraction_is_not_a_constant():
    """The whole defect in one assertion."""
    centers, _truth = _collapsing()
    ts = dense_phase_fraction_timeseries(_cache(centers))
    frac = ts["dense_fraction"].to_numpy()
    assert frac.std() > 0.05, f"dense_fraction is ~constant: {frac}"
    assert not np.allclose(frac, 1.0), "every chain labelled dense in every frame"


def test_dense_fraction_tracks_a_collapse():
    centers, truth = _collapsing()
    ts = dense_phase_fraction_timeseries(_cache(centers))
    frac = ts["dense_fraction"].to_numpy()

    assert frac[0] < 0.35, f"started at {frac[0]:.3f}; the uniform state is dilute"
    assert frac[-1] > 0.90, f"ended at {frac[-1]:.3f}; the collapsed state is dense"
    assert np.corrcoef(frac, truth)[0, 1] > 0.6, "does not follow the collapse"


def test_one_threshold_and_one_grid_are_used_for_the_whole_series():
    """Both have to be frame-independent, or the observable normalises away the
    change it exists to measure."""
    centers, _truth = _collapsing()
    ts = dense_phase_fraction_timeseries(_cache(centers))
    assert ts["dense_threshold"].nunique() == 1, "threshold moved between frames"


def test_an_explicit_threshold_is_honoured():
    """The number to quote in a methods section, and to hold fixed across runs."""
    centers, _truth = _collapsing()
    ts = dense_phase_fraction_timeseries(_cache(centers), dense_threshold=0.05)
    assert np.allclose(ts["dense_threshold"], 0.05)


@pytest.mark.parametrize("quantile,reference", [(0.80, 0.25), (0.85, 0.10),
                                                (0.90, 0.25), (0.80, 0.50)])
def test_the_reference_window_calibrates_to_one_minus_the_quantile(quantile, reference):
    """The exact property of a pooled-quantile calibration: averaged over the
    reference window, 1 - dense_quantile of chains are dense-phase.

    This holds for the window mean, not for its first frame -- the threshold is
    pooled over the whole window, and a single frame need not be representative
    of the pool. Growth above this baseline is the signal the series carries.
    """
    centers, _truth = _collapsing()
    n_frames = centers.shape[0]
    ts = dense_phase_fraction_timeseries(
        _cache(centers), dense_quantile=quantile, reference_fraction=reference)
    n_ref = max(1, int(round(n_frames * reference)))
    window_mean = ts["dense_fraction"].to_numpy()[:n_ref].mean()

    # NOT an exact equality, and the earlier +/-0.02 tolerance hid that. The cut
    # is an interpolated quantile of a finite sample and the comparison is
    # inclusive (>=), so the realised fraction is an order statistic: with 128
    # chains and q=0.85 it selects 20 of them, 0.15625 rather than 0.15. The
    # bound below is the resolution of one chain, which is the most this
    # construction can promise.
    one_chain = 1.0 / N
    assert window_mean == pytest.approx(1.0 - quantile, abs=2 * one_chain), (
        f"q={quantile} ref={reference}: window mean {window_mean:.5f}, "
        f"expected {1.0 - quantile:.5f} to within {2 * one_chain:.5f}")
    # and it is biased high, never low, because ties and interpolation can only
    # add chains at the boundary
    assert window_mean >= (1.0 - quantile) - one_chain


def test_smoothing_gives_the_count_grid_dynamic_range():
    """Unsmoothed, the occupied cells are all 1s and no cut can separate them."""
    centers, _truth = _collapsing()
    cache = _cache(centers)
    raw = compute_density_grid_for_frame(cache, 0, smoothing_cells=0.0)
    occupied = raw.density[raw.density > 0]
    assert set(np.unique(occupied.astype(int))) <= {1, 2}, (
        "expected a near-constant count field without smoothing")

    smooth = compute_density_grid_for_frame(cache, 0, smoothing_cells=1.5)
    vals = density_at_centers(centers[0], smooth)
    assert vals.std() > 0, "smoothed density has no variation between chains"
    assert np.unique(vals).size > N // 4, "smoothed density is still quantised"


def test_density_at_centers_matches_the_mask_used_for_labelling():
    """The helper must sample the same cells the classifier reads."""
    from oligomer_analysis.llps import classify_centers_by_density

    centers, _truth = _collapsing()
    cache = _cache(centers)
    df = compute_density_grid_for_frame(cache, 5, dense_threshold=0.05)
    vals = density_at_centers(centers[5], df)
    labels = classify_centers_by_density(centers[5], df)
    assert np.array_equal(labels, vals >= df.dense_threshold)


def test_a_flat_series_warns_rather_than_passing_as_a_result():
    """A constant fraction was the old bug's signature; it must announce itself."""
    static = np.repeat((np.random.default_rng(1).random((1, N, 3)) * L), 6, axis=0)
    with pytest.warns(UserWarning, match="exactly constant"):
        dense_phase_fraction_timeseries(_cache(static))


def test_an_out_of_box_centre_reads_the_cell_it_filled():
    """The centres were wrapped before histogramming but not when sampled, so a
    centre outside the cell read a different cell. A chain at x=105 in a 100 A
    box fills the x=5 cell and used to read an empty one."""
    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
    centers = np.array([[[105.0, 50.0, 50.0], [50.0, 50.0, 50.0]]])
    cache = ContactCache(
        frames=np.arange(1), cmaps=np.zeros((1, 2, 2), dtype=np.int32),
        nintra=np.zeros(1), ninter=np.zeros(1), ntotal=np.zeros(1),
        n_free=np.zeros(1), n_bound=np.zeros(1),
        seg_centers=centers, boxes=np.tile(box, (1, 1)), meta={},
    )
    df = compute_density_grid_for_frame(cache, 0, grid_shape=(10, 10, 10),
                                        smoothing_cells=0.0)
    vals = density_at_centers(centers[0], df)
    assert vals[0] == pytest.approx(1.0), (
        "the out-of-box centre read an empty cell, not the one it filled")
    assert df.density.sum() == pytest.approx(2.0), "a centre was dropped"


def test_the_classifier_and_the_sampler_cannot_disagree():
    """They used to duplicate the cell indexing, so a test comparing them passed
    while both read the wrong cell. The classifier now delegates."""
    from oligomer_analysis.llps import classify_centers_by_density

    centers, _truth = _collapsing()
    cache = _cache(centers)
    df = compute_density_grid_for_frame(cache, 5, dense_threshold=0.05)
    vals = density_at_centers(centers[5], df)
    assert np.array_equal(classify_centers_by_density(centers[5], df),
                          vals >= df.dense_threshold)


def test_a_nan_centre_does_not_land_in_an_edge_cell():
    """Clamping used to file an unrepresentable coordinate in a real cell."""
    centers = np.array([[[10.0, 10.0, 10.0], [np.nan, 10.0, 10.0]]])
    cache = ContactCache(
        frames=np.arange(1), cmaps=np.zeros((1, 2, 2), dtype=np.int32),
        nintra=np.zeros(1), ninter=np.zeros(1), ntotal=np.zeros(1),
        n_free=np.zeros(1), n_bound=np.zeros(1),
        seg_centers=centers, boxes=np.zeros((1, 6)), meta={},
    )
    df = compute_density_grid_for_frame(cache, 0, grid_shape=(8, 8, 8))
    vals = density_at_centers(centers[0], df)
    assert np.isnan(vals[1]), "a NaN centre was given a real density"


def test_both_consumers_share_one_dense_phase_definition(growth_lineages_llps):
    """figure_llps_coupling draws a time series and a per-lineage boxplot. They
    used different rules: the series used the calibrated threshold while the
    table rebuilt a per-frame cell quantile, which labelled every chain dense
    and made the boxplot three identical distributions of 1.0."""
    from oligomer_analysis.llps import (
        calibrate_dense_phase, dense_phase_fraction_timeseries,
        lineage_dense_phase_table,
    )

    cache, result = growth_lineages_llps
    cal = calibrate_dense_phase(cache)
    ts = dense_phase_fraction_timeseries(cache, calibration=cal)
    tab = lineage_dense_phase_table(cache, result, calibration=cal)

    assert np.allclose(ts["dense_threshold"], cal.dense_threshold)
    vals = tab["mean_dense_fraction_of_members"].dropna().to_numpy()
    assert vals.size > 0
    assert not np.allclose(vals, 1.0), (
        "every segment labelled fully dense -- the old degenerate definition")


def test_the_calibration_reports_physical_lengths():
    """A bandwidth in cells is a different physical width in every box, and on
    an elongated box it is anisotropic. The kernel must be isotropic in A."""
    from oligomer_analysis.llps import calibrate_dense_phase

    # a deliberately elongated box: 100 x 100 x 400
    centers = np.random.default_rng(0).random((4, N, 3)) * np.array([100.0, 100.0, 400.0])
    boxes = np.tile(np.array([100.0, 100.0, 400.0, 90.0, 90.0, 90.0]), (4, 1))
    cache = ContactCache(
        frames=np.arange(4), cmaps=np.zeros((4, N, N), dtype=np.int32),
        nintra=np.zeros(4), ninter=np.zeros(4), ntotal=np.zeros(4),
        n_free=np.zeros(4), n_bound=np.zeros(4),
        seg_centers=centers, boxes=boxes, meta={},
    )
    cal = calibrate_dense_phase(cache, grid_shape=(10, 10, 40), smoothing_length=20.0)
    assert np.allclose(cal.cell_lengths, [10.0, 10.0, 10.0])
    assert np.allclose(cal.smoothing_length, 20.0), cal.smoothing_length

    # with anisotropic cells the per-axis sigma in CELLS must differ so that the
    # physical width does not
    cal2 = calibrate_dense_phase(cache, grid_shape=(10, 10, 10), smoothing_length=20.0)
    assert np.allclose(cal2.cell_lengths, [10.0, 10.0, 40.0])
    assert not np.allclose(cal2.smoothing_cells[0], cal2.smoothing_cells[2])
    assert np.allclose(cal2.smoothing_length, 20.0)
    assert "A" in cal2.describe()


def test_a_quantile_calibrated_table_is_marked_relative():
    """The column records whether the baseline was constructed or measured."""
    from oligomer_analysis.llps import dense_phase_fraction_timeseries

    centers, _truth = _collapsing()
    rel = dense_phase_fraction_timeseries(_cache(centers))
    assert bool(rel["relative_to_reference"].iloc[0]) is True
    absolute = dense_phase_fraction_timeseries(_cache(centers), dense_threshold=0.05)
    assert bool(absolute["relative_to_reference"].iloc[0]) is False


def test_dense_phase_figure_draws_on_current_matplotlib(growth_lineages_llps):
    """3.7.0: ``boxplot(labels=...)`` raised TypeError on matplotlib >= 3.11."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from oligomer_analysis.publication_figures import figure_llps_coupling

    cache, result = growth_lineages_llps
    fig, axes, _ts, _tbl = figure_llps_coupling(cache, result)
    assert [t.get_text() for t in axes[1].get_xticklabels()] == [
        "on-pathway", "uncertain", "off-pathway"
    ]
    plt.close(fig)
