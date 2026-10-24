from __future__ import annotations

"""Minimal LLPS observables for aggregation trajectories.

The oligomer lineage analysis answers *which assemblies progress*.  LLPS analysis
asks *where* and *in what phase environment* those assemblies progress.  This
module provides lightweight observables that work directly from cached segment
centers, without requiring the original trajectory.

These tools are deliberately conservative: they give operational dense/dilute
phase labels that should be validated against visual inspection and, when
available, concentration-dependent replicate simulations.
"""

from dataclasses import dataclass
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .models import ContactCache, LineageTrackingResult


@dataclass(slots=True)
class DensityFrame:
    # NOTE: ``origin``/``span``/``periodic`` describe the coordinate transform
    # that was applied before histogramming. Any function that samples this
    # grid must apply the SAME transform, or it reads a different cell than the
    # one a point filled. See :func:`_to_grid_coords`.
    frame: int
    density: np.ndarray
    edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    dense_mask: np.ndarray
    dense_threshold: float
    box_lengths: np.ndarray
    # The coordinate transform applied before histogramming. Anything that
    # samples this grid must apply the SAME transform via _to_grid_coords, or
    # it reads a different cell than the one a point filled.
    origin: Optional[np.ndarray] = None
    span: Optional[np.ndarray] = None
    periodic: bool = False


def _smooth_density(density: np.ndarray, sigma_cells, periodic: bool) -> np.ndarray:
    """Blur a sparse count grid into a continuous local-density field.

    A histogram of chain centres is a set of unit spikes: at any resolution fine
    enough to locate a condensate, nearly every occupied cell holds exactly one
    chain, so the field has no contrast and no threshold can separate dense from
    dilute. Blurring over a neighbourhood makes the value at a cell reflect how
    crowded its surroundings are, which is what "dense phase" means.

    ``mode="wrap"`` is correct when the grid spans a periodic box, so a
    condensate sitting on a face is not cut in half.
    """
    sig = np.atleast_1d(np.asarray(sigma_cells, dtype=float))
    if not np.any(sig > 0):
        return density
    from scipy.ndimage import gaussian_filter

    # A sequence gives a different sigma per axis, which is what an anisotropic
    # grid needs: equal sigma in CELLS on a 202 x 202 x 607 A box is a kernel
    # three times wider along z in Angstroms.
    sigma = float(sig[0]) if sig.size == 1 else tuple(float(x) for x in sig)
    return gaussian_filter(
        density.astype(float),
        sigma=sigma,
        mode="wrap" if periodic else "nearest",
    )


def _to_grid_coords(
    centers: np.ndarray,
    origin: Optional[np.ndarray],
    span: Optional[np.ndarray],
    periodic: bool,
) -> np.ndarray:
    """Map coordinates into the grid's own frame, the one way for all callers.

    When the grid spans a periodic cell, a coordinate outside it is folded back
    *relative to the grid origin* -- not to ``[0, L)``, which is wrong whenever
    the origin is not zero. Folding here rather than clamping at the lookup is
    what keeps the cell a point fills and the cell it reads the same cell;
    clamping instead silently files an out-of-range point in an unrelated edge
    cell.
    """
    c = np.asarray(centers, dtype=float)
    if periodic and origin is not None and span is not None:
        with np.errstate(invalid="ignore"):
            c = origin + np.mod(c - origin, span)
    return c


def _valid_box_lengths(box6) -> Optional[np.ndarray]:
    box = np.asarray(box6, dtype=float)
    if box.shape == (6,) and np.all(box[:3] > 0):
        return box[:3].astype(float)
    return None


def compute_density_grid_for_frame(
    cache: ContactCache,
    frame_index: int,
    *,
    grid_shape=(24, 24, 24),
    dense_quantile: float = 0.85,
    smoothing_cells: float = 1.5,
    dense_threshold: Optional[float] = None,
    grid_range: Optional[np.ndarray] = None,
) -> DensityFrame:
    """Compute a coarse density grid from segment centers for one frame.

    For Martini proteins where each segment is one protein chain, this grid is a
    protein-center density field.  It is appropriate for detecting dense regions
    and condensate-like clusters, but not for atomistic solvent density.

    The raw histogram is smoothed over ``smoothing_cells`` grid cells before any
    threshold is applied.  Without that, the field is a set of unit spikes --
    with 128 chains in a 24^3 grid essentially every occupied cell holds exactly
    one chain -- the quantile of those counts is 1, ``density >= 1`` selects
    every occupied cell, and since every chain sits in an occupied cell the
    dense fraction is identically 1.0 for every frame of every trajectory.

    ``dense_threshold`` sets the cut in smoothed-density units.  When it is
    None the cut is this frame's own ``dense_quantile``, which makes the label
    *relative to this frame* -- useful for locating a condensate, useless for
    asking whether the system densified, because the threshold moves with the
    data.  :func:`dense_phase_fraction_timeseries` therefore derives one
    threshold and passes it in for every frame.
    """
    centers = np.asarray(cache.seg_centers[frame_index], dtype=float)
    box_lengths = _valid_box_lengths(cache.boxes[frame_index])

    if grid_range is not None:
        # An explicit, frame-independent grid. Required for any comparison
        # across time: a grid re-fitted to each frame's own extent shrinks with
        # a collapsing system, so each cell keeps roughly its old occupancy and
        # the density is scale-free.
        gr = np.asarray(grid_range, dtype=float).reshape(3, 2)
        origin = gr[:, 0].copy()
        span = gr[:, 1] - gr[:, 0]
        # Treat the supplied range as periodic only when it really is one
        # periodic cell of the cache's box; otherwise folding into it would
        # invent wrap-around that the box does not have.
        periodic = bool(
            box_lengths is not None and np.allclose(span, box_lengths, rtol=1e-6)
        )
    elif box_lengths is None:
        mins = np.nanmin(centers, axis=0)
        maxs = np.nanmax(centers, axis=0)
        origin = mins
        span = np.maximum(maxs - mins, 1.0)
        periodic = False
    else:
        origin = np.zeros(3, dtype=float)
        span = np.asarray(box_lengths, dtype=float)
        periodic = True

    centers = _to_grid_coords(centers, origin, span, periodic)
    ranges = [(float(origin[d]), float(origin[d] + span[d])) for d in range(3)]

    density, edges = np.histogramdd(centers, bins=grid_shape, range=ranges)
    density = _smooth_density(
        density, smoothing_cells, periodic=_valid_box_lengths(cache.boxes[frame_index]) is not None
    )
    if dense_threshold is not None:
        threshold = float(dense_threshold)
    else:
        positive = density[density > 0]
        threshold = (
            float(np.quantile(positive, dense_quantile)) if positive.size else np.inf
        )
    dense_mask = density >= threshold if np.isfinite(threshold) else np.zeros_like(density, dtype=bool)
    return DensityFrame(
        frame=int(cache.frames[frame_index]),
        density=density,
        edges=(edges[0], edges[1], edges[2]),
        dense_mask=dense_mask,
        dense_threshold=threshold,
        box_lengths=box_lengths,
        origin=origin,
        span=span,
        periodic=periodic,
    )


def density_at_centers(centers: np.ndarray, density_frame: DensityFrame) -> np.ndarray:
    """The smoothed density in the cell each centre occupies, one value per centre.

    This is the local crowding a chain experiences, and it is the distribution a
    dense/dilute cut has to be calibrated on. Calibrating on the distribution
    over *cells* instead does not work: the chains fill a tiny fraction of the
    volume, so every cell holding a chain already sits in the top few percent of
    cells and the cut selects all of them.
    """
    centers = _to_grid_coords(
        centers,
        getattr(density_frame, "origin", None),
        getattr(density_frame, "span", None),
        bool(getattr(density_frame, "periodic", False)),
    )
    edges = density_frame.edges
    out = np.zeros(centers.shape[0], dtype=float)
    for n, xyz in enumerate(centers):
        if not np.all(np.isfinite(xyz)):
            out[n] = np.nan
            continue
        idx = []
        for d in range(3):
            k = int(np.searchsorted(edges[d], xyz[d], side="right") - 1)
            k = max(0, min(k, density_frame.density.shape[d] - 1))
            idx.append(k)
        out[n] = float(density_frame.density[tuple(idx)])
    return out


def classify_centers_by_density(centers: np.ndarray, density_frame: DensityFrame) -> np.ndarray:
    """Classify centers as dense-phase (True) or dilute-phase (False)."""
    # Delegate, so the label and the sampled value can never disagree about
    # which cell a centre belongs to. They used to duplicate this indexing, and
    # a test that compared them passed while both read the wrong cell.
    vals = density_at_centers(centers, density_frame)
    return np.asarray(vals >= density_frame.dense_threshold, dtype=bool)


@dataclass(slots=True)
class DensePhaseCalibration:
    """One dense/dilute rule, fixed once and reused by every consumer.

    A dense-phase *fraction* only means something against a fixed rule. If the
    threshold, the grid or the kernel is re-derived per frame the measurement
    normalises away the change it exists to detect; if it is re-derived per run,
    two runs cannot be compared. This object carries the whole rule so it can be
    computed once, passed around, and reported.

    Every field belongs in a methods section. ``cell_lengths`` and
    ``smoothing_length`` are the physical sizes in Angstroms, which are the
    numbers that make the choice reproducible -- a bandwidth in *cells* means a
    different physical width in every box.
    """

    dense_threshold: float
    grid_shape: tuple
    grid_range: Optional[np.ndarray]
    smoothing_cells: np.ndarray
    cell_lengths: np.ndarray
    smoothing_length: np.ndarray
    dense_quantile: Optional[float]
    n_reference_frames: int

    def describe(self) -> str:
        cl = [round(float(x), 2) for x in np.atleast_1d(self.cell_lengths)]
        sl = [round(float(x), 2) for x in np.atleast_1d(self.smoothing_length)]
        how = (
            f"quantile {self.dense_quantile} of the local density at chain "
            f"positions over {self.n_reference_frames} reference frame(s)"
            if self.dense_quantile is not None
            else "supplied directly"
        )
        return (
            f"dense_threshold={self.dense_threshold:.6g} (chains per cell), "
            f"grid {tuple(self.grid_shape)} with cells {cl} A, "
            f"Gaussian bandwidth {sl} A, threshold {how}"
        )


def calibrate_dense_phase(
    cache: ContactCache,
    *,
    grid_shape=(24, 24, 24),
    dense_quantile: float = 0.85,
    smoothing_cells: Optional[float] = None,
    smoothing_length: Optional[float] = None,
    dense_threshold: Optional[float] = None,
    reference_fraction: float = 0.10,
) -> DensePhaseCalibration:
    """Derive one dense/dilute rule for a whole trajectory.

    ``smoothing_length`` is the Gaussian bandwidth in Angstroms and is the one
    to prefer: the same bandwidth in *cells* is a different physical width in
    every box, and on an elongated box it is anisotropic. It defaults to twice
    the mean cell edge, which is the scale over which a chain's neighbourhood is
    being averaged. ``smoothing_cells`` overrides it for backward compatibility.

    ``dense_threshold`` fixes the cut outright, which is the only way to compare
    runs: a threshold calibrated from each run's own opening frames measures
    change *within* that run and nothing between runs.
    """
    n_frames = len(cache.frames)
    if n_frames == 0:
        raise ValueError("cannot calibrate on an empty cache")
    if not 0.0 < reference_fraction <= 1.0:
        raise ValueError("reference_fraction must be in (0, 1]")

    # One grid for the run. With a box it is already frame-independent;
    # without one, use the union extent so a collapsing system is measured
    # against fixed space rather than a grid that shrinks with it.
    if _valid_box_lengths(cache.boxes[0]) is None:
        allc = np.asarray(cache.seg_centers, dtype=float).reshape(-1, 3)
        mins, maxs = np.nanmin(allc, axis=0), np.nanmax(allc, axis=0)
        span = np.maximum(maxs - mins, 1.0)
        grid_range = np.stack([mins, mins + span], axis=1)
        extent = span
    else:
        grid_range = None
        extent = np.asarray(_valid_box_lengths(cache.boxes[0]), dtype=float)

    shape = np.asarray(grid_shape, dtype=float)
    cell_lengths = extent / shape

    if smoothing_cells is not None:
        sig_cells = np.broadcast_to(np.asarray(smoothing_cells, dtype=float), (3,)).copy()
    else:
        if smoothing_length is None:
            smoothing_length = 2.0 * float(np.mean(cell_lengths))
        sig_cells = np.asarray(smoothing_length, dtype=float) / cell_lengths
    sig_length = sig_cells * cell_lengths

    n_ref = max(1, int(round(n_frames * reference_fraction)))
    quantile_used: Optional[float] = None
    if dense_threshold is None:
        quantile_used = float(dense_quantile)
        pooled = []
        for i in range(n_ref):
            ref = compute_density_grid_for_frame(
                cache, i, grid_shape=grid_shape, smoothing_cells=sig_cells,
                grid_range=grid_range, dense_threshold=np.inf,
            )
            vals = density_at_centers(cache.seg_centers[i], ref)
            pooled.append(vals[np.isfinite(vals)])
        pooled_arr = np.concatenate(pooled) if pooled else np.empty(0)
        dense_threshold = (
            float(np.quantile(pooled_arr, dense_quantile)) if pooled_arr.size else np.inf
        )
    else:
        n_ref = 0

    return DensePhaseCalibration(
        dense_threshold=float(dense_threshold),
        grid_shape=tuple(int(x) for x in grid_shape),
        grid_range=grid_range,
        smoothing_cells=sig_cells,
        cell_lengths=cell_lengths,
        smoothing_length=sig_length,
        dense_quantile=quantile_used,
        n_reference_frames=int(n_ref),
    )


def dense_phase_fraction_timeseries(
    cache: ContactCache,
    *,
    grid_shape=(24, 24, 24),
    dense_quantile: float = 0.85,
    smoothing_cells: Optional[float] = None,
    smoothing_length: Optional[float] = None,
    dense_threshold: Optional[float] = None,
    reference_fraction: float = 0.10,
    calibration: Optional[DensePhaseCalibration] = None,
) -> pd.DataFrame:
    """Fraction of protein chains located in dense grid cells over time.

    One threshold is used for the whole series, so the fraction can actually
    move.  A per-frame quantile cannot: it rises with the data, and a system
    collapsing from uniform into a single condensate reports the same number
    throughout.  Pass ``dense_threshold`` to set the cut explicitly in smoothed
    chains-per-cell; otherwise it is the ``dense_quantile`` of the local density
    *experienced by the chains* over the first ``reference_fraction`` of frames.
    So the default calibration says: a chain is dense-phase when it is more
    crowded than ``dense_quantile`` of chains were at the start of the run, and
    ``1 - dense_quantile`` of chains are dense-phase in the reference window by
    construction.  Growth above that is the signal.

    ``dense_threshold`` is the number to report in a methods section, and the
    one to keep fixed when comparing runs -- a threshold calibrated separately
    per run makes two runs incomparable.

    When the cut comes from ``dense_quantile`` rather than being supplied, the
    result is a *relative* high-density occupancy, not an absolute dense-phase
    fraction: roughly ``1 - dense_quantile`` of the reference observations are
    dense by construction, whether that reference state is dilute, already
    condensed, or simply different from another run's. Only the change away
    from that baseline carries information, absolute values must not be
    compared between runs, and the ``relative_to_reference`` column records
    which of the two a given table is.
    """
    n_frames = len(cache.frames)
    if n_frames == 0:
        return pd.DataFrame(
            columns=["frame", "dense_fraction", "n_dense", "n_total", "dense_threshold"]
        )

    cal = calibration or calibrate_dense_phase(
        cache, grid_shape=grid_shape, dense_quantile=dense_quantile,
        smoothing_cells=smoothing_cells, smoothing_length=smoothing_length,
        dense_threshold=dense_threshold, reference_fraction=reference_fraction,
    )

    rows = []
    for i in range(n_frames):
        df = compute_density_grid_for_frame(
            cache, i, grid_shape=cal.grid_shape,
            smoothing_cells=cal.smoothing_cells,
            dense_threshold=cal.dense_threshold, grid_range=cal.grid_range,
        )
        labels = classify_centers_by_density(cache.seg_centers[i], df)
        rows.append({
            "frame": int(cache.frames[i]),
            "dense_fraction": float(labels.mean()) if len(labels) else 0.0,
            "n_dense": int(labels.sum()),
            "n_total": int(len(labels)),
            "dense_threshold": float(df.dense_threshold),
            # True when the cut came from this run's own reference window, so
            # the baseline is a construction rather than a measurement.
            "relative_to_reference": cal.dense_quantile is not None,
        })
    out = pd.DataFrame(rows)

    # A constant series is the signature of the bug this replaced, so say so
    # rather than letting a flat line be read as "no phase separation".
    frac = out["dense_fraction"].to_numpy()
    if frac.size > 1 and float(frac.std()) == 0.0:
        warnings.warn(
            f"dense_fraction is exactly constant at {frac[0]:.3f} across "
            f"{frac.size} frames. That is usually a thresholding artefact rather "
            "than a physical result: check that grid_shape resolves the "
            "condensate, that smoothing_cells > 0, and that dense_threshold "
            "falls inside the range of the smoothed density. It is a real "
            "result only if the system genuinely never changes phase.",
            UserWarning,
            stacklevel=2,
        )
    return out


def radial_density_profile(cache: ContactCache, frame_index: int, *, center: Optional[np.ndarray] = None, n_bins: int = 30) -> pd.DataFrame:
    """Radial protein-center density profile for one frame.

    If ``center`` is None, the profile is measured around the center of geometry
    of all segment centers.  For a single droplet this approximates a condensate
    radial density profile.
    """
    centers = np.asarray(cache.seg_centers[frame_index], dtype=float)
    if center is None:
        center = np.nanmean(centers, axis=0)
    center = np.asarray(center, dtype=float)
    r = np.linalg.norm(centers - center[None, :], axis=1)
    rmax = max(float(np.nanmax(r)), 1.0)
    edges = np.linspace(0, rmax, n_bins + 1)
    counts, _ = np.histogram(r, bins=edges)
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    density = counts / np.maximum(shell_vol, 1e-12)
    return pd.DataFrame({
        "frame": int(cache.frames[frame_index]),
        "r_mid": 0.5 * (edges[1:] + edges[:-1]),
        "count": counts,
        "density": density,
    })


def lineage_dense_phase_table(
    cache: ContactCache,
    result: LineageTrackingResult,
    *,
    grid_shape=(24, 24, 24),
    dense_quantile: float = 0.85,
    smoothing_cells: Optional[float] = None,
    smoothing_length: Optional[float] = None,
    dense_threshold: Optional[float] = None,
    reference_fraction: float = 0.10,
    calibration: Optional[DensePhaseCalibration] = None,
) -> pd.DataFrame:
    """Estimate how much time each segment/lineage spends in the dense phase.

    Uses the same :class:`DensePhaseCalibration` as
    :func:`dense_phase_fraction_timeseries`, so a figure showing both is showing
    one definition. It did not: this function built a fresh per-frame grid with a
    per-frame quantile over occupied cells, which labelled every chain dense and
    returned exactly 1.0 for every segment -- on the reference run, all 146 of
    them -- while the time series next to it used the calibrated rule. Pass the
    same ``calibration`` object to both, or the same explicit
    ``dense_threshold``.
    """
    cal = calibration or calibrate_dense_phase(
        cache, grid_shape=grid_shape, dense_quantile=dense_quantile,
        smoothing_cells=smoothing_cells, smoothing_length=smoothing_length,
        dense_threshold=dense_threshold, reference_fraction=reference_fraction,
    )
    density_cache = [
        compute_density_grid_for_frame(
            cache, i, grid_shape=cal.grid_shape,
            smoothing_cells=cal.smoothing_cells,
            dense_threshold=cal.dense_threshold, grid_range=cal.grid_range,
        )
        for i in range(len(cache.frames))
    ]
    rows = []
    for seg in result.segments:
        dense_states = []
        for local_t, members in enumerate(seg.sets):
            idx = seg.start_idx + local_t
            if not members:
                continue
            centers = cache.seg_centers[idx, sorted(list(members)), :]
            labels = classify_centers_by_density(centers, density_cache[idx])
            dense_states.append(float(labels.mean()))
        rows.append({
            "segment_id": seg.segment_id,
            "lineage_id": seg.lineage_id,
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
            "duration": seg.duration,
            "mean_dense_fraction_of_members": float(np.mean(dense_states)) if dense_states else np.nan,
            "born_in_dense_phase": bool(dense_states[0] >= 0.5) if dense_states else False,
            "ends_in_dense_phase": bool(dense_states[-1] >= 0.5) if dense_states else False,
        })
    return pd.DataFrame(rows)


__all__ = [
    "DensePhaseCalibration",
    "calibrate_dense_phase",
    "density_at_centers",
    "DensityFrame",
    "compute_density_grid_for_frame",
    "classify_centers_by_density",
    "dense_phase_fraction_timeseries",
    "radial_density_profile",
    "lineage_dense_phase_table",
]
