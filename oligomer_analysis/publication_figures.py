from __future__ import annotations

"""Publication-oriented figure builders.

These functions compose the lower-level analysis tables into figures that are
useful for a manuscript: commitment predictors, LLPS coupling, and contact
hotspot enrichment.  They are intentionally explicit rather than highly styled,
so users can adapt them to journal-specific figure requirements.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .commitment import build_commitment_feature_table
from .hotspots import rank_contact_hotspots
from .llps import (
    calibrate_dense_phase,
    dense_phase_fraction_timeseries,
    lineage_dense_phase_table,
)
from .predictors import build_feature_table
from .models import ContactCache, LineageTrackingResult
from .lineage_plots import plot_whole_system_background, plot_lineage_graph, plot_lineage_fate


def figure_commitment_predictors(
    cache: ContactCache,
    result: LineageTrackingResult,
    *,
    inter_thresh: int,
    predictors: Sequence[str] = ("mean_contact_density", "mean_weighted_contacts", "contact_persistence", "mean_rg"),
    figsize=(14, 8),
):
    """Scatter selected predictors against future commitment score."""
    table = build_feature_table(cache, result, inter_thresh=inter_thresh, include_commitment=True)
    n = len(predictors)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, pred in zip(axes, predictors):
        if pred not in table.columns:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Missing predictor: {pred}", ha="center", va="center")
            continue
        ax.scatter(table[pred], table["max_commitment"], alpha=0.75)
        ax.set_xlabel(pred)
        ax.set_ylabel("Future commitment")
        ax.set_title(f"{pred} vs future productive fate")
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes, table


def figure_llps_coupling(
    cache: ContactCache,
    result: LineageTrackingResult,
    *,
    grid_shape=(24, 24, 24),
    dense_quantile: float = 0.85,
    dense_threshold: Optional[float] = None,
    smoothing_length: Optional[float] = None,
    figsize=(12, 5),
):
    """Plot dense-phase fraction and lineage dense-phase exposure by fate.

    Both panels use ONE :class:`~oligomer_analysis.llps.DensePhaseCalibration`.
    They used to disagree: the left panel used a run-level threshold calibrated
    on chain densities while the right rebuilt a per-frame cell-based quantile,
    which labelled everything dense and made the fate boxplot three identical
    distributions of 1.0.

    Pass ``dense_threshold`` to fix the cut absolutely, which is what comparing
    runs requires.
    """
    calibration = calibrate_dense_phase(
        cache, grid_shape=grid_shape, dense_quantile=dense_quantile,
        dense_threshold=dense_threshold, smoothing_length=smoothing_length,
    )
    dense_ts = dense_phase_fraction_timeseries(cache, calibration=calibration)
    lineage_dense = lineage_dense_phase_table(cache, result, calibration=calibration)
    class_map = {lin.lineage_id: lin.classification for lin in result.lineages}
    lineage_dense["classification"] = lineage_dense["lineage_id"].map(class_map).fillna("uncertain")

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    axes[0].plot(dense_ts["frame"], dense_ts["dense_fraction"], marker="o")
    axes[0].set_title("Dense-phase fraction over time")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Fraction of chains in dense cells")
    axes[0].grid(alpha=0.3)

    order = ["on-pathway", "uncertain", "off-pathway"]
    data = [lineage_dense.loc[lineage_dense["classification"] == cls, "mean_dense_fraction_of_members"].dropna().values for cls in order]
    axes[1].boxplot(data, labels=order, showfliers=True)
    axes[1].set_title("Dense-phase exposure by lineage fate")
    axes[1].set_ylabel("Mean dense fraction of members")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    return fig, axes, dense_ts, lineage_dense


def figure_contact_hotspots(
    cache: ContactCache,
    result: LineageTrackingResult,
    *,
    inter_thresh: int,
    top_n: int = 20,
    figsize=(9, 6),
):
    """Bar plot of segment-pair contacts enriched in on-pathway lineages."""
    top = rank_contact_hotspots(cache, result, inter_thresh=inter_thresh, top_n=top_n)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if top.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No contact hotspots available", ha="center", va="center")
        return fig, ax, top
    labels = [f"{int(r.seg_i)}-{int(r.seg_j)}" for _, r in top.iterrows()]
    ax.barh(labels[::-1], top["log2_enrichment_on_vs_off"].values[::-1])
    ax.set_xlabel("log2 enrichment: on-pathway / off-pathway")
    ax.set_title("On-pathway contact hotspot enrichment")
    ax.grid(axis="x", alpha=0.3)
    return fig, ax, top


def figure_system_overview(cache: ContactCache, result: LineageTrackingResult, **kwargs):
    """Thin publication alias for the calibrated whole-system background plot."""
    return plot_whole_system_background(cache, result, **kwargs)


def figure_lineage_overview(result: LineageTrackingResult, **kwargs):
    """Return lineage graph and lineage fate plot as a paired overview."""
    fig1, ax1 = plot_lineage_graph(result, **kwargs)
    fig2, ax2 = plot_lineage_fate(result, **kwargs)
    return (fig1, ax1), (fig2, ax2)


__all__ = [
    "figure_commitment_predictors",
    "figure_llps_coupling",
    "figure_contact_hotspots",
    "figure_system_overview",
    "figure_lineage_overview",
]
