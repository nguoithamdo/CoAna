# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

"""Memory-safe heatmap utilities for pathway/oligomer monitoring."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def plot_pathway_presence_heatmap_safe(
    df_pathways: pd.DataFrame,
    finished=None,
    *,
    min_duration: int = 10,
    label_column: str = "display_label",
    max_pathways: Optional[int] = 80,
    max_frames: Optional[int] = None,
    figsize: Tuple[float, float] = (16, 12),
    dpi: int = 100,
    fontsize_y: int = 6,
    fontsize_x: int = 8,
    show: bool = True,
):
    """Plot a memory-safe pathway presence/size heatmap.

    This function is designed for large simulations where a naive heatmap can
    create enormous Matplotlib canvases and raise ``MemoryError``. It caps the
    figure size, optionally limits the number of pathways, optionally downsamples
    frames, and avoids ``tight_layout()``.

    Parameters
    ----------
    df_pathways
        Long-format pathway table, usually from
        :func:`oligomer_analysis.pathway_timeline_dataframe`. Required columns:
        ``label``, ``frame``, ``size``. If present, ``display_label`` can be used
        for short labels such as ``oli1``.
    finished
        Optional list of old-style pathway objects. If supplied, pathways are
        filtered by ``min_duration`` using the pathway duration.
    min_duration
        Minimum pathway duration if ``finished`` is supplied.
    label_column
        Column used for y-axis labels. Defaults to ``display_label`` and falls
        back to ``label`` if not present.
    max_pathways
        Keep only the longest/largest pathways. Set to ``None`` to include all,
        but this can be visually crowded.
    max_frames
        Optional maximum number of frame columns to show. If exceeded, frames
        are evenly downsampled.
    figsize, dpi
        Fixed figure size and resolution. Keep these modest to avoid memory
        errors.

    Returns
    -------
    fig, ax, matrix, labels, frames
    """
    import matplotlib.pyplot as plt

    df = df_pathways.copy()
    if df.empty:
        print("No pathway data available.")
        return None, None, None, [], []

    if label_column not in df.columns:
        label_column = "label"

    if finished is not None:
        long_labels = [p.label if hasattr(p, "label") else p["label"] for p in finished
                       if (p.duration if hasattr(p, "duration") else p["duration"]) >= int(min_duration)]
        df = df[df["label"].isin(long_labels)]

    if df.empty:
        print("No pathways met the lifetime filter.")
        return None, None, None, [], []

    pathway_stats = (
        df.groupby(["label", label_column], dropna=False)
        .agg(duration=("frame", "nunique"), max_size=("size", "max"), mean_size=("size", "mean"))
        .reset_index()
        .sort_values(["duration", "max_size", "mean_size"], ascending=False)
    )

    if max_pathways is not None:
        pathway_stats = pathway_stats.head(int(max_pathways))

    keep_labels = set(pathway_stats["label"])
    df = df[df["label"].isin(keep_labels)]

    heatmap_frames = sorted(df["frame"].unique())
    if max_frames is not None and len(heatmap_frames) > int(max_frames):
        step_frame = max(1, len(heatmap_frames) // int(max_frames))
        heatmap_frames = heatmap_frames[::step_frame]
        df = df[df["frame"].isin(heatmap_frames)]

    labels_to_plot = pathway_stats[label_column].astype(str).tolist()
    full_labels_order = pathway_stats["label"].tolist()

    if not labels_to_plot or len(heatmap_frames) == 0:
        print("No pathways met the filter.")
        return None, None, None, [], []

    matrix = np.full((len(labels_to_plot), len(heatmap_frames)), np.nan)
    label_to_i = {lab: i for i, lab in enumerate(full_labels_order)}
    frame_to_j = {fr: j for j, fr in enumerate(heatmap_frames)}

    for _, row in df.iterrows():
        lab = row["label"]
        fr = row["frame"]
        if lab in label_to_i and fr in frame_to_j:
            matrix[label_to_i[lab], frame_to_j[fr]] = row["size"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest", resample=False)
    fig.colorbar(im, ax=ax, label="Pathway size")

    ax.set_yticks(range(len(labels_to_plot)))
    ax.set_yticklabels(labels_to_plot, fontsize=fontsize_y)

    step = max(1, len(heatmap_frames) // 20)
    xticks = np.arange(0, len(heatmap_frames), step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([heatmap_frames[i] for i in xticks], rotation=90, fontsize=fontsize_x)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Pathway")
    ax.set_title(
        f"Pathway presence / size over time\n"
        f"showing {len(labels_to_plot)} pathways, {len(heatmap_frames)} frames"
    )
    fig.subplots_adjust(left=0.25, right=0.92, bottom=0.18, top=0.88)
    if show:
        plt.show()
    return fig, ax, matrix, labels_to_plot, heatmap_frames


__all__ = ["plot_pathway_presence_heatmap_safe"]
