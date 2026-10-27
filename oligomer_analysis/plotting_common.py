# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

"""Shared plotting utilities and calibration controls.

The default plots are intentionally conservative because protein aggregation
simulations can produce oligomers ranging from dimers to very large assemblies.
When oligomer sizes approach ~100, naive plots become unreadable: node sizes blow
up, y-axis ticks become crowded, and transition graphs dominate the page.  The
:class:`PlotCalibration` object gives one consistent place to tune these choices.
"""

from dataclasses import dataclass
import re
from typing import Optional, Sequence, Tuple

import numpy as np

CLASS_COLORS = {
    "on-pathway": "forestgreen",
    "off-pathway": "firebrick",
    "uncertain": "goldenrod",
}

EVENT_COLORS = {
    "split": "royalblue",
    "merge": "darkviolet",
    "split_merge": "darkorange",
    "rearrangement": "gray",
    "growth": "seagreen",
    "shrink": "tomato",
    "death": "black",
    "birth": "dimgray",
    "continuation": "steelblue",
}


@dataclass(slots=True)
class PlotCalibration:
    """Calibration parameters for size-heavy aggregation figures.

    Use this when oligomer sizes can become large, e.g. 50--100 monomers.

    Parameters
    ----------
    max_oligomer_size
        Expected maximum oligomer size. If supplied, size axes default to
        ``0..max_oligomer_size`` for linear plots.
    size_ylim
        Explicit y-limits for size axes. Overrides ``max_oligomer_size``.
    size_tick_step
        Tick spacing for size axes. For max size 100, 10 or 20 is usually good.
    size_scale
        ``"linear"``, ``"log"``, or ``"symlog"`` for size axes.
        Linear is best for direct biological interpretation; log is useful when
        monomers/dimers and large aggregates coexist.
    frame_xlim
        Optional x-limits for frame axes.
    contact_scale
        ``"linear"`` or ``"log"`` for contact-count panels.
    contact_ylim
        Optional y-limits for contact-count panels.
    marker_size, line_width
        Global line-plot styling for dense trajectories.
    node_size_min, node_size_max
        Bounds for network node areas. This prevents 100-mers from producing
        huge nodes that obscure the graph.
    node_size_transform
        ``"sqrt"`` is recommended for sizes up to ~100 because visual area grows
        moderately while preserving ordering. Options: ``"linear"``, ``"sqrt"``,
        ``"log"``.
    matrix_max_size
        Optional maximum size to show in transition matrices. If None, show all.
    show_legend
        Global default for whether plot legends are shown. Individual plotting
        functions can override this with ``show_legend=True`` or ``False``.
    pathway_label_mode
        How long pathway labels are displayed in plots. ``"full"`` keeps labels
        like ``oli8_16_20_25_26``; ``"ordinal"``/``"short"`` displays only
        ``oli8``. This changes plot text only; the stored pathway label remains
        unchanged for lookup and reproducibility.
    """

    max_oligomer_size: Optional[int] = None
    size_ylim: Optional[Tuple[float, float]] = None
    size_tick_step: Optional[int] = None
    size_scale: str = "linear"
    frame_xlim: Optional[Tuple[float, float]] = None
    contact_scale: str = "linear"
    contact_ylim: Optional[Tuple[float, float]] = None
    marker_size: float = 4.0
    line_width: float = 1.5
    node_size_min: float = 250.0
    node_size_max: float = 1800.0
    node_size_transform: str = "sqrt"
    matrix_max_size: Optional[int] = None
    show_legend: bool = True
    pathway_label_mode: str = "full"


def resolve_plot_calibration(calibration: Optional[PlotCalibration | dict] = None, **overrides) -> PlotCalibration:
    """Return a :class:`PlotCalibration` from None, dict, or existing object."""
    if calibration is None:
        cal = PlotCalibration()
    elif isinstance(calibration, PlotCalibration):
        cal = calibration
    elif isinstance(calibration, dict):
        cal = PlotCalibration(**calibration)
    else:
        raise TypeError("calibration must be None, dict, or PlotCalibration")

    for key, value in overrides.items():
        if value is not None:
            if not hasattr(cal, key):
                raise ValueError(f"Unknown PlotCalibration field: {key}")
            setattr(cal, key, value)
    return cal



def format_oligomer_label(label, mode: str = "full") -> str:
    """Return a plot-friendly display label for an oligomer/pathway label.

    The original pathway label stores the initial membership, for example
    ``oli8_16_20_25_26``. That is useful for reproducibility but too long for
    legends and titles. Use ``mode="ordinal"`` or ``mode="short"`` to display
    only the ordinal prefix, e.g. ``oli8``.

    Supported modes
    ---------------
    full / long
        Keep the complete original label.
    ordinal / short / id / oli
        Keep only the leading ``oli<number>`` part. If the label does not match
        that pattern, use the text before the first underscore.
    none / hide
        Return an empty label.
    """
    text = str(label)
    key = str(mode or "full").strip().lower()
    if key in {"full", "long", "complete", "components"}:
        return text
    if key in {"ordinal", "short", "id", "oli", "compact"}:
        m = re.match(r"^(oli\d+)", text)
        if m:
            return m.group(1)
        return text.split("_", 1)[0]
    if key in {"none", "hide", "blank", ""}:
        return ""
    raise ValueError("label mode must be one of: full, ordinal/short/id/oli, none/hide")


def display_oligomer_label(label, calibration: Optional[PlotCalibration | dict] = None, label_mode: Optional[str] = None) -> str:
    """Display label using either explicit ``label_mode`` or ``PlotCalibration``.

    This helper changes only plot text. It never changes the actual pathway
    labels used to find pathways or save analysis output.
    """
    mode = label_mode
    if mode is None:
        mode = resolve_plot_calibration(calibration).pathway_label_mode
    return format_oligomer_label(label, mode=mode)


def legend_enabled(calibration: Optional[PlotCalibration | dict] = None, show_legend: Optional[bool] = None) -> bool:
    """Return whether legends should be shown for a plot.

    If ``show_legend`` is explicitly passed to a plotting function, it overrides
    ``PlotCalibration.show_legend``. Otherwise the calibration object supplies
    the default.
    """
    if show_legend is not None:
        return bool(show_legend)
    return bool(resolve_plot_calibration(calibration).show_legend)


def _auto_size_tick_step(ymax: float) -> int:
    if ymax <= 20:
        return 2
    if ymax <= 50:
        return 5
    if ymax <= 120:
        return 10
    if ymax <= 250:
        return 25
    return 50


def calibrate_size_axis(ax, calibration: Optional[PlotCalibration | dict] = None, *, observed_max: Optional[float] = None, ylabel: str = "Oligomer size"):
    """Apply consistent size-axis scaling/ticks.

    This is the key helper for simulations where oligomer size can reach 100 or
    more. It keeps all size plots comparable across whole-system, lineage, and
    single-pathway figures.
    """
    cal = resolve_plot_calibration(calibration)
    ax.set_ylabel(ylabel)
    ax.set_yscale(cal.size_scale)

    if cal.size_scale == "linear":
        if cal.size_ylim is not None:
            ymin, ymax = cal.size_ylim
        elif cal.max_oligomer_size is not None:
            ymin, ymax = 0, float(cal.max_oligomer_size)
        elif observed_max is not None and np.isfinite(observed_max):
            ymin, ymax = 0, max(2.0, float(observed_max) * 1.08)
        else:
            ymin, ymax = None, None

        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)
            step = cal.size_tick_step or _auto_size_tick_step(ymax)
            if step > 0:
                ax.set_yticks(np.arange(ymin, ymax + 0.5 * step, step))
    else:
        if cal.size_ylim is not None:
            ax.set_ylim(max(cal.size_ylim[0], 1e-6), cal.size_ylim[1])
        elif cal.max_oligomer_size is not None:
            ax.set_ylim(1, max(2, cal.max_oligomer_size))

    if cal.frame_xlim is not None:
        ax.set_xlim(*cal.frame_xlim)
    return ax


def calibrate_contact_axis(ax, calibration: Optional[PlotCalibration | dict] = None, *, ylabel: str = "Contacts"):
    """Apply contact-count axis scaling."""
    cal = resolve_plot_calibration(calibration)
    ax.set_ylabel(ylabel)
    ax.set_yscale(cal.contact_scale)
    if cal.contact_ylim is not None:
        ax.set_ylim(*cal.contact_ylim)
    if cal.frame_xlim is not None:
        ax.set_xlim(*cal.frame_xlim)
    return ax


def scale_node_sizes(values: Sequence[float], calibration: Optional[PlotCalibration | dict] = None) -> np.ndarray:
    """Scale numeric values to bounded matplotlib node areas.

    For large aggregates, never map size directly to node area.  This helper caps
    the visual area while retaining rank/order information.
    """
    cal = resolve_plot_calibration(calibration)
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x

    x = np.maximum(x, 0.0)
    if cal.node_size_transform == "sqrt":
        z = np.sqrt(x)
    elif cal.node_size_transform == "log":
        z = np.log1p(x)
    elif cal.node_size_transform == "linear":
        z = x
    else:
        raise ValueError("node_size_transform must be 'linear', 'sqrt', or 'log'")

    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or np.isclose(zmax, zmin):
        return np.full_like(z, (cal.node_size_min + cal.node_size_max) / 2.0, dtype=float)
    return cal.node_size_min + (z - zmin) / (zmax - zmin) * (cal.node_size_max - cal.node_size_min)


def matrix_display_slice(M: np.ndarray, *, min_size: int = 1, calibration: Optional[PlotCalibration | dict] = None):
    """Return matrix slice and tick labels for size-transition matrices.

    Parameters
    ----------
    M
        Matrix indexed by actual size. Row/column 0 is unused for oligomers.
    min_size
        Smallest size to display, typically 2 for oligomers.
    """
    cal = resolve_plot_calibration(calibration)
    max_size = M.shape[0] - 1
    if cal.matrix_max_size is not None:
        max_size = min(max_size, int(cal.matrix_max_size))
    if max_size < min_size:
        return M[0:0, 0:0], []
    labels = list(range(int(min_size), int(max_size) + 1))
    sub = M[min_size:max_size + 1, min_size:max_size + 1]
    return sub, labels


__all__ = [
    "CLASS_COLORS",
    "EVENT_COLORS",
    "PlotCalibration",
    "resolve_plot_calibration",
    "calibrate_size_axis",
    "calibrate_contact_axis",
    "scale_node_sizes",
    "matrix_display_slice",
    "format_oligomer_label",
    "display_oligomer_label",
]
