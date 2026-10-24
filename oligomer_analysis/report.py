"""Publication-grade HTML reporting for CoAnaPure / oligomer-analysis runs.

:func:`generate_html_report` turns a Stage A cache (plus the lineage analysis
derived from it) into a single self-contained HTML file: every figure is
embedded as base64 PNG, every style is inlined, so the report can be copied off
an HPC cluster, mailed, or archived as one file and will render identically
anywhere -- including printed to PDF from the browser.

The report covers the full analysis surface: run configuration and methods,
whole-system aggregation evolution, oligomer populations, contact maps,
split/merge-aware lineage and fate classification, reproducibility metadata,
and an appendix describing the engine's capabilities.

Figures are rendered headless (Agg canvas, no display needed -- SLURM/batch
safe) with a colorblind-validated palette.

Typical use::

    from oligomer_analysis import load_cache, generate_html_report
    generate_html_report("cache.npz", "report.html", title="A-beta 128-mer, 330 K")

or from the shell::

    coana report cache.npz -o report.html
"""

from __future__ import annotations

import base64
import html
import io
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .cache_io import load_cache
from .graph_tools import clusters_from_cmap
from .models import ContactCache, LineageTrackingResult, TargetStateConfig
from .system_analysis import (
    build_system_summary_dataframe,
    oligomer_sizes_by_frame_dataframe,
    oligomer_summary_by_frame_dataframe,
)

__all__ = ["generate_html_report"]


# --------------------------------------------------------------------------- #
# Palette (colorblind-validated reference palette; light mode)
# --------------------------------------------------------------------------- #

_C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
}

_SEQ_BLUES = [
    "#fcfcfb", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
    "#104281", "#0d366b",
]


def _blues_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("coana_blues", _SEQ_BLUES)


# --------------------------------------------------------------------------- #
# Figure helpers (pure-Figure Agg rendering; no pyplot state, headless-safe)
# --------------------------------------------------------------------------- #

def _new_fig(width: float = 9.2, height: float = 3.4):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), facecolor=_C["surface"], constrained_layout=True)
    FigureCanvasAgg(fig)
    return fig


def _style_ax(ax, *, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(_C["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C["baseline"])
        ax.spines[side].set_linewidth(1.0)
    ax.grid(True, color=_C["grid"], linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_C["muted"], labelsize=9, length=3)
    if xlabel:
        ax.set_xlabel(xlabel, color=_C["ink2"], fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=_C["ink2"], fontsize=10)


def _legend(ax) -> None:
    leg = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in leg.get_texts():
        text.set_color(_C["ink2"])


def _fig_b64(fig, dpi: int = 115) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img(b64: str, caption: str, alt: str) -> str:
    return (
        f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


# --------------------------------------------------------------------------- #
# Individual figures
# --------------------------------------------------------------------------- #

def _fig_free_bound(cache: ContactCache) -> str:
    fig = _new_fig()
    ax = fig.add_subplot(111)
    x = cache.frames
    ax.plot(x, cache.n_bound, color=_C["blue"], linewidth=2, label="Bound monomers")
    ax.plot(x, cache.n_free, color=_C["orange"], linewidth=2, label="Free monomers")
    _style_ax(ax, xlabel="Frame", ylabel="Monomer count")
    _legend(ax)
    return _fig_b64(fig)


def _fig_bound_fraction(cache: ContactCache) -> str:
    fig = _new_fig(height=2.9)
    ax = fig.add_subplot(111)
    total = cache.n_bound.astype(float) + cache.n_free.astype(float)
    total[total == 0] = np.nan
    frac = cache.n_bound / total
    ax.plot(cache.frames, frac, color=_C["blue"], linewidth=2)
    ax.fill_between(cache.frames, 0, frac, color=_C["blue"], alpha=0.10, linewidth=0)
    ax.set_ylim(0, 1.02)
    _style_ax(ax, xlabel="Frame", ylabel="Bound fraction")
    return _fig_b64(fig)


def _fig_contacts(cache: ContactCache) -> str:
    fig = _new_fig()
    ax = fig.add_subplot(111)
    ax.plot(cache.frames, cache.ntotal, color=_C["blue"], linewidth=2, label="Total contacts")
    ax.plot(cache.frames, cache.ninter, color=_C["orange"], linewidth=2, label="Inter-monomer contacts")
    _style_ax(ax, xlabel="Frame", ylabel="Residue-pair contacts")
    _legend(ax)
    return _fig_b64(fig)


def _fig_cluster_panels(df_system: pd.DataFrame) -> str:
    fig = _new_fig(height=3.0)
    panels = [
        ("n_clusters", "Oligomer count (size ≥ 2)"),
        ("largest_cluster", "Largest oligomer size"),
        ("mean_cluster_size", "Mean oligomer size"),
    ]
    for i, (col, label) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i)
        ax.plot(df_system["frame"], df_system[col], color=_C["blue"], linewidth=1.8)
        _style_ax(ax, xlabel="Frame", ylabel=label)
    return _fig_b64(fig)


def _fig_size_histogram(frames_clusters: List[List[frozenset]]) -> str:
    sizes = [len(c) for clusters in frames_clusters for c in clusters if len(c) >= 2]
    fig = _new_fig(height=3.0)
    ax = fig.add_subplot(111)
    if sizes:
        top = max(sizes)
        bins = np.arange(1.5, top + 1.5 + 1)
        ax.hist(sizes, bins=bins, color=_C["blue"], edgecolor=_C["surface"], linewidth=0.8)
        ax.set_yscale("log")
    _style_ax(ax, xlabel="Oligomer size (monomers)", ylabel="Occurrences (all frames, log)")
    return _fig_b64(fig)


def _fig_size_time_heatmap(frames_clusters: List[List[frozenset]], frames: np.ndarray) -> str:
    max_size = max(
        (len(c) for clusters in frames_clusters for c in clusters), default=1
    )
    max_size = max(max_size, 2)
    occ = np.zeros((max_size - 1, len(frames_clusters)), dtype=float)
    for t, clusters in enumerate(frames_clusters):
        for c in clusters:
            if len(c) >= 2:
                occ[len(c) - 2, t] += 1
    fig = _new_fig(height=3.6)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        occ,
        aspect="auto",
        origin="lower",
        cmap=_blues_cmap(),
        interpolation="nearest",
        extent=(float(frames[0]), float(frames[-1]), 1.5, max_size + 0.5),
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("Clusters of this size", color=_C["ink2"], fontsize=9)
    cb.ax.tick_params(colors=_C["muted"], labelsize=8)
    cb.outline.set_edgecolor(_C["baseline"])
    _style_ax(ax, xlabel="Frame", ylabel="Oligomer size")
    ax.grid(False)
    return _fig_b64(fig)


def _fig_contact_maps(cache: ContactCache) -> str:
    fig = _new_fig(height=4.2)
    mean_map = cache.cmaps.mean(axis=0)
    last_map = cache.cmaps[-1]
    for i, (mat, label) in enumerate(
        [(mean_map, "Time-averaged contact map"), (last_map, f"Final frame (frame {int(cache.frames[-1])})")],
        start=1,
    ):
        ax = fig.add_subplot(1, 2, i)
        im = ax.imshow(mat, cmap=_blues_cmap(), interpolation="nearest", origin="lower")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.ax.tick_params(colors=_C["muted"], labelsize=8)
        cb.outline.set_edgecolor(_C["baseline"])
        ax.set_title(label, color=_C["ink2"], fontsize=10)
        _style_ax(ax, xlabel="Monomer index", ylabel="Monomer index")
        ax.grid(False)
    return _fig_b64(fig)


def _fig_lineage_classes(result: LineageTrackingResult) -> str:
    from collections import Counter

    counts = Counter(lin.class_label for lin in result.lineages)
    order = ["on_pathway", "off_pathway", "uncertain"]
    labels = [lab for lab in order if lab in counts] + [
        lab for lab in counts if lab not in order
    ]
    values = [counts[lab] for lab in labels]
    colors = {
        "on_pathway": _C["blue"],
        "off_pathway": _C["orange"],
        "uncertain": _C["muted"],
    }
    fig = _new_fig(height=2.8, width=6.0)
    ax = fig.add_subplot(111)
    bars = ax.bar(
        range(len(labels)), values,
        color=[colors.get(lab, _C["aqua"]) for lab in labels],
        width=0.62, edgecolor=_C["surface"],
    )
    for rect, v in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2, rect.get_height(), f" {v}",
            ha="center", va="bottom", color=_C["ink2"], fontsize=10,
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([lab.replace("_", " ") for lab in labels], color=_C["ink2"])
    _style_ax(ax, ylabel="Lineages")
    return _fig_b64(fig)


def _lineage_package_figs(cache: ContactCache, result: LineageTrackingResult, calibration) -> List[Tuple[str, str]]:
    """Render the package's own lineage graph / fate figures, headless."""
    out: List[Tuple[str, str]] = []
    try:
        import matplotlib

        if not matplotlib.get_backend().lower().startswith("agg"):
            try:
                matplotlib.use("Agg", force=False)
            except Exception:
                pass
        import matplotlib.pyplot as plt

        from .lineage_plots import plot_lineage_fate, plot_lineage_graph

        try:
            fig, _ = plot_lineage_graph(result, calibration=calibration, node_size_by="max_size")
            out.append(
                (
                    _fig_b64(fig),
                    "Lineage graph: each node is one continuous oligomer episode; edges are "
                    "split/merge events linking parents to children.",
                )
            )
            plt.close(fig)
        except Exception:
            pass
        try:
            fig, _ = plot_lineage_fate(result, calibration=calibration)
            out.append(
                (
                    _fig_b64(fig),
                    "Lineage fate: size trajectory of every lineage over time, colored by "
                    "its on/off-pathway classification (fate-based, not size-based).",
                )
            )
            plt.close(fig)
        except Exception:
            pass
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0; background: %(page)s; color: %(ink)s;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px; line-height: 1.55;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 32px 28px 64px; }
header.rpt {
  border-bottom: 2px solid %(ink)s; padding-bottom: 18px; margin-bottom: 26px;
}
header.rpt h1 { margin: 0 0 4px; font-size: 30px; letter-spacing: -0.01em; }
header.rpt .sub { color: %(ink2)s; font-size: 16px; }
.chips { margin-top: 10px; }
.chip {
  display: inline-block; border: 1px solid %(baseline)s; border-radius: 999px;
  padding: 2px 12px; margin: 2px 6px 2px 0; font-size: 12.5px; color: %(ink2)s;
  background: %(surface)s;
}
h2 {
  font-size: 21px; margin: 40px 0 6px; padding-top: 10px;
  border-top: 1px solid %(grid)s;
}
h2 .no { color: %(muted)s; font-weight: 600; margin-right: 8px; }
h3 { font-size: 16px; margin: 22px 0 4px; color: %(ink)s; }
p.lead { color: %(ink2)s; margin: 4px 0 14px; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0 6px; }
.tile {
  flex: 1 1 150px; min-width: 150px; background: %(surface)s;
  border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 14px 16px;
}
.tile .v { font-size: 26px; font-weight: 650; letter-spacing: -0.01em; }
.tile .k { font-size: 12.5px; color: %(ink2)s; margin-top: 2px; }
figure {
  margin: 18px 0; background: %(surface)s; border: 1px solid rgba(11,11,11,0.10);
  border-radius: 10px; padding: 14px; overflow-x: auto;
}
figure img { max-width: 100%%; height: auto; display: block; margin: 0 auto; }
figcaption { color: %(ink2)s; font-size: 13px; margin-top: 10px; }
.tblwrap { overflow-x: auto; background: %(surface)s; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 6px 10px; margin: 14px 0; }
table.tbl { border-collapse: collapse; width: 100%%; font-size: 13.5px; }
table.tbl th {
  text-align: left; color: %(ink2)s; font-weight: 600; border-bottom: 1.5px solid %(baseline)s;
  padding: 7px 10px; white-space: nowrap;
}
table.tbl td { padding: 6px 10px; border-bottom: 1px solid %(grid)s; font-variant-numeric: tabular-nums; }
table.tbl tr:last-child td { border-bottom: none; }
table.tbl td.num, table.tbl th.num { text-align: right; }
.note { color: %(muted)s; font-size: 12.5px; margin: 6px 0 0; }
pre.code {
  background: %(surface)s; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px;
  padding: 14px 16px; overflow-x: auto; font-size: 12.8px; line-height: 1.5;
}
.toc { background: %(surface)s; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 12px 18px; margin: 18px 0; }
.toc a { color: %(blue)s; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.toc ol { margin: 6px 0; padding-left: 22px; columns: 2; }
footer.rpt {
  margin-top: 48px; padding-top: 14px; border-top: 1px solid %(grid)s;
  color: %(muted)s; font-size: 12.5px;
}
@media print {
  body { background: #ffffff; }
  .wrap { max-width: none; padding: 0; }
  h2 { break-before: page; border-top: none; }
  h2:first-of-type, h2.nopgbrk { break-before: auto; }
  figure, .tblwrap, .tile { break-inside: avoid; border-color: #ddd; }
}
""" % _C


def _fmt_num(v) -> str:
    if isinstance(v, float):
        return f"{v:,.3f}".rstrip("0").rstrip(".")
    return f"{v:,}"


def _tile(value: str, label: str) -> str:
    return f'<div class="tile"><div class="v">{value}</div><div class="k">{html.escape(label)}</div></div>'


def _df_to_html(df: pd.DataFrame, *, max_rows: int = 30, floatfmt: str = "%.3f") -> str:
    n = len(df)
    shown = df if n <= max_rows else df.head(max_rows)
    num_cols = {c for c in shown.columns if pd.api.types.is_numeric_dtype(shown[c])}

    head_cells = "".join(
        f'<th class="{"num" if c in num_cols else ""}">{html.escape(str(c))}</th>'
        for c in shown.columns
    )
    body_rows = []
    for _, row in shown.iterrows():
        cells = []
        for c in shown.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                txt = floatfmt % v
            else:
                txt = str(v)
            cls = ' class="num"' if c in num_cols else ""
            cells.append(f"<td{cls}>{html.escape(txt)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    note = (
        f'<p class="note">Showing first {max_rows} of {n:,} rows; the full table is '
        "available from the cache with the package's dataframe utilities.</p>"
        if n > max_rows
        else ""
    )
    return (
        '<div class="tblwrap"><table class="tbl">'
        f"<thead><tr>{head_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>{note}"
    )


def _kv_table(rows: Sequence[Tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><td style='color:{_C['ink2']};white-space:nowrap'>{html.escape(k)}</td>"
        f"<td>{html.escape(str(v))}</td></tr>"
        for k, v in rows
    )
    return f'<div class="tblwrap"><table class="tbl"><tbody>{body}</tbody></table></div>'


_CAPABILITIES: List[Tuple[str, str]] = [
    ("Trajectory formats",
     "PDB (single/multi-MODEL, streaming), GRO (single/multi-frame), TRR, and XTC via a "
     "native pure-Python 3dfcoord decompressor. No MDAnalysis required for any of them."),
    ("Constant-memory streaming",
     "Frames are parsed and consumed one at a time; peak memory stays at O(workers) frames "
     "regardless of trajectory size (multi-GB PDB files run in well under 1 GB RSS)."),
    ("Random frame access",
     "A byte-offset model index sidecar (built once with `coana index` or build_model_index) "
     "lets sparse frame subsets seek directly to the wanted models."),
    ("Parallel execution",
     "Threaded contact computation (NumPy/SciPy release the GIL), plus a process-parallel "
     "mode (parallel='process') that distributes parsing itself across CPU cores via the "
     "model index -- the scaling mode for HPC nodes."),
    ("Atom selection",
     "cfg.selection restricts the analysis to a subset of atoms before segmentation: "
     "'resname BB', 'not resname W ION', 'resid 1-128', 'chain A B', 'index 0-9999'."),
    ("Monomer segmentation",
     "Four segment modes: chain-id based ('mdanalysis'), equal atom blocks ('atom_blocks'), "
     "equal residue blocks ('residue_blocks'), and unequal per-monomer atom counts "
     "('atom_counts') for mixed systems."),
    ("Periodic boundary handling",
     "Orthorhombic and triclinic minimum-image PBC. The default reproduces the original "
     "MDAnalysis pipeline bit-for-bit (a leading CRYST1 in a multi-model PDB is dropped, "
     "with an explicit warning); broadcast_box=True applies the box to every frame for "
     "physically correct new analyses."),
    ("Checkpoint / resume",
     "checkpoint_every writes a crash-safe, configuration-fingerprinted checkpoint; "
     "resume=True lets a walltime-killed batch job continue where it stopped."),
    ("Progress reporting", "Rate and ETA lines suitable for interactive shells and SLURM logs."),
    ("Validation",
     "Cross-validated against MDAnalysis 2.9: contact maps, inter-monomer contact counts, "
     "free/bound monomer counts, boxes and frame numbering match exactly; segment centres "
     "agree to ~1e-4 A (CoAnaPure computes in float64). A benchmark module "
     "(python -m oligomer_analysis.coanapure.benchmark) re-runs the comparison on any "
     "trajectory where MDAnalysis is installed."),
    ("Command-line interface",
     "coana precompute / index / info / report / benchmark cover the full Stage A "
     "workflow without writing Python."),
    ("Cache & exports",
     "Compressed .npz cache identical to the original pipeline, per-frame CSV statistics, "
     "cache subsetting helpers, per-frame oligomer tables and Excel export."),
    ("Downstream analysis (Stage B)",
     "Split/merge-aware lineage tracking, fate-based on/off-pathway classification, "
     "commitment scoring, structural predictors, LLPS-style dense/dilute observables, "
     "contact hotspot enrichment, dashboards and this HTML report."),
]


def generate_html_report(
    cache: "ContactCache | str | Path",
    out_html: "str | Path | None" = None,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    author: Optional[str] = None,
    inter_thresh: int = 1,
    include_lineage: bool = True,
    target_config: Optional[TargetStateConfig] = None,
    lineage_kwargs: Optional[dict] = None,
    calibration=None,
    max_table_rows: int = 30,
    verbose: bool = True,
) -> str:
    """Render a self-contained HTML analysis report for a Stage A cache.

    Parameters
    ----------
    cache
        A loaded :class:`ContactCache` or a path to the ``.npz`` cache file.
    out_html
        Output path; defaults to ``<cache stem>_report.html`` next to the cache
        (or ``coanapure_report.html`` when a cache object is passed).
    inter_thresh
        Contact-count threshold defining a monomer-monomer edge (project rule:
        ``cmap[i, j] >= inter_thresh``).
    include_lineage
        Build the split/merge-aware lineage analysis and include the lineage /
        fate section.  Set False for a faster population-only report.
    target_config, lineage_kwargs
        Passed to :func:`~oligomer_analysis.lineage.build_lineages_from_cache`
        (defaults follow the documented V2 workflow).
    calibration
        Optional :class:`~oligomer_analysis.plotting_common.PlotCalibration`
        for the package-native lineage figures.

    Returns
    -------
    str
        The path of the written HTML file.
    """
    t_start = time.monotonic()

    cache_path: Optional[Path] = None
    if isinstance(cache, (str, Path)):
        cache_path = Path(cache)
        cache = load_cache(cache_path)
    if out_html is None:
        out_html = (
            cache_path.with_name(cache_path.stem + "_report.html")
            if cache_path is not None
            else Path("coanapure_report.html")
        )
    out_html = Path(out_html)

    meta = dict(cache.meta)
    n_frames = int(len(cache.frames))
    nseg = int(meta.get("nseg", cache.cmaps.shape[1]))
    has_box = bool(np.any(cache.boxes[:, :3] > 0))

    if verbose:
        print(f"CoAnaPure report: {n_frames} frames, {nseg} monomers", flush=True)

    # ---- population analysis ------------------------------------------------
    frames_clusters_full = [
        clusters_from_cmap(cache.cmaps[i], inter_thresh) for i in range(n_frames)
    ]
    df_system = build_system_summary_dataframe(cache, frames_clusters_full, cluster_min_size=2)
    df_olig = oligomer_sizes_by_frame_dataframe(cache, frames_clusters_full, min_size=2)
    df_olig_summary = (
        oligomer_summary_by_frame_dataframe(df_olig) if len(df_olig) else pd.DataFrame()
    )

    sizes_all = [len(c) for clusters in frames_clusters_full for c in clusters]
    max_size_seen = max(sizes_all) if sizes_all else 1
    final_bound_frac = (
        float(cache.n_bound[-1]) / max(1.0, float(cache.n_bound[-1] + cache.n_free[-1]))
    )

    # ---- lineage analysis ---------------------------------------------------
    lineage_result: Optional[LineageTrackingResult] = None
    lineage_error: Optional[str] = None
    df_lineage = pd.DataFrame()
    if include_lineage:
        if verbose:
            print("  building split/merge-aware lineages ...", flush=True)
        try:
            from .lineage import build_lineages_from_cache, lineage_summary_dataframe

            kwargs = dict(
                inter_thresh=inter_thresh,
                cluster_min_size=2,
                overlap_min=1,
                parent_frac_min=0.30,
                child_frac_min=0.30,
                union_cover_min=0.70,
                target_config=target_config or TargetStateConfig(),
            )
            kwargs.update(lineage_kwargs or {})
            lineage_result = build_lineages_from_cache(cache, **kwargs)
            df_lineage = lineage_summary_dataframe(lineage_result)
        except Exception as exc:  # keep the report usable even if Stage B fails
            lineage_error = f"{type(exc).__name__}: {exc}"
            lineage_result = None

    # ---- figures ------------------------------------------------------------
    if verbose:
        print("  rendering figures ...", flush=True)
    fig_free_bound = _fig_free_bound(cache)
    fig_bound_frac = _fig_bound_fraction(cache)
    fig_contacts = _fig_contacts(cache)
    fig_clusters = _fig_cluster_panels(df_system)
    fig_hist = _fig_size_histogram(frames_clusters_full)
    fig_heat = _fig_size_time_heatmap(frames_clusters_full, cache.frames)
    fig_cmaps = _fig_contact_maps(cache)
    fig_classes = _fig_lineage_classes(lineage_result) if lineage_result else None
    pkg_figs = (
        _lineage_package_figs(cache, lineage_result, calibration) if lineage_result else []
    )

    # ---- header / metadata --------------------------------------------------
    traj_file = str(meta.get("traj_file", "?"))
    traj_name = Path(traj_file).name if traj_file != "?" else "?"
    title = title or "Oligomer Aggregation Analysis"
    subtitle = subtitle or f"{traj_name} - {nseg} monomers, {n_frames} frames"
    today = time.strftime("%Y-%m-%d %H:%M")

    engine = str(meta.get("engine", "?"))
    selection = meta.get("selection")
    seg_mode = str(meta.get("segment_mode", "?"))

    config_rows: List[Tuple[str, str]] = [
        ("Trajectory", traj_file),
        ("Topology", str(meta.get("topology_file") or "from trajectory (first model)")),
        ("Format", str(meta.get("traj_format", "?"))),
        ("Frames analysed", f"{n_frames:,} (frame {meta.get('frames_min')} .. {meta.get('frames_max')})"),
        ("Contact cutoff", f"{meta.get('cutoff_contact')} Å (residue COG pairs)"),
        ("Intra-chain exclusion", f"|Δresidue| < {meta.get('intra_exclude')} ignored within a monomer"),
        ("Segment mode", seg_mode),
    ]
    if meta.get("atoms_per_monomer"):
        config_rows.append(("Atoms per monomer", str(meta["atoms_per_monomer"])))
    if meta.get("residues_per_monomer"):
        config_rows.append(("Residues per monomer", str(meta["residues_per_monomer"])))
    if meta.get("monomer_atom_counts"):
        counts = meta["monomer_atom_counts"]
        config_rows.append(
            ("Monomer atom counts", f"{len(counts)} monomers, {min(counts)}..{max(counts)} atoms")
        )
    config_rows += [
        ("Monomers (segments)", f"{nseg:,}"),
        ("Topology size", f"{meta.get('topology_n_atoms', '?'):,} atoms / {meta.get('topology_n_residues', '?'):,} residues"
         if isinstance(meta.get("topology_n_atoms"), int) else str(meta.get("topology_n_atoms", "?"))),
    ]
    if selection:
        config_rows.append(
            ("Atom selection", f"{selection}  ({meta.get('selection_n_atoms', '?'):,} atoms kept)")
        )
    config_rows += [
        ("Periodic boundaries", "yes (box stored per frame)" if has_box else
         "none (matches the original MDAnalysis pipeline for this input)"),
        ("Engine", f"{engine}, {meta.get('parallel', 'thread')} mode"),
        ("Edge rule (this report)", f"monomers i,j linked when cmap[i,j] ≥ {inter_thresh}"),
    ]

    # ---- tiles --------------------------------------------------------------
    tiles = [
        _tile(f"{n_frames:,}", "Frames"),
        _tile(f"{nseg:,}", "Monomers"),
        _tile(f"{max_size_seen}", "Largest oligomer"),
        _tile(f"{final_bound_frac * 100:.0f}%", "Final bound fraction"),
    ]
    if lineage_result is not None:
        n_on = sum(1 for l in lineage_result.lineages if l.class_label == "on_pathway")
        tiles += [
            _tile(f"{len(lineage_result.lineages)}", "Lineages tracked"),
            _tile(f"{n_on}", "On-pathway lineages"),
        ]

    # ---- section HTML -------------------------------------------------------
    sec: List[str] = []
    toc: List[Tuple[str, str]] = []

    def add_section(anchor: str, heading: str, body: str) -> None:
        toc.append((anchor, heading))
        sec.append(f'<h2 id="{anchor}"><span class="no">{len(toc)}.</span>{heading}</h2>{body}')

    add_section(
        "config", "Run configuration",
        '<p class="lead">Everything below was read from the cache metadata; the cache, '
        "not the trajectory, is the reproducible interface between the expensive Stage A "
        "computation and this report.</p>" + _kv_table(config_rows),
    )

    methods_html = f"""
<p class="lead">How the numbers in this report are defined.</p>
<p><strong>Contact definition.</strong> For every frame, the centre of geometry (COG) of each
residue/bead group is computed with periodic-boundary-safe unwrapping; two residues are in
contact when their COGs lie within <strong>{meta.get('cutoff_contact')} &#8491;</strong>.
Contacts between residues of the same monomer closer than {meta.get('intra_exclude')} positions
in sequence are excluded from the intra-monomer count. The monomer-level contact map
<code>cmap[i,j]</code> counts residue-residue contacts between monomers i and j.</p>
<p><strong>Oligomer definition.</strong> Monomers are nodes; an edge exists when
<code>cmap[i,j] &#8805; {inter_thresh}</code>. Connected components of this graph are the
clusters; size 1 is a free monomer, size &#8805; 2 is an oligomer.</p>
<p><strong>Lineage &amp; fate.</strong> Oligomer episodes are tracked through splits and merges;
a lineage is classified on/off-pathway by its <em>future fate</em> (whether its monomers
persistently commit to a target aggregate), never by size alone &mdash; temporary fragmentation
does not mean off-pathway.</p>
<p><strong>Engine.</strong> Stage A was computed by <strong>CoAnaPure</strong> ({html.escape(engine)}),
a pure NumPy/SciPy engine with no MDAnalysis dependency, cross-validated against MDAnalysis 2.9
(exact agreement on contact maps, inter-monomer contacts, free/bound counts and boxes; segment
centres to ~1e-4 &#8491;). See the capabilities appendix for the full feature set.</p>
"""
    add_section("methods", "Methods", methods_html)

    evolution_html = (
        _img(fig_free_bound, "Free vs bound monomers per frame. A monomer is bound when it has at "
             f"least one inter-monomer contact partner (edge rule: cmap &#8805; {inter_thresh}).",
             "Free and bound monomer counts over time")
        + _img(fig_bound_frac, "Fraction of all monomers bound in an oligomer.",
               "Bound monomer fraction over time")
        + _img(fig_contacts, "System-wide residue-pair contact counts per frame.",
               "Contact counts over time")
        + _img(fig_clusters, "Oligomer count, largest oligomer, and mean oligomer size per frame "
               "(clusters of size ≥ 2).", "Cluster statistics over time")
    )
    add_section("evolution", "System evolution", evolution_html)

    pop_html = (
        _img(fig_hist, "Pooled size distribution of all oligomers over all frames (log count).",
             "Oligomer size histogram")
        + _img(fig_heat, "Oligomer size occupancy over time: color gives the number of clusters "
               "of each size in each frame.", "Size-time occupancy heatmap")
    )
    if len(df_olig_summary):
        pop_html += "<h3>Per-frame oligomer summary</h3>" + _df_to_html(
            df_olig_summary, max_rows=max_table_rows
        )
    add_section("populations", "Oligomer populations", pop_html)

    add_section(
        "cmaps", "Contact maps",
        _img(fig_cmaps, "Monomer-monomer contact maps: residue-contact counts averaged over all "
             "frames (left) and in the final frame (right).", "Contact maps"),
    )

    if lineage_result is not None:
        lin_html = (
            '<p class="lead">Split/merge-aware tracking: an oligomer episode ends at every split '
            "or merge, and episodes are joined into lineages whose fate determines the "
            "classification.</p>"
        )
        if fig_classes:
            lin_html += _img(fig_classes, "Lineage classification counts (fate-based).",
                             "Lineage class counts")
        ev_types: Dict[str, int] = {}
        for ev in lineage_result.events:
            ev_types[ev.event_type] = ev_types.get(ev.event_type, 0) + 1
        if ev_types:
            lin_html += _kv_table(
                [("Episodes (segments)", f"{len(lineage_result.segments):,}")]
                + [(f"Events: {k}", f"{v:,}") for k, v in sorted(ev_types.items())]
            )
        if len(df_lineage):
            lin_html += "<h3>Lineage summary</h3>" + _df_to_html(df_lineage, max_rows=max_table_rows)
        for b64, cap in pkg_figs:
            lin_html += _img(b64, cap, cap.split(":")[0])
        add_section("lineage", "Lineage &amp; fate analysis", lin_html)
    elif include_lineage and lineage_error:
        add_section(
            "lineage", "Lineage &amp; fate analysis",
            f'<p class="lead">Lineage analysis could not be completed for this cache: '
            f"<code>{html.escape(lineage_error)}</code></p>",
        )

    # Reproducibility
    repro_cfg = {
        k: meta.get(k)
        for k in (
            "topology_file", "traj_file", "traj_format", "frames_min", "frames_max",
            "n_frames", "cutoff_contact", "intra_exclude", "segment_mode",
            "atoms_per_monomer", "residues_per_monomer", "n_monomers",
            "monomer_atom_counts", "selection", "fallback_box", "engine", "parallel",
        )
        if meta.get(k) is not None
    }
    repro_html = (
        '<p class="lead">The cache file plus the metadata below fully determine every number in '
        "this report.</p>"
        f"<pre class='code'>{html.escape(json.dumps(repro_cfg, indent=2))}</pre>"
        "<p>Regenerate this report from the cache at any time:</p>"
        f"<pre class='code'>coana report {html.escape(cache_path.name if cache_path else 'cache.npz')} "
        f"-o {html.escape(out_html.name)} --inter-thresh {inter_thresh}</pre>"
    )
    add_section("repro", "Reproducibility", repro_html)

    caps_html = '<p class="lead">What the CoAnaPure engine in this package provides.</p>'
    caps_html += '<div class="tblwrap"><table class="tbl"><tbody>'
    for name, desc in _CAPABILITIES:
        caps_html += (
            f"<tr><td style='color:{_C['ink2']};white-space:nowrap;vertical-align:top'>"
            f"<strong>{html.escape(name)}</strong></td><td>{html.escape(desc)}</td></tr>"
        )
    caps_html += "</tbody></table></div>"
    add_section("caps", "Appendix: CoAnaPure capabilities", caps_html)

    # ---- assemble -----------------------------------------------------------
    try:
        import scipy

        scipy_v = scipy.__version__
    except Exception:  # pragma: no cover
        scipy_v = "not installed"
    import matplotlib

    from . import __version__ as pkg_version

    toc_html = (
        '<nav class="toc"><strong>Contents</strong><ol>'
        + "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in toc)
        + "</ol></nav>"
    )

    chips = [
        f"Generated {today}",
        f"Engine {engine}",
        f"CoAna {pkg_version}",
    ]
    if author:
        chips.insert(0, f"Author: {author}")

    elapsed = time.monotonic() - t_start
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<header class="rpt">
  <h1>{html.escape(title)}</h1>
  <div class="sub">{html.escape(subtitle)}</div>
  <div class="chips">{''.join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)}</div>
</header>
<div class="tiles">{''.join(tiles)}</div>
{toc_html}
{''.join(sec)}
<footer class="rpt">
Generated by CoAna (CoAnaPure engine) in {elapsed:.1f}&nbsp;s
on {html.escape(platform.node())} &middot; Python {sys.version.split()[0]},
NumPy {np.__version__}, SciPy {scipy_v}, pandas {pd.__version__},
Matplotlib {matplotlib.__version__}.
</footer>
</div>
</body>
</html>
"""
    out_html.write_text(doc, encoding="utf-8")
    if verbose:
        size_kb = out_html.stat().st_size / 1024
        print(f"Report written: {out_html} ({size_kb:,.0f} KB, {elapsed:.1f} s)", flush=True)
    return str(out_html)
