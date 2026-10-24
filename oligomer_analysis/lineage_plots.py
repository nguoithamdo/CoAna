from __future__ import annotations

"""Figure utilities for split/merge-aware aggregation analysis.

This module implements the four core figure classes for protein aggregation:

1. whole-system background plots
2. lineage graph with split/merge events
3. fate plot showing lineage size vs time
4. deep-dive dashboards for representative on- and off-pathway lineages

All major size-bearing plots accept ``calibration`` so the same code works for
small oligomers and large assemblies up to ~100 monomers or more.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .lineage import representative_pathways_for_lineages
from .models import ContactCache, LineageTrackingResult
from .monitor import monitor_oligomer_label
from .system_analysis import build_system_summary_dataframe
from .plotting_common import (
    CLASS_COLORS,
    EVENT_COLORS,
    PlotCalibration,
    calibrate_contact_axis,
    calibrate_size_axis,
    resolve_plot_calibration,
    scale_node_sizes,
    legend_enabled,
)


def plot_whole_system_background(
    cache: ContactCache,
    result: LineageTrackingResult,
    *,
    figsize=(15, 8),
    calibration: Optional[PlotCalibration | dict] = None,
    max_oligomer_size: Optional[int] = None,
    size_tick_step: Optional[int] = None,
    size_scale: Optional[str] = None,
    show_legend: Optional[bool] = None,
):
    """Six-panel whole-system background dashboard.

    Use ``calibration=PlotCalibration(max_oligomer_size=100,
    size_tick_step=10)`` when large aggregates appear.
    """
    cal = resolve_plot_calibration(
        calibration,
        max_oligomer_size=max_oligomer_size,
        size_tick_step=size_tick_step,
        size_scale=size_scale,
    )
    df_system = build_system_summary_dataframe(cache, result.frames_clusters, cluster_min_size=2)
    inter_col = 'ninter_contacts' if 'ninter_contacts' in df_system.columns else 'inter_contacts'
    total_col = 'ntotal_contacts' if 'ntotal_contacts' in df_system.columns else 'total_contacts'

    observed_max = float(max(df_system['largest_cluster'].max(), df_system['mean_cluster_size'].max())) if len(df_system) else None

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)

    axes[0, 0].plot(df_system['frame'], df_system['n_clusters'], marker='o', markersize=cal.marker_size, linewidth=cal.line_width)
    axes[0, 0].set_title('Number of oligomeric clusters vs frame (size ≥ 2)')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('n_clusters')
    if cal.frame_xlim is not None:
        axes[0, 0].set_xlim(*cal.frame_xlim)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(df_system['frame'], df_system['largest_cluster'], marker='o', markersize=cal.marker_size, linewidth=cal.line_width)
    axes[0, 1].set_title('Largest oligomer size vs frame')
    axes[0, 1].set_xlabel('Frame')
    calibrate_size_axis(axes[0, 1], cal, observed_max=observed_max, ylabel='Largest size')
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(df_system['frame'], df_system['mean_cluster_size'], marker='o', markersize=cal.marker_size, linewidth=cal.line_width)
    axes[0, 2].set_title('Mean oligomer size vs frame (size ≥ 2)')
    axes[0, 2].set_xlabel('Frame')
    calibrate_size_axis(axes[0, 2], cal, observed_max=observed_max, ylabel='Mean size')
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(df_system['frame'], df_system[inter_col], marker='o', markersize=cal.marker_size, linewidth=cal.line_width, label='inter')
    axes[1, 0].plot(df_system['frame'], df_system[total_col], marker='o', markersize=cal.marker_size, linewidth=cal.line_width, label='total')
    axes[1, 0].set_title('System contact counts')
    axes[1, 0].set_xlabel('Frame')
    calibrate_contact_axis(axes[1, 0], cal, ylabel='Contacts')
    if legend_enabled(cal, show_legend):
        axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(df_system['frame'], df_system['n_free_monomers'], marker='o', markersize=cal.marker_size, linewidth=cal.line_width, label='free')
    axes[1, 1].plot(df_system['frame'], df_system['n_bound_monomers'], marker='o', markersize=cal.marker_size, linewidth=cal.line_width, label='bound')
    axes[1, 1].set_title('Free vs bound monomers')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Count')
    if cal.frame_xlim is not None:
        axes[1, 1].set_xlim(*cal.frame_xlim)
    if legend_enabled(cal, show_legend):
        axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    denom = (df_system['n_bound_monomers'] + df_system['n_free_monomers']).replace(0, np.nan)
    bound_fraction = df_system['n_bound_monomers'] / denom
    axes[1, 2].plot(df_system['frame'], bound_fraction, marker='o', markersize=cal.marker_size, linewidth=cal.line_width)
    axes[1, 2].set_title('Bound monomer fraction')
    axes[1, 2].set_xlabel('Frame')
    axes[1, 2].set_ylabel('Fraction bound')
    axes[1, 2].set_ylim(0, 1.05)
    if cal.frame_xlim is not None:
        axes[1, 2].set_xlim(*cal.frame_xlim)
    axes[1, 2].grid(alpha=0.3)

    return fig, axes, df_system


def plot_lineage_graph(
    result: LineageTrackingResult,
    *,
    figsize=(11, 8),
    layout_seed: int = 13,
    show_labels: bool = True,
    calibration: Optional[PlotCalibration | dict] = None,
    node_size_by: str = 'duration',
    show_legend: Optional[bool] = None,
):
    """Graph of segment-level events colored by lineage fate class.

    ``node_size_by`` can be ``duration``, ``max_size``, ``mean_size``, or
    ``contribution``. Node areas are always bounded by the calibration object.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError('plot_lineage_graph requires networkx') from exc

    cal = resolve_plot_calibration(calibration)
    lin_by_id = {lin.lineage_id: lin for lin in result.lineages}

    raw_values = []
    for seg in result.segments:
        lineage = lin_by_id.get(seg.lineage_id or '')
        if node_size_by == 'duration':
            raw_values.append(seg.duration)
        elif node_size_by == 'max_size':
            raw_values.append(seg.max_size)
        elif node_size_by == 'mean_size':
            raw_values.append(seg.mean_size)
        elif node_size_by == 'contribution':
            raw_values.append(lineage.contribution_score if lineage else 0.0)
        else:
            raise ValueError("node_size_by must be duration, max_size, mean_size, or contribution")

    scaled_sizes = scale_node_sizes(raw_values, cal)

    G = nx.DiGraph()
    for seg, node_area in zip(result.segments, scaled_sizes):
        lineage = lin_by_id.get(seg.lineage_id or '')
        class_label = lineage.class_label if lineage else 'uncertain'
        contribution = lineage.contribution_score if lineage else 0.0
        G.add_node(
            seg.segment_id,
            size=float(node_area),
            color=CLASS_COLORS.get(class_label, 'gray'),
            label=f"{seg.segment_id}\nsize≈{seg.mean_size:.1f}\nT={seg.duration}\nC={contribution:.2f}",
        )

    edge_labels = {}
    for ev in result.events:
        for parent in ev.parent_segment_ids:
            for child in ev.child_segment_ids:
                G.add_edge(parent, child, event_type=ev.event_type)
                edge_labels[(parent, child)] = ev.event_type

    pos = nx.spring_layout(G, seed=layout_seed)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=[G.nodes[n]['size'] for n in G.nodes],
        node_color=[G.nodes[n]['color'] for n in G.nodes],
        alpha=0.9,
    )
    edge_colors = [EVENT_COLORS.get(G.edges[e].get('event_type', 'rearrangement'), 'gray') for e in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=14, width=1.8, edge_color=edge_colors)
    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, labels={n: G.nodes[n]['label'] for n in G.nodes}, font_size=8)
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=7)
    if legend_enabled(cal, show_legend):
        from matplotlib.lines import Line2D
        class_handles = [
            Line2D([0], [0], marker='o', color='w', label=cls,
                   markerfacecolor=color, markeredgecolor='black', markersize=8)
            for cls, color in CLASS_COLORS.items()
        ]
        ax.legend(handles=class_handles, title='Lineage fate', loc='best', fontsize=8, title_fontsize=9)
    ax.set_title('Aggregation lineage graph (segments linked by split/merge-aware events)')
    ax.axis('off')
    return fig, ax


def plot_lineage_fate(
    result: LineageTrackingResult,
    *,
    figsize=(12, 7),
    annotate: bool = False,
    calibration: Optional[PlotCalibration | dict] = None,
    max_oligomer_size: Optional[int] = None,
    size_tick_step: Optional[int] = None,
    size_scale: Optional[str] = None,
    show_legend: Optional[bool] = None,
):
    """Fate plot: time on x, oligomer size on y, color by lineage class."""
    cal = resolve_plot_calibration(
        calibration,
        max_oligomer_size=max_oligomer_size,
        size_tick_step=size_tick_step,
        size_scale=size_scale,
    )
    seg_by_id = {seg.segment_id: seg for seg in result.segments}
    observed_max = max((seg.max_size for seg in result.segments), default=None)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for lin in result.lineages:
        color = CLASS_COLORS.get(lin.class_label, 'gray')
        for sid in lin.segment_ids:
            seg = seg_by_id[sid]
            ax.plot(seg.frames, seg.sizes, marker='o', markersize=cal.marker_size, linewidth=cal.line_width, color=color, alpha=0.85)
            if annotate and seg.frames:
                ax.text(seg.frames[0], seg.sizes[0], sid, fontsize=7, color=color)
    ax.set_title('Lineage fate plot (segment size vs time)')
    ax.set_xlabel('Frame')
    calibrate_size_axis(ax, cal, observed_max=observed_max, ylabel='Oligomer size')
    if legend_enabled(cal, show_legend):
        from matplotlib.lines import Line2D
        present_classes = [lin.class_label for lin in result.lineages]
        handles = [
            Line2D([0], [0], color=color, lw=2, marker='o', label=cls)
            for cls, color in CLASS_COLORS.items()
            if cls in present_classes
        ]
        if handles:
            ax.legend(handles=handles, title='Lineage fate', loc='best', fontsize=8, title_fontsize=9)
    ax.grid(alpha=0.3)
    return fig, ax


def _find_representative_segment_pathways(result: LineageTrackingResult):
    reps = representative_pathways_for_lineages(result)
    rep_by_label = {p.label: p for p in reps}
    return {lin.lineage_id: rep_by_label.get(lin.representative_segment_id or '') for lin in result.lineages}


def plot_on_off_pathway_dashboards(
    cache: ContactCache,
    result: LineageTrackingResult,
    *,
    inter_thresh: int,
    temperature_K: float = 330.0,
    on_lineage_id: Optional[str] = None,
    off_lineage_id: Optional[str] = None,
    calibration: Optional[PlotCalibration | dict] = None,
    show_legend: Optional[bool] = None,
):
    """Deep-dive dashboards for representative on- and off-pathway lineages."""
    on_candidates = [lin for lin in result.lineages if lin.class_label == 'on-pathway']
    off_candidates = [lin for lin in result.lineages if lin.class_label == 'off-pathway']
    if on_lineage_id is not None:
        on_candidates = [lin for lin in result.lineages if lin.lineage_id == on_lineage_id]
    if off_lineage_id is not None:
        off_candidates = [lin for lin in result.lineages if lin.lineage_id == off_lineage_id]

    on_lineage = max(on_candidates, key=lambda lin: (lin.contribution_score, lin.persistence_score), default=None)
    off_lineage = max(off_candidates, key=lambda lin: (lin.persistence_score, -lin.contribution_score), default=None)
    rep_by_lineage = _find_representative_segment_pathways(result)

    outputs = {}
    if on_lineage is not None and rep_by_lineage.get(on_lineage.lineage_id) is not None:
        pathway = rep_by_lineage[on_lineage.lineage_id]
        outputs['on_pathway'] = monitor_oligomer_label(
            cache=cache,
            finished=[pathway],
            label=pathway.label,
            inter_thresh=inter_thresh,
            temperature_K=temperature_K,
            transition_thr=1e-3,
            compactness_mode='rg',
            rg_use_mic=True,
            hide_member_curves=False,
            hide_oligomer_curve=True,
            calibration=calibration,
            show_legend=show_legend,
        )
    if off_lineage is not None and rep_by_lineage.get(off_lineage.lineage_id) is not None:
        pathway = rep_by_lineage[off_lineage.lineage_id]
        outputs['off_pathway'] = monitor_oligomer_label(
            cache=cache,
            finished=[pathway],
            label=pathway.label,
            inter_thresh=inter_thresh,
            temperature_K=temperature_K,
            transition_thr=1e-3,
            compactness_mode='rg',
            rg_use_mic=True,
            hide_member_curves=False,
            hide_oligomer_curve=True,
            calibration=calibration,
            show_legend=show_legend,
        )
    return outputs


__all__ = [
    'plot_whole_system_background',
    'plot_lineage_graph',
    'plot_lineage_fate',
    'plot_on_off_pathway_dashboards',
]
