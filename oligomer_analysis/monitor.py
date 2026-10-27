# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

from typing import Iterable, Optional

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .graph_tools import build_cluster_transition_matrix, intra_cluster_contacts, pathway_transition_matrix
from .metrics import (
    build_contact_graph_snapshot,
    compactness_timeseries,
    compute_free_energy_from_sizes,
    msd_pathway_peak_members,
    small_worldness_S,
)
from .models import ContactCache, Pathway
from .plotting_common import (
    PlotCalibration,
    calibrate_size_axis,
    calibrate_contact_axis,
    matrix_display_slice,
    resolve_plot_calibration,
    scale_node_sizes,
    display_oligomer_label,
    legend_enabled,
)

PANEL_NAMES = {
    1: "size",
    2: "intercontacts",
    3: "free_energy",
    4: "transition_matrix",
    5: "transition_graph",
    6: "snapshot_connections",
    7: "compactness",
    8: "msd",
    9: "small_world",
}
PANEL_ALIASES = {v: k for k, v in PANEL_NAMES.items()}


def _resolve_panel_id(panel) -> int:
    if isinstance(panel, int):
        if panel not in PANEL_NAMES:
            raise ValueError(f"panel must be 1..9, got {panel}")
        return panel
    if isinstance(panel, str):
        key = panel.strip().lower()
        if key in PANEL_ALIASES:
            return PANEL_ALIASES[key]
        raise ValueError(f"Unknown panel name '{panel}'")
    raise TypeError("panel must be an int or a str")


def _find_pathway(finished: Iterable[Pathway], label: str) -> Pathway:
    """Find a pathway by label, accepting either dataclass or dict pathways."""
    for pathway in finished:
        if _pathway_get(pathway, "label") == label:
            return pathway
    raise ValueError(f"Pathway label not found: {label}")


def _pathway_get(pathway, key):
    """Support both dataclass-style pathways and dict-style pathways."""
    return getattr(pathway, key) if hasattr(pathway, key) else pathway[key]

def _get_label_map(cache: ContactCache, use_segids: bool):
    if not use_segids:
        return None
    segids = cache.meta.get("segids")
    if segids is None:
        return None
    return {i: str(segids[i]) for i in range(len(segids))}


def _snapshot_indices(pathway: Pathway) -> tuple[int, int, int]:
    sizes = list(_pathway_get(pathway, "sizes"))
    t_peak = next(i for i, z in enumerate(sizes) if z == max(sizes))
    return 0, t_peak, int(_pathway_get(pathway, "duration")) - 1


def _transition_prob_matrix(T: np.ndarray, normalise: Optional[str] = "all") -> np.ndarray:
    M = T.astype(float)
    if normalise == "row":
        row = M.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        M = M / row
    elif normalise == "all":
        total = M.sum()
        if total > 0:
            M = M / total
    return M


def plot_size_transition_matrix(
    T: np.ndarray,
    normalise: Optional[str] = None,
    cmap: str = "Blues",
    *,
    min_size: int = 1,
    calibration: Optional[PlotCalibration | dict] = None,
) -> np.ndarray:
    """Plot a cluster-size transition matrix with calibrated size ticks.

    For oligomers up to size 100, use for example:
    ``PlotCalibration(max_oligomer_size=100, size_tick_step=10,
    matrix_max_size=100)``.
    """
    cal = resolve_plot_calibration(calibration)
    M = _transition_prob_matrix(T, normalise=normalise)
    plt.figure(figsize=(6, 5))
    M_plot, labels = matrix_display_slice(M, min_size=min_size, calibration=cal)
    im = plt.imshow(M_plot, origin="lower", cmap=cmap)
    plt.colorbar(im, shrink=0.8, label="counts" if normalise is None else "probability")

    step = cal.size_tick_step or max(1, len(labels) // 12)
    tick_idx = [i for i, lab in enumerate(labels) if (lab - labels[0]) % step == 0] if labels else []
    plt.xticks(tick_idx, [labels[i] for i in tick_idx])
    plt.yticks(tick_idx, [labels[i] for i in tick_idx])

    plt.xlabel("Size at frame t+1")
    plt.ylabel("Size at frame t")
    plt.title("Cluster-size transition matrix")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()
    return M


def plot_cluster_contacts(pathways: Iterable[Pathway], cache: ContactCache, min_duration: int = 1, *, calibration: Optional[PlotCalibration | dict] = None, label_mode: Optional[str] = None, show_legend: Optional[bool] = None) -> None:
    cal = resolve_plot_calibration(calibration)
    plt.figure(figsize=(8, 5))
    plotted = 0
    for p in pathways:
        if p.duration < min_duration:
            continue
        s = p.start_idx
        e = s + p.duration
        x = cache.frames[s:e]
        y = [intra_cluster_contacts(clu, cache.cmaps[s + t]) for t, clu in enumerate(p.sets)]
        plt.plot(x, y, marker="o", markersize=cal.marker_size, linewidth=cal.line_width, label=display_oligomer_label(p.label, cal, label_mode))
        plotted += 1
    if plotted:
        plt.title("Intra-cluster inter-segment contacts vs time")
        plt.xlabel("Frame")
        plt.ylabel("Residue contacts")
        if legend_enabled(cal, show_legend):
            plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.text(0.5, 0.5, f"No pathway lasts ≥ {min_duration} frames", ha="center", va="center", transform=plt.gca().transAxes)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_total_contacts_per_frame(cache: ContactCache) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(cache.frames, cache.ntotal, marker="o")
    plt.xlabel("Frame")
    plt.ylabel("Total residue contacts (intra+inter)")
    plt.title("Total contacts vs frame")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_total_intercontacts_per_frame(cache: ContactCache) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(cache.frames, cache.ninter, marker="o")
    plt.xlabel("Frame")
    plt.ylabel("Total inter-segment residue contacts")
    plt.title("Intercontacts vs frame")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pathway_sizes(
    frames: np.ndarray,
    pathways: Iterable[Pathway],
    min_duration: int = 1,
    *,
    calibration: Optional[PlotCalibration | dict] = None,
    label_mode: Optional[str] = None,
    show_legend: Optional[bool] = None,
) -> None:
    """Plot old linear pathway sizes with calibrated y-axis."""
    cal = resolve_plot_calibration(calibration)
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    plotted = 0
    observed_max = 0.0
    for p in pathways:
        if p.duration < min_duration:
            continue
        sidx = p.start_idx
        e = sidx + p.duration
        y = np.asarray(p.sizes, dtype=float)
        observed_max = max(observed_max, float(np.max(y)) if y.size else 0.0)
        ax.plot(frames[sidx:e], y, marker="o", markersize=cal.marker_size, linewidth=cal.line_width, label=display_oligomer_label(p.label, cal, label_mode))
        plotted += 1
    if plotted:
        ax.set_title(f"Oligomer size vs time (life ≥ {min_duration} frames)")
        ax.set_xlabel("Frame")
        calibrate_size_axis(ax, cal, observed_max=observed_max, ylabel="Cluster size")
        if legend_enabled(cal, show_legend):
            ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.text(0.5, 0.5, f"No pathway lasts ≥ {min_duration} frames", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def network_plot_global(
    M: np.ndarray,
    thr: float = 1e-3,
    seed: int = 40,
    remove_self_loops: bool = True,
    *,
    min_size: int = 1,
    calibration: Optional[PlotCalibration | dict] = None,
) -> None:
    """Directed global cluster-size transition graph with calibrated node sizes.

    The old implementation scaled node area directly as size * 500. That becomes
    unreadable when cluster size approaches 100. This version caps node area using
    ``PlotCalibration.node_size_min`` and ``node_size_max``.
    """
    import networkx as nx

    cal = resolve_plot_calibration(calibration)
    max_size = M.shape[0] - 1
    if cal.matrix_max_size is not None:
        max_size = min(max_size, int(cal.matrix_max_size))
    if max_size < min_size:
        raise ValueError("No size states available for the requested min_size/matrix_max_size")

    labels = np.arange(int(min_size), int(max_size) + 1)
    P = np.asarray(M)[min_size:max_size + 1, min_size:max_size + 1]

    G = nx.DiGraph()
    node_areas = scale_node_sizes(labels, cal)
    for idx, size_label in enumerate(labels):
        G.add_node(idx, size=float(node_areas[idx]), label=str(int(size_label)))

    edges_fwd, widths_fwd = [], []
    edges_rev, widths_rev = [], []
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] <= thr:
                continue
            if remove_self_loops and i == j:
                continue
            G.add_edge(i, j, weight=float(P[i, j]))
            if i < j:
                edges_fwd.append((i, j))
                widths_fwd.append(max(0.5, float(P[i, j]) * 40))
            else:
                edges_rev.append((i, j))
                widths_rev.append(max(0.5, float(P[i, j]) * 40))

    pos = nx.spring_layout(G, seed=seed, weight="weight")
    node_sizes = [G.nodes[n]["size"] for n in G.nodes]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes})

    edge_common = dict(arrows=True, arrowstyle='-|>', arrowsize=20, min_source_margin=20, min_target_margin=25)
    nx.draw_networkx_edges(G, pos, edgelist=edges_fwd, width=widths_fwd, edge_color="royalblue", connectionstyle="arc3,rad=0.2", **edge_common)
    nx.draw_networkx_edges(G, pos, edgelist=edges_rev, width=widths_rev, edge_color="orangered", connectionstyle="arc3,rad=-0.25", **edge_common)
    plt.title("Cluster-size Transition Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def _draw_transition_graph(ax, M: np.ndarray, thr: float = 1e-3, seed: int = 40, node_size_mode: str = "total", *, calibration: Optional[PlotCalibration | dict] = None) -> None:
    import networkx as nx

    cal = resolve_plot_calibration(calibration)
    P = np.asarray(M)[1:, 1:]
    if node_size_mode == "out":
        mass = P.sum(axis=1)
    elif node_size_mode == "in":
        mass = P.sum(axis=0)
    elif node_size_mode == "total":
        mass = P.sum(axis=1) + P.sum(axis=0)
    else:
        raise ValueError("node_size_mode must be one of: out, in, total")

    if mass.size and float(np.nanmax(mass)) > 0:
        node_sizes = scale_node_sizes(mass, cal)
    else:
        node_sizes = np.full(P.shape[0], (cal.node_size_min + cal.node_size_max) / 2.0)

    G = nx.DiGraph()
    for s in range(P.shape[0]):
        G.add_node(s, size=float(node_sizes[s]))

    edges_fwd, widths_fwd = [], []
    edges_rev, widths_rev = [], []
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] <= thr:
                continue
            if i == j:
                continue
            G.add_edge(i, j, weight=float(P[i, j]))
            w = float(P[i, j]) * 30.0
            if i < j:
                edges_fwd.append((i, j))
                widths_fwd.append(w)
            else:
                edges_rev.append((i, j))
                widths_rev.append(w)

    pos = nx.spring_layout(G, seed=seed, weight="weight")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[G.nodes[n]["size"] for n in G.nodes])
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: str(n + 1) for n in G.nodes})
    edge_common = dict(arrows=True, arrowstyle='-|>', arrowsize=16, min_source_margin=12, min_target_margin=14)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges_fwd, width=widths_fwd, connectionstyle="arc3,rad=0.2", **edge_common)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges_rev, width=widths_rev, connectionstyle="arc3,rad=-0.25", **edge_common)
    ax.set_title("Transition graph")
    ax.axis("off")


def _draw_chain_graph(ax, members, cmap, inter_thresh, title="", label_map=None, node_size: float = 420.0):
    import networkx as nx

    members = list(members)
    if len(members) == 0:
        ax.text(0.5, 0.5, "No members", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    G = nx.Graph()
    for m in members:
        G.add_node(m)
    for i in range(len(members) - 1):
        for j in range(i + 1, len(members)):
            a = members[i]
            b = members[j]
            if cmap[a, b] >= inter_thresh:
                G.add_edge(a, b, weight=int(cmap[a, b]))

    pos = {m: (k, 0.0) for k, m in enumerate(members)}
    labels = {m: (label_map[m] if (label_map and m in label_map) else str(m)) for m in G.nodes}
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=float(node_size))
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=8)
    nx.draw_networkx_edges(G, pos, ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def _fmt_member_list(members, per_line: int = 9, max_lines: int = 8) -> str:
    """Compact multi-line member-id listing for the panel-6 side notes.

    Plain ints (NumPy scalars would repr as ``np.int64(...)``), a fixed number
    of ids per line, and a hard line cap so an 80-mer cannot overflow into the
    neighbouring panels.
    """
    vals = [str(int(m)) for m in members]
    lines = [" ".join(vals[i : i + per_line]) for i in range(0, len(vals), per_line)]
    if len(lines) > max_lines:
        shown = max_lines * per_line
        lines = lines[:max_lines] + [f"... (+{len(vals) - shown} more)"]
    return "\n".join(lines)


def _draw_three_config_chain_graphs(ax, members_start, cmap_start, fr_start, members_peak, cmap_peak, fr_peak, members_last, cmap_last, fr_last, inter_thresh, label_map=None, node_size: float = 420.0):
    ax.axis("off")
    ax_top = ax.inset_axes([0.02, 0.68, 0.72, 0.30])
    ax_mid = ax.inset_axes([0.02, 0.36, 0.72, 0.30])
    ax_bot = ax.inset_axes([0.02, 0.04, 0.72, 0.30])

    _draw_chain_graph(ax_top, members_start, cmap_start, inter_thresh, title=f"Start (fr {fr_start})", label_map=label_map, node_size=node_size)
    _draw_chain_graph(ax_mid, members_peak, cmap_peak, inter_thresh, title=f"Peak (fr {fr_peak})", label_map=label_map, node_size=node_size)
    _draw_chain_graph(ax_bot, members_last, cmap_last, inter_thresh, title=f"Last (fr {fr_last})", label_map=label_map, node_size=node_size)

    ax.text(0.76, 0.95, f"Start ({len(members_start)}):\n{_fmt_member_list(members_start)}",
            transform=ax.transAxes, va="top", fontsize=7)
    ax.text(0.76, 0.62, f"Peak ({len(members_peak)}):\n{_fmt_member_list(members_peak)}",
            transform=ax.transAxes, va="top", fontsize=7)
    ax.text(0.76, 0.29, f"Last ({len(members_last)}):\n{_fmt_member_list(members_last)}",
            transform=ax.transAxes, va="top", fontsize=7)


def _draw_snapshot_network(ax, G, label_map=None, node_size: float = 700.0):
    import networkx as nx

    ax.axis("off")
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
        return
    pos = nx.circular_layout(G)
    labels = {n: str(label_map.get(n, n)) for n in G.nodes} if label_map else {n: str(n) for n in G.nodes}
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="black", width=1.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="royalblue", node_size=float(node_size))
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_color="white", font_size=10)


def draw_panel9_small_world(ax9, *, cache: ContactCache, pathway: Pathway, members_start, fr_start, members_peak, fr_peak, members_last, fr_last, inter_thresh: int, label_map=None, n_rand: int = 30, seed: int = 0, node_size: float = 700.0):
    ax9.axis("off")

    t_start, t_peak, t_last = _snapshot_indices(pathway)
    cmap_start = cache.cmaps[int(_pathway_get(pathway, "start_idx")) + t_start]
    cmap_peak = cache.cmaps[int(_pathway_get(pathway, "start_idx")) + t_peak]
    cmap_last = cache.cmaps[int(_pathway_get(pathway, "start_idx")) + t_last]

    Gs = build_contact_graph_snapshot(members_start, cmap_start, inter_thresh)
    Gp = build_contact_graph_snapshot(members_peak, cmap_peak, inter_thresh)
    Gl = build_contact_graph_snapshot(members_last, cmap_last, inter_thresh)

    Ss = small_worldness_S(Gs, n_rand=n_rand, seed=seed + 1)
    Sp = small_worldness_S(Gp, n_rand=n_rand, seed=seed + 2)
    Sl = small_worldness_S(Gl, n_rand=n_rand, seed=seed + 3)

    def fmt(x):
        return f"{x:.2f}" if np.isfinite(x) else "NaN"

    def make_card(parent_ax, rect):
        card = parent_ax.inset_axes(rect)
        card.set_xticks([])
        card.set_yticks([])
        card.set_facecolor("white")
        for spine in card.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_edgecolor("royalblue")
            spine.set_linestyle((0, (6, 4)))
        return card

    def add_network_area(card_ax):
        net_ax = card_ax.inset_axes([0.02, 0.06, 0.96, 0.86])
        net_ax.axis("off")
        return net_ax

    card1 = make_card(ax9, [0.02, 0.68, 0.96, 0.30])
    card2 = make_card(ax9, [0.02, 0.36, 0.96, 0.30])
    card3 = make_card(ax9, [0.02, 0.04, 0.96, 0.30])
    net1 = add_network_area(card1)
    net2 = add_network_area(card2)
    net3 = add_network_area(card3)

    card1.text(0.02, 0.98, f"Start (fr {fr_start})   S={fmt(Ss.S)}", transform=card1.transAxes, va="top", fontsize=10)
    card2.text(0.02, 0.98, f"Peak  (fr {fr_peak})   S={fmt(Sp.S)}", transform=card2.transAxes, va="top", fontsize=10)
    card3.text(0.02, 0.98, f"Last  (fr {fr_last})   S={fmt(Sl.S)}", transform=card3.transAxes, va="top", fontsize=10)

    _draw_snapshot_network(net1, Gs, label_map=label_map, node_size=node_size)
    _draw_snapshot_network(net2, Gp, label_map=label_map, node_size=node_size)
    _draw_snapshot_network(net3, Gl, label_map=label_map, node_size=node_size)

    return {"start": Ss, "peak": Sp, "last": Sl}


def _draw_panel(fig, ax, panel_id: int, *, cache: ContactCache, pathway: Pathway, frames: np.ndarray, sizes: np.ndarray, inter_thresh: int, compactness_mode: str = "density", rg_use_mic: bool = True, transition_thr: float = 1e-3, label_map=None, msd_max_lag: int = 150, hide_member_curves: bool = True, hide_oligomer_curve: bool = False, sw_n_rand: int = 30, sw_seed: int = 0, calibration: Optional[PlotCalibration | dict] = None, panel6_node_size: float = 420.0, panel9_node_size: float = 700.0, show_legend: Optional[bool] = None):
    TITLE_FS = 10
    LABEL_FS = 9
    cal = resolve_plot_calibration(calibration)
    s = int(_pathway_get(pathway, "start_idx"))
    dur = int(_pathway_get(pathway, "duration"))
    sets = list(_pathway_get(pathway, "sets"))
    label = _pathway_get(pathway, "label")
    display_label = display_oligomer_label(label, cal)

    if panel_id == 1:
        ax.plot(frames, sizes, marker="o", markersize=cal.marker_size, linewidth=cal.line_width)
        ax.axvspan(frames[0], frames[-1], alpha=0.10)
        ax.set_title(f"Size vs frame\n{display_label}", fontsize=TITLE_FS)
        ax.set_xlabel("Frame", fontsize=LABEL_FS)
        calibrate_size_axis(ax, cal, observed_max=float(np.max(sizes)) if len(sizes) else None, ylabel="Oligomer size")
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.95, f"life={dur}", transform=ax.transAxes, va="top", fontsize=8)
        return {}

    if panel_id == 2:
        intercontacts = np.array([intra_cluster_contacts(sets[t], cache.cmaps[s + t]) for t in range(dur)], dtype=float)
        ax.plot(frames, intercontacts, marker="o")
        ax.set_title("Intercontacts (within oligomer)", fontsize=TITLE_FS)
        ax.set_xlabel("Frame", fontsize=LABEL_FS)
        calibrate_contact_axis(ax, cal, ylabel="Contacts")
        ax.grid(alpha=0.3)
        return {"intercontacts": intercontacts}

    if panel_id == 3:
        profile = compute_free_energy_from_sizes(sizes, shift_min_to_zero=True)
        # Plot the uncertainty, not just the curve. The profile's whole vertical
        # extent is ln(c_max/c_min) and most bins hold a handful of frames, so
        # without error bars this panel reads as structure it cannot support.
        if profile.F_stderr is not None:
            ax.errorbar(
                profile.sizes, profile.F_over_kBT, yerr=profile.F_stderr,
                marker="o", capsize=2, linewidth=1.0, elinewidth=0.8,
            )
        else:
            ax.plot(profile.sizes, profile.F_over_kBT, marker="o")
        ax.set_title("Free-energy landscape", fontsize=TITLE_FS)
        ax.set_xlabel("Size", fontsize=LABEL_FS)
        ax.set_ylabel("F / (kB·T)", fontsize=LABEL_FS)
        ax.grid(alpha=0.3)
        return {"free_energy": profile}

    if panel_id == 4:
        T = pathway_transition_matrix(list(sizes))
        M = _transition_prob_matrix(T, normalise="all")
        Mp, labels = matrix_display_slice(M, min_size=1, calibration=cal)
        im = ax.imshow(Mp, origin="lower")
        ax.set_title("Transition matrix", fontsize=TITLE_FS)
        ax.set_xlabel("size(t+1)", fontsize=LABEL_FS)
        ax.set_ylabel("size(t)", fontsize=LABEL_FS)
        step = cal.size_tick_step or max(1, len(labels) // 8)
        tick_idx = [i for i, lab in enumerate(labels) if (lab - labels[0]) % step == 0] if labels else []
        ax.set_xticks(tick_idx)
        ax.set_yticks(tick_idx)
        ax.set_xticklabels([labels[i] for i in tick_idx], fontsize=8)
        ax.set_yticklabels([labels[i] for i in tick_idx], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        return {"T_counts": T, "M_prob": M}

    if panel_id == 5:
        T = pathway_transition_matrix(list(sizes))
        M = _transition_prob_matrix(T, normalise="all")
        try:
            _draw_transition_graph(ax, M, thr=transition_thr, node_size_mode="total", calibration=cal)
        except ImportError:
            ax.axis("off")
            ax.text(0.5, 0.5, "networkx not installed", ha="center", va="center", transform=ax.transAxes)
        return {"M_prob": M}

    if panel_id == 6:
        t_start, t_peak, t_last = _snapshot_indices(pathway)
        members_start = sorted(list(sets[t_start]))
        members_peak = sorted(list(sets[t_peak]))
        members_last = sorted(list(sets[t_last]))
        try:
            _draw_three_config_chain_graphs(
                ax,
                members_start,
                cache.cmaps[s + t_start],
                int(frames[t_start]),
                members_peak,
                cache.cmaps[s + t_peak],
                int(frames[t_peak]),
                members_last,
                cache.cmaps[s + t_last],
                int(frames[t_last]),
                inter_thresh=inter_thresh,
                label_map=label_map,
                node_size=panel6_node_size,
            )
        except ImportError:
            ax.axis("off")
            ax.text(0.5, 0.5, "networkx not installed", ha="center", va="center", transform=ax.transAxes)
        return {}

    if panel_id == 7:
        y = compactness_timeseries(cache, pathway, inter_thresh, mode=compactness_mode, rg_use_mic=rg_use_mic)
        if np.isnan(y).all():
            ax.axis("off")
            ax.text(0.5, 0.5, f"Compactness '{compactness_mode}' unavailable", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.plot(frames, y, marker="o")
            ax.grid(alpha=0.3)
            ax.set_xlabel("Frame", fontsize=LABEL_FS)
            ax.set_ylabel(compactness_mode, fontsize=LABEL_FS)
        ax.set_title(f"Compactness ({compactness_mode})", fontsize=TITLE_FS)
        return {"compactness": y}

    if panel_id == 8:
        try:
            msd_out = msd_pathway_peak_members(cache, pathway, max_lag=msd_max_lag)
            lags = msd_out["lags"]
            if not hide_member_curves:
                for curve in msd_out["msd_members"]:
                    ax.plot(lags, curve, linewidth=0.8, alpha=0.35)
            ax.plot(lags, msd_out["msd_mean"], linewidth=2.2, marker="o", label="mean MSD (members)")
            if not hide_oligomer_curve:
                ax.plot(lags, msd_out["msd_oligo"], linewidth=2.2, marker="o", label="oligomer center MSD")
            ax.set_title(f"MSD (pathway-only, peak N={len(msd_out['peak_members'])})", fontsize=TITLE_FS)
            ax.set_xlabel("Lag (frames)", fontsize=LABEL_FS)
            ax.set_ylabel("MSD", fontsize=LABEL_FS)
            ax.grid(alpha=0.3)
            if legend_enabled(cal, show_legend):
                ax.legend(fontsize=8, loc="upper left", frameon=True)
            return {"msd": msd_out}
        except Exception as exc:
            ax.axis("off")
            ax.text(0.5, 0.5, f"MSD unavailable\n{exc}", ha="center", va="center", transform=ax.transAxes)
            return {"msd": None}

    if panel_id == 9:
        t_start, t_peak, t_last = _snapshot_indices(pathway)
        members_start = sorted(list(sets[t_start]))
        members_peak = sorted(list(sets[t_peak]))
        members_last = sorted(list(sets[t_last]))
        try:
            sw_scores = draw_panel9_small_world(
                ax,
                cache=cache,
                pathway=pathway,
                members_start=members_start,
                fr_start=int(frames[t_start]),
                members_peak=members_peak,
                fr_peak=int(frames[t_peak]),
                members_last=members_last,
                fr_last=int(frames[t_last]),
                inter_thresh=inter_thresh,
                label_map=label_map,
                n_rand=sw_n_rand,
                seed=sw_seed,
                node_size=panel9_node_size,
            )
            return {"small_world": sw_scores}
        except ImportError:
            ax.axis("off")
            ax.text(0.5, 0.5, "networkx not installed", ha="center", va="center", transform=ax.transAxes)
            return {"small_world": None}

    raise ValueError(f"Unknown panel_id {panel_id}")


def plot_oligomer_panel(cache: ContactCache, finished: Iterable[Pathway], label: str, panel=1, *, inter_thresh: int = 1, use_segids: bool = False, compactness_mode: str = "density", rg_use_mic: bool = True, transition_thr: float = 1e-3, msd_max_lag: int = 150, hide_member_curves: bool = True, hide_oligomer_curve: bool = False, sw_n_rand: int = 30, sw_seed: int = 0, figsize=(8, 6), show: bool = True, calibration: Optional[PlotCalibration | dict] = None, label_mode: Optional[str] = None, show_legend: Optional[bool] = None, panel6_node_size: float = 420.0, panel9_node_size: float = 700.0):
    panel_id = _resolve_panel_id(panel)
    pathway = _find_pathway(finished, label)
    s = int(_pathway_get(pathway, "start_idx"))
    dur = int(_pathway_get(pathway, "duration"))
    frames = np.asarray(cache.frames[s : s + dur], dtype=int)
    sizes = np.asarray(_pathway_get(pathway, "sizes"), dtype=int)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111)
    extra = _draw_panel(
        fig,
        ax,
        panel_id,
        cache=cache,
        pathway=pathway,
        frames=frames,
        sizes=sizes,
        inter_thresh=inter_thresh,
        compactness_mode=compactness_mode,
        rg_use_mic=rg_use_mic,
        transition_thr=transition_thr,
        label_map=_get_label_map(cache, use_segids),
        msd_max_lag=msd_max_lag,
        hide_member_curves=hide_member_curves,
        hide_oligomer_curve=hide_oligomer_curve,
        sw_n_rand=sw_n_rand,
        sw_seed=sw_seed,
        calibration=calibration,
        panel6_node_size=panel6_node_size,
        panel9_node_size=panel9_node_size,
        show_legend=show_legend,
    )
    fig.suptitle(f"Panel {panel_id}: {PANEL_NAMES[panel_id]} — {display_oligomer_label(label, calibration, label_mode)}", fontsize=12)
    if show:
        plt.show()
    return fig, ax, extra


def monitor_oligomer_label(cache: ContactCache, finished: Iterable[Pathway], label: str, inter_thresh: int, temperature_K: float = 330.0, transition_thr: float = 1e-3, figsize=(16, 11), use_segids: bool = False, compactness_mode: str = "density", rg_use_mic: bool = True, msd_max_lag: int = 150, hide_member_curves: bool = True, hide_oligomer_curve: bool = False, sw_n_rand: int = 30, sw_seed: int = 0, show: bool = True, calibration: Optional[PlotCalibration | dict] = None, label_mode: Optional[str] = None, show_legend: Optional[bool] = None, panel6_node_size: float = 420.0, panel9_node_size: float = 700.0):
    pathway = _find_pathway(finished, label)
    s = int(_pathway_get(pathway, "start_idx"))
    dur = int(_pathway_get(pathway, "duration"))
    if dur <= 0:
        raise ValueError("This pathway has duration 0")

    frames = np.asarray(cache.frames[s : s + dur], dtype=int)
    sizes = np.asarray(_pathway_get(pathway, "sizes"), dtype=int)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.05, hspace=0.08)
    gs = fig.add_gridspec(3, 3)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]

    extras = {}
    for panel_id, ax in enumerate(axes, start=1):
        extra = _draw_panel(
            fig,
            ax,
            panel_id,
            cache=cache,
            pathway=pathway,
            frames=frames,
            sizes=sizes,
            inter_thresh=inter_thresh,
            compactness_mode=compactness_mode,
            rg_use_mic=rg_use_mic,
            transition_thr=transition_thr,
            label_map=_get_label_map(cache, use_segids),
            msd_max_lag=msd_max_lag,
            hide_member_curves=hide_member_curves,
            hide_oligomer_curve=hide_oligomer_curve,
            sw_n_rand=sw_n_rand,
            sw_seed=sw_seed,
            calibration=calibration,
            panel6_node_size=panel6_node_size,
            panel9_node_size=panel9_node_size,
            show_legend=show_legend,
        )
        extras.update(extra)

    fig.suptitle(f"Oligomer monitor dashboard — {display_oligomer_label(label, calibration, label_mode)}", fontsize=14)
    if show:
        plt.show()

    free_energy = compute_free_energy_from_sizes(sizes, shift_min_to_zero=True, temperature_K=temperature_K)
    return {
        "pathway": pathway,
        "frames": frames,
        "sizes": sizes,
        "T_counts": pathway_transition_matrix(list(sizes)),
        "free_energy_kBT": {
            "size": free_energy.sizes,
            "count": free_energy.counts,
            "F_over_kBT": free_energy.F_over_kBT,
            "F_joule": free_energy.F_joule,
        },
        "msd": extras.get("msd"),
        "small_world": extras.get("small_world"),
        "compactness": extras.get("compactness"),
    }



def _draw_panel9_style_snapshot_on_ax(
    ax,
    cache: ContactCache,
    members,
    frame: int,
    inter_thresh: int,
    label_map=None,
    n_rand: int = 30,
    seed: int = 0,
    frame_is_index: bool = False,
    node_size: float = 700.0,
):
    """Draw one Panel-9-style small-world snapshot on an existing axis."""
    if frame_is_index:
        idx = int(frame)
        if idx < 0 or idx >= len(cache.frames):
            raise ValueError(f"frame index out of range: {idx}")
        fr_num = int(cache.frames[idx])
    else:
        frame_to_idx = {int(f): i for i, f in enumerate(cache.frames)}
        fr_num = int(frame)
        if fr_num not in frame_to_idx:
            raise ValueError(f"frame {fr_num} not found in cache.frames")
        idx = frame_to_idx[fr_num]

    members = sorted(list(members))
    cmap = cache.cmaps[idx]
    G = build_contact_graph_snapshot(members, cmap, inter_thresh)
    score = small_worldness_S(G, n_rand=n_rand, seed=seed)

    ax.axis("off")
    card = ax.inset_axes([0.02, 0.02, 0.96, 0.96])
    card.set_xticks([])
    card.set_yticks([])
    card.set_facecolor("white")
    for spine in card.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_edgecolor("royalblue")
        spine.set_linestyle((0, (6, 4)))
    net_ax = card.inset_axes([0.02, 0.08, 0.96, 0.88])
    net_ax.axis("off")
    fmt = lambda x: f"{x:.2f}" if np.isfinite(x) else "NaN"
    card.text(0.02, 0.98, f"Frame {fr_num}   |   N={len(members)}   |   S={fmt(score.S)}", transform=card.transAxes, va="top", fontsize=11)
    _draw_snapshot_network(net_ax, G, label_map=label_map, node_size=node_size)
    return {"frame": fr_num, "members": members, "S": score.S, "C": score.C, "L": score.L, "Cr": score.Cr, "Lr": score.Lr, "n": score.n, "m": score.m}


def plot_small_world_range(
    cache: ContactCache,
    finished: Iterable[Pathway],
    target_label: str,
    i: int,
    j: int,
    inter_thresh: int,
    *,
    by: str = "pathway_index",
    n_rand: int = 30,
    seed: int = 0,
    label_map=None,
    ncols: int = 8,
    cell_figsize: tuple[float, float] = (3.0, 3.0),
    show: bool = True,
    label_mode: str = "full",
    panel9_node_size: float = 700.0,
):
    """Plot a series of Panel-9-style small-world snapshots for one pathway label.

    Parameters
    ----------
    by
        ``"pathway_index"`` interprets ``i`` and ``j`` as pathway-local indices.
        ``"absolute_frame"`` interprets them as trajectory frame numbers.
    ncols
        Number of snapshots per row. Default is 8.
    """
    pathway = next((p for p in finished if _pathway_get(p, "label") == target_label), None)
    if pathway is None:
        raise ValueError(f"Pathway label not found: {target_label}")

    start_idx = int(_pathway_get(pathway, "start_idx"))
    duration = int(_pathway_get(pathway, "duration"))
    sets = list(_pathway_get(pathway, "sets"))

    if by == "pathway_index":
        t_start = max(0, int(i))
        t_end = min(duration - 1, int(j))
        if t_start > t_end:
            raise ValueError(f"Invalid pathway index range after clipping: {t_start}..{t_end}")
        selected_t = list(range(t_start, t_end + 1))
    elif by == "absolute_frame":
        selected_t = []
        for t in range(duration):
            cache_idx = start_idx + t
            fr = int(cache.frames[cache_idx])
            if int(i) <= fr <= int(j):
                selected_t.append(t)
        if not selected_t:
            raise ValueError(f"No frames found for label '{target_label}' in frame range [{i}, {j}]")
    else:
        raise ValueError("`by` must be either 'pathway_index' or 'absolute_frame'")

    n_panels = len(selected_t)
    nrows = max(1, math.ceil(n_panels / ncols))
    fig_w = cell_figsize[0] * ncols
    fig_h = cell_figsize[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    rows = []
    for k, t in enumerate(selected_t):
        ax = axes[k]
        cache_idx = start_idx + t
        fr_num = int(cache.frames[cache_idx])
        members_t = sets[t]
        info = _draw_panel9_style_snapshot_on_ax(
            ax=ax,
            cache=cache,
            members=members_t,
            frame=fr_num,
            inter_thresh=inter_thresh,
            label_map=label_map,
            n_rand=n_rand,
            seed=seed + k,
            frame_is_index=False,
            node_size=panel9_node_size,
        )
        rows.append({
            "label": target_label,
            "display_label": display_oligomer_label(target_label, label_mode=label_mode),
            "pathway_t": int(t),
            "frame": int(fr_num),
            "n_members": int(len(info["members"])),
            "members": info["members"],
            "S": info["S"],
            "C": info["C"],
            "L": info["L"],
            "Cr": info["Cr"],
            "Lr": info["Lr"],
            "n": info["n"],
            "m": info["m"],
        })

    for k in range(n_panels, len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        f"Small-world snapshots — {display_oligomer_label(target_label, label_mode=label_mode)}\nrange=({i}, {j}), by={by}, {ncols} frames per row",
        fontsize=14,
    )
    if show:
        plt.show()
    return fig, axes, pd.DataFrame(rows)



def plot_panel9_style_snapshot(cache: ContactCache, members, frame: int, inter_thresh: int, label_map=None, n_rand: int = 30, seed: int = 0, figsize=(6, 4), frame_is_index: bool = False, show: bool = True, node_size: float = 700.0):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    out = _draw_panel9_style_snapshot_on_ax(
        ax=ax,
        cache=cache,
        members=members,
        frame=frame,
        inter_thresh=inter_thresh,
        label_map=label_map,
        n_rand=n_rand,
        seed=seed,
        frame_is_index=frame_is_index,
        node_size=node_size,
    )
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax, out


def draw_transition_graph_from_prob(
    M: np.ndarray,
    thr: float = 1e-3,
    min_size: int = 2,
    max_size: Optional[int] = None,
    calibration: Optional[PlotCalibration | dict] = None,
    title: str = "Cluster-size transition graph",
):
    """Draw a directed transition graph from a size-transition probability matrix.

    This is a user-friendly alias around the global transition-network plotting
    logic. ``M[a, b]`` is interpreted as the probability of transitioning from
    oligomer size ``a`` to size ``b``. Use ``min_size=2`` to exclude monomers.

    Returns the NetworkX graph for further inspection.
    """
    import networkx as nx

    cal = resolve_plot_calibration(calibration)
    if max_size is not None:
        cal.matrix_max_size = int(max_size)
    max_plot_size = np.asarray(M).shape[0] - 1
    if cal.matrix_max_size is not None:
        max_plot_size = min(max_plot_size, int(cal.matrix_max_size))
    if max_plot_size < int(min_size):
        raise ValueError("No size states available for requested min_size/max_size")

    labels = np.arange(int(min_size), int(max_plot_size) + 1)
    P = np.asarray(M, dtype=float)[int(min_size):max_plot_size + 1, int(min_size):max_plot_size + 1]

    G = nx.DiGraph()
    node_areas = scale_node_sizes(labels, cal)
    for idx, size_label in enumerate(labels):
        G.add_node(int(size_label), size=float(node_areas[idx]), label=str(int(size_label)))

    edges_fwd, widths_fwd, edges_rev, widths_rev = [], [], [], []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            prob = float(P[i, j])
            if prob <= thr or a == b:
                continue
            G.add_edge(int(a), int(b), weight=prob)
            width = max(0.5, prob * 40.0)
            if a < b:
                edges_fwd.append((int(a), int(b)))
                widths_fwd.append(width)
            else:
                edges_rev.append((int(a), int(b)))
                widths_rev.append(width)

    pos = nx.spring_layout(G, seed=40, weight="weight")
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=[G.nodes[n]["size"] for n in G.nodes],
                           node_color="lightsteelblue", edgecolors="black")
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=10)
    edge_common = dict(arrows=True, arrowstyle="-|>", arrowsize=18, min_source_margin=15, min_target_margin=18)
    nx.draw_networkx_edges(G, pos, edgelist=edges_fwd, width=widths_fwd, edge_color="royalblue",
                           connectionstyle="arc3,rad=0.2", **edge_common)
    nx.draw_networkx_edges(G, pos, edgelist=edges_rev, width=widths_rev, edge_color="orangered",
                           connectionstyle="arc3,rad=-0.25", **edge_common)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return G
