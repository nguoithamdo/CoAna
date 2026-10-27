# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

"""Interactive Plotly views for aggregation lineage analysis.

These functions are optional notebook tools.  They require plotly, and the
filter-panel helper additionally requires ipywidgets.  They are not needed for
non-interactive batch analysis.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Interactive lineage plots require plotly. Install with: pip install plotly") from exc
    return go


def _select_lineages(
    lineage_result,
    lineage_ids: Optional[Sequence[str]] = None,
    class_labels: Optional[Sequence[str]] = None,
    min_max_size: Optional[int] = None,
    min_duration: Optional[int] = None,
    min_contribution: Optional[float] = None,
):
    """Select lineages using common filters."""
    selected = []
    class_set = set(class_labels) if class_labels is not None else None
    lineage_set = set(lineage_ids) if lineage_ids is not None else None

    for lin in lineage_result.lineages:
        if lineage_set is not None and lin.lineage_id not in lineage_set:
            continue
        if class_set is not None and lin.class_label not in class_set:
            continue
        if min_max_size is not None and lin.max_size < min_max_size:
            continue
        if min_duration is not None and lin.duration < min_duration:
            continue
        if min_contribution is not None and lin.contribution_score < min_contribution:
            continue
        selected.append(lin)
    return selected


def build_temporal_network_dataframe(
    lineage_result,
    lineage_ids: Optional[Sequence[str]] = None,
    class_labels: Optional[Sequence[str]] = None,
    min_max_size: Optional[int] = None,
    min_duration: Optional[int] = None,
    min_contribution: Optional[float] = None,
    node_size_by: str = "max_size",
    y_mode: str = "lineage",
    node_size_min: float = 8.0,
    node_size_max: float = 26.0,
):
    """Build segment/event dataframes for structured temporal lineage plotting."""
    selected_lineages = _select_lineages(
        lineage_result,
        lineage_ids=lineage_ids,
        class_labels=class_labels,
        min_max_size=min_max_size,
        min_duration=min_duration,
        min_contribution=min_contribution,
    )
    if not selected_lineages:
        return None, None, [], [], []

    selected_lineage_ids = {lin.lineage_id for lin in selected_lineages}
    lineages_by_id = {lin.lineage_id: lin for lin in lineage_result.lineages}
    selected_segments = [seg for seg in lineage_result.segments if seg.lineage_id in selected_lineage_ids]

    if not selected_segments:
        return None, None, [], [], []

    if y_mode == "lineage":
        ordered_lineages = sorted(
            selected_lineages,
            key=lambda lin: (lin.start_frame, lin.class_label, -lin.max_size, lin.lineage_id),
        )
        y_base = {lin.lineage_id: i for i, lin in enumerate(ordered_lineages)}
        segs_by_lin = {}
        for seg in selected_segments:
            segs_by_lin.setdefault(seg.lineage_id, []).append(seg)

        y_pos = {}
        for lin_id, segs in segs_by_lin.items():
            segs = sorted(segs, key=lambda s: (s.start_frame, s.end_frame, str(s.segment_id)))
            offsets = [0.0] if len(segs) == 1 else np.linspace(-0.28, 0.28, len(segs))
            for seg, off in zip(segs, offsets):
                y_pos[seg.segment_id] = y_base[lin_id] + float(off)

        y_tickvals = [y_base[lin.lineage_id] for lin in ordered_lineages]
        y_ticktext = [
            f"{lin.lineage_id} | {lin.class_label} | max={lin.max_size} | C={lin.contribution_score:.2f}"
            for lin in ordered_lineages
        ]

    elif y_mode == "class":
        class_order = ["on-pathway", "uncertain", "off-pathway"]
        segs_by_class = {}
        for seg in selected_segments:
            lin = lineages_by_id[seg.lineage_id]
            segs_by_class.setdefault(lin.class_label, []).append(seg)

        y_pos, y_tickvals, y_ticktext = {}, [], []
        current_y = 0
        for cls in class_order:
            segs = segs_by_class.get(cls, [])
            if not segs:
                continue

            by_lin = {}
            for seg in segs:
                by_lin.setdefault(seg.lineage_id, []).append(seg)

            lineage_order = sorted(
                by_lin.keys(),
                key=lambda lid: (lineages_by_id[lid].start_frame, -lineages_by_id[lid].max_size, lid),
            )
            for lid in lineage_order:
                group = sorted(by_lin[lid], key=lambda s: (s.start_frame, s.end_frame, str(s.segment_id)))
                offsets = [0.0] if len(group) == 1 else np.linspace(-0.28, 0.28, len(group))
                for seg, off in zip(group, offsets):
                    y_pos[seg.segment_id] = current_y + float(off)
                lin = lineages_by_id[lid]
                y_tickvals.append(current_y)
                y_ticktext.append(f"{cls} | {lid} | max={lin.max_size} | C={lin.contribution_score:.2f}")
                current_y += 1
            current_y += 1
    else:
        raise ValueError("y_mode must be 'lineage' or 'class'")

    raw_values = []
    for seg in selected_segments:
        lin = lineages_by_id[seg.lineage_id]
        if node_size_by == "max_size":
            value = seg.max_size
        elif node_size_by == "mean_size":
            value = seg.mean_size
        elif node_size_by == "duration":
            value = seg.duration
        elif node_size_by == "contribution":
            value = lin.contribution_score
        else:
            raise ValueError("node_size_by must be one of: max_size, mean_size, duration, contribution")
        raw_values.append(float(value))

    raw_values = np.asarray(raw_values, dtype=float)
    if raw_values.size == 0:
        return None, None, [], [], []
    if raw_values.max() > raw_values.min():
        node_sizes = node_size_min + (node_size_max - node_size_min) * (
            raw_values - raw_values.min()
        ) / (raw_values.max() - raw_values.min())
    else:
        node_sizes = np.full_like(raw_values, 0.5 * (node_size_min + node_size_max))

    seg_rows = []
    for seg, node_size in zip(selected_segments, node_sizes):
        lin = lineages_by_id[seg.lineage_id]
        x_mid = 0.5 * (seg.start_frame + seg.end_frame)
        seg_rows.append({
            "segment_id": seg.segment_id,
            "lineage_id": seg.lineage_id,
            "class_label": lin.class_label,
            "contribution_score": lin.contribution_score,
            "growth_score": getattr(lin, "growth_score", np.nan),
            "persistence_score": getattr(lin, "persistence_score", np.nan),
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
            "duration": seg.duration,
            "max_size": seg.max_size,
            "mean_size": seg.mean_size,
            "x": x_mid,
            "y": y_pos[seg.segment_id],
            "node_size": float(node_size),
        })
    df_seg = pd.DataFrame(seg_rows)
    selected_segment_ids = set(df_seg["segment_id"])

    event_rows = []
    for ev in lineage_result.events:
        for parent_id in ev.parent_segment_ids:
            for child_id in ev.child_segment_ids:
                if parent_id not in selected_segment_ids or child_id not in selected_segment_ids:
                    continue
                p = df_seg.loc[df_seg["segment_id"] == parent_id].iloc[0]
                c = df_seg.loc[df_seg["segment_id"] == child_id].iloc[0]
                event_rows.append({
                    "event_id": ev.event_id,
                    "event_type": ev.event_type,
                    "frame_number": ev.frame_number,
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "x0": p["x"],
                    "y0": p["y"],
                    "x1": c["x"],
                    "y1": c["y"],
                    "xm": 0.5 * (p["x"] + c["x"]),
                    "ym": 0.5 * (p["y"] + c["y"]),
                    "description": getattr(ev, "description", ""),
                    "parents": str(ev.parent_segment_ids),
                    "children": str(ev.child_segment_ids),
                })
    df_event = pd.DataFrame(event_rows)
    return df_seg, df_event, y_tickvals, y_ticktext, selected_lineages


def plot_lineage_temporal_network(
    lineage_result,
    lineage_ids: Optional[Sequence[str]] = None,
    class_labels: Optional[Sequence[str]] = None,
    min_max_size: Optional[int] = None,
    min_duration: Optional[int] = None,
    min_contribution: Optional[float] = None,
    node_size_by: str = "max_size",
    y_mode: str = "lineage",
    highlight_frame: Optional[int] = None,
    show_segment_labels: bool = True,
    show_event_labels: bool = True,
    show_legend: bool = True,
    figure_width: int = 1500,
    figure_height: int = 900,
    node_size_min: float = 8.0,
    node_size_max: float = 26.0,
):
    """Structured interactive temporal lineage network.

    x-axis is frame; y-axis is lineage or class lane.  Nodes are oligomer
    segments, horizontal bars are segment lifetimes, and edges are split/merge
    or continuation events.
    """
    go = _require_plotly()
    df_seg, df_event, y_tickvals, y_ticktext, _ = build_temporal_network_dataframe(
        lineage_result,
        lineage_ids=lineage_ids,
        class_labels=class_labels,
        min_max_size=min_max_size,
        min_duration=min_duration,
        min_contribution=min_contribution,
        node_size_by=node_size_by,
        y_mode=y_mode,
        node_size_min=node_size_min,
        node_size_max=node_size_max,
    )
    if df_seg is None or len(df_seg) == 0:
        print("No lineages passed the filter.")
        return None

    class_colors = {"on-pathway": "forestgreen", "off-pathway": "firebrick", "uncertain": "goldenrod"}
    event_colors = {
        "continuation": "steelblue", "growth": "seagreen", "shrink": "tomato",
        "split": "royalblue", "merge": "darkviolet", "split_merge": "darkorange",
        "rearrangement": "gray", "birth": "dimgray", "death": "black",
    }

    fig = go.Figure()

    for cls, sub in df_seg.groupby("class_label"):
        x_lines, y_lines = [], []
        for _, r in sub.iterrows():
            x_lines += [r["start_frame"], r["end_frame"], None]
            y_lines += [r["y"], r["y"], None]
        fig.add_trace(go.Scatter(
            x=x_lines, y=y_lines, mode="lines",
            line=dict(color=class_colors.get(cls, "gray"), width=5),
            opacity=0.28, hoverinfo="skip", showlegend=False, name=f"{cls} lifetime",
        ))

    if df_event is not None and len(df_event) > 0:
        for ev_type, sub in df_event.groupby("event_type"):
            xs, ys = [], []
            for _, r in sub.iterrows():
                xs += [r["x0"], r["x1"], None]
                ys += [r["y0"], r["y1"], None]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=event_colors.get(ev_type, "gray"), width=2),
                opacity=0.75, hoverinfo="skip", name=f"event: {ev_type}",
            ))
        if show_event_labels:
            for ev_type, sub in df_event.groupby("event_type"):
                fig.add_trace(go.Scatter(
                    x=sub["xm"], y=sub["ym"], mode="markers",
                    marker=dict(size=8, color=event_colors.get(ev_type, "gray"), symbol="diamond",
                                line=dict(color="black", width=0.5)),
                    customdata=np.stack([
                        sub["event_id"], sub["event_type"], sub["frame_number"],
                        sub["parent_id"], sub["child_id"], sub["parents"],
                        sub["children"], sub["description"],
                    ], axis=1),
                    hovertemplate=(
                        "<b>event %{customdata[0]}</b><br>"
                        "type: %{customdata[1]}<br>"
                        "frame: %{customdata[2]}<br>"
                        "parent segment: %{customdata[3]}<br>"
                        "child segment: %{customdata[4]}<br>"
                        "parents: %{customdata[5]}<br>"
                        "children: %{customdata[6]}<br>"
                        "%{customdata[7]}<extra></extra>"
                    ),
                    name=f"{ev_type} marker", showlegend=False,
                ))

    if highlight_frame is not None:
        fig.add_vline(x=highlight_frame, line_width=2, line_dash="dash", line_color="black",
                      annotation_text=f"frame {highlight_frame}", annotation_position="top")

    for cls, sub in df_seg.groupby("class_label"):
        if highlight_frame is None:
            opacity = np.full(len(sub), 0.95)
        else:
            active = (sub["start_frame"] <= highlight_frame) & (highlight_frame <= sub["end_frame"])
            opacity = np.where(active.values, 1.0, 0.12)
        text_vals = sub["segment_id"].astype(str) if show_segment_labels else None
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers+text" if show_segment_labels else "markers",
            text=text_vals, textposition="top center",
            marker=dict(size=sub["node_size"], color=class_colors.get(cls, "gray"),
                        opacity=opacity, line=dict(color="black", width=1)),
            customdata=np.stack([
                sub["segment_id"], sub["lineage_id"], sub["class_label"],
                sub["start_frame"], sub["end_frame"], sub["duration"],
                sub["max_size"], sub["mean_size"], sub["contribution_score"],
                sub["growth_score"], sub["persistence_score"],
            ], axis=1),
            hovertemplate=(
                "<b>segment %{customdata[0]}</b><br>"
                "lineage: %{customdata[1]}<br>"
                "class: %{customdata[2]}<br>"
                "start: %{customdata[3]}<br>"
                "end: %{customdata[4]}<br>"
                "duration: %{customdata[5]}<br>"
                "max size: %{customdata[6]}<br>"
                "mean size: %{customdata[7]:.2f}<br>"
                "contribution: %{customdata[8]:.3f}<br>"
                "growth score: %{customdata[9]:.3f}<br>"
                "persistence score: %{customdata[10]:.3f}<extra></extra>"
            ),
            name=cls,
        ))

    title = "Structured temporal aggregation lineage network"
    if highlight_frame is not None:
        title += f" | highlighted frame = {highlight_frame}"

    fig.update_layout(
        title=title, width=figure_width, height=figure_height, template="plotly_white",
        hovermode="closest",
        showlegend=bool(show_legend),
        xaxis=dict(title="Frame", showgrid=True, zeroline=False),
        yaxis=dict(
            title="Lineage lane" if y_mode == "lineage" else "Class / lineage lane",
            tickmode="array", tickvals=y_tickvals, ticktext=y_ticktext, automargin=True,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
    )
    return fig


def make_lineage_temporal_animation(
    lineage_result,
    lineage_ids: Optional[Sequence[str]] = None,
    class_labels: Optional[Sequence[str]] = None,
    min_max_size: Optional[int] = None,
    min_duration: Optional[int] = None,
    min_contribution: Optional[float] = None,
    node_size_by: str = "max_size",
    y_mode: str = "lineage",
    show_segment_labels: bool = False,
    show_event_labels: bool = True,
    show_legend: bool = True,
    frame_step: int = 5,
    figure_width: int = 1500,
    figure_height: int = 900,
    node_size_min: float = 8.0,
    node_size_max: float = 26.0,
):
    """Build an animated Plotly figure with play button and frame slider."""
    go = _require_plotly()
    df_seg, _, _, _, _ = build_temporal_network_dataframe(
        lineage_result, lineage_ids=lineage_ids, class_labels=class_labels,
        min_max_size=min_max_size, min_duration=min_duration,
        min_contribution=min_contribution, node_size_by=node_size_by,
        y_mode=y_mode, node_size_min=node_size_min, node_size_max=node_size_max,
    )
    if df_seg is None or len(df_seg) == 0:
        print("No lineages passed the filter.")
        return None

    frame_min = int(df_seg["start_frame"].min())
    frame_max = int(df_seg["end_frame"].max())
    frames = list(range(frame_min, frame_max + 1, int(frame_step)))

    fig = plot_lineage_temporal_network(
        lineage_result, lineage_ids=lineage_ids, class_labels=class_labels,
        min_max_size=min_max_size, min_duration=min_duration,
        min_contribution=min_contribution, node_size_by=node_size_by,
        y_mode=y_mode, highlight_frame=frame_min, show_segment_labels=show_segment_labels,
        show_event_labels=show_event_labels, show_legend=show_legend, figure_width=figure_width,
        figure_height=figure_height, node_size_min=node_size_min, node_size_max=node_size_max,
    )

    anim_frames = []
    for fr in frames:
        fr_fig = plot_lineage_temporal_network(
            lineage_result, lineage_ids=lineage_ids, class_labels=class_labels,
            min_max_size=min_max_size, min_duration=min_duration,
            min_contribution=min_contribution, node_size_by=node_size_by,
            y_mode=y_mode, highlight_frame=fr, show_segment_labels=show_segment_labels,
            show_event_labels=show_event_labels, show_legend=show_legend, figure_width=figure_width,
            figure_height=figure_height, node_size_min=node_size_min, node_size_max=node_size_max,
        )
        anim_frames.append(go.Frame(data=fr_fig.data, name=str(fr),
                                    layout=go.Layout(title=f"Structured temporal aggregation lineage network | frame = {fr}")))
    fig.frames = anim_frames
    fig.update_layout(
        updatemenus=[{
            "type": "buttons", "showactive": False, "x": 0, "y": -0.08,
            "xanchor": "left", "yanchor": "top",
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 500, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate", "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": 0, "x": 0.1, "y": -0.08, "len": 0.85,
            "currentvalue": {"prefix": "Frame: ", "font": {"size": 14}},
            "steps": [
                {"label": str(fr), "method": "animate",
                 "args": [[str(fr)], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate", "transition": {"duration": 0}}]}
                for fr in frames
            ],
        }],
    )
    return fig


def export_lineage_temporal_html(fig, filename: str = "lineage_temporal_network.html"):
    """Export Plotly interactive figure to standalone HTML."""
    fig.write_html(filename, include_plotlyjs="cdn", full_html=True)
    print(f"Saved HTML to: {filename}")


def lineage_network_filter_panel(lineage_result):
    """Jupyter widget panel for interactively filtering the temporal lineage network."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("The filter panel requires ipywidgets. Install with: pip install ipywidgets") from exc

    class_options = ["on-pathway", "uncertain", "off-pathway"]
    min_size_widget = widgets.IntSlider(
        value=2, min=2, max=max([lin.max_size for lin in lineage_result.lineages], default=100),
        step=1, description="Min size:", continuous_update=False, layout=widgets.Layout(width="500px"),
    )
    min_duration_widget = widgets.IntSlider(
        value=1, min=1, max=max([lin.duration for lin in lineage_result.lineages], default=100),
        step=1, description="Min T:", continuous_update=False, layout=widgets.Layout(width="500px"),
    )
    min_contribution_widget = widgets.FloatSlider(
        value=0.0, min=0.0, max=1.0, step=0.05, description="Min C:",
        continuous_update=False, layout=widgets.Layout(width="500px"),
    )
    class_widget = widgets.SelectMultiple(
        options=class_options, value=tuple(class_options), description="Classes:",
        layout=widgets.Layout(width="500px", height="90px"),
    )
    node_size_widget = widgets.Dropdown(
        options=["max_size", "mean_size", "duration", "contribution"], value="max_size",
        description="Node size:", layout=widgets.Layout(width="500px"),
    )
    y_mode_widget = widgets.Dropdown(
        options=["lineage", "class"], value="lineage", description="Y mode:",
        layout=widgets.Layout(width="500px"),
    )
    labels_widget = widgets.Checkbox(value=False, description="Show segment labels")
    event_labels_widget = widgets.Checkbox(value=True, description="Show event hover markers")
    legend_widget = widgets.Checkbox(value=True, description="Show legend")
    output = widgets.Output()

    def update_plot(change=None):
        with output:
            output.clear_output(wait=True)
            fig = plot_lineage_temporal_network(
                lineage_result,
                class_labels=list(class_widget.value),
                min_max_size=min_size_widget.value,
                min_duration=min_duration_widget.value,
                min_contribution=min_contribution_widget.value,
                node_size_by=node_size_widget.value,
                y_mode=y_mode_widget.value,
                show_segment_labels=labels_widget.value,
                show_event_labels=event_labels_widget.value,
                show_legend=legend_widget.value,
            )
            if fig is not None:
                fig.show()

    controls = widgets.VBox([
        min_size_widget, min_duration_widget, min_contribution_widget,
        class_widget, node_size_widget, y_mode_widget, labels_widget, event_labels_widget, legend_widget,
    ])
    for w in [min_size_widget, min_duration_widget, min_contribution_widget, class_widget,
              node_size_widget, y_mode_widget, labels_widget, event_labels_widget, legend_widget]:
        w.observe(update_plot, names="value")
    display(widgets.HBox([controls]), output)
    update_plot()


__all__ = [
    "build_temporal_network_dataframe",
    "plot_lineage_temporal_network",
    "make_lineage_temporal_animation",
    "export_lineage_temporal_html",
    "lineage_network_filter_panel",
]
