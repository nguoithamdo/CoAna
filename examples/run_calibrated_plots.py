# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Example: calibrating plots for large oligomers up to size 100.

This script shows how to use the same calibration object across whole-system,
lineage, transition, and dashboard figures.
"""

from oligomer_analysis import (
    PlotCalibration,
    TargetStateConfig,
    build_lineages_from_cache,
    load_cache,
    monitor_oligomer_label,
    plot_lineage_fate,
    plot_lineage_graph,
    plot_whole_system_background,
)
from oligomer_analysis.monitor import network_plot_global
from oligomer_analysis.system_analysis import build_cluster_transition_matrix

cache = load_cache("cache_cut6.5_compactness.npz")

lineage_result = build_lineages_from_cache(
    cache,
    inter_thresh=1,
    cluster_min_size=2,
    overlap_min=1,
    parent_frac_min=0.30,
    child_frac_min=0.30,
    union_cover_min=0.70,
    target_config=TargetStateConfig(min_size=10, min_duration=5, min_contribution=0.70),
)

# One calibration object for all large-size figures.
cal = PlotCalibration(
    max_oligomer_size=100,
    size_tick_step=10,
    matrix_max_size=100,
    node_size_min=250,
    node_size_max=1800,
    node_size_transform="sqrt",
    marker_size=3.5,
    line_width=1.4,
)

plot_whole_system_background(cache, lineage_result, calibration=cal)
plot_lineage_fate(lineage_result, calibration=cal)
plot_lineage_graph(lineage_result, calibration=cal, node_size_by="max_size")

T_global = build_cluster_transition_matrix(lineage_result.frames_clusters, min_size=2)
M_global = T_global.astype(float)
if M_global.sum() > 0:
    M_global /= M_global.sum()
network_plot_global(M_global, min_size=2, calibration=cal)

# For the old 9-panel dashboard, pass calibration as well.
# monitor_oligomer_label(
#     cache=cache,
#     finished=finished,
#     label="oli8_16_20_25_26",
#     inter_thresh=1,
#     calibration=cal,
# )
