# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Example: turn legends on/off in V3.1.2 plots.

This assumes you already created `cache`, `finished`, `frames_clusters`, and
`lineage_result` using the normal workflow.
"""

from oligomer_analysis import PlotCalibration
from oligomer_analysis.monitor import plot_pathway_sizes

# Global default: all supported legends are hidden unless overridden.
cal_no_legend = PlotCalibration(
    max_oligomer_size=100,
    size_tick_step=10,
    matrix_max_size=100,
    node_size_min=10,
    node_size_max=120,
    pathway_label_mode="ordinal",
    show_legend=False,
)

# Example calls:
# plot_pathway_sizes(cache.frames, finished, min_duration=1, calibration=cal_no_legend)
# plot_whole_system_background(cache, lineage_result, calibration=cal_no_legend)
# plot_lineage_fate(lineage_result, calibration=cal_no_legend)
# plot_lineage_graph(lineage_result, calibration=cal_no_legend, show_labels=False)
# monitor_oligomer_label(cache, finished, label=finished[0].label, inter_thresh=1, calibration=cal_no_legend)

# Per-plot override if global calibration hides legends:
# plot_pathway_sizes(cache.frames, finished, min_duration=1, calibration=cal_no_legend, show_legend=True)
