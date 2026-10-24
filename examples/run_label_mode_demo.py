"""Example: choose full or short oligomer/pathway labels in plots.

This changes only the text shown in legends/titles. The original full labels are
kept internally and should still be used for lookup.
"""

from oligomer_analysis import PlotCalibration, format_oligomer_label
from oligomer_analysis.monitor import plot_pathway_sizes

# Example label conversion
print(format_oligomer_label("oli8_16_20_25_26", mode="full"))      # oli8_16_20_25_26
print(format_oligomer_label("oli8_16_20_25_26", mode="ordinal"))   # oli8

# Use a global setting in PlotCalibration.
cal = PlotCalibration(
    max_oligomer_size=100,
    size_tick_step=10,
    matrix_max_size=100,
    node_size_min=10,
    node_size_max=120,
    node_size_transform="sqrt",
    pathway_label_mode="ordinal",  # show oli8 instead of oli8_16_20_25_26
)

# After you already have `cache` and `finished`:
# plot_pathway_sizes(cache.frames, finished, min_duration=10, calibration=cal)
# monitor_oligomer_label(cache, finished, label=finished[0].label, inter_thresh=1, calibration=cal)
# plot_oligomer_panel(cache, finished, label=finished[0].label, panel=1, inter_thresh=1, calibration=cal)

# Or override per call:
# plot_pathway_sizes(cache.frames, finished, min_duration=10, label_mode="full")
# plot_pathway_sizes(cache.frames, finished, min_duration=10, label_mode="ordinal")
