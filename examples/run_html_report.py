# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Generate the self-contained HTML analysis report from a Stage A cache.

Equivalent CLI:

    coana report cache_coanapure_1956frames.npz -o CoAnaPure_Analysis_Report.html \
        --title "128-mer Oligomer Aggregation Analysis" --max-size 100
"""

from oligomer_analysis import PlotCalibration, generate_html_report

generate_html_report(
    "cache_coanapure_1956frames.npz",
    "CoAnaPure_Analysis_Report.html",
    title="128-mer Oligomer Aggregation Analysis",
    subtitle="Martini system: initial1.pdb + md1_centered_filtered_1956frames.pdb",
    inter_thresh=1,
    calibration=PlotCalibration(
        max_oligomer_size=100,
        matrix_max_size=100,
        node_size_transform="sqrt",
    ),
)
