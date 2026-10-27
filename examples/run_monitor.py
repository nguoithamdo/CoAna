# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from oligomer_analysis import analyze_from_cache, load_cache, monitor_oligomer_label

cache = load_cache("cache_cut6.5_compactness.npz")
res = analyze_from_cache(
    cache,
    inter_thresh=1,
    size_thresh=3,
    continue_min_size=1,
    overlap_min=1,
    shrink_tolerance=3,
    dur_store_min=5,
    do_plots=False,
)

monitor_oligomer_label(
    cache=cache,
    finished=res["finished"],
    label="oli8_16_20_25_26",
    inter_thresh=1,
    compactness_mode="rg",
    rg_use_mic=True,
    hide_member_curves=False,
    hide_oligomer_curve=True,
)
