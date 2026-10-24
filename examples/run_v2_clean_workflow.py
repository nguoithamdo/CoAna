"""Clean V2 workflow example for protein aggregation analysis.

This script assumes Stage A precomputation has already produced a cache file.
To build the cache first, run:

    python examples/run_precompute_v2.py

Then edit `cache_path` and thresholds here, and run:

    python examples/run_v2_clean_workflow.py

Stage A precomputation is intentionally not run here because it is expensive.
"""

from oligomer_analysis import (
    PlotCalibration,
    TargetStateConfig,
    build_feature_table,
    build_lineages_from_cache,
    figure_commitment_predictors,
    figure_contact_hotspots,
    figure_llps_coupling,
    lineage_summary_dataframe,
    load_cache,
    plot_lineage_fate,
    plot_lineage_graph,
    plot_whole_system_background,
    target_state_table,
)

cache_path = "cache_cut6.5_compactness.npz"
inter_thresh = 1

cache = load_cache(cache_path)

target_config = TargetStateConfig(
    min_size=10,
    min_duration=5,
    min_contribution=0.70,
)

cal = PlotCalibration(
    max_oligomer_size=100,
    size_tick_step=10,
    matrix_max_size=100,
    node_size_transform="sqrt",
)

lineage_result = build_lineages_from_cache(
    cache,
    inter_thresh=inter_thresh,
    cluster_min_size=2,
    overlap_min=1,
    parent_frac_min=0.30,
    child_frac_min=0.30,
    union_cover_min=0.70,
    target_config=target_config,
)

summary = lineage_summary_dataframe(lineage_result)
features = build_feature_table(cache, lineage_result, inter_thresh=inter_thresh)
targets = target_state_table(lineage_result)

print("\n=== Lineage summary ===")
print(summary.head(20).to_string(index=False))
print("\n=== Feature table ===")
print(features.head().to_string(index=False))
print("\n=== Target states ===")
print(targets.head().to_string(index=False))

plot_whole_system_background(cache, lineage_result, calibration=cal)
plot_lineage_graph(lineage_result, calibration=cal, node_size_by="max_size")
plot_lineage_fate(lineage_result, calibration=cal)

figure_commitment_predictors(cache, lineage_result, inter_thresh=inter_thresh)
figure_llps_coupling(cache, lineage_result)
figure_contact_hotspots(cache, lineage_result, inter_thresh=inter_thresh)
