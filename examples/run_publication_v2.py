"""V2 publication workflow: commitment, predictors, LLPS, and hotspots."""

from oligomer_analysis import (
    PlotCalibration,
    TargetStateConfig,
    build_lineages_from_cache,
    build_feature_table,
    compute_lineage_commitment_scores,
    figure_commitment_predictors,
    figure_contact_hotspots,
    figure_llps_coupling,
    load_cache,
    plot_lineage_fate,
    plot_lineage_graph,
    plot_whole_system_background,
)

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

cal = PlotCalibration(max_oligomer_size=100, size_tick_step=10, matrix_max_size=100)

# Core figures
plot_whole_system_background(cache, lineage_result, calibration=cal)
plot_lineage_graph(lineage_result, calibration=cal, node_size_by="max_size")
plot_lineage_fate(lineage_result, calibration=cal)

# Publication discovery tables/figures
features = build_feature_table(cache, lineage_result, inter_thresh=1)
commitment = compute_lineage_commitment_scores(lineage_result)
print(features.head())
print(commitment.head())

figure_commitment_predictors(cache, lineage_result, inter_thresh=1)
figure_llps_coupling(cache, lineage_result)
figure_contact_hotspots(cache, lineage_result, inter_thresh=1)
