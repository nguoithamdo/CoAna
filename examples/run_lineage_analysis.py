from oligomer_analysis import (
    TargetStateConfig,
    build_lineages_from_cache,
    lineage_summary_dataframe,
    load_cache,
    plot_lineage_fate,
    plot_lineage_graph,
    plot_on_off_pathway_dashboards,
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

summary = lineage_summary_dataframe(lineage_result)
print(summary.head(20))

plot_whole_system_background(cache, lineage_result)
plot_lineage_graph(lineage_result)
plot_lineage_fate(lineage_result)
plot_on_off_pathway_dashboards(cache, lineage_result, inter_thresh=1)
