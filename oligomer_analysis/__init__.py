"""Oligomer analysis toolkit for MD trajectory contact clustering, pathway monitoring,
and split/merge-aware protein aggregation lineage analysis.

The package is intentionally importable without MDAnalysis.  Stage B/cache analysis,
lineage tracking, and plotting can be used from a saved cache.  Stage A trajectory
precomputation imports MDAnalysis lazily only when :func:`precompute_cache` is called.
"""

from .cache_io import load_cache, validate_cache, save_contact_cache, subset_cache_by_indices, subset_cache_by_frames
from .models import (
    ContactCache,
    FrameContacts,
    Lineage,
    LineageEvent,
    LineageTrackingResult,
    OligomerSegment,
    Pathway,
    PrecomputeConfig,
    TargetStateConfig,
)
from .plotting_common import PlotCalibration, format_oligomer_label, display_oligomer_label, legend_enabled
from .monitor import (
    monitor_oligomer_label,
    plot_oligomer_panel,
    plot_panel9_style_snapshot,
    plot_small_world_range,
    draw_transition_graph_from_prob,
)
from .tracking import analyze_from_cache
from .lineage import (
    attach_species_compositions,
    build_lineages_from_cache,
    directed_fate_summary,
    directed_fates,
    filter_lineages,
    lineage_summary_dataframe,
    lineage_warnings,
    representative_pathways_for_lineages,
    sweep_verdict,
    target_size_sweep,
)
from .lineage_plots import (
    plot_lineage_fate,
    plot_lineage_graph,
    plot_on_off_pathway_dashboards,
    plot_whole_system_background,
)

from .commitment import (
    build_commitment_feature_table,
    compute_lineage_commitment_scores,
    compute_segment_commitment_scores,
    find_target_segments,
    target_state_table,
)
from .predictors import (
    build_feature_table,
    compute_lineage_features,
    compute_segment_features,
)
from .llps import (
    DensePhaseCalibration,
    calibrate_dense_phase,
    dense_phase_fraction_timeseries,
    lineage_dense_phase_table,
    radial_density_profile,
)
from .hotspots import (
    contact_enrichment_on_vs_off,
    contact_frequency_by_fate,
    hotspot_block_dataframe,
    rank_contact_hotspots,
    region_contact_table,
    residue_hotspot_matrix,
    residue_hotspot_profile,
    top_residue_pairs,
)
from .chemistry import (
    CONTACT_CLASSES,
    classify_residues,
    contact_class_census,
    contact_class_dataframe,
    contact_class_matrix,
    contact_class_summary,
    describe_contact_classes,
    hydrophobic_fraction_timeseries,
    salt_bridge_timeseries,
)
from .convergence import (
    ConvergenceReport,
    assess_convergence,
    convergence_dataframe,
    integrated_autocorrelation_time,
)
from .compare import compare_caches, group_summary, summarize_cache
from .mechanism import (
    ResidenceEvent,
    exchange_statistics,
    mechanism_summary,
    residence_dataframe,
    residence_events,
)
from .network import (
    state_labels_by_frame,
    transition_counts,
    transition_dataframe,
    transition_matrix,
    write_lineage_gexf,
    write_transition_gexf,
)
from .publication_figures import (
    figure_commitment_predictors,
    figure_contact_hotspots,
    figure_llps_coupling,
    figure_lineage_overview,
    figure_system_overview,
)

from .heatmaps import plot_pathway_presence_heatmap_safe
from .export_utils import save_oligomer_frame_excel, save_all_monitoring_csvs

from .system_analysis import (
    build_cluster_transition_matrix,
    build_system_summary_dataframe,
    collect_cluster_sizes,
    filter_clusters_by_size,
    pathway_timeline_dataframe,
    oligomer_sizes_by_frame_dataframe,
    oligomer_summary_by_frame_dataframe,
)


from .interactive import (
    build_temporal_network_dataframe,
    plot_lineage_temporal_network,
    make_lineage_temporal_animation,
    export_lineage_temporal_html,
    lineage_network_filter_panel,
)


from .coanapure import build_model_index, precompute_cache_pure
from .report import generate_html_report
from .timeline import (
    build_timeline_payload,
    generate_lineage_timeline,
    render_timeline_html,
)

__version__ = "3.6.2"


def precompute_cache(*args, **kwargs):
    """Run Stage A cache precomputation with the original MDAnalysis engine.

    MDAnalysis is imported lazily here so the rest of the package remains usable
    on machines where only cached `.npz` analysis is needed.  The default Stage A
    engine is :func:`precompute_cache_pure` (CoAnaPure), which needs no
    MDAnalysis and reads PDB, GRO, and TRR trajectories directly; use this
    function for other formats or to cross-validate the two engines.
    """
    try:
        from .precompute import precompute_cache as _precompute_cache
    except Exception as exc:  # pragma: no cover - depends on optional MDAnalysis
        raise ImportError(
            "precompute_cache requires MDAnalysis. Install MDAnalysis, or use "
            "precompute_cache_pure(...) (the CoAnaPure engine, no MDAnalysis "
            "needed) to build caches from PDB/GRO/TRR trajectories."
        ) from exc
    return _precompute_cache(*args, **kwargs)


__all__ = [
    "SpeciesMap",
    "SegmentationReport",
    "cluster_compositions",
    "composition_timeseries",
    "species_bound_free",
    "species_contact_dataframe",
    "species_contact_matrix",
    "validate_segmentation",
    "__version__",
    "precompute_cache_pure",
    "build_model_index",
    "generate_html_report",
    "generate_lineage_timeline",
    "build_timeline_payload",
    "render_timeline_html",
    "save_all_monitoring_csvs",
    "save_oligomer_frame_excel",
    "plot_pathway_presence_heatmap_safe",
    "oligomer_summary_by_frame_dataframe",
    "oligomer_sizes_by_frame_dataframe",
    "draw_transition_graph_from_prob",
    "subset_cache_by_frames",
    "subset_cache_by_indices",
    "save_contact_cache",
    "lineage_network_filter_panel",
    "export_lineage_temporal_html",
    "make_lineage_temporal_animation",
    "plot_lineage_temporal_network",
    "build_temporal_network_dataframe",
    "display_oligomer_label",
    "legend_enabled",
    "format_oligomer_label",
    "pdb_chain_id_summary",
    "debug_lineage_events",
    "debug_lineage_summary",
    "debug_frame_transition",
    "debug_cluster_overlap",
    "debug_pathway",
    "debug_edge_changes_focus",
    "debug_edge_changes",
    "debug_row_sums",
    "debug_cmap_pairs",
    "debug_show_big_clusters",
    "PlotCalibration",
    "ContactCache",
    "FrameContacts",
    "Lineage",
    "LineageEvent",
    "LineageTrackingResult",
    "OligomerSegment",
    "Pathway",
    "PrecomputeConfig",
    "TargetStateConfig",
    "analyze_from_cache",
    "build_cluster_transition_matrix",
    "build_lineages_from_cache",
    "directed_fates",
    "directed_fate_summary",
    "build_system_summary_dataframe",
    "collect_cluster_sizes",
    "filter_clusters_by_size",
    "lineage_summary_dataframe",
    "load_cache",
    "monitor_oligomer_label",
    "pathway_timeline_dataframe",
    "plot_lineage_fate",
    "plot_lineage_graph",
    "plot_on_off_pathway_dashboards",
    "plot_oligomer_panel",
    "plot_panel9_style_snapshot",
    "plot_small_world_range",
    "plot_whole_system_background",
    "precompute_cache",
    "representative_pathways_for_lineages",
    "validate_cache",
    "build_commitment_feature_table",
    "build_feature_table",
    "compute_lineage_commitment_scores",
    "compute_lineage_features",
    "compute_segment_commitment_scores",
    "compute_segment_features",
    "contact_enrichment_on_vs_off",
    "contact_frequency_by_fate",
    "DensePhaseCalibration",
    "calibrate_dense_phase",
    "dense_phase_fraction_timeseries",
    "figure_commitment_predictors",
    "figure_contact_hotspots",
    "figure_lineage_overview",
    "figure_llps_coupling",
    "figure_system_overview",
    "find_target_segments",
    "lineage_dense_phase_table",
    "radial_density_profile",
    "rank_contact_hotspots",
    "target_state_table",
    # v3.4 -- species-aware lineages, chemistry, convergence, comparison,
    # residue hotspots, mechanism, networks
    "attach_species_compositions",
    "filter_lineages",
    "lineage_warnings",
    "target_size_sweep",
    "sweep_verdict",
    "parse_species_spec",
    "monomer_atom_counts_from_spec",
    "partition_coefficients",
    "partition_summary",
    "free_monomer_concentration",
    "CONTACT_CLASSES",
    "classify_residues",
    "contact_class_census",
    "contact_class_matrix",
    "contact_class_dataframe",
    "contact_class_summary",
    "describe_contact_classes",
    "salt_bridge_timeseries",
    "hydrophobic_fraction_timeseries",
    "ConvergenceReport",
    "assess_convergence",
    "convergence_dataframe",
    "integrated_autocorrelation_time",
    "summarize_cache",
    "compare_caches",
    "group_summary",
    "residue_hotspot_matrix",
    "residue_hotspot_profile",
    "top_residue_pairs",
    "region_contact_table",
    "hotspot_block_dataframe",
    "ResidenceEvent",
    "residence_events",
    "residence_dataframe",
    "mechanism_summary",
    "exchange_statistics",
    "state_labels_by_frame",
    "transition_counts",
    "transition_matrix",
    "transition_dataframe",
    "write_lineage_gexf",
    "write_transition_gexf",
]


from .debug_tools import (
    debug_show_big_clusters,
    debug_cmap_pairs,
    debug_row_sums,
    debug_edge_changes,
    debug_edge_changes_focus,
    debug_pathway,
    debug_cluster_overlap,
    debug_frame_transition,
    debug_lineage_summary,
    debug_lineage_events,
    pdb_chain_id_summary,
)
from .species import (
    SpeciesMap,
    SegmentationReport,
    cluster_compositions,
    composition_timeseries,
    free_monomer_concentration,
    monomer_atom_counts_from_spec,
    parse_species_spec,
    partition_coefficients,
    partition_summary,
    species_bound_free,
    species_contact_dataframe,
    species_contact_matrix,
    validate_segmentation,
)
