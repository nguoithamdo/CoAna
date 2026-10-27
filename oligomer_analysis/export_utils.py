# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

"""Export utilities for monitoring tables."""

from pathlib import Path
from typing import Optional

import pandas as pd


def save_oligomer_frame_excel(
    df_oligomers: pd.DataFrame,
    df_oligomer_summary: pd.DataFrame,
    output_file: str | Path = "oligomer_frame_analysis.xlsx",
    *,
    oligomers_sheet: str = "Oligomers_By_Frame",
    summary_sheet: str = "Frame_Summary",
) -> Path:
    """Save per-oligomer and per-frame summary tables into one Excel file.

    The file has two sheets:
    1. ``Oligomers_By_Frame``: one row per oligomer in each frame.
    2. ``Frame_Summary``: one row per frame with number/sizes of oligomers.
    """
    output_file = Path(output_file)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_oligomers.to_excel(writer, sheet_name=oligomers_sheet, index=False)
        df_oligomer_summary.to_excel(writer, sheet_name=summary_sheet, index=False)
    return output_file


def save_all_monitoring_csvs(
    cache,
    *,
    res=None,
    finished=None,
    lineage_result=None,
    inter_thresh: int = 1,
    out_prefix: str = "monitor",
):
    """Save common monitoring CSV files from cache, old pathways, and lineages."""
    from .system_analysis import (
        build_system_summary_dataframe,
        pathway_timeline_dataframe,
    )
    from .lineage import lineage_summary_dataframe
    from .commitment import target_state_table
    from .predictors import build_feature_table

    out_prefix = str(out_prefix)
    df_contact_stats = pd.DataFrame({
        "frame": cache.frames,
        "nintra": cache.nintra,
        "ninter": cache.ninter,
        "ntotal": cache.ntotal,
        "n_free": cache.n_free,
        "n_bound": cache.n_bound,
    })
    df_contact_stats.to_csv(f"{out_prefix}_contact_stats.csv", index=False)

    if res is not None:
        frames_clusters = res["frames_clusters"]
        df_system = build_system_summary_dataframe(cache, frames_clusters, cluster_min_size=2)
        df_system.to_csv(f"{out_prefix}_whole_system.csv", index=False)
        if finished is None:
            finished = res.get("finished", None)

    if finished is not None:
        df_pathways = pathway_timeline_dataframe(cache, finished, min_duration=1, label_mode="ordinal")
        df_pathways.to_csv(f"{out_prefix}_pathway_timeline.csv", index=False)

    if lineage_result is not None:
        df_lineages = lineage_summary_dataframe(lineage_result)
        df_lineages.to_csv(f"{out_prefix}_lineage_summary.csv", index=False)
        df_targets = target_state_table(lineage_result)
        df_targets.to_csv(f"{out_prefix}_target_states.csv", index=False)
        df_features = build_feature_table(cache, lineage_result, inter_thresh=inter_thresh)
        df_features.to_csv(f"{out_prefix}_feature_table.csv", index=False)

    print(f"Saved monitoring CSV files with prefix: {out_prefix}")


__all__ = ["save_oligomer_frame_excel", "save_all_monitoring_csvs"]
