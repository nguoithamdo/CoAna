# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .graph_tools import build_cluster_transition_matrix, filter_clusters_by_size
from .plotting_common import format_oligomer_label


def _pathway_get(pathway, key):
    """Support both dataclass-style pathways and dict-style pathways."""
    return getattr(pathway, key) if hasattr(pathway, key) else pathway[key]


def build_system_summary_dataframe(cache, frames_clusters: Sequence[Sequence[frozenset]], *, cluster_min_size: int = 2) -> pd.DataFrame:
    """Build a per-frame whole-system summary table.

    Clusters are counted only when size >= ``cluster_min_size``. Monomers remain available
    through the separate ``n_free_monomers`` and ``n_bound_monomers`` columns.

    The dataframe exposes both the concise contact column names used in the package
    (``inter_contacts``, ``total_contacts``, ``intra_contacts``) and the explicit notebook-
    friendly aliases (``ninter_contacts``, ``ntotal_contacts``, ``nintra_contacts``).
    """
    rows = []
    for i, clusters in enumerate(frames_clusters):
        oligomers = filter_clusters_by_size(clusters, min_size=cluster_min_size)
        sizes = sorted((len(c) for c in oligomers), reverse=True)
        inter_contacts = int(cache.ninter[i])
        total_contacts = int(cache.ntotal[i])
        intra_contacts = int(cache.nintra[i])
        rows.append({
            "frame": int(cache.frames[i]),
            "n_clusters": int(len(oligomers)),
            "largest_cluster": int(max(sizes)) if sizes else 0,
            "mean_cluster_size": float(np.mean(sizes)) if sizes else 0.0,
            "n_free_monomers": int(cache.n_free[i]),
            "n_bound_monomers": int(cache.n_bound[i]),
            "total_contacts": total_contacts,
            "inter_contacts": inter_contacts,
            "intra_contacts": intra_contacts,
            "ntotal_contacts": total_contacts,
            "ninter_contacts": inter_contacts,
            "nintra_contacts": intra_contacts,
        })
    return pd.DataFrame(rows)


def collect_cluster_sizes(frames_clusters: Sequence[Sequence[frozenset]], *, min_size: int = 2) -> list[int]:
    """Flatten all cluster sizes across all frames, excluding objects smaller than ``min_size``."""
    return [len(cluster) for frame_clusters in frames_clusters for cluster in frame_clusters if len(cluster) >= int(min_size)]


def pathway_timeline_dataframe(cache, finished: Iterable, *, min_duration: int = 1, label_mode: str = "full") -> pd.DataFrame:
    """Build a long-format dataframe of pathway size vs frame."""
    rows = []
    for p in finished:
        duration = int(_pathway_get(p, "duration"))
        if duration < int(min_duration):
            continue
        label = _pathway_get(p, "label")
        display_label = format_oligomer_label(label, label_mode)
        start_idx = int(_pathway_get(p, "start_idx"))
        sizes = list(_pathway_get(p, "sizes"))
        for k, size in enumerate(sizes):
            rows.append({
                "label": label,
                "display_label": display_label,
                "frame": int(cache.frames[start_idx + k]),
                "size": int(size),
            })
    return pd.DataFrame(rows)




def oligomer_sizes_by_frame_dataframe(
    cache,
    frames_clusters: Sequence[Sequence[frozenset]],
    *,
    min_size: int = 2,
    include_members: bool = True,
) -> pd.DataFrame:
    """Create a dataframe showing every oligomer in every frame.

    One row is one oligomer/cluster in one frame. Use ``min_size=2`` to exclude
    monomers.
    """
    rows = []
    for frame_index, clusters in enumerate(frames_clusters):
        frame_number = int(cache.frames[frame_index])
        oligomers = [c for c in clusters if len(c) >= int(min_size)]
        oligomers = sorted(oligomers, key=lambda c: (-len(c), sorted(list(c))))
        for oligomer_index, cluster in enumerate(oligomers, start=1):
            members = sorted(list(cluster))
            row = {
                "frame": frame_number,
                "frame_index": int(frame_index),
                "oligomer_index_in_frame": int(oligomer_index),
                "oligomer_size": int(len(members)),
            }
            if include_members:
                row["members"] = ",".join(map(str, members))
                row["members_list"] = members
            rows.append(row)
    return pd.DataFrame(rows)


def oligomer_summary_by_frame_dataframe(df_oligomers: pd.DataFrame) -> pd.DataFrame:
    """Create one compact row per frame from ``oligomer_sizes_by_frame_dataframe``.

    The output contains the number of oligomers, list of oligomer sizes, largest
    oligomer, and mean oligomer size for each frame.
    """
    if df_oligomers.empty:
        return pd.DataFrame(columns=[
            "frame", "n_oligomers", "oligomer_sizes", "largest_oligomer", "mean_oligomer_size"
        ])
    return (
        df_oligomers
        .groupby("frame")
        .agg(
            n_oligomers=("oligomer_size", "count"),
            oligomer_sizes=("oligomer_size", list),
            largest_oligomer=("oligomer_size", "max"),
            mean_oligomer_size=("oligomer_size", "mean"),
        )
        .reset_index()
    )


__all__ = [
    "build_cluster_transition_matrix",
    "build_system_summary_dataframe",
    "collect_cluster_sizes",
    "filter_clusters_by_size",
    "pathway_timeline_dataframe",
    "oligomer_sizes_by_frame_dataframe",
    "oligomer_summary_by_frame_dataframe",
]
