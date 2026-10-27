# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

"""Lineage-resolved commitment analysis for protein aggregation.

This module turns the lineage graph into a quantitative fate map.  The central
quantity is the **commitment score**: how much of a transient oligomer survives
into a later productive target aggregate.  This is deliberately fate-based.  It
prevents a common mistake in aggregation analysis: calling species on/off
pathway only from instantaneous size or compactness.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .models import LineageTrackingResult, OligomerSegment, TargetStateConfig


@dataclass(slots=True)
class TargetState:
    """One segment that satisfies the user-defined productive aggregate state."""

    segment_id: str
    lineage_id: Optional[str]
    start_frame: int
    end_frame: int
    duration: int
    max_size: int
    union_members: frozenset[int]


def find_target_segments(result: LineageTrackingResult, config: Optional[TargetStateConfig] = None) -> list[OligomerSegment]:
    """Return segments that satisfy the productive aggregate definition.

    A target aggregate is intentionally configurable.  For a first pass, use a
    persistent large assembly, for example size >= 10 for at least 5 frames.
    """
    cfg = config or result.target_config
    return [seg for seg in result.segments if seg.max_size >= cfg.min_size and seg.duration >= cfg.min_duration]


def target_state_table(result: LineageTrackingResult, config: Optional[TargetStateConfig] = None) -> pd.DataFrame:
    """Tidy table of productive target states."""
    targets = find_target_segments(result, config)
    rows = []
    for seg in targets:
        rows.append({
            "segment_id": seg.segment_id,
            "lineage_id": seg.lineage_id,
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
            "duration": seg.duration,
            "max_size": seg.max_size,
            "mean_size": seg.mean_size,
            "n_union_members": len(seg.union_members),
        })
    return pd.DataFrame(rows)


def state_commitment_to_targets(state: frozenset[int], frame: int, targets: Iterable[OligomerSegment]) -> float:
    """Maximum future contribution of one cluster state to later target states.

    commitment = max_{future target A} |state ∩ A| / |state|

    Values near 1 mean most monomers in this state later become part of a
    productive aggregate.  Values near 0 indicate no productive future fate.
    """
    if not state:
        return 0.0
    best = 0.0
    for target in targets:
        for target_frame, target_state in zip(target.frames, target.sets):
            if int(target_frame) <= int(frame):
                continue
            best = max(best, len(state & target_state) / float(len(state)))
    return float(best)


def compute_segment_commitment_scores(result: LineageTrackingResult, config: Optional[TargetStateConfig] = None) -> pd.DataFrame:
    """Compute future commitment scores for every segment.

    This returns a publication-ready table for testing whether early structural
    descriptors predict future productive aggregation.
    """
    cfg = config or result.target_config
    targets = find_target_segments(result, cfg)
    rows = []
    for seg in result.segments:
        state_scores = [state_commitment_to_targets(state, frame, targets) for frame, state in zip(seg.frames, seg.sets)]
        max_commitment = max(state_scores, default=0.0)
        mean_commitment = float(np.mean(state_scores)) if state_scores else 0.0
        target_reached = seg in targets
        if target_reached:
            max_commitment = max(1.0, max_commitment)
        rows.append({
            "segment_id": seg.segment_id,
            "lineage_id": seg.lineage_id,
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
            "duration": seg.duration,
            "initial_size": seg.sizes[0] if seg.sizes else 0,
            "max_size": seg.max_size,
            "mean_size": seg.mean_size,
            "target_reached": bool(target_reached),
            "max_commitment": float(max_commitment),
            "mean_commitment": float(mean_commitment),
            "class_hint": "on-pathway" if (target_reached or max_commitment >= cfg.min_contribution) else "not-committed",
        })
    return pd.DataFrame(rows)


def compute_lineage_commitment_scores(result: LineageTrackingResult, config: Optional[TargetStateConfig] = None) -> pd.DataFrame:
    """Compute commitment and fate summaries for each lineage."""
    cfg = config or result.target_config
    seg_df = compute_segment_commitment_scores(result, cfg)
    lin_by_id = {lin.lineage_id: lin for lin in result.lineages}
    rows = []
    for lineage_id, sub in seg_df.groupby("lineage_id", dropna=False):
        if pd.isna(lineage_id):
            continue
        lin = lin_by_id.get(str(lineage_id))
        rows.append({
            "lineage_id": str(lineage_id),
            "classification": lin.classification if lin is not None else "unknown",
            "n_segments": int(len(sub)),
            "target_reached": bool(sub["target_reached"].any()),
            "max_commitment": float(sub["max_commitment"].max()),
            "mean_segment_commitment": float(sub["max_commitment"].mean()),
            "max_size": int(sub["max_size"].max()),
            "total_segment_duration": int(sub["duration"].sum()),
            "growth_score": float(lin.growth_score) if lin is not None else np.nan,
            "trap_score": float(lin.trap_score) if lin is not None else np.nan,
            "persistence_score": float(lin.persistence_score) if lin is not None else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["max_commitment", "max_size"], ascending=[False, False]) if rows else pd.DataFrame()


def build_commitment_feature_table(result: LineageTrackingResult, features_df: Optional[pd.DataFrame] = None, config: Optional[TargetStateConfig] = None) -> pd.DataFrame:
    """Join commitment labels with optional structural features.

    The returned table is the main input for publication-style analyses such as:
    compactness vs future commitment, small-worldness vs fate, or early contact
    persistence vs productive aggregation.
    """
    commit = compute_segment_commitment_scores(result, config)
    if features_df is None or features_df.empty:
        return commit
    return features_df.merge(commit, on=["segment_id", "lineage_id"], how="left", suffixes=("", "_commitment"))


__all__ = [
    "TargetState",
    "find_target_segments",
    "target_state_table",
    "state_commitment_to_targets",
    "compute_segment_commitment_scores",
    "compute_lineage_commitment_scores",
    "build_commitment_feature_table",
]
