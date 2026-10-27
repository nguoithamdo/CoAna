# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

"""Structural and dynamical predictors of aggregation fate.

These features are designed for the scientific question: which early oligomer
properties predict future productive aggregation?  They should be joined with
commitment scores from :mod:`oligomer_analysis.commitment` rather than used as
circular definitions of on/off-pathway state.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .metrics import compactness_timeseries, small_worldness_S
from .models import ContactCache, LineageTrackingResult, OligomerSegment, Pathway


def _segment_as_pathway(segment: OligomerSegment) -> Pathway:
    return Pathway(
        label=segment.segment_id,
        start_idx=segment.start_idx,
        start_frame=segment.start_frame,
        duration=segment.duration,
        sizes=list(segment.sizes),
        sets=list(segment.sets),
    )


def _snapshot_graph(members, cmap, inter_thresh: int):
    import networkx as nx

    members = sorted(list(members))
    G = nx.Graph()
    G.add_nodes_from(members)
    for i in range(len(members) - 1):
        for j in range(i + 1, len(members)):
            a, b = members[i], members[j]
            if cmap[a, b] >= inter_thresh:
                G.add_edge(a, b, weight=int(cmap[a, b]))
    return G


def segment_contact_persistence(cache: ContactCache, segment: OligomerSegment, inter_thresh: int) -> float:
    """Fraction of internal segment-pair edges that persist across the segment.

    High persistence suggests a stable nucleus or trap; low persistence suggests
    a loose encounter complex.  This is intentionally separate from future fate.

    Each pair is scored only over the frames in which **both** partners belong to
    the segment.  Dividing instead by the whole segment duration would charge a
    pair for the frames before it could exist at all, which makes a growing
    assembly look unstable purely because it recruited chains late: a nucleus
    that grows 2 -> 11 chains with every contact present in every frame it could
    be scores 0.4 under that denominator and 1.0 under this one.  Since this
    predictor is tested against future fate, that artefact would be
    anti-correlated with growth by construction.
    """
    edge_counts: dict[tuple[int, int], int] = {}
    edge_coresident: dict[tuple[int, int], int] = {}
    for local_t, members in enumerate(segment.sets):
        idx = segment.start_idx + local_t
        mem = sorted(list(members))
        cmap = cache.cmaps[idx]
        for i in range(len(mem) - 1):
            for j in range(i + 1, len(mem)):
                a, b = mem[i], mem[j]
                edge_coresident[(a, b)] = edge_coresident.get((a, b), 0) + 1
                if cmap[a, b] >= inter_thresh:
                    edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1
    if not edge_counts:
        return 0.0
    return float(
        np.mean(
            [
                count / max(1, edge_coresident[edge])
                for edge, count in edge_counts.items()
            ]
        )
    )


def segment_exchange_fraction(segment: OligomerSegment) -> float:
    """Membership exchange over a segment.

    0 means initial and final membership are identical; high values indicate
    exchange/reorganization, which may be relevant for LLPS-like fluidity.
    """
    if segment.duration <= 1:
        return 0.0
    initial, final = set(segment.initial_set), set(segment.final_set)
    union = initial | final
    if not union:
        return 0.0
    return float(1.0 - len(initial & final) / len(union))


def compute_segment_features(
    cache: ContactCache,
    result: LineageTrackingResult,
    *,
    inter_thresh: int,
    n_rand_smallworld: int = 20,
    smallworld_min_size: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute predictor features for every oligomer segment.

    Small-worldness is reported only for snapshots with enough monomers by
    default, because small-world statistics are unstable for tiny oligomers.
    """
    rows = []
    for k, seg in enumerate(result.segments):
        pathway = _segment_as_pathway(seg)
        density = compactness_timeseries(cache, pathway, inter_thresh, mode="density")
        degree = compactness_timeseries(cache, pathway, inter_thresh, mode="degree")
        weighted = compactness_timeseries(cache, pathway, inter_thresh, mode="weighted")
        rg = compactness_timeseries(cache, pathway, inter_thresh, mode="rg")

        sw_values = []
        for local_t, members in enumerate(seg.sets):
            if len(members) < int(smallworld_min_size):
                continue
            idx = seg.start_idx + local_t
            G = _snapshot_graph(members, cache.cmaps[idx], inter_thresh)
            sw = small_worldness_S(G, n_rand=n_rand_smallworld, seed=seed + k + local_t)
            sw_values.append(sw.S)

        sizes = np.asarray(seg.sizes, dtype=float)
        if seg.duration > 1:
            growth_rate = float((sizes[-1] - sizes[0]) / max(1.0, seg.duration - 1))
            size_slope = float(np.polyfit(np.arange(seg.duration, dtype=float), sizes, 1)[0])
        else:
            growth_rate = 0.0
            size_slope = 0.0

        rows.append({
            "segment_id": seg.segment_id,
            "lineage_id": seg.lineage_id,
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
            "duration": seg.duration,
            "initial_size": int(seg.sizes[0]) if seg.sizes else 0,
            "final_size": int(seg.sizes[-1]) if seg.sizes else 0,
            "max_size": seg.max_size,
            "mean_size": seg.mean_size,
            "growth_rate": growth_rate,
            "size_slope": size_slope,
            "mean_contact_density": float(np.nanmean(density)) if len(density) else np.nan,
            "mean_degree": float(np.nanmean(degree)) if len(degree) else np.nan,
            "mean_weighted_contacts": float(np.nanmean(weighted)) if len(weighted) else np.nan,
            "mean_rg": float(np.nanmean(rg)) if len(rg) else np.nan,
            "contact_persistence": segment_contact_persistence(cache, seg, inter_thresh),
            "exchange_fraction": segment_exchange_fraction(seg),
            "mean_smallworld_S": float(np.nanmean(sw_values)) if sw_values else np.nan,
            "n_smallworld_snapshots": int(len(sw_values)),
        })
    return pd.DataFrame(rows)


def compute_lineage_features(segment_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate segment features to lineage-level predictors."""
    if segment_features.empty:
        return pd.DataFrame()
    agg = segment_features.groupby("lineage_id", dropna=False).agg(
        n_segments=("segment_id", "count"),
        total_segment_duration=("duration", "sum"),
        max_size=("max_size", "max"),
        mean_size=("mean_size", "mean"),
        mean_growth_rate=("growth_rate", "mean"),
        mean_contact_density=("mean_contact_density", "mean"),
        mean_degree=("mean_degree", "mean"),
        mean_weighted_contacts=("mean_weighted_contacts", "mean"),
        mean_rg=("mean_rg", "mean"),
        mean_contact_persistence=("contact_persistence", "mean"),
        mean_exchange_fraction=("exchange_fraction", "mean"),
        mean_smallworld_S=("mean_smallworld_S", "mean"),
    ).reset_index()
    return agg


def build_feature_table(cache: ContactCache, result: LineageTrackingResult, *, inter_thresh: int, include_commitment: bool = True) -> pd.DataFrame:
    """One-call segment feature table, optionally joined with commitment scores."""
    features = compute_segment_features(cache, result, inter_thresh=inter_thresh)
    if not include_commitment:
        return features
    from .commitment import build_commitment_feature_table

    return build_commitment_feature_table(result, features)


__all__ = [
    "segment_contact_persistence",
    "segment_exchange_fraction",
    "compute_segment_features",
    "compute_lineage_features",
    "build_feature_table",
]
