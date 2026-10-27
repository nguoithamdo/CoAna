# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

"""Split/merge-aware oligomer lineage tracking for protein aggregation.

This module is the scientifically important upgrade over linear pathway tracking.
The old :func:`oligomer_analysis.tracking.analyze_from_cache` answers:

    "Which connected component best overlaps with a previous component?"

For protein aggregation, the better question is:

    "Which assembly lineages progress toward productive aggregates, and which
    become transient species, side branches, or kinetic traps?"

Therefore this module tracks **segments**, **events**, and **lineages**:

- A segment is one continuous connected oligomer episode.
- Events connect segments through continuation-breaking transitions such as
  split, merge, birth, death, or rearrangement.
- A lineage is the graph component connecting related segments.

A temporary split such as ``10 -> 6 + 4 -> 10`` is therefore represented as one
lineage with explicit split/merge events.  It is not automatically off-pathway;
classification is fate-based.

In a mixture, fate alone is still not the whole question.  Passing a
:class:`~oligomer_analysis.species.SpeciesMap` to
:func:`build_lineages_from_cache` records what each assembly is *made of* at
every frame, so the summary table carries per-species counts beside the fate
label.  That join is what lets the analysis ask whether the productive lineages
are the mixed ones -- the molecular form of "does the second component nucleate
the first?" -- instead of inferring it from two separate tables.
"""

from itertools import count
from typing import TYPE_CHECKING, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .species import SpeciesMap

from .graph_tools import clusters_from_cmap, filter_clusters_by_size
from .metrics import compactness_timeseries
from .models import (
    ContactCache,
    Lineage,
    LineageEvent,
    LineageTrackingResult,
    OligomerSegment,
    Pathway,
    TargetStateConfig,
)


# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------

def _segment_as_pathway(segment: OligomerSegment) -> Pathway:
    """Convert an :class:`OligomerSegment` into the old Pathway object.

    This adapter allows the new lineage machinery to reuse the mature 9-panel
    dashboard without rewriting every plot at once.
    """
    return Pathway(
        label=segment.segment_id,
        start_idx=int(segment.start_idx),
        start_frame=int(segment.start_frame),
        duration=int(segment.duration),
        sizes=list(segment.sizes),
        sets=list(segment.sets),
    )


# ---------------------------------------------------------------------------
# Low-level overlap/event utilities
# ---------------------------------------------------------------------------

def _overlap_matrix(prev_clusters: Sequence[FrozenSet[int]], curr_clusters: Sequence[FrozenSet[int]]) -> np.ndarray:
    M = np.zeros((len(prev_clusters), len(curr_clusters)), dtype=int)
    for i, prev in enumerate(prev_clusters):
        for j, curr in enumerate(curr_clusters):
            M[i, j] = len(prev & curr)
    return M


def _fraction_matrices(overlap: np.ndarray, prev_clusters: Sequence[FrozenSet[int]], curr_clusters: Sequence[FrozenSet[int]]) -> Tuple[np.ndarray, np.ndarray]:
    parent_fraction = np.zeros_like(overlap, dtype=float)
    child_fraction = np.zeros_like(overlap, dtype=float)
    for i, prev in enumerate(prev_clusters):
        if len(prev) > 0:
            parent_fraction[i, :] = overlap[i, :] / float(len(prev))
    for j, curr in enumerate(curr_clusters):
        if len(curr) > 0:
            child_fraction[:, j] = overlap[:, j] / float(len(curr))
    return parent_fraction, child_fraction


def _meaningful_overlap(
    ov: int,
    prev_size: int,
    curr_size: int,
    overlap_min: int,
    parent_frac_min: float,
    child_frac_min: float,
) -> bool:
    """Return True if an overlap is large enough to imply inheritance.

    We allow either a large fraction of the parent or a large fraction of the
    child to match.  This asymmetry is useful for aggregation: a small child may
    carry only part of a large parent during a split, while a large child may be
    assembled from several smaller parents during a merge.
    """
    if ov < int(overlap_min):
        return False
    if prev_size <= 0 or curr_size <= 0:
        return False
    return (ov / prev_size >= float(parent_frac_min)) or (ov / curr_size >= float(child_frac_min))


def _coverage_of_parent(parent: FrozenSet[int], children: Sequence[FrozenSet[int]]) -> float:
    if not parent:
        return 0.0
    union_children: set[int] = set()
    for child in children:
        union_children.update(child)
    return len(parent & union_children) / float(len(parent))


def _coverage_of_child(child: FrozenSet[int], parents: Sequence[FrozenSet[int]]) -> float:
    if not child:
        return 0.0
    union_parents: set[int] = set()
    for parent in parents:
        union_parents.update(parent)
    return len(child & union_parents) / float(len(child))


def _bipartite_components(
    parent_to_children: Dict[int, List[int]],
    child_to_parents: Dict[int, List[int]],
) -> List[Tuple[List[int], List[int]]]:
    """Connected components of the parent-child overlap graph."""
    seen_p: set[int] = set()
    seen_c: set[int] = set()
    components: List[Tuple[List[int], List[int]]] = []

    for start_p in sorted(parent_to_children):
        if start_p in seen_p or not parent_to_children[start_p]:
            continue
        stack_p = [start_p]
        comp_p: set[int] = set()
        comp_c: set[int] = set()
        while stack_p:
            p = stack_p.pop()
            if p in seen_p:
                continue
            seen_p.add(p)
            comp_p.add(p)
            for c in parent_to_children.get(p, []):
                if c not in seen_c:
                    seen_c.add(c)
                    comp_c.add(c)
                    for p2 in child_to_parents.get(c, []):
                        if p2 not in seen_p:
                            stack_p.append(p2)
        components.append((sorted(comp_p), sorted(comp_c)))

    for c in sorted(child_to_parents):
        if c not in seen_c and child_to_parents[c]:
            # Defensive branch for pathological graphs; normally every child with
            # a parent is reached from the parent loop above.
            components.append((sorted(child_to_parents[c]), [c]))
            seen_c.add(c)
    return components


def _classify_event_type(
    parent_indices: Sequence[int],
    child_indices: Sequence[int],
    prev_clusters: Sequence[FrozenSet[int]],
    curr_clusters: Sequence[FrozenSet[int]],
    union_cover_min: float,
) -> str:
    """Classify a connected parent-child component into an event type.

    Split and merge calls require union coverage.  If the overlap graph is
    connected but coverage is insufficient, the event is labeled
    ``rearrangement`` rather than over-interpreted as a clean split/merge.
    """
    n_parent = len(parent_indices)
    n_child = len(child_indices)
    if n_parent == 0 and n_child > 0:
        return "birth"
    if n_child == 0 and n_parent > 0:
        return "death"

    if n_parent == 1 and n_child == 1:
        parent = prev_clusters[parent_indices[0]]
        child = curr_clusters[child_indices[0]]
        if len(child) > len(parent):
            return "growth"
        if len(child) < len(parent):
            return "shrink"
        return "continuation"

    if n_parent == 1 and n_child > 1:
        parent = prev_clusters[parent_indices[0]]
        children = [curr_clusters[j] for j in child_indices]
        return "split" if _coverage_of_parent(parent, children) >= union_cover_min else "rearrangement"

    if n_parent > 1 and n_child == 1:
        parents = [prev_clusters[i] for i in parent_indices]
        child = curr_clusters[child_indices[0]]
        return "merge" if _coverage_of_child(child, parents) >= union_cover_min else "rearrangement"

    if n_parent > 1 and n_child > 1:
        # Many-to-many transitions are real in dense aggregation systems.  We do
        # not force them into a linear story.
        return "split_merge"

    return "rearrangement"


def _new_segment(
    segment_counter,
    cluster: FrozenSet[int],
    frame_idx: int,
    frame_number: int,
    *,
    parent_ids: Iterable[str] = (),
    event_start: str = "birth",
) -> OligomerSegment:
    return OligomerSegment(
        segment_id=f"seg{next(segment_counter)}",
        start_idx=int(frame_idx),
        end_idx=int(frame_idx),
        frames=[int(frame_number)],
        sizes=[int(len(cluster))],
        sets=[frozenset(cluster)],
        parent_ids=list(parent_ids),
        child_ids=[],
        event_start=event_start,
        event_end="active",
    )


def _extend_segment(segment: OligomerSegment, cluster: FrozenSet[int], frame_idx: int, frame_number: int) -> None:
    segment.end_idx = int(frame_idx)
    segment.frames.append(int(frame_number))
    segment.sizes.append(int(len(cluster)))
    segment.sets.append(frozenset(cluster))


def _close_segment(segment: OligomerSegment, event_end: str) -> None:
    segment.event_end = event_end


def _connected_components_from_pairs(nodes: Iterable[str], pairs: Iterable[Tuple[str, str]]) -> List[List[str]]:
    adjacency: Dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in pairs:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen: set[str] = set()
    components: List[List[str]] = []
    for node in adjacency:
        if node in seen:
            continue
        stack = [node]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(adjacency.get(cur, ()))
        components.append(sorted(comp, key=lambda x: int(x.replace("seg", "")) if x.startswith("seg") else x))
    return components


# ---------------------------------------------------------------------------
# Fate metrics and classification
# ---------------------------------------------------------------------------

def _segment_future_contribution(segment: OligomerSegment, target_segments: Sequence[OligomerSegment]) -> float:
    """Maximum future monomer continuity into a productive target segment.

    This is the key fate metric.  For a segment state O(t), we ask how much of
    its monomer content survives into a later target aggregate A(t').

        contribution = |O(t) ∩ A(t')| / |O(t)|
    """
    best = 0.0
    for state_frame, state in zip(segment.frames, segment.sets):
        if not state:
            continue
        for target in target_segments:
            for target_frame, target_state in zip(target.frames, target.sets):
                if target_frame <= state_frame:
                    continue
                best = max(best, len(state & target_state) / float(len(state)))
    return float(best)


def _segment_growth_score(segment: OligomerSegment) -> float:
    if segment.duration <= 1:
        return 0.0
    diffs = np.diff(np.asarray(segment.sizes, dtype=float))
    pos = np.clip(diffs, 0, None).sum()
    neg = np.clip(-diffs, 0, None).sum()
    denom = max(1.0, pos + neg)
    return float((pos - neg) / denom)


def _segment_reversibility_score(segment: OligomerSegment) -> float:
    """Size-direction oscillation within one segment.

    High values mean repeated grow/shrink changes without clear commitment.  This
    supports trap/off-pathway classification but does not decide it alone.
    """
    if segment.duration <= 2:
        return 0.0
    diffs = np.diff(np.asarray(segment.sizes, dtype=float))
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) <= 1:
        return 0.0
    return float(np.sum(signs[1:] != signs[:-1]) / max(1, len(signs) - 1))


def _lineage_structural_score(cache: ContactCache, segment: Optional[OligomerSegment], inter_thresh: int) -> float:
    if segment is None:
        return float("nan")
    try:
        pathway = _segment_as_pathway(segment)
        density = compactness_timeseries(cache, pathway, inter_thresh, mode="density")
        if np.isnan(density).all():
            return float("nan")
        return float(np.nanmean(density))
    except Exception:
        return float("nan")


def _classify_lineage(
    *,
    target_reached: bool,
    contribution_score: float,
    growth_score: float,
    trap_score: float,
    lineage_duration: int,
    config: TargetStateConfig,
) -> str:
    """Transparent first-pass fate classifier.

    This is intentionally a ruleset, not a black-box model.  It gives a robust
    scientific starting point and makes every on/off call auditable.

    Persistence deliberately plays no part here.  It used to be accepted as an
    argument and never read, while every lineage's ``notes`` claimed the
    classification weighed it -- so the stated method and the actual one
    disagreed.  Duration enters through ``lineage_duration`` against
    ``config.min_duration``; ``persistence_score`` is that duration divided by
    the length of the whole trajectory, which is reported for ranking but is not
    a fate criterion, because a lineage occupying 5% of a long run can be fully
    productive.
    """
    on_thr = float(config.on_path_contribution_threshold or config.min_contribution)
    off_thr = float(config.off_path_contribution_threshold)

    if target_reached:
        return "on-pathway"
    if contribution_score >= on_thr and trap_score < 0.70:
        return "on-pathway"

    short_lived = lineage_duration < int(config.min_duration)
    no_productive_fate = contribution_score <= off_thr
    non_growing = growth_score <= 0.05
    if no_productive_fate and (short_lived or trap_score >= 0.50 or non_growing):
        return "off-pathway"

    return "uncertain"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_lineages_from_cache(
    cache: ContactCache,
    *,
    inter_thresh: int,
    cluster_min_size: int = 2,
    overlap_min: int = 1,
    parent_frac_min: float = 0.30,
    child_frac_min: float = 0.30,
    union_cover_min: float = 0.70,
    target_config: Optional[TargetStateConfig] = None,
    species_map: Optional["SpeciesMap"] = None,
) -> LineageTrackingResult:
    """Build split/merge-aware aggregation lineages from cached contact maps.

    Parameters
    ----------
    cache:
        Stage A cache.
    inter_thresh:
        Contact threshold converting a contact map to graph edges.
    cluster_min_size:
        Minimum cluster size to track as an oligomer.  The default, 2, excludes
        monomers because size 1 is a free/bound monomer state rather than an
        oligomeric assembly.
    overlap_min, parent_frac_min, child_frac_min:
        Thresholds deciding whether two clusters across adjacent frames are
        meaningfully related.
    union_cover_min:
        Extra threshold required before declaring a clean split or merge.  This
        prevents noisy partial overlaps from being over-interpreted.
    target_config:
        User-defined productive aggregate state for fate classification.
    species_map:
        Optional composition map.  When given, every segment gains a per-frame
        species count and every lineage gains its overall and peak composition,
        so fate and composition can be read from one table.

    Returns
    -------
    LineageTrackingResult
        Frames, segments, split/merge events, lineages, the target definition,
        and any non-fatal warnings raised while building (see
        :func:`lineage_warnings`).

    Notes
    -----
    The default ``cluster_min_size=2`` tracks every transient dimer, which in a
    dense box produces far more lineages than there are chains -- most of them
    two frames long and meaningless.  That is the correct raw output, but it is
    not a result: filter it with :func:`filter_lineages` before drawing any
    conclusion.  The build warns when the count runs away.
    """
    if target_config is None:
        target_config = TargetStateConfig()

    raw_frames_clusters = [clusters_from_cmap(cmap, inter_thresh) for cmap in cache.cmaps]
    frames_clusters = [filter_clusters_by_size(frame, min_size=cluster_min_size) for frame in raw_frames_clusters]

    segment_counter = count(0)
    event_counter = count(0)
    segments_by_id: Dict[str, OligomerSegment] = {}
    events: List[LineageEvent] = []

    # Initial frame: every oligomer-sized connected component starts a segment.
    active_segments: Dict[int, str] = {}
    if frames_clusters:
        frame0 = int(cache.frames[0])
        for idx, cluster in enumerate(frames_clusters[0]):
            seg = _new_segment(segment_counter, cluster, 0, frame0, event_start="birth")
            segments_by_id[seg.segment_id] = seg
            active_segments[idx] = seg.segment_id

    for t in range(len(frames_clusters) - 1):
        prev_clusters = frames_clusters[t]
        curr_clusters = frames_clusters[t + 1]
        prev_frame = int(cache.frames[t])
        next_frame = int(cache.frames[t + 1])

        overlap = _overlap_matrix(prev_clusters, curr_clusters)
        parent_frac, child_frac = _fraction_matrices(overlap, prev_clusters, curr_clusters)

        parent_to_children: Dict[int, List[int]] = {i: [] for i in range(len(prev_clusters))}
        child_to_parents: Dict[int, List[int]] = {j: [] for j in range(len(curr_clusters))}

        for i, prev in enumerate(prev_clusters):
            for j, curr in enumerate(curr_clusters):
                ov = int(overlap[i, j])
                if _meaningful_overlap(ov, len(prev), len(curr), overlap_min, parent_frac_min, child_frac_min):
                    parent_to_children[i].append(j)
                    child_to_parents[j].append(i)

        next_active_segments: Dict[int, str] = {}
        continued_parents: set[int] = set()
        continued_children: set[int] = set()

        # True one-to-one relations are continuous segments.  The segment may
        # grow or shrink, but no branching event is introduced.
        for i in range(len(prev_clusters)):
            children = parent_to_children.get(i, [])
            if len(children) != 1:
                continue
            j = children[0]
            if len(child_to_parents.get(j, [])) != 1:
                continue
            seg_id = active_segments.get(i)
            if seg_id is None:
                continue
            _extend_segment(segments_by_id[seg_id], curr_clusters[j], t + 1, next_frame)
            next_active_segments[j] = seg_id
            continued_parents.add(i)
            continued_children.add(j)

        # Remove one-to-one continuations and classify the remaining overlap
        # graph as split/merge/rearrangement events.
        reduced_parent_to_children = {
            i: [j for j in js if i not in continued_parents and j not in continued_children]
            for i, js in parent_to_children.items()
        }
        reduced_child_to_parents = {
            j: [i for i in is_ if i not in continued_parents and j not in continued_children]
            for j, is_ in child_to_parents.items()
        }

        for parent_indices, child_indices in _bipartite_components(reduced_parent_to_children, reduced_child_to_parents):
            if not parent_indices and not child_indices:
                continue

            event_type = _classify_event_type(parent_indices, child_indices, prev_clusters, curr_clusters, union_cover_min)
            parent_segment_ids = [active_segments[i] for i in parent_indices if i in active_segments]
            child_segment_ids: List[str] = []

            for seg_id in parent_segment_ids:
                _close_segment(segments_by_id[seg_id], event_type)

            # Directed links follow the meaningful overlaps only.  The event
            # still records every parent and child of the component, but a
            # parent is not an ancestor of a child it shares no material with.
            for j in child_indices:
                own_parents = [active_segments[i] for i in child_to_parents.get(j, [])
                               if i in parent_indices and i in active_segments]
                seg = _new_segment(
                    segment_counter,
                    curr_clusters[j],
                    t + 1,
                    next_frame,
                    parent_ids=own_parents,
                    event_start=event_type,
                )
                segments_by_id[seg.segment_id] = seg
                child_segment_ids.append(seg.segment_id)
                next_active_segments[j] = seg.segment_id
                for parent_id in own_parents:
                    segments_by_id[parent_id].child_ids.append(seg.segment_id)

            local_overlap = overlap[np.ix_(parent_indices, child_indices)] if parent_indices and child_indices else np.zeros((len(parent_indices), len(child_indices)), dtype=int)
            local_parent_frac = parent_frac[np.ix_(parent_indices, child_indices)] if parent_indices and child_indices else np.zeros_like(local_overlap, dtype=float)
            local_child_frac = child_frac[np.ix_(parent_indices, child_indices)] if parent_indices and child_indices else np.zeros_like(local_overlap, dtype=float)
            description = (
                f"{event_type}: {len(parent_segment_ids)} parent segment(s) -> "
                f"{len(child_segment_ids)} child segment(s) between frames {prev_frame} and {next_frame}."
            )
            events.append(
                LineageEvent(
                    event_id=f"ev{next(event_counter)}",
                    event_type=event_type,
                    frame_idx=int(t),
                    prev_frame=prev_frame,
                    next_frame=next_frame,
                    parent_segment_ids=parent_segment_ids,
                    child_segment_ids=child_segment_ids,
                    parent_sets=[prev_clusters[i] for i in parent_indices],
                    child_sets=[curr_clusters[j] for j in child_indices],
                    overlap_matrix=local_overlap,
                    parent_fraction_matrix=local_parent_frac,
                    child_fraction_matrix=local_child_frac,
                    description=description,
                    notes="Split/merge events preserve lineage provenance; temporary fragmentation is not automatically off-pathway.",
                )
            )

        # Parents with no meaningful children die.
        for i in range(len(prev_clusters)):
            if i in continued_parents or parent_to_children.get(i):
                continue
            seg_id = active_segments.get(i)
            if seg_id is None:
                continue
            _close_segment(segments_by_id[seg_id], "death")
            events.append(
                LineageEvent(
                    event_id=f"ev{next(event_counter)}",
                    event_type="death",
                    frame_idx=int(t),
                    prev_frame=prev_frame,
                    next_frame=next_frame,
                    parent_segment_ids=[seg_id],
                    child_segment_ids=[],
                    parent_sets=[prev_clusters[i]],
                    child_sets=[],
                    overlap_matrix=np.zeros((1, 0), dtype=int),
                    description=f"death: segment {seg_id} has no meaningful oligomeric child at frame {next_frame}.",
                )
            )

        # Children with no meaningful parents are births.
        for j in range(len(curr_clusters)):
            if j in continued_children or child_to_parents.get(j):
                continue
            seg = _new_segment(segment_counter, curr_clusters[j], t + 1, next_frame, event_start="birth")
            segments_by_id[seg.segment_id] = seg
            next_active_segments[j] = seg.segment_id
            events.append(
                LineageEvent(
                    event_id=f"ev{next(event_counter)}",
                    event_type="birth",
                    frame_idx=int(t),
                    prev_frame=prev_frame,
                    next_frame=next_frame,
                    parent_segment_ids=[],
                    child_segment_ids=[seg.segment_id],
                    parent_sets=[],
                    child_sets=[curr_clusters[j]],
                    overlap_matrix=np.zeros((0, 1), dtype=int),
                    description=f"birth: new oligomeric segment {seg.segment_id} appears at frame {next_frame}.",
                )
            )

        active_segments = next_active_segments

    for seg_id in active_segments.values():
        if segments_by_id[seg_id].event_end == "active":
            _close_segment(segments_by_id[seg_id], "terminal")

    segments = sorted(segments_by_id.values(), key=lambda seg: (seg.start_idx, seg.segment_id))

    # Lineages are connected components of the segment graph induced by events.
    segment_ids = [seg.segment_id for seg in segments]
    event_pairs = [(p, c) for ev in events for p in ev.parent_segment_ids for c in ev.child_segment_ids]
    lineage_components = _connected_components_from_pairs(segment_ids, event_pairs)

    target_segments = [seg for seg in segments if seg.max_size >= target_config.min_size and seg.duration >= target_config.min_duration]

    lineages: List[Lineage] = []
    total_frames = max(1, len(cache.frames))
    event_by_id = {ev.event_id: ev for ev in events}

    for lin_idx, component in enumerate(lineage_components):
        lineage_id = f"lin{lin_idx}"
        component_segments = [segments_by_id[sid] for sid in component]
        for seg in component_segments:
            seg.lineage_id = lineage_id

        component_set = set(component)
        lineage_events = [ev for ev in events if set(ev.parent_segment_ids + ev.child_segment_ids) & component_set]
        event_ids = [ev.event_id for ev in lineage_events]
        event_types = [ev.event_type for ev in lineage_events]

        frames_union = sorted({fr for seg in component_segments for fr in seg.frames})
        lineage_duration = len(frames_union)
        lineage_start_frame = int(frames_union[0]) if frames_union else -1
        lineage_end_frame = int(frames_union[-1]) if frames_union else -1
        max_size = max((seg.max_size for seg in component_segments), default=0)
        first_seg = min(component_segments, key=lambda seg: (seg.start_idx, seg.segment_id), default=None)
        initial_size = int(first_seg.sizes[0]) if first_seg is not None and first_seg.sizes else 0

        target_reached = any(seg in target_segments for seg in component_segments)
        contribution_score = max((_segment_future_contribution(seg, target_segments) for seg in component_segments), default=0.0)
        if target_reached:
            contribution_score = max(1.0, contribution_score)

        if initial_size > 0:
            growth_score = float((max_size - initial_size) / initial_size)
        else:
            growth_score = 0.0

        persistence_score = float(lineage_duration / total_frames)
        reversibility_score = float(np.mean([_segment_reversibility_score(seg) for seg in component_segments])) if component_segments else 0.0
        representative = max(component_segments, key=lambda seg: (seg.duration, seg.max_size), default=None)
        representative_id = representative.segment_id if representative is not None else None
        structural_score = _lineage_structural_score(cache, representative, inter_thresh)

        n_deaths = sum(1 for x in event_types if x == "death")
        n_events = max(1, len(event_types))
        low_contribution = 1.0 - min(1.0, contribution_score)
        low_growth = 1.0 if growth_score <= 0 else max(0.0, 0.2 - min(0.2, growth_score)) / 0.2
        death_rate = n_deaths / n_events
        trap_score = float(np.clip(0.45 * low_contribution + 0.25 * low_growth + 0.20 * reversibility_score + 0.10 * death_rate, 0.0, 1.0))

        class_label = _classify_lineage(
            target_reached=bool(target_reached),
            contribution_score=float(contribution_score),
            growth_score=float(growth_score),
            trap_score=float(trap_score),
            lineage_duration=int(lineage_duration),
            config=target_config,
        )

        notes = (
            "Classification is fate-based. A temporary split is not off-pathway by itself; "
            "the lineage is judged by target-state contribution, growth, duration, and trap score. "
            "persistence_score is reported for ranking but is not a classification criterion."
        )
        lineages.append(
            Lineage(
                lineage_id=lineage_id,
                segment_ids=component,
                event_ids=event_ids,
                representative_segment_id=representative_id,
                class_label=class_label,
                target_reached=bool(target_reached),
                contribution_score=float(contribution_score),
                growth_score=float(growth_score),
                persistence_score=float(persistence_score),
                reversibility_score=float(reversibility_score),
                structural_score=float(structural_score),
                trap_score=float(trap_score),
                max_size=int(max_size),
                duration=int(lineage_duration),
                start_frame=int(lineage_start_frame),
                end_frame=int(lineage_end_frame),
                notes=notes,
            )
        )

    class_rank = {"on-pathway": 0, "uncertain": 1, "off-pathway": 2}
    lineages.sort(key=lambda lin: (class_rank.get(lin.class_label, 9), -lin.contribution_score, -lin.max_size, lin.lineage_id))

    result = LineageTrackingResult(
        frames_clusters=frames_clusters,
        segments=segments,
        events=events,
        lineages=lineages,
        target_config=target_config,
        species_map=species_map,
    )
    if species_map is not None:
        attach_species_compositions(result, species_map)
    result.warnings = lineage_warnings(result, nseg=int(np.asarray(cache.cmaps).shape[1]),
                                       cluster_min_size=int(cluster_min_size))
    return result


# ---------------------------------------------------------------------------
# Species composition
# ---------------------------------------------------------------------------

def attach_species_compositions(result: LineageTrackingResult, smap: "SpeciesMap") -> None:
    """Fill in per-segment and per-lineage species composition, in place.

    Segments store a ``(duration, n_species)`` count array parallel to their
    sizes.  Lineages store the distinct chains of each species that ever joined
    (``composition``), the composition of the largest state reached
    (``peak_composition``), and the fraction of their states that held more than
    one species (``hetero_fraction``).

    Splitting this out from the build lets an existing result be annotated after
    the fact, which matters because Stage B is cheap and Stage A is not.
    """
    seg_species = np.asarray(smap.seg_species, dtype=np.int64)
    n_species = int(smap.n_species)
    nseg = seg_species.size

    for seg in result.segments:
        comps = np.zeros((len(seg.sets), n_species), dtype=np.int32)
        for t, members in enumerate(seg.sets):
            if not members:
                continue
            idx = np.fromiter(members, dtype=np.int64, count=len(members))
            if idx.max(initial=-1) >= nseg:
                raise ValueError(
                    f"segment {seg.segment_id} references chain "
                    f"{int(idx.max())} but the species map covers {nseg} chains."
                )
            comps[t] = np.bincount(seg_species[idx], minlength=n_species)
        seg.compositions = comps

    seg_by_id = result.segment_dict
    for lin in result.lineages:
        members: set = set()
        n_states = 0
        n_mixed = 0
        best_size = -1
        peak = np.zeros(n_species, dtype=np.int32)
        for sid in lin.segment_ids:
            seg = seg_by_id[sid]
            for state, size, comp in zip(seg.sets, seg.sizes, seg.compositions):
                members.update(state)
                n_states += 1
                if int((comp > 0).sum()) > 1:
                    n_mixed += 1
                if size > best_size:
                    best_size = size
                    peak = comp
        idx = np.fromiter(members, dtype=np.int64, count=len(members))
        lin.species_names = list(smap.names)
        lin.composition = (
            np.bincount(seg_species[idx], minlength=n_species).astype(np.int32)
            if idx.size
            else np.zeros(n_species, dtype=np.int32)
        )
        lin.peak_composition = np.asarray(peak, dtype=np.int32)
        lin.hetero_fraction = float(n_mixed / n_states) if n_states else float("nan")


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def lineage_warnings(
    result: LineageTrackingResult,
    *,
    nseg: Optional[int] = None,
    cluster_min_size: int = 2,
) -> List[str]:
    """Non-fatal problems that make a lineage table easy to misread.

    Every check here corresponds to a way the raw output has actually been
    misinterpreted:

    * far more lineages than chains -- the tracker is counting transient dimers,
      not assemblies, and the unfiltered table cannot be summarised;
    * a target size no assembly ever reaches -- every lineage is then
      off-pathway or uncertain by construction, which is a statement about the
      threshold, not the physics;
    * a target size at or above the number of chains -- only a fully condensed
      box can qualify;
    * a table dominated by two-frame lineages -- the median duration says how
      much of it is noise.
    """
    messages: List[str] = []
    n_lin = len(result.lineages)
    if n_lin == 0:
        messages.append("no lineages were built; check inter_thresh and cluster_min_size.")
        return messages

    durations = np.array([lin.duration for lin in result.lineages], dtype=float)
    max_sizes = np.array([lin.max_size for lin in result.lineages], dtype=float)

    if nseg and n_lin > 2 * nseg:
        messages.append(
            f"{n_lin} lineages from {nseg} chains (cluster_min_size="
            f"{cluster_min_size}): most are transient sub-oligomers. Filter with "
            "filter_lineages(result, min_duration=...) before summarising -- raw "
            "counts here are not a result."
        )

    target_size = int(result.target_config.min_size)
    if nseg and target_size >= nseg:
        messages.append(
            f"target min_size={target_size} is at or above the {nseg} chains in "
            "the box, so only complete condensation can reach it."
        )
    elif max_sizes.max(initial=0) < target_size:
        messages.append(
            f"no lineage reaches the target size {target_size} (largest is "
            f"{int(max_sizes.max(initial=0))}); every fate label below is a "
            "statement about the threshold, not about the trajectory."
        )

    median_duration = float(np.median(durations))
    if median_duration <= 2 and n_lin > 10:
        frac = float((durations <= 2).mean())
        messages.append(
            f"{100.0 * frac:.0f}% of lineages last two frames or fewer "
            f"(median {median_duration:.0f}); treat the table as raw output."
        )
    return messages


def filter_lineages(
    result: LineageTrackingResult,
    *,
    min_duration: int = 1,
    min_max_size: int = 0,
    class_labels: Optional[Sequence[str]] = None,
) -> LineageTrackingResult:
    """Return a result holding only the lineages that pass a persistence filter.

    A raw lineage table from a dense box is dominated by two-frame dimers.  The
    fix that has actually worked is a duration floor: keeping only lineages that
    persist for ``min_duration`` frames leaves the assemblies worth discussing,
    and conclusions should be checked to be stable across a range of floors
    rather than quoted at one.

    Segments and events are filtered to those belonging to the surviving
    lineages, so the returned object is self-consistent and can be passed to any
    downstream table or plot.  ``frames_clusters`` is shared with the original --
    it describes the trajectory, not the selection.
    """
    keep = [
        lin
        for lin in result.lineages
        if lin.duration >= int(min_duration)
        and lin.max_size >= int(min_max_size)
        and (class_labels is None or lin.class_label in set(class_labels))
    ]
    keep_segments = {sid for lin in keep for sid in lin.segment_ids}
    keep_events = {eid for lin in keep for eid in lin.event_ids}
    return LineageTrackingResult(
        frames_clusters=result.frames_clusters,
        segments=[seg for seg in result.segments if seg.segment_id in keep_segments],
        events=[ev for ev in result.events if ev.event_id in keep_events],
        lineages=keep,
        target_config=result.target_config,
        species_map=result.species_map,
        warnings=list(result.warnings),
    )


def lineage_summary_dataframe(result: LineageTrackingResult):
    """Return a tidy table of lineage fate scores and classifications."""
    import pandas as pd

    rows = []
    for lin in result.lineages:
        row = {
            "lineage_id": lin.lineage_id,
            "classification": lin.classification,
            "class_label": lin.class_label,
            "target_reached": lin.target_reached,
            "n_segments": len(lin.segment_ids),
            "n_events": len(lin.event_ids),
            "max_size": lin.max_size,
            "duration": lin.duration,
            "start_frame": lin.start_frame,
            "end_frame": lin.end_frame,
            "persistence_score": lin.persistence_score,
            "growth_score": lin.growth_score,
            "target_contribution_score": lin.contribution_score,
            "contribution_score": lin.contribution_score,
            "trap_score": lin.trap_score,
            "reversibility_score": lin.reversibility_score,
            "structural_score": lin.structural_score,
            "representative_segment_id": lin.representative_segment_id,
        }
        # Composition columns appear only when the lineages were built with a
        # species map, so a one-component table keeps its original shape.
        if lin.species_names and lin.composition is not None:
            for k, name in enumerate(lin.species_names):
                row[f"n_{name}"] = int(lin.composition[k])
                row[f"peak_{name}"] = int(
                    lin.peak_composition[k] if lin.peak_composition is not None else 0
                )
            row["n_species_present"] = int((lin.composition > 0).sum())
            row["is_hetero"] = bool((lin.composition > 0).sum() > 1)
            row["hetero_fraction"] = float(lin.hetero_fraction)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    class_order = {"on-pathway": 0, "uncertain": 1, "off-pathway": 2}
    df = pd.DataFrame(rows)
    df["_class_order"] = df["class_label"].map(class_order).fillna(9)
    df = df.sort_values(["_class_order", "target_contribution_score", "max_size", "duration"], ascending=[True, False, False, False])
    return df.drop(columns=["_class_order"])


def target_size_sweep(
    cache: ContactCache,
    sizes: Sequence[int],
    *,
    inter_thresh: int = 1,
    cluster_min_size: int = 2,
    min_duration: int = 1,
    target_config: Optional[TargetStateConfig] = None,
    species_map: Optional["SpeciesMap"] = None,
    **build_kwargs,
):
    """Re-classify the same trajectory against several target sizes.

    On/off-pathway labels are defined relative to a target aggregate size, and
    that size is a choice.  A conclusion that only holds at one value of it is a
    property of the threshold; this sweep is how that gets checked rather than
    assumed.  Lineages are rebuilt once per size and summarised into one row
    each, so the trend -- and its stability -- is visible in a single table.

    Returns a DataFrame with one row per target size: the on/off/uncertain
    counts, the on-pathway fraction, and the largest assembly observed (which
    does not depend on the target and so should be constant down the column).
    """
    import pandas as pd

    base = target_config or TargetStateConfig()
    rows = []
    for size in sizes:
        cfg = TargetStateConfig(
            min_size=int(size),
            min_duration=base.min_duration,
            min_contribution=base.min_contribution,
            off_path_contribution_threshold=base.off_path_contribution_threshold,
        )
        result = build_lineages_from_cache(
            cache,
            inter_thresh=inter_thresh,
            cluster_min_size=cluster_min_size,
            target_config=cfg,
            species_map=species_map,
            **build_kwargs,
        )
        if min_duration > 1:
            result = filter_lineages(result, min_duration=min_duration)
        labels = [lin.class_label for lin in result.lineages]
        n_total = len(labels)
        n_on = sum(1 for x in labels if x == "on-pathway")
        n_off = sum(1 for x in labels if x == "off-pathway")
        rows.append(
            {
                "target_size": int(size),
                "n_lineages": n_total,
                "n_on_pathway": n_on,
                "n_off_pathway": n_off,
                "n_uncertain": n_total - n_on - n_off,
                "on_fraction": (n_on / n_total if n_total else float("nan")),
                "max_size_observed": max((lin.max_size for lin in result.lineages), default=0),
                "min_duration_filter": int(min_duration),
            }
        )
    return pd.DataFrame(rows)


def sweep_verdict(sweep) -> str:
    """One-line reading of a :func:`target_size_sweep` table.

    Reports whether the on-pathway fraction keeps the same ordering across the
    swept target sizes.  A conclusion that survives this is about the
    trajectory; one that does not is about the threshold.
    """
    if len(sweep) < 2:
        return "sweep needs at least two target sizes to say anything."
    frac = sweep["on_fraction"].to_numpy(dtype=float)
    spread = float(np.nanmax(frac) - np.nanmin(frac))
    monotone = bool(np.all(np.diff(frac) <= 1e-12) or np.all(np.diff(frac) >= -1e-12))
    sizes = sweep["target_size"].tolist()
    head = (
        f"on-fraction over target sizes {sizes}: "
        f"{np.nanmin(frac):.2f}-{np.nanmax(frac):.2f} (spread {spread:.2f})"
    )
    if "max_size_observed" in sweep.columns:
        largest = int(np.nanmax(sweep["max_size_observed"].to_numpy(dtype=float)))
        if largest < min(sizes):
            return (
                head + f" -- uninformative: no lineage reaches even the smallest "
                f"target (largest observed {largest}); lower the targets or extend the run."
            )
    if spread <= 0.10:
        return head + " -- stable; the conclusion does not depend on the target size."
    if monotone:
        return head + " -- varies monotonically; quote the trend, not one value."
    return head + " -- non-monotonic and wide; the target size is driving the result."


def representative_pathways_for_lineages(result: LineageTrackingResult) -> List[Pathway]:
    """Return one Pathway-like representative per lineage.

    This is the bridge from the lineage-aware analysis to the old monitor.  Each
    lineage is represented by its most persistent / largest segment.
    """
    seg_by_id = result.segment_dict
    pathways: List[Pathway] = []
    for lin in result.lineages:
        rep = seg_by_id.get(lin.representative_segment_id or "")
        if rep is None:
            continue
        pathways.append(_segment_as_pathway(rep))
    return pathways


# ---------------------------------------------------------------------------
# Directed fates (3.5.0)
# ---------------------------------------------------------------------------

def directed_fates(result: LineageTrackingResult) -> "pd.DataFrame":
    """Fate of every nucleation event, following time forward only.

    A *lineage* groups every segment ever joined by any split or merge, in both
    directions, so a fragment shed from a large aggregate, or a cluster that only
    merges into one much later, shares that aggregate's lineage and its
    ``on-pathway`` label.  This table asks a narrower question of each birth (a
    segment with no parent): is a target state reachable *forward* from it,
    through its own descendants?

    ``fate`` is ``on-pathway`` when a target segment is reachable forward,
    ``uncertain`` when some descendant is still alive in the last frame without
    a target having been reached, and ``off-pathway`` otherwise.
    """
    import pandas as pd

    segs = sorted(result.segments, key=lambda s: (s.start_idx, s.segment_id))
    cfg = result.target_config
    last_idx = max((s.end_idx for s in segs), default=-1)
    names = list(result.species_map.names) if result.species_map is not None else []

    first_target: Dict[str, float] = {}
    fmax: Dict[str, int] = {}
    fend: Dict[str, int] = {}
    for seg in reversed(segs):
        is_target = seg.max_size >= cfg.min_size and seg.duration >= cfg.min_duration
        ft = float(seg.start_idx) if is_target else float("inf")
        mx, end = int(seg.max_size), int(seg.end_idx)
        for cid in seg.child_ids:
            if cid in first_target:
                ft = min(ft, first_target[cid])
                mx = max(mx, fmax[cid])
                end = max(end, fend[cid])
        first_target[seg.segment_id], fmax[seg.segment_id], fend[seg.segment_id] = ft, mx, end

    rows = []
    for seg in segs:
        if seg.parent_ids:
            continue
        ft = first_target[seg.segment_id]
        reaches = np.isfinite(ft)
        alive = fend[seg.segment_id] >= last_idx
        fate = "on-pathway" if reaches else ("uncertain" if alive else "off-pathway")
        row = {
            "root_id": seg.segment_id,
            "lineage_id": seg.lineage_id,
            "start_frame": seg.start_frame,
            "start_idx": int(seg.start_idx),
            "born_in_first_frame": int(seg.start_idx) == 0,
            "initial_size": int(seg.sizes[0]) if seg.sizes else 0,
            "reaches_target": bool(reaches),
            "frames_to_target": (int(ft) - int(seg.start_idx)) if reaches else None,
            "forward_max_size": int(fmax[seg.segment_id]),
            "forward_end_idx": int(fend[seg.segment_id]),
            "forward_span": int(fend[seg.segment_id]) - int(seg.start_idx) + 1,
            "alive_at_end": bool(alive),
            "fate": fate,
        }
        if seg.compositions is not None and names:
            comp0 = np.asarray(seg.compositions[0])
            for i, name in enumerate(names):
                row[f"n_{name}_at_birth"] = int(comp0[i])
            row["is_hetero_at_birth"] = int((comp0 > 0).sum()) > 1
        rows.append(row)
    return pd.DataFrame(rows)


def directed_fate_summary(df: "pd.DataFrame", *, min_span: int = 1) -> "pd.DataFrame":
    """On/off/uncertain counts of nucleation events, optionally split by composition.

    ``min_span`` drops births whose whole forward subtree lasts fewer frames,
    the directed counterpart of the ``duration >= 50`` persistence filter.
    """
    import pandas as pd

    if df.empty:
        return pd.DataFrame(columns=["group", "n", "on", "off", "uncertain", "on_fraction"])
    d = df[df["forward_span"] >= int(min_span)]
    groups = [("all", d)]
    if "is_hetero_at_birth" in d.columns:
        groups += [("mixed at birth", d[d["is_hetero_at_birth"]]),
                   ("single-species at birth", d[~d["is_hetero_at_birth"].astype(bool)])]
    rows = []
    for name, sub in groups:
        n = len(sub)
        on = int((sub["fate"] == "on-pathway").sum())
        off = int((sub["fate"] == "off-pathway").sum())
        unc = int((sub["fate"] == "uncertain").sum())
        rows.append({"group": name, "n": n, "on": on, "off": off, "uncertain": unc,
                     "on_fraction": on / n if n else float("nan")})
    return pd.DataFrame(rows)
