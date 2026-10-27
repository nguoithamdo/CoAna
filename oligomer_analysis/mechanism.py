# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Does the second component join the aggregate, or catalyse it?

When a minority species speeds up the aggregation of a majority species, two
mechanisms look identical in every bulk observable:

**Incorporation.**  The guest becomes part of the product.  It is present when
the assembly commits to growing, and it is still there at the end.  The result
is a co-aggregate.

**Catalysis.**  The guest lowers the barrier to nucleation and then leaves.  It
is present before and at commitment, departs, and the assembly carries on
without it.  This is what heterogeneous primary nucleation means at the level of
a single chain, and it is the claim that kinetic experiments can support but not
demonstrate.

Telling them apart needs the identity of every guest chain, the assembly it is
in at every frame, and when that assembly committed -- which is exactly what the
lineage layer plus a species map provide.  This module turns those into per-visit
records and classifies each one.

Two further outcomes fall out of the same records and are worth naming, because
both are routinely mistaken for the first two:

**Passenger.**  The guest joins an assembly that had already committed.  It is
recruited *by* the aggregate rather than contributing to its formation -- the
same end state as incorporation, a different mechanism.

**Bystander.**  The guest touches assemblies briefly, nothing grows, nothing
commits.  In a system where most encounters are of this kind, an enrichment in
bulk binding says very little.

Every classification is a transparent rule over reported numbers, so a visit can
always be re-classified under a different definition without recomputing
anything.

Whose commitment counts
-----------------------
Since 3.5.0 every rule is judged on the guest's **own assembly**, followed through
time by chain identity (``mode="branch"``, the default).  An assembly's successor
in the next frame is the cluster that holds most of its host chains, and its
predecessor is found the same way backwards.  An assembly *commits* when it holds
at least ``commit_size`` host chains for ``commit_duration`` consecutive frames
along that track; the commit frame is the first frame of that run.

Versions up to 3.4.0 judged commitment, survival and post-departure growth on the
whole lineage instead (kept as ``mode="lineage"``).  A lineage is every segment
ever linked by a split or merge, including clusters that only merge in later, so a
guest sitting on a small assembly that never grew could be labelled ``passenger``
or ``catalytic`` because a different branch had committed, or because its own
cluster merged into that branch after the guest had left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import LineageTrackingResult

__all__ = [
    "ResidenceEvent",
    "residence_events",
    "residence_dataframe",
    "mechanism_summary",
    "exchange_statistics",
]


@dataclass(slots=True)
class ResidenceEvent:
    """One continuous visit of a guest chain to one assembly lineage."""

    chain: int
    species: str
    lineage_id: str
    start_idx: int
    end_idx: int
    start_frame: int
    end_frame: int
    duration: int

    host_at_entry: int
    host_at_exit: int
    host_max_during: int
    host_growth_during: int
    host_growth_after: float

    committed: bool
    commit_idx: Optional[int]
    present_at_commit: bool
    joined_after_commit: bool
    survives_to_lineage_end: bool
    lineage_end_idx: int

    classification: str
    reason: str
    commit_frame: Optional[int] = None
    # Did the guest's own assembly commit only AFTER the guest left? Data, not a
    # label: one such visit is a temporal coincidence, and whether guests raise
    # the chance of a later commitment is an ensemble question, to be answered
    # against pseudo-guests (random host chains). None when the visit is already
    # credited with a commitment, or the assembly never commits on its own line.
    later_commit_idx: Optional[int] = None
    later_commit_frame: Optional[int] = None
    departure_to_commit_lag: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "chain": self.chain,
            "species": self.species,
            "lineage_id": self.lineage_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration": self.duration,
            "host_at_entry": self.host_at_entry,
            "host_at_exit": self.host_at_exit,
            "host_max_during": self.host_max_during,
            "host_growth_during": self.host_growth_during,
            "host_growth_after": self.host_growth_after,
            "committed": self.committed,
            "commit_idx": self.commit_idx,
            "commit_frame": self.commit_frame,
            "present_at_commit": self.present_at_commit,
            "joined_after_commit": self.joined_after_commit,
            "survives_to_end": self.survives_to_lineage_end,
            "classification": self.classification,
            "reason": self.reason,
            "later_commit_idx": self.later_commit_idx,
            "later_commit_frame": self.later_commit_frame,
            "departure_to_commit_lag": self.departure_to_commit_lag,
        }


def _lineage_frame_tables(
    result: LineageTrackingResult,
    host_index: int,
) -> Tuple[Dict[int, Dict[int, str]], Dict[str, Dict[int, int]], Dict[str, Dict[int, int]]]:
    """Per-frame membership and host content, keyed by lineage.

    Returns ``(lineage_of, seg_host_of_chain, lineage_host)`` where

    * ``lineage_of[frame_idx][chain]`` is the lineage a chain belongs to;
    * ``seg_host_of_chain[frame_idx][chain]`` is the host-species count of the
      chain's own connected assembly at that frame;
    * ``lineage_host[lineage_id][frame_idx]`` is the host count of the lineage's
      *largest* concurrent assembly, which is the branch that carries the
      lineage after a split.
    """
    lineage_of: Dict[int, Dict[int, str]] = {}
    seg_host: Dict[int, Dict[int, int]] = {}
    lineage_host: Dict[str, Dict[int, int]] = {}

    for seg in result.segments:
        if seg.lineage_id is None:
            continue
        if seg.compositions is None:
            raise ValueError(
                "segments carry no species composition; build the lineages with "
                "build_lineages_from_cache(..., species_map=smap)."
            )
        for t, members in enumerate(seg.sets):
            frame_idx = int(seg.start_idx) + t
            host_here = int(seg.compositions[t][host_index])
            frame_map = lineage_of.setdefault(frame_idx, {})
            host_map = seg_host.setdefault(frame_idx, {})
            for chain in members:
                frame_map[int(chain)] = seg.lineage_id
                host_map[int(chain)] = host_here
            per_lin = lineage_host.setdefault(seg.lineage_id, {})
            if host_here > per_lin.get(frame_idx, -1):
                per_lin[frame_idx] = host_here
    return lineage_of, seg_host, lineage_host


def _commit_index(host_series: Dict[int, int], commit_size: int) -> Optional[int]:
    """First frame index at which a lineage's host content reaches the target."""
    hits = [f for f, h in host_series.items() if h >= commit_size]
    return min(hits) if hits else None


# ---------------------------------------------------------------------------
# Branch-local tracks (mode="branch")
# ---------------------------------------------------------------------------


class _Tracks:
    """Every assembly in every frame, linked to its own past and future.

    Built once per result from ``frames_clusters``, keeping only the clusters
    that belong to one of the result's segments -- so a result passed through
    :func:`filter_lineages` has the filtered-out assemblies removed here too,
    not just from the output table.  Links follow host chains, never lineage
    membership, so an assembly's history is that of the material it is actually
    made of.

    Commitment state is per assembly and follows its own main-predecessor chain:
    ``back_len[t][k]`` is the number of consecutive frames, ending at ``t``,
    that the chain has held ``commit_size`` host chains, and ``ever_commit[t][k]``
    the first frame of the earliest such run that lasted ``commit_duration``
    frames.  Nothing is shared between siblings of a split, and a commitment is
    never forgotten when the assembly later dips below the threshold.
    """

    def __init__(self, result: LineageTrackingResult, host_index: int,
                 commit_size: int, commit_duration: int) -> None:
        smap = result.species_map
        is_host = np.asarray(smap.seg_species) == host_index
        self.commit_size = int(commit_size)
        self.commit_duration = int(commit_duration)
        allowed: Dict[int, set] = {}
        for seg in result.segments:
            for t, members in enumerate(seg.sets):
                allowed.setdefault(int(seg.start_idx) + t, set()).add(
                    frozenset(int(c) for c in members))
        self.clusters: List[List[frozenset]] = []
        for t, fr in enumerate(result.frames_clusters):
            keep = allowed.get(t, set())
            self.clusters.append([cl for cl in (frozenset(int(c) for c in x) for x in fr)
                                  if cl in keep])
        self.n_frames = len(self.clusters)
        self.where: List[Dict[int, int]] = []
        self.hosts: List[List[frozenset]] = []
        for fr in self.clusters:
            m: Dict[int, int] = {}
            hs: List[frozenset] = []
            for k, cl in enumerate(fr):
                for c in cl:
                    m[int(c)] = k
                hs.append(frozenset(int(c) for c in cl if is_host[int(c)]))
            self.where.append(m)
            self.hosts.append(hs)

        self.succ: List[List[Optional[int]]] = [
            [self._link(t, k, t + 1) for k in range(len(self.clusters[t]))]
            for t in range(self.n_frames)
        ]
        self.pred: List[List[Optional[int]]] = [
            [self._link(t, k, t - 1) for k in range(len(self.clusters[t]))]
            for t in range(self.n_frames)
        ]

        # Commitment along each assembly's own main-predecessor chain.
        self.back_len: List[List[int]] = []
        self.ever_commit: List[List[Optional[int]]] = []
        for t in range(self.n_frames):
            lens: List[int] = []
            ever: List[Optional[int]] = []
            for k in range(len(self.clusters[t])):
                p = self.pred[t][k] if t > 0 else None
                prev_len = self.back_len[t - 1][p] if p is not None else 0
                prev_ever = self.ever_commit[t - 1][p] if p is not None else None
                n = prev_len + 1 if self.qualifies(t, k) else 0
                if prev_ever is None and n >= self.commit_duration:
                    prev_ever = t - n + 1
                lens.append(n)
                ever.append(prev_ever)
            self.back_len.append(lens)
            self.ever_commit.append(ever)

        # Last frame each assembly's successor chain reaches.
        self.track_end: List[List[int]] = [[] for _ in range(self.n_frames)]
        for t in range(self.n_frames - 1, -1, -1):
            row = []
            for k in range(len(self.clusters[t])):
                s = self.succ[t][k]
                row.append(self.track_end[t + 1][s] if s is not None else t)
            self.track_end[t] = row

    def _link(self, t: int, k: int, t2: int) -> Optional[int]:
        """The cluster at ``t2`` holding most of cluster ``(t, k)``'s material."""
        if t2 < 0 or t2 >= self.n_frames:
            return None
        key = self.hosts[t][k] or self.clusters[t][k]
        best, best_n, best_size = None, 0, -1
        for k2, cl in enumerate(self.clusters[t2]):
            n = len(key & cl)
            if n > best_n or (n == best_n and n > 0 and len(cl) > best_size):
                best, best_n, best_size = k2, n, len(cl)
        return best

    def host(self, t: int, k: int) -> int:
        return len(self.hosts[t][k])

    def qualifies(self, t: int, k: int) -> bool:
        return len(self.hosts[t][k]) >= self.commit_size

    def ancestor(self, t: int, k: int, t0: int) -> Optional[int]:
        """The cluster at frame ``t0 <= t`` on (t, k)'s main-predecessor chain."""
        cur = k
        for tt in range(t, t0, -1):
            cur = self.pred[tt][cur]
            if cur is None:
                return None
        return cur

    def visit_commit(self, a: int, ks: Sequence[int]) -> Optional[int]:
        """Commit frame of the assemblies a guest occupied from frame ``a``.

        Each assembly the guest is in during the visit is judged on its own
        history (its main-predecessor chain), so riding a small cluster into an
        assembly that committed long ago is joining a committed assembly.  If
        none of them has committed by the guest's departure, a run of
        ``commit_size`` host chains still going at departure is followed along
        the departed assembly's successor chain, and counts if it lasts
        ``commit_duration`` frames.  The commit frame is the first frame of the
        run.  ``None`` when there is none.
        """
        b = a + len(ks) - 1
        for t, k in zip(range(a, b + 1), ks):
            if self.ever_commit[t][k] is not None:
                return self.ever_commit[t][k]
        t, k = b, ks[-1]
        run = self.back_len[t][k]
        while run > 0:
            nxt = self.succ[t][k]
            if nxt is None:
                break
            t, k = t + 1, nxt
            if not self.qualifies(t, k):
                break
            run += 1
            if run >= self.commit_duration:
                return t - run + 1
        return None

    def later_commit(self, b: int, k: int) -> Optional[int]:
        """First commitment of the departed assembly ``(b, k)`` that starts after ``b``.

        Walks forward along the successor chain only while the assembly keeps its
        identity -- while it is the main predecessor of its successor. Once it is
        absorbed into an assembly whose main line is something else, the walk
        stops: that assembly's commitment is not this one's, which is the merge
        case 3.5.0's branch rule exists to exclude. Returns the first frame of the
        committing run, or ``None``.
        """
        t = b
        while True:
            e = self.ever_commit[t][k]
            if e is not None:
                return e if e > b else None
            s = self.succ[t][k]
            if s is None or t + 1 >= self.n_frames:
                return None
            if self.pred[t + 1][s] != k:
                return None
            t, k = t + 1, s

    def follow(self, t: int, k: int, steps: int) -> Tuple[Optional[int], Optional[int]]:
        """Walk the successor chain; return the last (frame, cluster) reached."""
        last_t, last_k = None, None
        cur_t, cur_k = t, k
        for _ in range(int(steps)):
            nxt = self.succ[cur_t][cur_k]
            if nxt is None:
                break
            cur_t, cur_k = cur_t + 1, nxt
            last_t, last_k = cur_t, cur_k
        return last_t, last_k


def _branch_visits(tracks: _Tracks, chain: int) -> List[Tuple[int, int]]:
    """Continuous stays of one chain in assemblies.

    A stay continues while the chain is in an assembly in consecutive frames and
    that assembly shares at least one *host* chain with the previous frame's, so
    hopping straight from one assembly to an unrelated one starts a new visit --
    also when another guest makes the same hop.  When either assembly holds no
    host chain (a guest-only cluster, or one that gains its first host), any
    other shared chain continues the stay.
    """
    visits: List[Tuple[int, int]] = []
    start: Optional[int] = None
    prev_k: Optional[int] = None
    for t in range(tracks.n_frames):
        k = tracks.where[t].get(chain)
        if k is None:
            if start is not None:
                visits.append((start, t - 1))
            start, prev_k = None, None
            continue
        if start is not None:
            h_now, h_prev = tracks.hosts[t][k], tracks.hosts[t - 1][prev_k]
            if h_now and h_prev:
                shared = h_now & h_prev
            else:
                shared = (tracks.clusters[t][k] & tracks.clusters[t - 1][prev_k]) - {chain}
            if not shared:
                visits.append((start, t - 1))
                start = t
        else:
            start = t
        prev_k = k
    if start is not None:
        visits.append((start, tracks.n_frames - 1))
    return visits


def residence_events(
    result: LineageTrackingResult,
    *,
    guest_species: str,
    host_species: Optional[str] = None,
    commit_size: Optional[int] = None,
    after_window: int = 10,
    growth_min: int = 1,
    end_tolerance: int = 1,
    frames: Optional[Sequence[int]] = None,
    commit_duration: Optional[int] = None,
    mode: str = "branch",
) -> List[ResidenceEvent]:
    """Per-visit records for every chain of ``guest_species``.

    Parameters
    ----------
    guest_species:
        The species whose mechanism is in question -- the minority component in
        a "does X nucleate Y?" experiment.  Passing the majority species instead
        is *not* a fair control: its chains are then judged against an assembly
        of the minority species.  A fair control designates a random subset of
        host chains as pseudo-guests and keeps the rest as host, so both are
        judged against the same assembly.
    host_species:
        The species whose assembly is being counted.  Defaults to the most
        abundant species that is not the guest.
    commit_size:
        Host chains an assembly must reach to count as committed.  Defaults to
        the target size the lineages were built with, so the definition of
        "productive" stays the same one used for fate classification.
    after_window:
        How many frames past a departure to look when asking whether the
        assembly kept growing without the guest.  Growth measured over this
        window is what separates catalysis from a coincidental exit.
    growth_min:
        Host chains an assembly must gain during a visit before that visit can
        be called promoting.
    end_tolerance:
        A visit ending within this many frames of the end of its assembly (or of
        the trajectory) counts as surviving to the end, which absorbs the
        single-frame flicker that contact-based clustering produces at an
        assembly's surface.
    commit_duration:
        Consecutive frames an assembly must hold ``commit_size`` host chains to
        count as committed (``mode="branch"`` only).  Defaults to the target
        duration the lineages were built with, so a one-frame touch of the
        threshold is not a commitment.
    mode:
        ``"branch"`` (default) judges every rule on the guest's own assembly,
        tracked by chain identity.  ``"lineage"`` reproduces the 3.4.0 rules,
        which judged commitment on the whole split/merge lineage; it is kept
        only so old tables can be compared with new ones.
    """
    if mode not in ("branch", "lineage"):
        raise ValueError(f"mode must be 'branch' or 'lineage', not {mode!r}")
    smap = result.species_map
    if smap is None:
        raise ValueError(
            "this result has no species map; rebuild with "
            "build_lineages_from_cache(..., species_map=smap)."
        )
    guest_index = smap.index(guest_species)
    if host_species is None:
        counts = smap.counts()
        order = np.argsort(-counts)
        host_candidates = [int(k) for k in order if int(k) != guest_index]
        if not host_candidates:
            raise ValueError(
                "only one species is present, so there is no host to aggregate; "
                "a mechanism analysis needs at least two."
            )
        host_index = host_candidates[0]
    else:
        host_index = smap.index(host_species)
    if host_index == guest_index:
        raise ValueError("guest_species and host_species must differ.")

    if commit_size is None:
        commit_size = int(result.target_config.min_size)
    if commit_duration is None:
        commit_duration = int(result.target_config.min_duration)

    lineage_of, seg_host, lineage_host = _lineage_frame_tables(result, host_index)
    if not lineage_of:
        return []

    if mode == "branch":
        return _branch_residence_events(
            result,
            lineage_of=lineage_of,
            guest_index=guest_index,
            host_index=host_index,
            commit_size=int(commit_size),
            commit_duration=int(commit_duration),
            after_window=after_window,
            growth_min=growth_min,
            end_tolerance=end_tolerance,
            frames=frames,
        )

    all_frames = sorted(lineage_of)
    lineage_last: Dict[str, int] = {}
    for lin_id, series in lineage_host.items():
        lineage_last[lin_id] = max(series)
    commit_idx: Dict[str, Optional[int]] = {
        lin_id: _commit_index(series, commit_size)
        for lin_id, series in lineage_host.items()
    }

    frame_numbers = list(frames) if frames is not None else None

    def frame_number(idx: int) -> int:
        if frame_numbers is None:
            return int(idx)
        return int(frame_numbers[idx]) if 0 <= idx < len(frame_numbers) else int(idx)

    guest_chains = np.flatnonzero(smap.seg_species == guest_index)
    events: List[ResidenceEvent] = []

    for chain in guest_chains:
        chain = int(chain)
        # Walk the frame axis and cut a new visit whenever the lineage changes
        # (including to or from "not in any tracked assembly").
        current: Optional[str] = None
        start: Optional[int] = None
        previous_idx: Optional[int] = None

        def close(end_idx: int) -> None:
            if current is None or start is None:
                return
            events.append(
                _build_event(
                    chain=chain,
                    species=smap.names[guest_index],
                    lineage_id=current,
                    start_idx=start,
                    end_idx=end_idx,
                    seg_host=seg_host,
                    lineage_host=lineage_host,
                    lineage_last=lineage_last,
                    commit_idx=commit_idx,
                    commit_size=commit_size,
                    after_window=after_window,
                    growth_min=growth_min,
                    end_tolerance=end_tolerance,
                    frame_number=frame_number,
                )
            )

        for idx in all_frames:
            lin_here = lineage_of.get(idx, {}).get(chain)
            contiguous = previous_idx is not None and idx == previous_idx + 1
            if lin_here != current or not contiguous:
                close(previous_idx if previous_idx is not None else idx)
                current = lin_here
                start = idx if lin_here is not None else None
            previous_idx = idx
        close(previous_idx if previous_idx is not None else 0)

    events.sort(key=lambda e: (e.chain, e.start_idx))
    return events


def _branch_residence_events(
    result: LineageTrackingResult,
    *,
    lineage_of: Dict[int, Dict[int, str]],
    guest_index: int,
    host_index: int,
    commit_size: int,
    commit_duration: int,
    after_window: int,
    growth_min: int,
    end_tolerance: int,
    frames: Optional[Sequence[int]],
) -> List[ResidenceEvent]:
    """``residence_events`` with every rule judged on the guest's own assembly."""
    smap = result.species_map
    tracks = _Tracks(result, host_index, commit_size, commit_duration)
    frame_numbers = list(frames) if frames is not None else None

    def frame_number(idx: int) -> int:
        if frame_numbers is None:
            return int(idx)
        return int(frame_numbers[idx]) if 0 <= idx < len(frame_numbers) else int(idx)

    last_frame = tracks.n_frames - 1
    events: List[ResidenceEvent] = []
    for chain in np.flatnonzero(np.asarray(smap.seg_species) == guest_index):
        chain = int(chain)
        for a, b in _branch_visits(tracks, chain):
            ks = [tracks.where[t][chain] for t in range(a, b + 1)]
            during = [tracks.host(t, k) for t, k in zip(range(a, b + 1), ks)]
            host_entry, host_exit, host_max = during[0], during[-1], max(during)

            commit = tracks.visit_commit(a, ks)
            committed = commit is not None
            later = None if committed else tracks.later_commit(b, ks[-1])
            present_at_commit = bool(committed and commit >= a)
            joined_after_commit = bool(committed and commit < a)

            k_exit = ks[-1]
            track_end = int(tracks.track_end[b][k_exit])
            survives = bool(b >= track_end - int(end_tolerance)
                            or b >= last_frame - int(end_tolerance))
            t_after, k_after = tracks.follow(b, k_exit, int(after_window))
            growth_after = (
                float(tracks.host(t_after, k_after) - host_exit)
                if t_after is not None else float("nan")
            )
            growth_during = host_max - host_entry
            label, reason = _classify_residence(
                committed=committed,
                present_at_commit=present_at_commit,
                joined_after_commit=joined_after_commit,
                survives=survives,
                growth_during=growth_during,
                growth_after=growth_after,
                growth_min=growth_min,
                commit_size=commit_size,
            )
            events.append(
                ResidenceEvent(
                    chain=chain,
                    species=smap.names[guest_index],
                    lineage_id=lineage_of.get(a, {}).get(chain, ""),
                    start_idx=int(a),
                    end_idx=int(b),
                    start_frame=frame_number(a),
                    end_frame=frame_number(b),
                    duration=int(b - a + 1),
                    host_at_entry=int(host_entry),
                    host_at_exit=int(host_exit),
                    host_max_during=int(host_max),
                    host_growth_during=int(growth_during),
                    host_growth_after=growth_after,
                    committed=committed,
                    commit_idx=(int(commit) if commit is not None else None),
                    present_at_commit=present_at_commit,
                    joined_after_commit=joined_after_commit,
                    survives_to_lineage_end=survives,
                    lineage_end_idx=track_end,
                    later_commit_idx=later,
                    later_commit_frame=(frame_number(later) if later is not None else None),
                    departure_to_commit_lag=(int(later - b) if later is not None else None),
                    classification=label,
                    reason=reason,
                    commit_frame=(frame_number(commit) if commit is not None else None),
                )
            )
    events.sort(key=lambda e: (e.chain, e.start_idx))
    return events


def _build_event(
    *,
    chain: int,
    species: str,
    lineage_id: str,
    start_idx: int,
    end_idx: int,
    seg_host: Dict[int, Dict[int, int]],
    lineage_host: Dict[str, Dict[int, int]],
    lineage_last: Dict[str, int],
    commit_idx: Dict[str, Optional[int]],
    commit_size: int,
    after_window: int,
    growth_min: int,
    end_tolerance: int,
    frame_number,
) -> ResidenceEvent:
    host_series = lineage_host.get(lineage_id, {})
    during = [
        seg_host.get(f, {}).get(chain, 0) for f in range(start_idx, end_idx + 1)
    ]
    during = [h for h in during if h is not None]
    host_entry = int(during[0]) if during else 0
    host_exit = int(during[-1]) if during else 0
    host_max = int(max(during)) if during else 0

    last_idx = int(lineage_last.get(lineage_id, end_idx))
    after_idx = min(end_idx + int(after_window), last_idx)
    # A lineage can be absent from individual frames (every one of its clusters
    # briefly falls below the size floor), so look back to the newest frame it
    # actually occupies rather than assuming after_idx is present.
    candidates = [f for f in host_series if end_idx <= f <= after_idx]
    host_after = host_series[max(candidates)] if candidates else None
    growth_after = (
        float(host_after - host_exit) if host_after is not None else float("nan")
    )

    commit = commit_idx.get(lineage_id)
    committed = commit is not None
    present_at_commit = bool(committed and start_idx <= commit <= end_idx)
    joined_after_commit = bool(committed and start_idx > commit)
    survives = bool(end_idx >= last_idx - int(end_tolerance))
    growth_during = host_max - host_entry

    label, reason = _classify_residence(
        committed=committed,
        present_at_commit=present_at_commit,
        joined_after_commit=joined_after_commit,
        survives=survives,
        growth_during=growth_during,
        growth_after=growth_after,
        growth_min=growth_min,
        commit_size=commit_size,
    )
    return ResidenceEvent(
        chain=chain,
        species=species,
        lineage_id=lineage_id,
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        start_frame=frame_number(start_idx),
        end_frame=frame_number(end_idx),
        duration=int(end_idx - start_idx + 1),
        host_at_entry=host_entry,
        host_at_exit=host_exit,
        host_max_during=host_max,
        host_growth_during=int(growth_during),
        host_growth_after=growth_after,
        committed=committed,
        commit_idx=(int(commit) if commit is not None else None),
        present_at_commit=present_at_commit,
        joined_after_commit=joined_after_commit,
        survives_to_lineage_end=survives,
        lineage_end_idx=last_idx,
        later_commit_idx=(int(commit) if commit is not None and commit > end_idx else None),
        later_commit_frame=(frame_number(commit)
                            if commit is not None and commit > end_idx else None),
        departure_to_commit_lag=(int(commit - end_idx)
                                 if commit is not None and commit > end_idx else None),
        classification=label,
        reason=reason,
        commit_frame=(frame_number(commit) if commit is not None else None),
    )


def _classify_residence(
    *,
    committed: bool,
    present_at_commit: bool,
    joined_after_commit: bool,
    survives: bool,
    growth_during: int,
    growth_after: float,
    growth_min: int,
    commit_size: int,
) -> Tuple[str, str]:
    """The rule set, written out so every call can be audited or overridden.

    Order matters: commitment is checked first because it is the event the whole
    mechanistic distinction hangs on, and only then is the guest's relationship
    to it examined.
    """
    if committed and present_at_commit and survives:
        return (
            "incorporated",
            f"present when the assembly reached {commit_size} host chains and "
            "still there at the end -- part of the product",
        )
    if committed and present_at_commit and not survives:
        kept_going = not np.isfinite(growth_after) or growth_after >= 0
        if kept_going:
            return (
                "catalytic",
                f"present at commitment ({commit_size} host chains), then left "
                "while the assembly held or kept growing -- promotes formation "
                "without joining the product",
            )
        return (
            "transient",
            "present at commitment but the assembly shrank after departure; "
            "the visit did not leave a stable aggregate behind",
        )
    if committed and joined_after_commit:
        return (
            "passenger",
            "joined an assembly that had already committed -- recruited by the "
            "aggregate rather than contributing to its formation",
        )
    if committed and not present_at_commit and not joined_after_commit:
        # The guest was gone before the assembly reached commit_size: the one
        # case in which it is absent from the commitment frame *and* from the
        # product, so it used to fall through every branch above and be
        # reported as a bystander whose stated reason ("no commitment") was the
        # opposite of the truth.
        #
        # This is deliberately *not* called catalytic.  A visit that precedes
        # commitment establishes association, not causation: the assembly could
        # have reached commit_size long after, for unrelated reasons.  Calling
        # it catalytic would assert the mechanism the data cannot distinguish,
        # so it gets its own label and the growth floor that "promoting" uses.
        if growth_during >= growth_min:
            return (
                "pre-commitment",
                f"gone before the assembly reached {commit_size} host chains, "
                f"which gained {growth_during} during the visit -- associated "
                "with growth that preceded commitment, which is weaker "
                "evidence than catalysis: the visit and the later commitment "
                "are only adjacent in time, not shown to be causally linked",
            )
        return (
            "bystander",
            f"left before the assembly reached {commit_size} host chains, "
            f"gaining {growth_during} during the visit -- below the promotion "
            f"threshold of {growth_min}, so an encounter rather than a "
            "mechanism",
        )
    if not committed and growth_during >= growth_min:
        return (
            "promoting",
            f"assembly gained {growth_during} host chains during the visit, but no "
            f"commitment to {commit_size} host chains is attributed to the visit -- "
            "growth without commitment",
        )
    return (
        "bystander",
        f"no commitment attributed to the visit, and the {growth_during} host "
        f"chains gained during the "
        f"visit are below the promotion threshold of {growth_min} -- an "
        "encounter, not a mechanism",
    )


def residence_dataframe(events: Sequence[ResidenceEvent]) -> pd.DataFrame:
    """Tidy table of residence events, one row per visit."""
    if not events:
        return pd.DataFrame(
            columns=[
                "chain", "species", "lineage_id", "start_frame", "end_frame",
                "duration", "host_at_entry", "host_at_exit", "host_max_during",
                "host_growth_during", "host_growth_after", "committed",
                "commit_idx", "commit_frame", "present_at_commit", "joined_after_commit",
                "survives_to_end", "classification", "reason",
                "later_commit_idx", "later_commit_frame", "departure_to_commit_lag",
            ]
        )
    return pd.DataFrame([e.to_dict() for e in events])


def mechanism_summary(
    events: Sequence[ResidenceEvent],
    *,
    n_guest_chains: Optional[int] = None,
) -> pd.DataFrame:
    """Counts, shares and mean residence time per mechanism class.

    ``visit_share`` weights every visit equally; ``frame_share`` weights by how
    long each visit lasted.  The two differ exactly when one class is made of
    many brief encounters and another of a few long ones, which is the usual
    situation -- reporting only the first makes bystanders look like the
    dominant mechanism.
    """
    # Every label _classify_residence can return must appear here: the loop
    # below iterates this list, so an unregistered class is silently dropped
    # from the summary while still counting toward the share denominators.
    order = [
        "incorporated", "catalytic", "pre-commitment", "passenger",
        "promoting", "transient", "bystander",
    ]
    df = residence_dataframe(events)
    if df.empty:
        return pd.DataFrame(columns=["classification", "n_visits", "visit_share"])

    total_visits = len(df)
    total_frames = float(df["duration"].sum())
    rows = []
    for label in order:
        sub = df[df["classification"] == label]
        if sub.empty:
            continue
        rows.append(
            {
                "classification": label,
                "n_visits": int(len(sub)),
                "visit_share": len(sub) / total_visits,
                "frame_share": float(sub["duration"].sum()) / total_frames
                if total_frames
                else float("nan"),
                "mean_duration": float(sub["duration"].mean()),
                "median_duration": float(sub["duration"].median()),
                "n_chains": int(sub["chain"].nunique()),
                "chain_share": (
                    sub["chain"].nunique() / n_guest_chains
                    if n_guest_chains
                    else float("nan")
                ),
                "mean_growth_during": float(sub["host_growth_during"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("frame_share", ascending=False).reset_index(drop=True)


def exchange_statistics(
    events: Sequence[ResidenceEvent],
    *,
    n_frames: int,
    n_guest_chains: int,
    frame_interval_ns: Optional[float] = None,
) -> Dict[str, float]:
    """How fast the guest population turns over.

    A guest that is bound most of the time but exchanges constantly is doing
    something different from one that binds once and stays, even when the two
    give the same average bound fraction.  Reported as visits per chain per
    frame, and per microsecond when the frame spacing is known.
    """
    df = residence_dataframe(events)
    n_visits = len(df)
    bound_frames = float(df["duration"].sum()) if n_visits else 0.0
    denom = float(n_frames * n_guest_chains)
    out: Dict[str, float] = {
        "n_visits": float(n_visits),
        "n_guest_chains": float(n_guest_chains),
        "visits_per_chain": (n_visits / n_guest_chains if n_guest_chains else float("nan")),
        "mean_residence_frames": (bound_frames / n_visits if n_visits else float("nan")),
        "bound_fraction": (bound_frames / denom if denom else float("nan")),
        "exchange_rate_per_chain_per_frame": (
            n_visits / denom if denom else float("nan")
        ),
    }
    if frame_interval_ns:
        per_ns = out["exchange_rate_per_chain_per_frame"] / float(frame_interval_ns)
        out["exchange_rate_per_chain_per_us"] = per_ns * 1000.0
        out["mean_residence_ns"] = out["mean_residence_frames"] * float(frame_interval_ns)
    return out
