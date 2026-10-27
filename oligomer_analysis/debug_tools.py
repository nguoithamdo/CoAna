# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

"""Complete debugging utilities for contact-map oligomer analysis.

All edge decisions use the same rule as the main analysis:

    edge(i, j) = cmap[i, j] >= inter_thresh
"""

from typing import Sequence
import numpy as np

from .graph_tools import clusters_from_cmap


def _pathway_get(pathway, key):
    """Support both dataclass-style pathways and dict-style pathways."""
    return getattr(pathway, key) if hasattr(pathway, key) else pathway[key]


def _frame_index_map(cache):
    """Map actual frame number -> cache index."""
    return {int(fr): i for i, fr in enumerate(cache.frames)}


def debug_show_big_clusters(cache, inter_thresh: int, size_thresh: int, fr0: int, fr1: int, max_print: int = 20):
    """Print clusters with size >= size_thresh for frames fr0..fr1 inclusive."""
    f2i = _frame_index_map(cache)
    for fr in range(int(fr0), int(fr1) + 1):
        if fr not in f2i:
            print(fr, "(not in cache)")
            continue
        clusters = clusters_from_cmap(cache.cmaps[f2i[fr]], inter_thresh)
        big = [c for c in clusters if len(c) >= int(size_thresh)]
        out = sorted([sorted(list(c)) for c in big], key=len, reverse=True)
        if len(out) > max_print:
            out = out[:max_print] + [["..."]]
        print(fr, out)


def debug_cmap_pairs(cache, segments: Sequence[int], fr0: int, fr1: int, inter_thresh: int):
    """Print cmap[a,b] for all a<b among selected segments across frames."""
    f2i = _frame_index_map(cache)
    segs = list(segments)
    print(f"Segments: {segs}")
    print(f"EDGE rule: cmap[i,j] >= {inter_thresh}\n")
    for fr in range(int(fr0), int(fr1) + 1):
        if fr not in f2i:
            print(fr, "(not in cache)")
            continue
        cmap = cache.cmaps[f2i[fr]]
        print(f"Frame {fr}:")
        for ai in range(len(segs) - 1):
            for bi in range(ai + 1, len(segs)):
                a, b = int(segs[ai]), int(segs[bi])
                val = int(cmap[a, b])
                ok = "EDGE" if val >= inter_thresh else "----"
                print(f"  cmap[{a:>3},{b:>3}] = {val:>6}   {ok}")
        print("")


def debug_row_sums(cache, segments: Sequence[int], fr0: int, fr1: int):
    """Print total intercontact row sums for selected segments."""
    f2i = _frame_index_map(cache)
    segs = list(segments)
    for fr in range(int(fr0), int(fr1) + 1):
        if fr not in f2i:
            print(fr, "(not in cache)")
            continue
        row = cache.cmaps[f2i[fr]].sum(axis=1)
        print(f"Frame {fr}: " + ", ".join([f"s{seg} sum={int(row[int(seg)])}" for seg in segs]))


def debug_edge_changes(cache, fr: int, inter_thresh: int, max_edges: int = 50):
    """For transition fr-1 -> fr, list edges lost and gained."""
    f2i = _frame_index_map(cache)
    fr = int(fr)
    if fr not in f2i or (fr - 1) not in f2i:
        print("Need both fr and fr-1 in cache")
        return

    A0 = cache.cmaps[f2i[fr - 1]] >= inter_thresh
    A1 = cache.cmaps[f2i[fr]] >= inter_thresh
    np.fill_diagonal(A0, False)
    np.fill_diagonal(A1, False)

    lost = [(int(i), int(j)) for i, j in np.argwhere(A0 & ~A1) if i < j]
    gained = [(int(i), int(j)) for i, j in np.argwhere(~A0 & A1) if i < j]

    print(f"Frame {fr-1} -> {fr} | EDGE rule cmap >= {inter_thresh}")
    print("Lost edges:", lost[:max_edges], "..." if len(lost) > max_edges else "")
    print("Gained edges:", gained[:max_edges], "..." if len(gained) > max_edges else "")


def debug_edge_changes_focus(cache, fr: int, inter_thresh: int, focus: Sequence[int]):
    """For transition fr-1 -> fr, list lost/gained edges only among selected segments."""
    f2i = _frame_index_map(cache)
    fr = int(fr)
    if fr not in f2i or (fr - 1) not in f2i:
        print("Need both fr and fr-1 in cache")
        return

    focus = sorted(set(int(x) for x in focus))
    A0 = cache.cmaps[f2i[fr - 1]] >= inter_thresh
    A1 = cache.cmaps[f2i[fr]] >= inter_thresh
    np.fill_diagonal(A0, False)
    np.fill_diagonal(A1, False)

    lost, gained = [], []
    for i in range(len(focus) - 1):
        for j in range(i + 1, len(focus)):
            a, b = focus[i], focus[j]
            if A0[a, b] and not A1[a, b]:
                lost.append((a, b))
            if (not A0[a, b]) and A1[a, b]:
                gained.append((a, b))

    print(f"Frame {fr-1}->{fr} | focus={focus} | cmap >= {inter_thresh}")
    print("Lost:", lost)
    print("Gained:", gained)


def debug_pathway(cache, finished, label: str, inter_thresh: int, pad: int = 2, max_members: int = 50):
    """Print per-frame membership for a tracked pathway and nearby edge changes."""
    p = next((x for x in finished if _pathway_get(x, "label") == label), None)
    if p is None:
        raise ValueError(f"Label not found: {label}")

    start_idx = int(_pathway_get(p, "start_idx"))
    duration = int(_pathway_get(p, "duration"))
    sets = list(_pathway_get(p, "sets"))
    end_idx = start_idx + duration
    frames = cache.frames

    print(f"Pathway {label}")
    print(f" start_frame={int(frames[start_idx])}, duration={duration}, indices={start_idx}:{end_idx}")
    print(f" final_set={sorted(list(sets[-1])) if sets else []}\n")

    for t in range(duration):
        fr = int(frames[start_idx + t])
        members = sorted(list(sets[t]))
        if len(members) > max_members:
            members = members[:max_members] + ["..."]
        print(f"  frame {fr}: size={len(sets[t]):>3} members={members}")

    print("\nEdge changes near pathway window:")
    fr_start = int(frames[max(start_idx - pad, 0)])
    fr_end = int(frames[min(end_idx + pad - 1, len(frames) - 1)])
    for fr in range(fr_start + 1, fr_end + 1):
        debug_edge_changes(cache, fr, inter_thresh)


def debug_cluster_overlap(prev_clusters, next_clusters, min_overlap: int = 1):
    """Print overlap matrix between two cluster lists."""
    prev_clusters = list(prev_clusters)
    next_clusters = list(next_clusters)
    O = np.zeros((len(prev_clusters), len(next_clusters)), dtype=int)
    for i, a in enumerate(prev_clusters):
        for j, b in enumerate(next_clusters):
            O[i, j] = len(a & b)

    print("Overlap matrix:")
    print(O)
    print("\nNonzero overlaps:")
    for i in range(O.shape[0]):
        for j in range(O.shape[1]):
            if O[i, j] >= min_overlap:
                print(f"  prev[{i}] size={len(prev_clusters[i])} -> next[{j}] size={len(next_clusters[j])}: overlap={O[i,j]}")
    return O


def debug_frame_transition(cache, frame: int, inter_thresh: int, cluster_min_size: int = 2):
    """Inspect cluster overlap from frame-1 to frame."""
    f2i = _frame_index_map(cache)
    frame = int(frame)
    if frame not in f2i or (frame - 1) not in f2i:
        print("Need both frame and frame-1 in cache")
        return None

    c0 = [c for c in clusters_from_cmap(cache.cmaps[f2i[frame - 1]], inter_thresh) if len(c) >= cluster_min_size]
    c1 = [c for c in clusters_from_cmap(cache.cmaps[f2i[frame]], inter_thresh) if len(c) >= cluster_min_size]
    print(f"Transition {frame-1} -> {frame}")
    print("Previous clusters:", [sorted(c) for c in c0])
    print("Next clusters:", [sorted(c) for c in c1])
    return debug_cluster_overlap(c0, c1)


def debug_lineage_summary(result, max_rows: int = 20):
    """Print a compact summary of lineage objects."""
    rows = []
    for lin in result.lineages:
        rows.append((
            lin.lineage_id,
            lin.class_label,
            lin.start_frame,
            lin.end_frame,
            lin.duration,
            lin.max_size,
            f"{lin.contribution_score:.3f}",
            f"{lin.growth_score:.3f}",
            len(lin.segment_ids),
        ))
    rows = sorted(rows, key=lambda x: (x[5], x[4]), reverse=True)
    print("lineage_id | class | start | end | duration | max_size | contribution | growth | n_segments")
    for row in rows[:max_rows]:
        print(" | ".join(map(str, row)))


def debug_lineage_events(result, lineage_id: str | None = None, event_type: str | None = None, max_rows: int = 50):
    """Print lineage events, optionally filtered by lineage or event type."""
    n = 0
    for ev in result.events:
        if event_type is not None and ev.event_type != event_type:
            continue
        if lineage_id is not None:
            parent_ok = any(
                (result.segment_by_id(pid) is not None and result.segment_by_id(pid).lineage_id == lineage_id)
                for pid in ev.parent_segment_ids
            )
            child_ok = any(
                (result.segment_by_id(cid) is not None and result.segment_by_id(cid).lineage_id == lineage_id)
                for cid in ev.child_segment_ids
            )
            if not (parent_ok or child_ok):
                continue
        print(
            f"{ev.event_id}: {ev.event_type} frame={ev.frame_number} "
            f"parents={ev.parent_segment_ids} children={ev.child_segment_ids} {ev.description}"
        )
        n += 1
        if n >= max_rows:
            break
    if n == 0:
        print("No matching events.")





def pdb_chain_id_summary(pdb_file: str, atoms_per_monomer: int | None = None):
    """Diagnose PDB chain-ID reuse without using MDAnalysis.

    This is useful for large Martini systems. Standard PDB chain identifiers are
    stored in one character, so files with more than ~62 chains often reuse
    chain IDs. If your physical monomers are ordered as fixed atom blocks, the
    precompute stage should use ``segment_mode='atom_blocks'`` rather than PDB
    chain IDs.

    Parameters
    ----------
    pdb_file
        Path to a PDB trajectory/topology file.
    atoms_per_monomer
        Optional expected atom/bead count per physical monomer, for example 83.

    Returns
    -------
    dict
        Summary with total atom count, unique PDB chain IDs, and inferred number
        of monomers if ``atoms_per_monomer`` is provided.
    """
    from collections import Counter

    total_atoms = 0
    chain_counts = Counter()

    with open(pdb_file, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                total_atoms += 1
                chain_id = line[21:22]
                if chain_id == " ":
                    chain_id = "_blank"
                chain_counts[chain_id] += 1

    summary = {
        "pdb_file": str(pdb_file),
        "total_atoms": int(total_atoms),
        "unique_pdb_chain_ids": int(len(chain_counts)),
        "chain_id_counts": dict(chain_counts),
        "atoms_per_monomer": atoms_per_monomer,
        "expected_monomers_from_atom_blocks": None,
        "atom_count_divisible_by_atoms_per_monomer": None,
    }

    if atoms_per_monomer is not None:
        atoms_per_monomer = int(atoms_per_monomer)
        summary["expected_monomers_from_atom_blocks"] = total_atoms / atoms_per_monomer
        summary["atom_count_divisible_by_atoms_per_monomer"] = (total_atoms % atoms_per_monomer == 0)

    print("PDB chain-ID summary")
    print("--------------------")
    print(f"File: {pdb_file}")
    print(f"Total ATOM/HETATM records: {total_atoms}")
    print(f"Unique one-character PDB chain IDs: {len(chain_counts)}")

    if atoms_per_monomer is not None:
        print(f"Expected monomers from atom blocks ({atoms_per_monomer} atoms/monomer): {total_atoms / atoms_per_monomer}")
        if total_atoms % atoms_per_monomer != 0:
            print("WARNING: total atoms is not divisible by atoms_per_monomer.")

    print("\nLargest chain-ID counts:")
    for cid, count in chain_counts.most_common(20):
        print(f"  {cid!r}: {count}")

    return summary

__all__ = [
    "pdb_chain_id_summary",
    "debug_show_big_clusters",
    "debug_cmap_pairs",
    "debug_row_sums",
    "debug_edge_changes",
    "debug_edge_changes_focus",
    "debug_pathway",
    "debug_cluster_overlap",
    "debug_frame_transition",
    "debug_lineage_summary",
    "debug_lineage_events",
]
