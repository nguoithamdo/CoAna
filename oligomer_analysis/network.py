"""Export the analysis as networks other tools already read.

Two graphs come out of a lineage analysis, and neither is easy to share in the
package's own formats.

**The lineage graph** -- segments as nodes, split/merge/growth events as edges --
is the provenance record of the whole trajectory.  Written as GEXF it opens in
Gephi and Cytoscape, where a collaborator can lay it out and interrogate it
without installing anything from here.

**The state-transition network** -- coarse states as nodes, observed transitions
as weighted edges -- is the object the transition-network literature works in.
Emitting it in the same form makes the results directly comparable with that
work instead of merely adjacent to it, and the row-normalised matrix is the
Markov transition matrix those methods start from.

States can be plain cluster size, or size plus composition in a mixture
(``"12(10A+2B)"``), which is what distinguishes a decamer that recruited two
guest chains from a pure one.

GEXF is written directly rather than through a graph library, so the export has
no dependency of its own and the file is small and diffable.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from .graph_tools import clusters_from_cmap, filter_clusters_by_size
from .models import LineageTrackingResult

__all__ = [
    "state_labels_by_frame",
    "transition_counts",
    "transition_matrix",
    "transition_dataframe",
    "write_lineage_gexf",
    "write_transition_gexf",
]


# --------------------------------------------------------------------------
# coarse states and their transitions
# --------------------------------------------------------------------------


def _state_label(size: int, comp: Optional[np.ndarray], names: Sequence[str]) -> str:
    if comp is None:
        return str(int(size))
    parts = [f"{int(c)}{n}" for n, c in zip(names, comp) if c]
    return f"{int(size)}({'+'.join(parts)})" if parts else str(int(size))


def state_labels_by_frame(
    cache,
    *,
    inter_thresh: int = 1,
    min_size: int = 2,
    species_map=None,
    max_size: Optional[int] = None,
) -> List[List[Tuple[frozenset, str]]]:
    """Per frame, every cluster with its coarse state label.

    ``max_size`` caps the resolved sizes: everything at or above it collapses
    into one state (``">=N"``), which keeps the network readable when a run ends
    in a single large aggregate whose size fluctuates by one chain from frame to
    frame.

    This is a *capped macrostate*, not an absorbing one: a cluster that drops
    back below the cap leaves it again, so a ``">=10" -> "9"`` transition is
    normal and expected. Treating ``">=N"`` as absorbing -- as an absorbing
    state could not leave itself -- would misread the exported network.
    """
    cmaps = np.asarray(cache.cmaps)
    names = list(species_map.names) if species_map is not None else []
    seg_species = (
        np.asarray(species_map.seg_species, dtype=np.int64)
        if species_map is not None
        else None
    )
    n_species = len(names)

    out: List[List[Tuple[frozenset, str]]] = []
    for f in range(cmaps.shape[0]):
        rows: List[Tuple[frozenset, str]] = []
        clusters = filter_clusters_by_size(
            clusters_from_cmap(cmaps[f], inter_thresh), min_size=min_size
        )
        for clu in clusters:
            size = len(clu)
            if max_size is not None and size >= int(max_size):
                rows.append((clu, f">={int(max_size)}"))
                continue
            comp = None
            if seg_species is not None:
                idx = np.fromiter(clu, dtype=np.int64, count=size)
                comp = np.bincount(seg_species[idx], minlength=n_species)
            rows.append((clu, _state_label(size, comp, names)))
        out.append(rows)
    return out


def transition_counts(
    cache,
    *,
    inter_thresh: int = 1,
    min_size: int = 2,
    species_map=None,
    max_size: Optional[int] = None,
    include_solution: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """Observed transition counts between coarse states.

    A cluster's successor is the cluster in the next frame it shares the most
    members with, which is the same "largest overlap" rule the whole-system
    transition matrix already uses.  With ``include_solution`` a pseudo-state
    ``solution`` absorbs clusters that dissolve and emits clusters that appear,
    so every row and column balances and the matrix can be normalised into a
    proper Markov transition matrix.

    Returns ``(state_names, counts)`` with ``counts[i, j]`` the number of
    observed ``i -> j`` steps.
    """
    per_frame = state_labels_by_frame(
        cache,
        inter_thresh=inter_thresh,
        min_size=min_size,
        species_map=species_map,
        max_size=max_size,
    )
    order: Dict[str, int] = {}

    def index_of(label: str) -> int:
        if label not in order:
            order[label] = len(order)
        return order[label]

    if include_solution:
        index_of("solution")

    pairs: List[Tuple[int, int]] = []
    for prev, nxt in zip(per_frame[:-1], per_frame[1:]):
        matched_next = set()
        for clu_prev, label_prev in prev:
            best, best_overlap, best_pos = None, 0, None
            for pos, (clu_next, label_next) in enumerate(nxt):
                overlap = len(clu_prev & clu_next)
                if overlap > best_overlap:
                    best, best_overlap, best_pos = label_next, overlap, pos
            if best is None:
                if include_solution:
                    pairs.append((index_of(label_prev), index_of("solution")))
                continue
            matched_next.add(best_pos)
            pairs.append((index_of(label_prev), index_of(best)))
        if include_solution:
            for pos, (_clu_next, label_next) in enumerate(nxt):
                if pos not in matched_next:
                    pairs.append((index_of("solution"), index_of(label_next)))

    n = len(order)
    counts = np.zeros((n, n), dtype=np.int64)
    for i, j in pairs:
        counts[i, j] += 1
    names = [None] * n
    for label, idx in order.items():
        names[idx] = label
    return names, counts


def transition_matrix(counts: np.ndarray) -> np.ndarray:
    """Row-normalise a count matrix into transition probabilities.

    Rows with no observed transitions are left as zeros rather than filled with
    a uniform distribution: an unvisited state has no measured dynamics, and
    inventing them would put structure into the model that the trajectory never
    showed.
    """
    counts = np.asarray(counts, dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, counts / totals, 0.0)


def transition_dataframe(
    names: Sequence[str], counts: np.ndarray, *, min_count: int = 1
) -> pd.DataFrame:
    """Long-form edge table with counts and transition probabilities."""
    probs = transition_matrix(counts)
    rows = []
    for i, src in enumerate(names):
        for j, dst in enumerate(names):
            c = int(counts[i, j])
            if c < int(min_count):
                continue
            rows.append(
                {
                    "source": src,
                    "target": dst,
                    "count": c,
                    "probability": float(probs[i, j]),
                    "self_transition": bool(i == j),
                }
            )
    cols = ["source", "target", "count", "probability", "self_transition"]
    if not rows:
        # No edge cleared min_count (a one-frame run, or a single state). The
        # columns still have to exist, or sorting raises KeyError and takes the
        # CSV export down with it.
        return pd.DataFrame(columns=cols)
    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# GEXF writers
# --------------------------------------------------------------------------


_ATTR_TYPES = {int: "integer", float: "float", bool: "boolean", str: "string"}


def _attr(value) -> str:
    """Escape a value for use inside a double-quoted XML attribute.

    ``xml.sax.saxutils.escape`` handles ``&``, ``<`` and ``>`` but deliberately
    leaves quotation marks alone, which is correct for element text and wrong
    here: a species or state name containing a double quote would close the
    attribute and emit invalid GEXF. The species spec accepts such a name, so
    this is reachable from user input rather than hypothetical.
    """
    return escape(str(value), {chr(34): "&quot;", chr(39): "&apos;"})


def _gexf_document(
    nodes: Sequence[Tuple[str, str, Dict[str, object]]],
    edges: Sequence[Tuple[str, str, float, Dict[str, object]]],
    *,
    directed: bool,
    description: str,
) -> str:
    """Assemble a GEXF 1.2 document from nodes and edges."""
    node_attrs: Dict[str, str] = {}
    for _nid, _label, attrs in nodes:
        for key, value in attrs.items():
            node_attrs.setdefault(key, _ATTR_TYPES.get(type(value), "string"))
    edge_attrs: Dict[str, str] = {}
    for _s, _t, _w, attrs in edges:
        for key, value in attrs.items():
            edge_attrs.setdefault(key, _ATTR_TYPES.get(type(value), "string"))

    node_keys = list(node_attrs)
    edge_keys = list(edge_attrs)

    out: List[str] = []
    out.append('<?xml version="1.0" encoding="UTF-8"?>')
    out.append('<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">')
    out.append(f"  <meta>\n    <description>{escape(description)}</description>\n  </meta>")
    out.append(f'  <graph mode="static" defaultedgetype="{"directed" if directed else "undirected"}">')

    if node_keys:
        out.append('    <attributes class="node">')
        for i, key in enumerate(node_keys):
            out.append(
                f'      <attribute id="{i}" title="{_attr(key)}" type="{node_attrs[key]}"/>'
            )
        out.append("    </attributes>")
    if edge_keys:
        out.append('    <attributes class="edge">')
        for i, key in enumerate(edge_keys):
            out.append(
                f'      <attribute id="{i}" title="{_attr(key)}" type="{edge_attrs[key]}"/>'
            )
        out.append("    </attributes>")

    out.append("    <nodes>")
    for nid, label, attrs in nodes:
        out.append(f'      <node id="{_attr(nid)}" label="{_attr(label)}">')
        if attrs:
            out.append("        <attvalues>")
            for i, key in enumerate(node_keys):
                if key in attrs:
                    out.append(
                        f'          <attvalue for="{i}" value="{_attr(_fmt(attrs[key]))}"/>'
                    )
            out.append("        </attvalues>")
        out.append("      </node>")
    out.append("    </nodes>")

    out.append("    <edges>")
    for k, (src, dst, weight, attrs) in enumerate(edges):
        out.append(
            f'      <edge id="{k}" source="{_attr(src)}" '
            f'target="{_attr(dst)}" weight="{float(weight):g}">'
        )
        if attrs:
            out.append("        <attvalues>")
            for i, key in enumerate(edge_keys):
                if key in attrs:
                    out.append(
                        f'          <attvalue for="{i}" value="{_attr(_fmt(attrs[key]))}"/>'
                    )
            out.append("        </attvalues>")
        out.append("      </edge>")
    out.append("    </edges>")
    out.append("  </graph>")
    out.append("</gexf>")
    return "\n".join(out) + "\n"


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_lineage_gexf(
    result: LineageTrackingResult,
    path: str | os.PathLike,
    *,
    min_duration: int = 1,
) -> str:
    """Write the segment/event provenance graph as GEXF.

    Nodes are segments, carrying their lineage, fate label, size range, frame
    span and -- in a mixture -- their peak composition.  Edges are the events
    that connect them, typed as split, merge, growth and so on, so the graph can
    be filtered by event type in Gephi.

    ``min_duration`` drops short-lived lineages before export; a raw graph from a
    dense box has thousands of two-frame nodes and no layout makes that readable.
    """
    from .lineage import filter_lineages

    if min_duration > 1:
        result = filter_lineages(result, min_duration=min_duration)

    lin_by_id = result.lineage_dict
    nodes: List[Tuple[str, str, Dict[str, object]]] = []
    for seg in result.segments:
        lin = lin_by_id.get(seg.lineage_id or "")
        attrs: Dict[str, object] = {
            "lineage": seg.lineage_id or "",
            "fate": lin.class_label if lin else "unknown",
            "start_frame": int(seg.start_frame),
            "end_frame": int(seg.end_frame),
            "duration": int(seg.duration),
            "max_size": int(seg.max_size),
            "mean_size": float(seg.mean_size),
            "event_start": seg.event_start,
            "event_end": seg.event_end,
        }
        if seg.compositions is not None and lin is not None and lin.species_names:
            peak = seg.peak_composition
            for k, name in enumerate(lin.species_names):
                attrs[f"peak_{name}"] = int(peak[k]) if peak is not None else 0
        nodes.append((seg.segment_id, f"{seg.segment_id} ({seg.max_size})", attrs))

    known = {seg.segment_id for seg in result.segments}
    edges: List[Tuple[str, str, float, Dict[str, object]]] = []
    for ev in result.events:
        for parent in ev.parent_segment_ids:
            for child in ev.child_segment_ids:
                if parent in known and child in known:
                    edges.append(
                        (
                            parent,
                            child,
                            1.0,
                            {
                                "event_type": ev.event_type,
                                "event_id": ev.event_id,
                                "frame": int(ev.next_frame),
                            },
                        )
                    )

    doc = _gexf_document(
        nodes,
        edges,
        directed=True,
        description=(
            "Oligomer lineage provenance: nodes are continuous assembly "
            "episodes, edges are split/merge/growth events."
        ),
    )
    path = str(path)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    return path


def write_transition_gexf(
    names: Sequence[str],
    counts: np.ndarray,
    path: str | os.PathLike,
    *,
    min_count: int = 1,
    include_self: bool = False,
) -> str:
    """Write a coarse state-transition network as GEXF.

    Node weight is how many transitions *left* the state; edge weight is the
    transition probability, with the raw count kept alongside so a rare
    high-probability edge can be told from a well-sampled one.  Self-transitions
    are excluded by default -- in a network this dense they dominate every edge
    weight and hide the actual pathway.

    The node attribute is ``n_departures``, not an occupancy: a row sum counts
    the transitions out of a state, which is one fewer than the frames spent in
    it and is zero for a state only ever entered last. For the two-frame path
    ``A -> B`` the departures are ``A=1, B=0`` even though both states were
    occupied once. Pass per-frame occupancies separately if true residence is
    wanted.
    """
    probs = transition_matrix(counts)
    departures = np.asarray(counts, dtype=np.float64).sum(axis=1)

    nodes = [
        (
            name,
            name,
            {
                "n_departures": int(departures[i]),
                "outgoing": int(counts[i].sum()),
                "incoming": int(counts[:, i].sum()),
            },
        )
        for i, name in enumerate(names)
    ]
    edges = []
    for i, src in enumerate(names):
        for j, dst in enumerate(names):
            if i == j and not include_self:
                continue
            c = int(counts[i, j])
            if c < int(min_count):
                continue
            edges.append((src, dst, float(probs[i, j]), {"count": c}))

    doc = _gexf_document(
        nodes,
        edges,
        directed=True,
        description=(
            "Coarse state-transition network; edge weight is the observed "
            "transition probability."
        ),
    )
    path = str(path)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    return path
