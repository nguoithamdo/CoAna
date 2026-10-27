# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Time-ordered interactive lineage timeline.

A force-directed lineage graph (:func:`~oligomer_analysis.plot_lineage_graph`)
shows topology well and chronology not at all.  This module renders the same
segments and events onto a **real frame axis**, so splits and merges read left
to right in the order they happened, and inlines the result into one
self-contained HTML page that opens from disk.

Two steps, usable separately or together:

* :func:`build_timeline_payload` — cache (or a prebuilt lineage result) into a
  compact JSON-ready dictionary;
* :func:`render_timeline_html` — payload into the standalone page.

:func:`generate_lineage_timeline` does both in one call, and is what the
``coana timeline`` subcommand invokes.

The payload deliberately carries slightly more than the page draws by default
(member sets, event provenance coverage, coarse size traces).  Those are what
the inspector reveals on click, and they are cheap relative to the segment and
event records themselves.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .cache_io import load_cache, subset_cache_by_frames
from .lineage import build_lineages_from_cache
from .models import ContactCache, LineageTrackingResult, TargetStateConfig

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "build_timeline_payload",
    "render_timeline_html",
    "generate_lineage_timeline",
    "default_template_path",
]

#: Lineage-construction defaults.  These mirror the documented defaults of
#: :func:`~oligomer_analysis.build_lineages_from_cache`; every one is
#: overridable per call and exposed as a ``coana timeline`` flag.
DEFAULT_BUILD_PARAMS: Dict[str, Any] = dict(
    inter_thresh=1,
    cluster_min_size=2,
    overlap_min=1,
    parent_frac_min=0.30,
    child_frac_min=0.30,
    union_cover_min=0.70,
)


def default_template_path() -> Path:
    """Path to the page template shipped with the package."""
    return Path(__file__).resolve().parent / "templates" / "timeline_template.html"


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _segment_records(result: LineageTrackingResult) -> list:
    seg_lineage = {}
    for lineage in result.lineages:
        for seg_id in lineage.segment_ids:
            seg_lineage[seg_id] = lineage.lineage_id

    records = []
    for seg in result.segments:
        records.append({
            "id": seg.segment_id,
            "lin": seg_lineage.get(seg.segment_id, seg.lineage_id or ""),
            "f0": int(seg.start_frame),
            "f1": int(seg.end_frame),
            "dur": int(seg.duration),
            "max": int(seg.max_size),
            "mean": _round(seg.mean_size, 2),
            "s0": int(seg.sizes[0]) if seg.sizes else 0,
            "s1": int(seg.sizes[-1]) if seg.sizes else 0,
            "ev0": seg.event_start,
            "ev1": seg.event_end,
            "par": list(seg.parent_ids),
            "chi": list(seg.child_ids),
            # Coarse size trace: enough to draw a profile when zoomed in without
            # bloating the payload for very long segments.
            "trace": [int(v) for v in seg.sizes[:: max(1, len(seg.sizes) // 24)]][:25],
        })
    return records


def _event_records(result: LineageTrackingResult) -> list:
    records = []
    for ev in result.events:
        # Cluster sizes across an event are not a mass balance: monomers below
        # cluster_min_size are untracked, so a fragment can pick one up (or drop
        # one) in the same frame step it splits.  Record the difference so the
        # page can say "4 -> 2 + 3, +1 recruited" instead of leaving the reader
        # to wonder why the arithmetic fails.
        p_union = set().union(*ev.parent_sets) if ev.parent_sets else set()
        c_union = set().union(*ev.child_sets) if ev.child_sets else set()
        gained, lost = len(c_union - p_union), len(p_union - c_union)

        # Provenance strength — the same quantity the event classifier tests
        # against union_cover_min to decide between a clean split/merge and a
        # rearrangement.  Exporting it lets the page justify the label it shows.
        coverage = None
        if len(ev.parent_sets) == 1 and len(ev.child_sets) > 1:
            parent = ev.parent_sets[0]
            if parent:
                coverage = len(parent & c_union) / len(parent)
        elif len(ev.parent_sets) > 1 and len(ev.child_sets) == 1:
            child = ev.child_sets[0]
            if child:
                coverage = len(child & p_union) / len(child)

        entry = {
            "id": ev.event_id,
            "type": ev.event_type,
            "f": int(ev.next_frame),
            "fprev": int(ev.prev_frame),
            "par": list(ev.parent_segment_ids),
            "chi": list(ev.child_segment_ids),
            "psz": [len(s) for s in ev.parent_sets],
            "csz": [len(s) for s in ev.child_sets],
            # The actual monomer ids, aligned with par/chi.  The page keeps them
            # behind a click ("member sets" in the inspector) so the default
            # view stays uncluttered.
            "pm": [sorted(int(m) for m in s) for s in ev.parent_sets],
            "cm": [sorted(int(m) for m in s) for s in ev.child_sets],
            # `description` is omitted on purpose: the page rebuilds the same
            # sentence from type/parents/children/frames, which keeps the
            # payload roughly a megabyte smaller.
        }
        if gained:
            entry["g"] = gained
        if lost:
            entry["l"] = lost
        if coverage is not None:
            entry["cov"] = _round(coverage, 2)
        records.append(entry)
    return records


def _lineage_records(result: LineageTrackingResult) -> list:
    return [{
        "id": lineage.lineage_id,
        "cls": lineage.class_label,
        "rep": lineage.representative_segment_id or "",
        "nseg": len(lineage.segment_ids),
        "nev": len(lineage.event_ids),
        "max": int(lineage.max_size),
        "dur": int(lineage.duration),
        "f0": int(lineage.start_frame),
        "f1": int(lineage.end_frame),
        "target": bool(lineage.target_reached),
        "contribution": _round(lineage.contribution_score),
        "growth": _round(lineage.growth_score),
        "persistence": _round(lineage.persistence_score),
        "reversibility": _round(lineage.reversibility_score),
        "structural": _round(lineage.structural_score),
        "trap": _round(lineage.trap_score),
    } for lineage in result.lineages]


def build_timeline_payload(
    cache: Union[ContactCache, str, Path],
    *,
    frame_lo: Optional[int] = None,
    frame_hi: Optional[int] = None,
    target_config: Optional[TargetStateConfig] = None,
    lineage_result: Optional[LineageTrackingResult] = None,
    verbose: bool = True,
    **build_params: Any,
) -> Dict[str, Any]:
    """Build the time-ordered lineage payload for one cache.

    Parameters
    ----------
    cache
        A :class:`ContactCache` or a path to a Stage A ``.npz``.
    frame_lo, frame_hi
        Inclusive frame window.  ``None`` on either side means "to the end".
        Windowing a long trajectory is the cheapest way to keep the page
        responsive: cost grows with the number of segments, not frames.
    target_config
        Operational definition of a productive aggregate, which decides the
        on/off-pathway colouring.  Defaults to
        :class:`~oligomer_analysis.TargetStateConfig` defaults (size >= 10 held
        for >= 5 frames) — revisit it for systems much smaller or larger than
        the 128-monomer reference run.
    lineage_result
        Skip lineage construction and export this result instead.  Use it to
        avoid rebuilding lineages you already have in hand; ``cache`` is then
        used only for metadata and must match.
    **build_params
        Overrides for :data:`DEFAULT_BUILD_PARAMS`, forwarded to
        :func:`build_lineages_from_cache`.
    """
    unknown = set(build_params) - set(DEFAULT_BUILD_PARAMS)
    if unknown:
        raise TypeError(
            f"unknown build parameter(s): {sorted(unknown)}; "
            f"expected any of {sorted(DEFAULT_BUILD_PARAMS)}"
        )
    params = {**DEFAULT_BUILD_PARAMS, **build_params}

    cache_name = os.path.basename(str(cache)) if isinstance(cache, (str, Path)) else "<in-memory cache>"
    if isinstance(cache, (str, Path)):
        cache = load_cache(cache)

    if frame_lo is not None or frame_hi is not None:
        lo = int(frame_lo) if frame_lo is not None else int(cache.frames[0])
        hi = int(frame_hi) if frame_hi is not None else int(cache.frames[-1])
        cache = subset_cache_by_frames(cache, range(lo, hi + 1))

    t0 = time.perf_counter()
    if lineage_result is None:
        lineage_result = build_lineages_from_cache(
            cache, target_config=target_config, **params)
    build_seconds = time.perf_counter() - t0
    if verbose:
        print(f"lineages built in {build_seconds:.1f} s: "
              f"{len(lineage_result.segments)} segments, "
              f"{len(lineage_result.events)} events, "
              f"{len(lineage_result.lineages)} lineages", flush=True)

    segments = _segment_records(lineage_result)
    events = _event_records(lineage_result)
    lineages = _lineage_records(lineage_result)

    type_counts: Dict[str, int] = {}
    for ev in events:
        type_counts[ev["type"]] = type_counts.get(ev["type"], 0) + 1
    class_counts: Dict[str, int] = {}
    for lineage in lineages:
        class_counts[lineage["cls"]] = class_counts.get(lineage["cls"], 0) + 1

    target = target_config or lineage_result.target_config
    return {
        "meta": {
            "cache": cache_name,
            "n_frames": int(len(cache.frames)),
            "frame_first": int(cache.frames[0]),
            "frame_last": int(cache.frames[-1]),
            "n_segments": len(segments),
            "n_events": len(events),
            "n_lineages": len(lineages),
            "event_type_counts": type_counts,
            "class_counts": class_counts,
            "params": params,
            "target": {
                "min_size": int(target.min_size),
                "min_duration": int(target.min_duration),
                "min_contribution": float(target.min_contribution),
            } if target is not None else None,
            "build_seconds": _round(build_seconds, 1),
        },
        "segments": segments,
        "events": events,
        "lineages": lineages,
    }


def _check_js(html: str, *, verbose: bool = True) -> None:
    """Parse the page's inline script with node, when node is available."""
    node = shutil.which("node")
    if not node:
        if verbose:
            print("note: node not found, skipping the JavaScript syntax check")
        return
    start = html.index("<script>") + len("<script>")
    end = html.index("</script>", start)
    script = html[start:end]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
        fh.write(script)
        tmp = fh.name
    try:
        proc = subprocess.run([node, "--check", tmp], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("JavaScript syntax error in the built page:\n" + proc.stderr)
        if verbose:
            print(f"javascript OK ({len(script.splitlines())} lines parsed by node)")
    finally:
        os.unlink(tmp)


def render_timeline_html(
    payload: Dict[str, Any],
    out_html: Union[str, Path],
    *,
    template: Optional[Union[str, Path]] = None,
    check_js: bool = False,
    verbose: bool = True,
) -> str:
    """Inline ``payload`` into the page template and write the standalone HTML.

    The result has no external dependencies beyond a Google Fonts stylesheet,
    so it opens straight from disk and can be shared as a single file.

    ``check_js=True`` parses the page's inline script with ``node`` before
    writing, when node is on PATH.  It is off by default because node is not a
    dependency of this package; the shipped toolchain turns it on.
    """
    template_path = Path(template) if template is not None else default_template_path()
    if not template_path.is_file():
        raise FileNotFoundError(f"page template not found: {template_path}")

    html = template_path.read_text(encoding="utf-8")
    if "{{DATA}}" not in html:
        raise ValueError(f"template has no {{{{DATA}}}} placeholder: {template_path}")

    data = json.dumps(payload, separators=(",", ":"))
    # str.replace would rescan the multi-megabyte payload for backslash escapes;
    # splitting once keeps the substitution literal and cheap.
    head, tail = html.split("{{DATA}}", 1)
    html = head + data + tail

    if check_js:
        _check_js(html, verbose=verbose)

    out_path = Path(out_html)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    if verbose:
        print(f"written {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return str(out_path)


def generate_lineage_timeline(
    cache: Union[ContactCache, str, Path],
    out_html: Union[str, Path] = "LINEAGE_TIMELINE.html",
    *,
    frame_lo: Optional[int] = None,
    frame_hi: Optional[int] = None,
    target_config: Optional[TargetStateConfig] = None,
    lineage_result: Optional[LineageTrackingResult] = None,
    template: Optional[Union[str, Path]] = None,
    payload_json: Optional[Union[str, Path]] = None,
    check_js: bool = False,
    verbose: bool = True,
    **build_params: Any,
) -> str:
    """Build a standalone interactive lineage timeline from a Stage A cache.

    One call from cache to page; this is what ``coana timeline`` runs.

    >>> generate_lineage_timeline("cache.npz", "TIMELINE.html", inter_thresh=1)

    ``payload_json`` additionally writes the intermediate payload, which is
    useful for diffing two runs or feeding a different renderer.
    """
    payload = build_timeline_payload(
        cache, frame_lo=frame_lo, frame_hi=frame_hi, target_config=target_config,
        lineage_result=lineage_result, verbose=verbose, **build_params)

    if payload_json is not None:
        json_path = Path(payload_json)
        if json_path.parent and not json_path.parent.exists():
            json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        if verbose:
            print(f"written {json_path} ({json_path.stat().st_size / 1e6:.2f} MB)")

    return render_timeline_html(payload, out_html, template=template,
                                check_js=check_js, verbose=verbose)
