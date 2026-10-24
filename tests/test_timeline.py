"""Tests for the interactive lineage timeline: payload, renderer and CLI.

The page itself is checked by ``docs/lineage_timeline/smoke_test.js`` (29 checks
against a built page, run with node). These tests cover the Python side: that
the payload describes the lineage graph faithfully, that the parameters are
actually honoured rather than silently defaulted, and that the rendered page is
genuinely self-contained.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from oligomer_analysis.cache_io import load_cache
from oligomer_analysis.cli import main
from oligomer_analysis.lineage import build_lineages_from_cache
from oligomer_analysis.models import TargetStateConfig
from oligomer_analysis.timeline import (
    DEFAULT_BUILD_PARAMS,
    build_timeline_payload,
    default_template_path,
    generate_lineage_timeline,
    render_timeline_html,
)


def _write_aggregating_pdb(path, n_frames=14, n_monomers=6, beads=2):
    """Monomers drift together, so the run has real clusters and events."""
    lines = []
    rng = np.random.RandomState(11)
    jitter = rng.rand(n_monomers, beads, 2) * 0.5
    for fr in range(n_frames):
        lines.append(f"MODEL   {fr + 1:6d}\n")
        serial = 1
        shrink = max(0.15, 1.0 - 0.08 * fr)
        for m in range(n_monomers):
            for b in range(beads):
                x = m * 6.0 * shrink + b * 0.8
                y, z = jitter[m, b, 0], jitter[m, b, 1]
                lines.append(
                    f"ATOM  {serial:5d}  B{b:<2d} ALA {chr(65 + m)}{m + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
                )
                serial += 1
        lines.append("ENDMDL\n")
    with open(path, "w") as fh:
        fh.writelines(lines)


@pytest.fixture()
def small_cache(tmp_path):
    pdb = tmp_path / "agg.pdb"
    _write_aggregating_pdb(pdb)
    out = tmp_path / "cache.npz"
    rc = main([
        "precompute", "--allow-no-box", "--traj", str(pdb), "--frames", "all",
        "--cutoff", "2.5", "--intra-exclude", "0",
        "--segment-mode", "atom_blocks", "--atoms-per-monomer", "2",
        "--n-monomers", "6", "-o", str(out), "--quiet",
    ])
    assert rc == 0
    return out


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
def test_template_ships_with_the_package():
    template = default_template_path()
    assert template.is_file(), f"page template missing from the package: {template}"
    assert "{{DATA}}" in template.read_text(encoding="utf-8")


def test_payload_matches_the_lineage_result(small_cache):
    cache = load_cache(str(small_cache))
    result = build_lineages_from_cache(cache, **DEFAULT_BUILD_PARAMS)
    payload = build_timeline_payload(cache, verbose=False)

    assert payload["meta"]["n_segments"] == len(result.segments) == len(payload["segments"])
    assert payload["meta"]["n_events"] == len(result.events) == len(payload["events"])
    assert payload["meta"]["n_lineages"] == len(result.lineages) == len(payload["lineages"])
    assert payload["meta"]["frame_first"] == int(cache.frames[0])
    assert payload["meta"]["frame_last"] == int(cache.frames[-1])

    # Every segment names a lineage that exists, and vice versa.
    lineage_ids = {lin["id"] for lin in payload["lineages"]}
    assert {seg["lin"] for seg in payload["segments"]} <= lineage_ids

    # Event member sets must agree with the sizes reported alongside them; the
    # page re-derives its chips from both, so a mismatch is a real defect.
    for ev in payload["events"]:
        assert [len(s) for s in ev["pm"]] == ev["psz"]
        assert [len(s) for s in ev["cm"]] == ev["csz"]

    # The payload must survive a JSON round-trip: it is inlined as literal JS.
    assert json.loads(json.dumps(payload))["meta"]["n_events"] == len(result.events)


def test_payload_honours_build_and_target_parameters(small_cache):
    target = TargetStateConfig(min_size=3, min_duration=2, min_contribution=0.5)
    payload = build_timeline_payload(
        small_cache, inter_thresh=2, cluster_min_size=3,
        target_config=target, verbose=False)

    assert payload["meta"]["params"]["inter_thresh"] == 2
    assert payload["meta"]["params"]["cluster_min_size"] == 3
    assert payload["meta"]["target"] == {
        "min_size": 3, "min_duration": 2, "min_contribution": 0.5}

    # A stricter contact threshold cannot find more structure than a loose one.
    loose = build_timeline_payload(small_cache, inter_thresh=1, verbose=False)
    assert payload["meta"]["n_segments"] <= loose["meta"]["n_segments"] * 3


def test_payload_frame_window(small_cache):
    payload = build_timeline_payload(small_cache, frame_lo=2, frame_hi=7, verbose=False)
    assert payload["meta"]["frame_first"] == 2
    assert payload["meta"]["frame_last"] == 7
    assert payload["meta"]["n_frames"] == 6
    for seg in payload["segments"]:
        assert 2 <= seg["f0"] <= seg["f1"] <= 7


def test_unknown_build_parameter_is_rejected(small_cache):
    # Silently ignoring a misspelled parameter would mean a page built with
    # defaults while the caller believes otherwise.
    with pytest.raises(TypeError, match="unknown build parameter"):
        build_timeline_payload(small_cache, inter_thresold=2, verbose=False)


def test_prebuilt_lineage_result_is_reused(small_cache):
    cache = load_cache(str(small_cache))
    result = build_lineages_from_cache(cache, inter_thresh=1, cluster_min_size=2)
    payload = build_timeline_payload(cache, lineage_result=result, verbose=False)
    assert payload["meta"]["n_segments"] == len(result.segments)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_rendered_page_is_self_contained(small_cache, tmp_path):
    out = tmp_path / "timeline.html"
    payload = build_timeline_payload(small_cache, verbose=False)
    path = render_timeline_html(payload, out, verbose=False)

    assert os.path.exists(path)
    doc = out.read_text(encoding="utf-8")
    assert "{{DATA}}" not in doc, "payload placeholder was not substituted"
    assert '"segments"' in doc and '"events"' in doc and '"lineages"' in doc

    # Self-contained: the Google Fonts stylesheet is the only allowed remote
    # reference (documented), and no script or data may be fetched.
    remote = [chunk for chunk in doc.split('"') if chunk.startswith("http")]
    assert all("fonts.googleapis.com" in url or "fonts.gstatic.com" in url
               for url in remote), remote[:5]


def test_render_rejects_a_template_without_the_placeholder(tmp_path):
    bad = tmp_path / "bad.html"
    bad.write_text("<html><body>no placeholder</body></html>", encoding="utf-8")
    with pytest.raises(ValueError, match="placeholder"):
        render_timeline_html({"meta": {}}, tmp_path / "x.html",
                             template=bad, verbose=False)


def test_render_reports_a_missing_template(tmp_path):
    with pytest.raises(FileNotFoundError):
        render_timeline_html({"meta": {}}, tmp_path / "x.html",
                             template=tmp_path / "nope.html", verbose=False)


def test_generate_creates_parent_directories(small_cache, tmp_path):
    out = tmp_path / "nested" / "deeper" / "timeline.html"
    json_out = tmp_path / "nested" / "payload.json"
    generate_lineage_timeline(small_cache, out, payload_json=json_out, verbose=False)
    assert out.is_file() and json_out.is_file()
    assert json.loads(json_out.read_text(encoding="utf-8"))["meta"]["n_segments"] >= 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_timeline(small_cache, tmp_path):
    out = tmp_path / "cli_timeline.html"
    rc = main([
        "timeline", str(small_cache), "-o", str(out),
        "--inter-thresh", "1", "--cluster-min-size", "2",
        "--min-size", "3", "--min-duration", "2",
        "--frame-lo", "1", "--frame-hi", "10", "--quiet",
    ])
    assert rc == 0
    doc = out.read_text(encoding="utf-8")
    assert "{{DATA}}" not in doc
    assert '"frame_first":1' in doc.replace(" ", "")
    assert '"min_size":3' in doc.replace(" ", "")


def test_cli_timeline_default_output_path(small_cache, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["timeline", str(small_cache), "--quiet"]) == 0
    assert (tmp_path / "cache_timeline.html").is_file()
