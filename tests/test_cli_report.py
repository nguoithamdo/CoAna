# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests for the ``coana`` command-line interface and the HTML report module."""

from __future__ import annotations

import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from oligomer_analysis.cache_io import load_cache
from oligomer_analysis.cli import _parse_frames_spec, main
from oligomer_analysis.report import generate_html_report


# --------------------------------------------------------------------------- #
# Synthetic aggregating trajectory: monomers drift together over time so the
# report has real clusters, events and lineages to show.
# --------------------------------------------------------------------------- #
def _write_aggregating_pdb(path, n_frames=12, n_monomers=6, beads=2):
    lines = []
    rng = np.random.RandomState(11)
    jitter = rng.rand(n_monomers, beads, 2) * 0.5
    for fr in range(n_frames):
        lines.append(f"MODEL   {fr + 1:6d}\n")
        serial = 1
        shrink = max(0.15, 1.0 - 0.08 * fr)  # monomers approach each other
        for m in range(n_monomers):
            for b in range(beads):
                x = m * 6.0 * shrink + b * 0.8
                y = jitter[m, b, 0]
                z = jitter[m, b, 1]
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
# CLI
# --------------------------------------------------------------------------- #
def test_parse_frames_spec():
    assert _parse_frames_spec("all") is None
    assert _parse_frames_spec("0:5") == [0, 1, 2, 3, 4]
    assert _parse_frames_spec("0:10:3") == [0, 3, 6, 9]
    assert _parse_frames_spec("3,1,7") == [3, 1, 7]
    assert _parse_frames_spec("42") == [42]
    with pytest.raises(SystemExit):
        _parse_frames_spec("5:5")


def test_cli_precompute_and_info(small_cache, capsys):
    cache = load_cache(str(small_cache))
    assert len(cache.frames) == 12
    assert cache.meta["nseg"] == 6

    assert main(["info", str(small_cache)]) == 0
    out = capsys.readouterr().out
    assert "6 monomers" in out and "frames : 12" in out


def test_cli_index(tmp_path):
    pdb = tmp_path / "t.pdb"
    _write_aggregating_pdb(pdb, n_frames=4)
    assert main(["index", str(pdb)]) == 0
    assert os.path.exists(str(pdb) + ".coanapure-index.npz")


def test_cli_version(capsys):
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert "coana 3." in out and "CoAnaPure" in out  # renamed in 3.6.0


def test_cli_frames_subset_and_selection(tmp_path):
    pdb = tmp_path / "t.pdb"
    _write_aggregating_pdb(pdb, n_frames=6)
    out = tmp_path / "c.npz"
    rc = main([
        "precompute", "--allow-no-box", "--traj", str(pdb), "--frames", "1:6:2",
        "--cutoff", "2.5", "--intra-exclude", "0",
        "--segment-mode", "atom_blocks", "--atoms-per-monomer", "2",
        "--n-monomers", "6", "--selection", "resname ALA",
        "-o", str(out), "--quiet",
    ])
    assert rc == 0
    cache = load_cache(str(out))
    assert list(cache.frames) == [1, 3, 5]
    assert cache.meta["selection"] == "resname ALA"


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def test_report_python_api(small_cache, tmp_path):
    out_html = tmp_path / "report.html"
    path = generate_html_report(
        str(small_cache), out_html, title="Synthetic aggregation", verbose=False
    )
    assert os.path.exists(path)
    doc = out_html.read_text(encoding="utf-8")

    # structure
    assert "<title>Synthetic aggregation</title>" in doc
    for anchor in ("config", "methods", "evolution", "populations", "cmaps", "repro", "caps"):
        assert f'id="{anchor}"' in doc, anchor
    # figures are embedded, not referenced
    assert doc.count("data:image/png;base64,") >= 6
    # capabilities appendix names the engine features
    for feat in ("Checkpoint / resume", "Atom selection", "Parallel execution",
                 "Trajectory formats", "Validation"):
        assert feat in doc, feat
    # methods reflect the run configuration
    assert "2.5" in doc  # cutoff
    # self-contained: no external resources
    assert 'src="http' not in doc and "@import" not in doc


def test_report_includes_lineage_section(small_cache, tmp_path):
    out_html = tmp_path / "rep.html"
    generate_html_report(str(small_cache), out_html, verbose=False)
    doc = out_html.read_text(encoding="utf-8")
    # the aggregating trajectory must produce a lineage section (not the error note)
    assert 'id="lineage"' in doc
    assert "Lineages tracked" in doc


def test_report_no_lineage_flag(small_cache, tmp_path):
    out_html = tmp_path / "rep2.html"
    generate_html_report(str(small_cache), out_html, include_lineage=False, verbose=False)
    doc = out_html.read_text(encoding="utf-8")
    assert 'id="lineage"' not in doc
    assert 'id="caps"' in doc


def test_report_cli(small_cache, tmp_path):
    out_html = tmp_path / "cli_report.html"
    rc = main([
        "report", str(small_cache), "-o", str(out_html),
        "--title", "CLI report", "--max-size", "6", "--quiet",
    ])
    assert rc == 0
    doc = out_html.read_text(encoding="utf-8")
    assert "<title>CLI report</title>" in doc


def test_report_default_output_path(small_cache):
    path = generate_html_report(str(small_cache), verbose=False)
    assert path.endswith("cache_report.html")
    assert os.path.exists(path)


def test_report_counts_on_pathway_lineages(small_cache, tmp_path):
    """The tile used to compare against "on_pathway" and always showed 0."""
    import re

    from oligomer_analysis.lineage import build_lineages_from_cache
    from oligomer_analysis.models import TargetStateConfig

    cache = load_cache(str(small_cache))
    target = TargetStateConfig(min_size=3, min_duration=2)
    result = build_lineages_from_cache(cache, inter_thresh=1, target_config=target)
    n_on = sum(1 for lin in result.lineages if lin.class_label == "on-pathway")
    assert n_on > 0
    out = tmp_path / "r.html"
    generate_html_report(cache, out, inter_thresh=1, target_config=target, verbose=False)
    page = out.read_text(encoding="utf-8")
    tile = re.search(r'<div class="v">(\d+)</div><div class="k">On-pathway lineages</div>', page)
    assert tile is not None and int(tile.group(1)) == n_on
