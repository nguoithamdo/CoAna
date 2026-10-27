# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests for the v3.4 feature set.

Every expected number below is derived from the construction of the synthetic
system in :mod:`synthetic`, never from the code under test.  The two fixtures:

* ``two_species_trajectory`` -- three frames whose contact matrices can be
  written down by hand (frame 1: 100 inter-chain residue contacts, of which 12
  are salt bridges).
* ``growth_trajectory`` -- forty frames in which one assembly nucleates and
  grows on :data:`synthetic.GROWTH_SCHEDULE`; one B chain is incorporated and
  one is catalytic by construction.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from synthetic import (  # noqa: E402
    GROWTH_SCHEDULE,
    cache_from_cmaps,
    growth_trajectory,
    two_species_trajectory,
)

from oligomer_analysis import (  # noqa: E402
    CONTACT_CLASSES,
    SpeciesMap,
    assess_convergence,
    attach_species_compositions,
    build_lineages_from_cache,
    classify_residues,
    compare_caches,
    contact_class_summary,
    exchange_statistics,
    filter_lineages,
    free_monomer_concentration,
    group_summary,
    hotspot_block_dataframe,
    integrated_autocorrelation_time,
    lineage_summary_dataframe,
    lineage_warnings,
    load_cache,
    mechanism_summary,
    monomer_atom_counts_from_spec,
    parse_species_spec,
    partition_summary,
    region_contact_table,
    residence_events,
    residue_hotspot_matrix,
    salt_bridge_timeseries,
    subset_cache_by_indices,
    sweep_verdict,
    target_size_sweep,
    transition_counts,
    transition_matrix,
    write_lineage_gexf,
    write_transition_gexf,
)
from oligomer_analysis.cache_io import save_contact_cache  # noqa: E402
from oligomer_analysis.compare import summarize_cache  # noqa: E402
from oligomer_analysis.coanapure import precompute_cache_pure  # noqa: E402
from oligomer_analysis.graph_tools import _clusters_bfs, clusters_from_cmap  # noqa: E402
from oligomer_analysis.models import ContactCache, PrecomputeConfig, TargetStateConfig  # noqa: E402

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _precompute(gro: str, n_frames: int, spec: str, out, *, hotspot_blocks=3):
    cfg = PrecomputeConfig(
        topology_file=None,
        traj_file=gro,
        frames=list(range(n_frames)),
        cutoff_contact=6.0,
        intra_exclude=2,
        species_spec=spec,
        contact_classes=True,
        hotspot_blocks=hotspot_blocks,
    )
    precompute_cache_pure(cfg, str(out), verbose=False)
    return load_cache(out)


@pytest.fixture(scope="module")
def two_species(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("two_species")
    info = two_species_trajectory(tmp, n_a=3, n_b=2)
    cache = _precompute(info["gro"], info["n_frames"], info["species_spec"], tmp / "c.npz")
    return info, cache, tmp / "c.npz"


@pytest.fixture(scope="module")
def growth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("growth")
    info = growth_trajectory(tmp)
    cache = _precompute(
        info["gro"], info["n_frames"], info["species_spec"], tmp / "growth.npz", hotspot_blocks=4
    )
    return info, cache, tmp / "growth.npz"


GROWTH_TARGET = TargetStateConfig(min_size=4, min_duration=5)


@pytest.fixture(scope="module")
def growth_lineages(growth):
    _info, cache, _path = growth
    smap = SpeciesMap.from_cache(cache)
    result = build_lineages_from_cache(
        cache, inter_thresh=1, cluster_min_size=2, target_config=GROWTH_TARGET, species_map=smap
    )
    return cache, smap, result


# ---------------------------------------------------------------------------
# #7 species spec grammar
# ---------------------------------------------------------------------------


def test_parse_species_spec_grammar():
    assert parse_species_spec("aSyn:295x50,Ab40:92x10") == [("aSyn", 295, 50), ("Ab40", 92, 10)]
    assert parse_species_spec(" A:4x6 , B:2x2 ") == [("A", 4, 6), ("B", 2, 2)]
    assert monomer_atom_counts_from_spec("A:4x2,B:2x1") == [4, 4, 2]


@pytest.mark.parametrize("bad", ["", "A:4", "A:4x", "A4x6", "A:0x6", "A:4x0", "A:4x6,A:2x2"])
def test_parse_species_spec_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_species_spec(bad)


# ---------------------------------------------------------------------------
# #2 Stage A extras: chemistry and residue hotspots are conserved
# ---------------------------------------------------------------------------


def test_stage_a_records_species_and_classes(two_species):
    info, cache, _ = two_species
    names = [d["name"] for d in cache.meta["species"]]
    assert names == ["A", "B"]
    assert [d["count"] for d in cache.meta["species"]] == [3, 2]
    assert [d["n_residues"] for d in cache.meta["species"]] == [4, 2]
    census = cache.meta["contact_class_census"]
    # ALA + 2 LEU apolar, ASP + 2 GLU anion, LYS cation, PHE aromatic (x3 A chains)
    assert census["apolar"] == 3 + 2
    assert census["anion"] == 3 + 2
    assert census["cation"] == 3
    assert census["aromatic"] == 3


def test_class_matrix_conserves_contacts(two_species):
    _info, cache, _ = two_species
    for f in range(cache.cmaps.shape[0]):
        raw = int(np.triu(cache.cmaps[f], 1).sum())
        cls = int(np.triu(cache.class_cmaps[f]).sum())
        assert cls == raw, f"frame {f}: class matrix {cls} != cmap {raw}"
        assert np.array_equal(cache.class_cmaps[f], cache.class_cmaps[f].T)
    # every chain pair is within the cutoff on the contact line, so the count
    # is just residues x residues over every chain pair
    assert int(np.triu(cache.cmaps[0], 1).sum()) == 0
    assert int(np.triu(cache.cmaps[1], 1).sum()) == 3 * 16 + 6 * 8 + 1 * 4
    assert int(np.triu(cache.cmaps[2], 1).sum()) == 3 * 16


def test_salt_bridges_by_hand(two_species):
    _info, cache, _ = two_species
    # frame 1: ASP-LYS both ways in each of 3 AA pairs, LYS-GLU in each of 6 AB pairs
    assert list(salt_bridge_timeseries(cache)) == [0, 6 + 6, 6]


def test_hotspot_tensor_conserves_contacts(two_species):
    _info, cache, _ = two_species
    hot = np.asarray(cache.hotspot_maps)
    assert hot.shape[0] == 3  # blocks
    assert hot.shape[1:3] == (2, 2)
    assert int(hot.sum()) // 2 == int(np.triu(cache.cmaps, 1).sum())
    total = hot.sum(axis=0)
    assert np.array_equal(total[0, 1], total[1, 0].T)


def test_contact_class_summary_channels(two_species):
    _info, cache, _ = two_species
    summary = contact_class_summary(cache, last_fraction=1.0)
    assert np.isclose(summary["fraction_late"].sum(), 1.0)
    row = summary[(summary.class_a == "cation") & (summary.class_b == "anion")]
    assert len(row) == 1 and row["channel"].iloc[0] == "salt bridge"
    assert np.isclose(row["mean_all"].iloc[0], (0 + 12 + 6) / 3)


def test_classify_residues_never_raises_on_solvent():
    idx = classify_residues(["ALA", "W", "ION", "PEG", "LYS", "GLU", "PHE"])
    labels = [CONTACT_CLASSES[int(i)] for i in idx]
    assert labels == ["apolar", "other", "other", "other", "cation", "anion", "aromatic"]


# ---------------------------------------------------------------------------
# cache persistence of the new fields
# ---------------------------------------------------------------------------


def test_cache_round_trip_and_subset(two_species, tmp_path):
    _info, cache, _ = two_species
    out = tmp_path / "rt.npz"
    save_contact_cache(cache, out, save_csv_stats=False)
    back = load_cache(out)
    assert np.array_equal(back.class_cmaps, cache.class_cmaps)
    assert np.array_equal(back.hotspot_maps, cache.hotspot_maps)
    assert back.class_names == list(cache.class_names)
    assert back.species_names == ["A", "B"]

    sub = subset_cache_by_indices(cache, [1, 2])
    assert sub.class_cmaps.shape[0] == 2
    assert sub.hotspot_maps is None
    assert sub.meta.get("hotspot_dropped_on_subset") is True


def test_species_map_from_cache_uses_recorded_names(two_species):
    _info, cache, _ = two_species
    smap = SpeciesMap.from_cache(cache)
    assert list(smap.names) == ["A", "B"]
    assert list(smap.seg_species) == [0, 0, 0, 1, 1]


# ---------------------------------------------------------------------------
# #6 clustering speed-up keeps the reference answer
# ---------------------------------------------------------------------------


def test_csgraph_clusters_match_bfs_reference():
    rng = np.random.default_rng(7)
    for _ in range(25):
        n = int(rng.integers(2, 40))
        cmap = rng.integers(0, 3, size=(n, n)).astype(np.int32)
        cmap = np.triu(cmap, 1)
        cmap = cmap + cmap.T
        fast = clusters_from_cmap(cmap, 1)
        ref = _clusters_bfs(cmap >= 1)
        assert fast == ref  # identical content *and* order


# ---------------------------------------------------------------------------
# #7 partition coefficient and free concentration
# ---------------------------------------------------------------------------


def test_partition_by_hand(growth):
    _info, cache, _ = growth
    smap = SpeciesMap.from_cache(cache)
    summ = partition_summary(cache, smap, last_fraction=0.25).set_index("species")
    # last 10 frames: assembly is 6A + 1B out of 6A + 2B
    assert np.isclose(summ.loc["A", "K_mean"], (6 / 7) / (6 / 8))
    assert np.isclose(summ.loc["B", "K_mean"], (1 / 7) / (2 / 8))
    assert summ.loc["A", "K_sd"] == 0.0

    free = free_monomer_concentration(cache, smap, last_fraction=0.25).set_index("species")
    assert free.loc["A", "n_free_mean"] == 0.0
    assert free.loc["B", "n_free_mean"] == 1.0
    assert bool(free.loc["A", "dilute_phase"]) is False
    assert bool(free.loc["B", "dilute_phase"]) is True
    # one chain in a 50 nm box: 1 / (N_A * 1.25e-19 L)
    expected_uM = 1.0 / (6.02214076e23 * 125000.0 * 1e-24) * 1e6
    assert np.isclose(free.loc["B", "c_free_uM"], expected_uM, rtol=1e-4)


# ---------------------------------------------------------------------------
# #1 species x lineage join
# ---------------------------------------------------------------------------


def test_lineage_carries_species_composition(growth_lineages):
    _cache, _smap, result = growth_lineages
    assert len(result.lineages) == 1
    lin = result.lineages[0]
    assert lin.classification == "on-pathway"
    assert list(lin.composition) == [6, 2]  # every chain that ever belonged
    assert list(lin.peak_composition) == [6, 1]  # makeup of the largest state
    # states at frames 5..39 (35); mixed from frame 15 on (25)
    assert np.isclose(lin.hetero_fraction, 25 / 35)

    df = lineage_summary_dataframe(result)
    assert bool(df["is_hetero"].iloc[0]) is True
    assert int(df["peak_A"].iloc[0]) == 6 and int(df["peak_B"].iloc[0]) == 1
    assert int(df["n_species_present"].iloc[0]) == 2


def test_attach_after_the_fact_matches_build_time(growth):
    _info, cache, _ = growth
    smap = SpeciesMap.from_cache(cache)
    plain = build_lineages_from_cache(cache, inter_thresh=1, target_config=GROWTH_TARGET)
    assert plain.lineages[0].composition is None
    attach_species_compositions(plain, smap)
    assert list(plain.lineages[0].peak_composition) == [6, 1]


def test_one_species_table_keeps_original_columns(growth):
    _info, cache, _ = growth
    plain = build_lineages_from_cache(cache, inter_thresh=1, target_config=GROWTH_TARGET)
    df = lineage_summary_dataframe(plain)
    assert "is_hetero" not in df.columns and "peak_A" not in df.columns


# ---------------------------------------------------------------------------
# #3 guardrails: warnings, persistence filter, target sweep
# ---------------------------------------------------------------------------


def _churning_dimer_cache(nseg=20, n_frames=60):
    """A different transient dimer every frame -- the noise trap.

    Consecutive frames use disjoint pairs, so no dimer can be linked to the
    next frame's and every lineage lasts exactly one frame.
    """
    cmaps = np.zeros((n_frames, nseg, nseg), dtype=np.int32)
    for f in range(n_frames):
        a = (2 * f) % nseg
        b = (2 * f + 1) % nseg
        cmaps[f, a, b] = cmaps[f, b, a] = 5
    return cache_from_cmaps(cmaps)


def test_lineage_warnings_fire_on_dimer_churn():
    cache = _churning_dimer_cache()
    result = build_lineages_from_cache(
        cache, inter_thresh=1, cluster_min_size=2, target_config=TargetStateConfig(min_size=25)
    )
    assert len(result.lineages) > 2 * 20
    msgs = lineage_warnings(result, nseg=20, cluster_min_size=2)
    joined = " ".join(msgs).lower()
    assert "lineages" in joined and "chains" in joined  # more lineages than chains
    assert "target" in joined  # target size >= nseg is unreachable
    assert "duration" in joined or "frames" in joined  # median duration tiny
    assert result.warnings == msgs


def test_filter_lineages_is_self_consistent():
    cache = _churning_dimer_cache()
    result = build_lineages_from_cache(cache, inter_thresh=1, cluster_min_size=2)
    kept = filter_lineages(result, min_duration=2)
    assert len(kept.lineages) == 0
    assert len(kept.segments) == 0 and len(kept.events) == 0
    kept_all = filter_lineages(result, min_duration=1)
    assert len(kept_all.lineages) == len(result.lineages)
    ids = {lin.lineage_id for lin in kept_all.lineages}
    assert all(seg.lineage_id in ids for seg in kept_all.segments)


def test_target_size_sweep_and_verdict(growth):
    _info, cache, _ = growth
    sweep = target_size_sweep(
        cache, [2, 4, 7, 8], inter_thresh=1, target_config=GROWTH_TARGET
    )
    assert list(sweep["target_size"]) == [2, 4, 7, 8]
    assert list(sweep["on_fraction"]) == [1.0, 1.0, 1.0, 0.0]
    assert "monoton" in sweep_verdict(sweep)
    stable = sweep[sweep.target_size <= 7]
    assert "stable" in sweep_verdict(stable)


# ---------------------------------------------------------------------------
# #4 convergence
# ---------------------------------------------------------------------------


def test_autocorrelation_time_constant_and_ar1():
    assert integrated_autocorrelation_time(np.ones(50)) == 0.5
    rng = np.random.default_rng(1)
    x = np.zeros(20000)
    phi = 0.9
    for i in range(1, x.size):
        x[i] = phi * x[i - 1] + rng.normal()
    tau = integrated_autocorrelation_time(x)
    # exact integrated time for AR(1) is (1+phi)/(2(1-phi)) = 9.5
    assert 7.0 < tau < 12.0


def test_convergence_step_change_is_not_converged():
    step = np.array([0] * 5 + [1] * 35, dtype=float)
    rep = assess_convergence(None, "x", series=step)
    assert rep.converged is False
    assert np.isinf(rep.geweke_z) and rep.geweke_z < 0
    flat = assess_convergence(None, "x", series=np.full(40, 3.0))
    assert flat.converged is True and flat.geweke_z == 0.0
    assert np.isclose(flat.relative_drift, 0.0, atol=1e-12)
    empty = assess_convergence(None, "x", series=np.zeros(40))
    assert empty.converged is True


def test_convergence_corrected_sem_exceeds_naive():
    rng = np.random.default_rng(3)
    x = np.zeros(4000)
    for i in range(1, x.size):
        x[i] = 0.8 * x[i - 1] + rng.normal()
    rep = assess_convergence(None, "x", series=x)
    assert rep.sem_corrected > 2 * rep.sem_naive


def test_growth_run_reports_unconverged(growth):
    _info, cache, _ = growth
    rep = assess_convergence(cache, "largest_cluster")
    assert rep.converged is False and rep.n_frames == 40


# ---------------------------------------------------------------------------
# #5 comparison and pooling guard
# ---------------------------------------------------------------------------


def test_compare_caches_and_group_summary(growth):
    _info, _cache, path = growth
    labels = ["1.2a_m", "1.2a_pp", "1.2b_m"]

    def desc(label):
        cond, ff = label.split("_")
        return {"cond": cond, "ff": ff}

    df = compare_caches(
        [path, path, path], labels=labels, descriptor_fn=desc,
        target_config=GROWTH_TARGET, last_fraction=0.25,
    )
    assert list(df["label"]) == labels
    assert list(df["on_fraction"]) == [1.0, 1.0, 1.0]
    assert list(df["largest_cluster_late"]) == [7.0, 7.0, 7.0]
    assert np.allclose(df["K_B"], (1 / 7) / (2 / 8))

    by_cond = group_summary(df, ["cond"]).set_index("cond")
    assert by_cond.loc["1.2a", "n_runs"] == 2
    assert by_cond.loc["1.2a", "mixed_columns"] == "ff"  # the _m/_pp pooling trap
    assert by_cond.loc["1.2b", "mixed_columns"] == ""
    both = group_summary(df, ["cond", "ff"])
    assert (both["mixed_columns"] == "").all()


def test_compare_skips_unreadable_when_asked(growth, tmp_path):
    _info, _cache, path = growth
    bad = tmp_path / "missing.npz"
    with pytest.raises(Exception):
        compare_caches([path, bad], include_lineage=False)
    df = compare_caches([path, bad], include_lineage=False, on_error="skip")
    assert df["error"].notna().sum() == 1


# ---------------------------------------------------------------------------
# #8 residue-resolved hotspots
# ---------------------------------------------------------------------------


def test_residue_hotspots_by_hand(growth):
    info, cache, _ = growth
    m = residue_hotspot_matrix(cache, "A", "B")
    assert m.shape == (4, 2)
    # every A residue touches every B residue on the line:
    # frames 15-19: 3A x 1B, 20-24: 4A x 2B, 25-29: 4A x 1B, 30-39: 6A x 1B
    expected_pairs = 5 * 3 * 1 + 5 * 4 * 2 + 5 * 4 * 1 + 10 * 6 * 1
    assert m.sum() == expected_pairs * 4 * 2
    assert np.array_equal(m, residue_hotspot_matrix(cache, "B", "A").T)

    regions = region_contact_table(cache, "A", "B", {"N": (1, 2), "C": (3, 4)})
    assert np.allclose(regions["contact_share"], 0.5)
    assert np.allclose(regions["enrichment"], 1.0)

    blocks = hotspot_block_dataframe(cache, "A", "B")
    assert list(blocks["contacts"]) == [0.0, 5 * 3 * 8, (5 * 8 + 5 * 4) * 8, 10 * 6 * 8]


def test_residue_hotspots_require_stage_a_option():
    cache = _churning_dimer_cache()
    with pytest.raises(ValueError, match="hotspot-blocks"):
        residue_hotspot_matrix(cache, 0, 0)


# ---------------------------------------------------------------------------
# #9 mechanism
# ---------------------------------------------------------------------------


def test_residence_events_classify_by_construction(growth_lineages):
    _cache, _smap, result = growth_lineages
    events = residence_events(result, guest_species="B", commit_size=4)
    by_chain = {e.chain: e for e in events}
    assert set(by_chain) == {6, 7}

    b0 = by_chain[6]
    assert (b0.start_idx, b0.end_idx, b0.duration) == (15, 39, 25)
    assert (b0.host_at_entry, b0.host_at_exit, b0.host_growth_during) == (3, 6, 3)
    assert b0.present_at_commit and b0.survives_to_lineage_end
    assert b0.classification == "incorporated"

    b1 = by_chain[7]
    assert (b1.start_idx, b1.end_idx, b1.duration) == (20, 24, 5)
    assert b1.host_at_entry == 4 and b1.host_growth_after == 2.0
    assert b1.present_at_commit and not b1.survives_to_lineage_end
    assert b1.classification == "catalytic"

    summ = mechanism_summary(events, n_guest_chains=2).set_index("classification")
    assert np.isclose(summ.loc["incorporated", "frame_share"], 25 / 30)
    assert np.isclose(summ.loc["catalytic", "visit_share"], 0.5)

    ex = exchange_statistics(events, n_frames=40, n_guest_chains=2, frame_interval_ns=10.0)
    assert np.isclose(ex["bound_fraction"], 30 / 80)
    assert np.isclose(ex["exchange_rate_per_chain_per_frame"], 2 / 80)
    assert np.isclose(ex["exchange_rate_per_chain_per_us"], 2 / 80 * 100)
    assert np.isclose(ex["mean_residence_ns"], 150.0)


def test_majority_species_control(growth_lineages):
    _cache, _smap, result = growth_lineages
    events = residence_events(result, guest_species="A", host_species="B", commit_size=1)
    labels = sorted(e.classification for e in events)
    # A0-A2 were there before B0 arrived (commit); A3-A5 joined afterwards
    assert labels == ["incorporated"] * 3 + ["passenger"] * 3


def test_mechanism_needs_species_map(growth):
    _info, cache, _ = growth
    plain = build_lineages_from_cache(cache, inter_thresh=1, target_config=GROWTH_TARGET)
    with pytest.raises(ValueError, match="species map"):
        residence_events(plain, guest_species="B")


# ---------------------------------------------------------------------------
# #10 transition network and GEXF
# ---------------------------------------------------------------------------


def test_transition_counts_follow_schedule(growth):
    _info, cache, _ = growth
    smap = SpeciesMap.from_cache(cache)
    names, counts = transition_counts(cache, species_map=smap)
    idx = {n: i for i, n in enumerate(names)}
    expected_path = ["solution", "2(2A)", "3(3A)", "4(3A+1B)", "6(4A+2B)", "5(4A+1B)", "7(6A+1B)"]
    assert set(expected_path) <= set(names)
    for a, b in zip(expected_path, expected_path[1:]):
        assert counts[idx[a], idx[b]] == 1
    assert counts[idx["2(2A)"], idx["2(2A)"]] == 4
    assert counts[idx["7(6A+1B)"], idx["7(6A+1B)"]] == 9
    assert counts.sum() == 35  # 40 frames minus the 5 empty ones

    P = transition_matrix(counts)
    rows = P.sum(axis=1)
    assert np.all((np.isclose(rows, 1.0)) | (rows == 0.0))
    empty = transition_matrix(np.array([[0, 0], [1, 0]]))
    assert np.array_equal(empty[0], [0.0, 0.0])  # unvisited row stays zero


def test_gexf_files_are_well_formed(growth_lineages, tmp_path):
    cache, smap, result = growth_lineages
    lin_path = tmp_path / "lin.gexf"
    write_lineage_gexf(result, lin_path)
    root = ET.parse(lin_path).getroot()
    ns = {"g": root.tag.split("}")[0].strip("{")}
    assert len(root.findall(".//g:node", ns)) == 1
    assert len(root.findall(".//g:edge", ns)) == 0

    names, counts = transition_counts(cache, species_map=smap)
    tr_path = tmp_path / "tr.gexf"
    write_transition_gexf(names, counts, tr_path, include_self=False)
    root = ET.parse(tr_path).getroot()
    assert len(root.findall(".//g:node", ns)) == 7
    assert len(root.findall(".//g:edge", ns)) == 6


# ---------------------------------------------------------------------------
# CLI: every new subcommand runs against the growth cache
# ---------------------------------------------------------------------------


def _coana(*args, cwd):
    proc = subprocess.run(
        [sys.executable, "-m", "oligomer_analysis.cli", *map(str, args)],
        cwd=str(cwd), capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_cli_subcommands_end_to_end(growth, tmp_path):
    _info, _cache, path = growth
    cache = str(path)

    rc, out = _coana("info", cache, cwd=tmp_path)
    assert rc == 0 and "A:4x6, B:2x2" in out and "hotspots in 4 block" in out

    rc, out = _coana("partition", cache, "--last-fraction", "0.25", cwd=tmp_path)
    assert rc == 0 and "1.142857" in out and "0.571429" in out

    rc, out = _coana("chemistry", cache, cwd=tmp_path)
    assert rc == 0 and "salt bridge" in out

    rc, out = _coana("hotspots", cache, "--species-a", "A", "--species-b", "B",
                     "--regions", "N:1-2,C:3-4", cwd=tmp_path)
    assert rc == 0 and "enrichment" in out

    rc, out = _coana("converged", cache, "--all", cwd=tmp_path)
    assert rc == 1 and "False" in out  # a growth run is not converged

    rc, out = _coana("lineage", cache, "--min-size", "4", "--min-duration", "5",
                     "--sweep", "2,4,8", "-o", tmp_path / "lin.csv", cwd=tmp_path)
    assert rc == 0 and "on-pathway" in out and "mixed" in out
    assert (tmp_path / "lin.csv").exists()

    rc, out = _coana("mechanism", cache, "--guest", "B", "--min-size", "4",
                     "--min-duration", "5", "--commit-size", "4", cwd=tmp_path)
    assert rc == 0 and "incorporated" in out and "catalytic" in out

    for sub in ("r1_m", "r1_pp"):
        d = tmp_path / sub
        d.mkdir()
        (d / "cache.npz").write_bytes(path.read_bytes())
    rc, out = _coana("compare", tmp_path / "r1_m" / "cache.npz", tmp_path / "r1_pp" / "cache.npz",
                     "--descriptor-regex", "(?P<cond>r[0-9]+)_(?P<ff>[a-z]+)",
                     "--group-by", "cond", "--no-lineage", cwd=tmp_path)
    assert rc == 0 and "varies in ff" in out

    rc, out = _coana("export", cache, "--what", "both", "--composition-states",
                     "--min-size", "4", "--out-dir", tmp_path / "gexf", cwd=tmp_path)
    assert rc == 0
    assert (tmp_path / "gexf" / "growth_transitions.gexf").exists()
    assert (tmp_path / "gexf" / "growth_lineage.gexf").exists()


def test_sweep_verdict_uninformative_when_no_lineage_reaches_smallest_target():
    """A sweep whose every target exceeds the largest lineage says so instead
    of calling a row of zeros 'stable' (seen on a 0.5 us smoke test)."""
    import pandas as pd

    sweep = pd.DataFrame(
        {
            "target_size": [6, 8, 10],
            "n_lineages": [14, 14, 14],
            "n_on_pathway": [0, 0, 0],
            "n_off_pathway": [14, 14, 14],
            "n_uncertain": [0, 0, 0],
            "on_fraction": [0.0, 0.0, 0.0],
            "max_size_observed": [3, 3, 3],
            "min_duration_filter": [50, 50, 50],
        }
    )
    verdict = sweep_verdict(sweep)
    assert "uninformative" in verdict and "largest observed 3" in verdict
    # Once the smallest target is reachable the ordinary reading applies.
    sweep["max_size_observed"] = 7
    assert "stable" in sweep_verdict(sweep)


def test_cli_compare_grouped_also_writes_per_run_rows(growth, tmp_path):
    """A grouped campaign table is only defensible next to its per-run rows,
    so -o writes both instead of replacing one with the other."""
    _info, _cache, path = growth
    runs = tmp_path / "runs"
    for name in ("1.2a_m", "1.2a_pp"):
        d = runs / name
        d.mkdir(parents=True)
        shutil.copy(str(path), str(d / "cache.npz"))

    out = tmp_path / "campaign.csv"
    rc, _stdout = _coana(
        "compare",
        str(runs / "1.2a_m" / "cache.npz"),
        str(runs / "1.2a_pp" / "cache.npz"),
        "--descriptor-regex", r"runs[/\\](?P<cond>[0-9.]+[a-z]+)_(?P<ff>[a-z]+)[/\\]",
        "--group-by", "cond",
        "--min-size", "4",
        "--min-duration", "5",
        "-o", out,
        cwd=tmp_path,
    )
    assert rc == 0
    grouped = pd.read_csv(out)
    per_run = pd.read_csv(tmp_path / "campaign_runs.csv")
    assert list(grouped["cond"]) == ["1.2a"] and int(grouped["n_runs"].iloc[0]) == 2
    # The pooling the grouped row hides is visible in the per-run rows.
    assert grouped["mixed_columns"].iloc[0] == "ff"
    assert len(per_run) == 2 and sorted(per_run["ff"]) == ["m", "pp"]


def test_group_summary_does_not_flag_integer_metrics_as_descriptors():
    """Only real descriptors belong in mixed_columns: an integer result column
    that differs between replicates is not evidence of pooling (it flagged
    every group on the 60-run campaign until this was fixed)."""
    df = pd.DataFrame(
        {
            "label": ["r1", "r2", "r3", "r4"],
            "cond": ["1.2a", "1.2a", "1.2b", "1.2b"],
            "ff": ["m", "pp", "m", "m"],
            "stage": ["prod5us", "prod5us", "prod5us", "prod1us"],
            "n_on_pathway": [3, 7, 5, 5],          # integer metric, varies
            "n_frames": [5001, 5001, 5001, 1001],  # integer metric, varies
            "largest_cluster_late": [10.0, 12.0, 20.0, 22.0],
        }
    )
    grouped = group_summary(df, ["cond"], metrics=["largest_cluster_late"])
    mixed = dict(zip(grouped["cond"], grouped["mixed_columns"]))
    # 1.2a genuinely pools two force fields; 1.2b pools two trajectory lengths.
    assert mixed["1.2a"] == "ff"
    assert mixed["1.2b"] == "stage"
    assert "n_on_pathway" not in " ".join(mixed.values())
    assert "n_frames" not in " ".join(mixed.values())


def test_cli_compare_honours_the_requested_target_size(growth, tmp_path):
    """`coana compare` accepted no target arguments and passed none down, so
    --min-size was silently ignored and every run was classified against the
    default target of 10. The growth fixture peaks at 7 chains, so a target of
    4 is reached and one of 8 is not -- if the flag is honoured."""
    _info, _cache, path = growth

    def on_fraction(min_size):
        out = tmp_path / f"cmp{min_size}.csv"
        rc, stdout = _coana(
            "compare", str(path),
            "--min-size", str(min_size), "--min-duration", "1",
            "--lineage-min-duration", "1", "--inter-thresh", "1",
            "-o", out, cwd=tmp_path,
        )
        assert rc == 0, stdout
        return float(pd.read_csv(out)["on_fraction"].iloc[0])

    assert on_fraction(4) > 0.0
    assert on_fraction(8) == 0.0


def test_cli_compare_labels_caches_that_share_a_file_name(growth, tmp_path):
    """Every campaign cache is named cache.npz and lives in a directory named
    for its condition. Labelling by stem alone made all runs indistinguishable
    and the descriptor regex -- matched against the label -- found nothing."""
    _info, _cache, path = growth
    for cond in ("1.2a_m", "1.2a_pp"):
        d = tmp_path / "runs" / cond
        d.mkdir(parents=True)
        shutil.copy(str(path), str(d / "cache.npz"))

    out = tmp_path / "cmp.csv"
    rc, _stdout = _coana(
        "compare",
        str(tmp_path / "runs" / "1.2a_m" / "cache.npz"),
        str(tmp_path / "runs" / "1.2a_pp" / "cache.npz"),
        "--descriptor-regex", r"runs[/\\](?P<cond>[0-9.]+[a-z]+)_(?P<ff>[a-z]+)[/\\]",
        "--no-lineage", "-o", out,
        cwd=tmp_path,
    )
    assert rc == 0
    df = pd.read_csv(out)
    # Directory-qualified, so the two runs are distinguishable...
    assert sorted(df["label"]) == ["1.2a_m/cache", "1.2a_pp/cache"]
    # ...and the regex, matched against the path, populated both columns.
    assert sorted(df["ff"]) == ["m", "pp"]
    assert set(df["cond"]) == {"1.2a"}


def test_cli_mechanism_names_the_host_species(growth, tmp_path):
    """The header said 'in host the majority species' rather than naming it, so
    the output did not record which species the classification was against."""
    _info, _cache, path = growth
    rc, out = _coana("mechanism", str(path), "--guest", "B", "--min-size", "4",
                     "--min-duration", "5", cwd=tmp_path)
    assert rc == 0
    assert "guest : B (2 chains)" in out
    assert "host  : A (6 chains" in out
    assert "most abundant other species" in out
    assert "the majority species" not in out


def test_partition_flags_single_species_as_degenerate(growth, tmp_path):
    """In a one-component box K compares a species with itself, so it is 1 by
    construction. Reporting "K_mean 1.000, K_sd 0.000" unqualified reads as a
    precise measurement of no enrichment; the table must say otherwise."""
    _info, cache, path = growth

    two = partition_summary(cache, SpeciesMap.from_cache(cache))
    assert not two["degenerate"].any()          # A and B present: K is meaningful

    one_meta = dict(cache.meta)
    one_meta["species"] = [{"name": "A", "n_atoms": 4, "count": 8}]
    one = ContactCache(
        frames=cache.frames, cmaps=cache.cmaps, nintra=cache.nintra,
        ninter=cache.ninter, ntotal=cache.ntotal, n_free=cache.n_free,
        n_bound=cache.n_bound, seg_centers=cache.seg_centers, boxes=cache.boxes,
        meta=one_meta,
    )
    summary = partition_summary(one, SpeciesMap.from_cache(one))
    assert bool(summary["degenerate"].all())
    assert abs(float(summary["K_mean"].iloc[0]) - 1.0) < 1e-12

    # The CLI must say so out loud, not only in a column nobody reads.
    one_path = tmp_path / "one_species.npz"
    save_contact_cache(one, one_path, save_csv_stats=False)
    rc, out = _coana("partition", str(one_path), cwd=tmp_path)
    assert rc == 0, out
    assert "identically 1" in out and "not a measurement" in out

    # ...and must not cry wolf on a genuine two-species box.
    rc, out_two = _coana("partition", str(path), cwd=tmp_path)
    assert rc == 0, out_two
    assert "identically 1" not in out_two


def _two_cluster_cmaps(n_frames=8):
    """5 host chains (A) + 2 guest chains (B), in two competing assemblies.

    cluster I  = A0, A1, B0, B1   -> 4 chains, 2 of them host
    cluster II = A2, A3, A4       -> 3 chains, 3 of them host
    A ranking by total chains picks I, a ranking by host chains picks II.
    """
    nseg = 7                       # A0..A4 = 0..4, B0..B1 = 5..6
    cmaps = np.zeros((n_frames, nseg, nseg), dtype=np.int32)
    for f in range(n_frames):
        for i, j in [(0, 1), (0, 5), (1, 5), (5, 6)]:      # cluster I
            cmaps[f, i, j] = cmaps[f, j, i] = 5
        for i, j in [(2, 3), (3, 4), (2, 4)]:              # cluster II
            cmaps[f, i, j] = cmaps[f, j, i] = 5
    return cmaps


def test_partition_reports_when_the_cluster_choice_made_the_answer(tmp_path):
    """K must not quietly depend on which cluster is called 'the assembly'.

    The guest is 2 of 7 chains, so x_bulk = 2/7. Inside cluster I it is 2 of 4,
    giving K = (2/4)/(2/7) = 1.75; inside cluster II it is absent, K = 0. Both
    numbers are correct for their own definition of largest, which is exactly
    why the table has to say that the two disagree.
    """
    meta = {"species": [{"name": "A", "n_atoms": 4, "count": 5},
                        {"name": "B", "n_atoms": 4, "count": 2}]}
    cache = cache_from_cmaps(_two_cluster_cmaps(), meta=meta)
    smap = SpeciesMap.from_cache(cache)

    summary = partition_summary(cache, smap).set_index("species")
    assert abs(float(summary.loc["B", "K_mean"]) - 1.75) < 1e-9

    # The bias, made visible: same system, ranking by the host species only.
    assert abs(float(summary.loc["B", "K_host_ranked"]) - 0.0) < 1e-9
    assert float(summary.loc["A", "K_host_ranked"]) > float(summary.loc["A", "K_mean"])
    assert float(summary.loc["B", "rank_agreement"]) == 0.0

    # A system with one unambiguous assembly must not be flagged.
    solid = np.zeros((8, 7, 7), dtype=np.int32)
    for f in range(8):
        for i in range(6):
            for j in range(i + 1, 6):
                solid[f, i, j] = solid[f, j, i] = 5
    agree = partition_summary(cache_from_cmaps(solid, meta=meta), smap)
    assert float(agree["rank_agreement"].iloc[0]) == 1.0
    assert np.allclose(agree["K_mean"], agree["K_host_ranked"])

    # ...and the CLI has to say so out loud, not only in a column.
    path = tmp_path / "two_clusters.npz"
    save_contact_cache(cache, path, save_csv_stats=False)
    rc, out = _coana("partition", str(path), cwd=tmp_path)
    assert rc == 0, out
    assert "K_host_ranked" in out
    assert "property of the definition" in out


def test_compare_table_carries_the_selection_rule_check(tmp_path):
    """A campaign table is where K gets quoted from, so it has to show whether
    the number depends on which cluster was called the assembly. The fixture is
    the same two competing assemblies: ranked by total the guest looks enriched
    (K = 1.75), ranked by host it is absent (K = 0)."""
    meta = {"species": [{"name": "A", "n_atoms": 4, "count": 5},
                        {"name": "B", "n_atoms": 4, "count": 2}]}
    path = tmp_path / "two_clusters.npz"
    save_contact_cache(cache_from_cmaps(_two_cluster_cmaps(), meta=meta), path,
                       save_csv_stats=False)

    df = compare_caches([str(path)], include_lineage=False)
    assert "K_B_host_ranked" in df.columns and "rank_agreement" in df.columns
    assert abs(float(df["K_B"].iloc[0]) - 1.75) < 1e-9
    assert abs(float(df["K_B_host_ranked"].iloc[0])) < 1e-9
    assert float(df["rank_agreement"].iloc[0]) == 0.0


def _two_residue_pdb(tmp_path):
    """Two 4-atom chains: centroids 9 A apart, nearest atoms 3 A apart."""
    lines = ["CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1"]
    serial = 0
    for chain, xs in (("A", (0.0, 2.0, 4.0, 6.0)), ("B", (9.0, 11.0, 13.0, 15.0))):
        for x in xs:
            serial += 1
            lines.append(
                "ATOM  {:5d} {:<4s} {:>3s} {:1s}{:4d}    "
                "{:8.3f}{:8.3f}{:8.3f}  1.00  0.00".format(
                    serial, "BB", "ALA", chain, 1, x, 0.0, 0.0))
        lines.append("TER")
    lines.append("END")
    path = tmp_path / "two_residues.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("mode,expect_contact", [("residue_cog", False),
                                                 ("min_atom", True)])
def test_contact_mode_changes_what_counts_as_a_contact(tmp_path, mode, expect_contact):
    """The two conventions must disagree on a geometry built to separate them.

    Centroids are 9 A apart and the closest atoms 3 A apart, so at a 5 A cutoff
    only the any-atom criterion sees a contact. This is the difference that made
    coana and OLIGAMIX disagree on 17% of frames of a real trajectory.
    """
    pdb = _two_residue_pdb(tmp_path)
    out = tmp_path / f"cache_{mode}.npz"
    cfg = PrecomputeConfig(
        topology_file=None,
        traj_file=str(pdb),
        frames=[0],
        cutoff_contact=5.0,
        intra_exclude=2,
        segment_mode="atom_blocks",
        atoms_per_monomer=4,
        n_monomers=2,
        contact_mode=mode,
    )
    precompute_cache_pure(cfg, str(out), verbose=False)

    cache = load_cache(str(out))
    contacts = int(np.asarray(cache.cmaps[0])[0, 1])
    assert (contacts > 0) is expect_contact, (
        f"{mode}: expected {'a contact' if expect_contact else 'no contact'}, "
        f"got {contacts}")
    # the convention must be recorded, or a reader cannot tell which one ran
    assert cache.meta.get("contact_mode") == mode


def test_contact_mode_rejects_an_unknown_value(tmp_path):
    """A typo must fail loudly rather than silently falling back."""
    pdb = _two_residue_pdb(tmp_path)
    cfg = PrecomputeConfig(
        topology_file=None, traj_file=str(pdb), frames=[0],
        cutoff_contact=5.0, intra_exclude=2, segment_mode="atom_blocks",
        atoms_per_monomer=4, n_monomers=2, contact_mode="minimum_atom",
    )
    with pytest.raises(ValueError, match="unknown contact_mode"):
        precompute_cache_pure(cfg, str(tmp_path / "bad.npz"), verbose=False)



# ---------------------------------------------------------------------------
# #9 mechanism: a guest that leaves before commitment
# ---------------------------------------------------------------------------
#
# _build_event sets present_at_commit only when start_idx <= commit <= end_idx
# and joined_after_commit only when start_idx > commit.  A visit that ends
# before the assembly reaches commit_size therefore sets neither, so it used to
# fall past every branch and land on "bystander" -- whose stated reason, "no
# commitment and no growth during the visit", was false on both counts.  That is
# pre-commitment catalysis, the mechanism the module exists to detect.

def _classify(**kw):
    from oligomer_analysis.mechanism import _classify_residence

    base = dict(
        committed=True, present_at_commit=False, joined_after_commit=False,
        survives=False, growth_during=0, growth_after=float("nan"),
        growth_min=1, commit_size=4,
    )
    base.update(kw)
    return _classify_residence(**base)


def test_guest_that_leaves_before_commitment_is_not_a_bystander():
    """Present early, gone before commit_size, host went on to commit.

    Deliberately NOT labelled "catalytic": association with growth that
    preceded commitment is weaker evidence than catalysis, which would require
    the departure and the commitment to be causally linked rather than merely
    adjacent in time.
    """
    label, reason = _classify(growth_during=3)
    assert label == "pre-commitment", f"got {label!r}: {reason}"
    # the reason must not deny the commitment that did happen
    assert "no commitment" not in reason
    # ... and must not claim more than association
    assert "causal" in reason or "not shown" in reason


def test_pre_commitment_visit_without_growth_stays_a_bystander():
    """The growth floor still has to be cleared, or a coincidental early visit
    would be credited with preceding a later commitment."""
    label, reason = _classify(growth_during=0)
    assert label == "bystander"
    # ... but the reason must describe what actually happened
    assert "no commitment" not in reason
    assert "4" in reason


def test_pre_commitment_growth_floor_is_honoured():
    assert _classify(growth_during=2, growth_min=3)[0] == "bystander"
    assert _classify(growth_during=3, growth_min=3)[0] == "pre-commitment"


def test_the_established_mechanism_labels_are_unchanged():
    """GAP-01, restored. Pins the five labels that existed before FIX-02, so
    adding pre-commitment cannot silently move a case between them.

    Originally written with FIX-02, then deleted by a patch script that
    truncated this file before the audit was under version control.
    """
    assert _classify(present_at_commit=True, survives=True)[0] == "incorporated"
    assert _classify(present_at_commit=True, survives=False,
                     growth_after=0.0)[0] == "catalytic"
    assert _classify(present_at_commit=True, survives=False,
                     growth_after=-2.0)[0] == "transient"
    assert _classify(joined_after_commit=True)[0] == "passenger"
    assert _classify(committed=False, growth_during=5)[0] == "promoting"
    assert _classify(committed=False, growth_during=0)[0] == "bystander"


def test_a_bystander_reason_never_claims_zero_growth_it_did_not_measure():
    """Sub-threshold growth is still growth; the text must say what it was."""
    for committed in (True, False):
        _lbl, reason = _classify(committed=committed, growth_during=2, growth_min=3)
        assert "no growth" not in reason
        assert "2" in reason and "3" in reason


def test_every_label_the_ruleset_emits_is_registered_in_the_summary():
    """An unregistered class is silently dropped from mechanism_summary while
    still counting toward its share denominators, so the taxonomy and the
    summary must not drift apart.

    Exercises the ruleset over every combination of its decision inputs rather
    than parsing its source, so a future branch is covered automatically.
    """
    import itertools
    from oligomer_analysis.mechanism import _classify_residence, mechanism_summary
    import inspect

    emitted = set()
    for committed, present, joined, survives in itertools.product((True, False), repeat=4):
        for growth_during in (0, 1, 5):
            for growth_after in (-2.0, 0.0, 3.0, float("nan")):
                label, _reason = _classify_residence(
                    committed=committed, present_at_commit=present,
                    joined_after_commit=joined, survives=survives,
                    growth_during=growth_during, growth_after=growth_after,
                    growth_min=1, commit_size=4,
                )
                emitted.add(label)

    registered = inspect.getsource(mechanism_summary).split("order = [")[1].split("]")[0]
    missing = sorted(lbl for lbl in emitted if f'"{lbl}"' not in registered)
    assert not missing, f"labels the ruleset emits but the summary drops: {missing}"


# ---------------------------------------------------------------------------
# #10 chemistry: the class is decided by charge, not by the residue name
# ---------------------------------------------------------------------------

def test_neutral_protonation_variants_are_not_charged():
    """A name saying "protonated" does not mean "charged".

    ASPP/ASPH/ASH are protonated -- hence electrically neutral -- Asp, and
    GLUP/GLUH/GLH protonated Glu; the proton neutralises the carboxylate.
    LYSN/LSN/LYN are deprotonated (neutral) Lys and ARGN is neutral Arg.
    Filing any of them as anion or cation puts them in the cation-anion
    channel, which is the salt-bridge channel, so a low-pH box would report
    salt bridges between side chains carrying no charge at all.
    """
    from oligomer_analysis.chemistry import RESIDUE_CLASS

    for neutral in ("ASPP", "ASPH", "ASH", "GLUP", "GLUH", "GLH",
                    "LYSN", "LSN", "LYN", "ARGN",
                    "HID", "HIE", "HSD", "HSE", "HISD", "HISE"):
        cls = RESIDUE_CLASS.get(neutral, "other")
        assert cls not in ("cation", "anion"), f"{neutral} classified as {cls}"

    # the genuinely charged forms must stay charged
    for resname, want in (("ASP", "anion"), ("GLU", "anion"),
                          ("ARG", "cation"), ("LYS", "cation"),
                          ("LYSH", "cation"), ("HSP", "cation"),
                          ("HISH", "cation"), ("HIP", "cation")):
        assert RESIDUE_CLASS[resname] == want, f"{resname} should be {want}"


def test_a_neutral_pair_contributes_no_salt_bridge():
    """The end-to-end consequence: ASPP-ARGN is not a salt bridge."""
    from oligomer_analysis.chemistry import CONTACT_CLASSES, classify_residues

    idx = classify_residues(["ASPP", "ARGN"])
    cation, anion = CONTACT_CLASSES.index("cation"), CONTACT_CLASSES.index("anion")
    assert cation not in idx and anion not in idx, (
        "a neutral Asp/Arg pair still lands in the salt-bridge channel")


# ---------------------------------------------------------------------------
# #11 compare: a numeric descriptor must not be mistaken for a measurement
# ---------------------------------------------------------------------------

def test_a_numeric_descriptor_is_still_checked_for_pooling(growth):
    """Grouping by temperature while salt varies must say so.

    group_summary used to tell descriptors from results by dtype, which holds
    for the path regex (groupdict values are strings) but not for a
    descriptor_fn returning numbers.  A numeric condition then looked like a
    measurement, so the group mean pooled two salt concentrations and reported
    mixed_columns='' -- indistinguishable from a legitimate group mean.
    """
    _info, _cache, path = growth
    labels = ["t300_salt0", "t300_salt150"]
    salt = {"t300_salt0": 0, "t300_salt150": 150}

    df = compare_caches(
        [str(path), str(path)], labels=labels,
        descriptor_fn=lambda name: {"temperature": 300, "salt_mM": salt[name]},
    )
    assert df.attrs["descriptor_columns"] == ["temperature", "salt_mM"]
    assert pd.api.types.is_numeric_dtype(df["salt_mM"]), (
        "the regression needs salt_mM to be a numeric column")

    grouped = group_summary(df, ["temperature"])
    assert len(grouped) == 1
    assert int(grouped["n_runs"].iloc[0]) == 2
    mixed = grouped["mixed_columns"].iloc[0]
    assert "salt_mM" in mixed, f"pooled two salt conditions silently: {mixed!r}"

    # result metrics must NOT be reported as descriptors, or the warning drowns
    for metric in ("n_lineages", "largest_cluster_mean", "n_frames"):
        assert metric not in mixed, f"{metric} reported as a descriptor"


def test_group_summary_accepts_descriptor_columns_directly(growth):
    """attrs can be dropped by pandas operations, so the list can be passed."""
    _info, _cache, path = growth
    df = compare_caches(
        [str(path), str(path)], labels=["a", "b"],
        descriptor_fn=lambda name: {"rep": 1 if name == "a" else 2, "cond": "x"},
    )
    stripped = df.copy()
    stripped.attrs = {}                      # simulate attrs being lost
    # without provenance the numeric descriptor is missed (documented fallback)
    assert group_summary(stripped, ["cond"])["mixed_columns"].iloc[0] == ""
    # stating it explicitly restores the check
    grouped = group_summary(stripped, ["cond"], descriptor_columns=["rep", "cond"])
    assert "rep" in grouped["mixed_columns"].iloc[0]


# ---------------------------------------------------------------------------
# #12 remaining audit fixes: normalisation, empty networks, GEXF, Geweke
# ---------------------------------------------------------------------------

def test_class_fractions_sum_to_one_over_the_emitted_pairs():
    """normalize=True must divide by DISTINCT contacts, not the symmetric sum.

    The matrix is symmetric, so summing all of it double-counts every
    heterotypic pair. contact_class_dataframe emits only the upper triangle, so
    the emitted fractions used to sum to U/(U+H) -- exactly 0.5 when every
    contact is heterotypic -- instead of 1.
    """
    from oligomer_analysis.chemistry import (
        CONTACT_CLASSES, contact_class_dataframe, contact_class_matrix,
    )

    nc = len(CONTACT_CLASSES)
    cat, an = CONTACT_CLASSES.index("cation"), CONTACT_CLASSES.index("anion")
    ap = CONTACT_CLASSES.index("apolar")

    for name, entries in (
        ("one heterotypic contact", {(cat, an): 1}),
        ("all heterotypic", {(cat, an): 30, (ap, cat): 70}),
        ("mixed", {(cat, an): 12, (ap, ap): 88}),
        ("all homotypic", {(ap, ap): 50, (cat, cat): 50}),
    ):
        m = np.zeros((1, nc, nc), dtype=np.int64)
        for (i, j), n in entries.items():
            m[0, i, j] = n
            if i != j:
                m[0, j, i] = n
        cmaps = np.ones((1, 2, 2), dtype=np.int32)
        cmaps[0, 0, 0] = cmaps[0, 1, 1] = 0
        cache = cache_from_cmaps(
            cmaps, class_cmaps=m, meta={"contact_classes": list(CONTACT_CLASSES)})

        total = sum(entries.values())
        norm = contact_class_matrix(cache, normalize=True)
        for (i, j), n in entries.items():
            assert np.isclose(norm[0, i, j], n / total), (
                f"{name}: {CONTACT_CLASSES[i]}-{CONTACT_CLASSES[j]} "
                f"got {norm[0, i, j]:.4f}, want {n / total:.4f}")
        emitted = contact_class_dataframe(cache, normalize=True)["contacts"].sum()
        assert np.isclose(emitted, 1.0), f"{name}: fractions sum to {emitted:.4f}"


def test_an_empty_transition_table_is_empty_not_an_exception():
    """No edge clears min_count -> an empty frame with the right columns."""
    from oligomer_analysis.network import transition_dataframe

    df = transition_dataframe(["A"], np.zeros((1, 1), dtype=np.int64))
    assert df.empty
    for col in ("source", "target", "count", "probability", "self_transition"):
        assert col in df.columns


def test_gexf_attributes_survive_a_quote_in_a_name(tmp_path):
    """escape() leaves `"` alone, which closes the attribute and breaks the XML."""
    from oligomer_analysis.network import write_transition_gexf

    quote = chr(34)
    names = ["A" + quote + "B", "C"]
    counts = np.array([[0, 2], [1, 0]], dtype=np.int64)
    out = tmp_path / "t.gexf"
    write_transition_gexf(names, counts, out)
    xml = out.read_text(encoding="utf-8")
    ET.fromstring(xml)                       # raises if the quote broke the XML
    assert "&quot;" in xml


def test_node_weight_is_named_for_what_it_counts(tmp_path):
    """A row sum counts departures, not occupancy: A -> B gives A=1, B=0."""
    from oligomer_analysis.network import write_transition_gexf

    out = tmp_path / "t.gexf"
    write_transition_gexf(["A", "B"], np.array([[0, 1], [0, 0]], dtype=np.int64),
                          out, include_self=True)
    xml = out.read_text(encoding="utf-8")
    assert "n_departures" in xml and "occupancy" not in xml


def test_overlapping_geweke_windows_are_refused(growth):
    """Overlapping windows share frames, biasing z toward 0 -- toward calling a
    drifting run converged."""
    _info, cache, _path = growth
    with pytest.raises(ValueError, match="overlap"):
        assess_convergence(cache, "largest_cluster",
                           first_fraction=0.6, last_fraction=0.5)
    # the disjoint defaults must still work
    assert assess_convergence(cache, "largest_cluster").n_frames > 0


def test_a_descriptor_may_not_be_named_after_a_measurement(growth):
    """It would replace that column's value, and group_summary excludes metric
    names from its pooling check, so the swap would never be reported."""
    _info, _cache, path = growth
    for bad in ("largest_cluster_late", "n_lineages", "converged"):
        with pytest.raises(ValueError, match="collide"):
            compare_caches([str(path)], labels=["r1"],
                           descriptor_fn=lambda name, k=bad: {k: 1})
    # a non-colliding name is still fine
    df = compare_caches([str(path)], labels=["r1"],
                        descriptor_fn=lambda name: {"cond_salt": 150})
    assert int(df["cond_salt"].iloc[0]) == 150


# ---------------------------------------------------------------------------
# #13 gaps closed after an external review of the fixes above
# ---------------------------------------------------------------------------

def test_a_declared_descriptor_list_does_not_switch_off_the_heuristic(growth):
    """Declared provenance must ADD to the non-numeric heuristic, not replace it.

    compare_caches sets attrs even when descriptor_fn is None, so if a declared
    list replaced the heuristic an empty list would switch the check off and a
    built-in string descriptor such as `species` would stop being examined.
    """
    _info, _cache, path = growth
    df = compare_caches([str(path), str(path)], labels=["r1", "r2"])
    assert df.attrs["descriptor_columns"] == []          # no descriptor_fn

    # a non-numeric column that varies must still be caught
    df["forcefield"] = ["m22", "m30"]
    grouped = group_summary(df, ["n_chains"])
    assert "forcefield" in grouped["mixed_columns"].iloc[0]

    # and a declared numeric descriptor is caught *as well as*, not instead of
    df.attrs["descriptor_columns"] = ["salt_mM"]
    df["salt_mM"] = [0, 150]
    mixed = group_summary(df, ["n_chains"])["mixed_columns"].iloc[0]
    assert "salt_mM" in mixed and "forcefield" in mixed


def test_geweke_windows_that_round_into_overlap_are_refused(growth):
    """Fractions summing to <= 1 can still overlap once each size is rounded
    and clamped up to 2: n=10 with 0.1/0.9 gives sizes 2 and 9."""
    _info, cache, _path = growth
    series = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="overlap"):
        assess_convergence(cache, "largest_cluster", series=series,
                           first_fraction=0.1, last_fraction=0.9)
    # a genuinely disjoint split of the same series is fine
    rep = assess_convergence(cache, "largest_cluster", series=series,
                             first_fraction=0.2, last_fraction=0.5)
    assert rep.n_frames == 10


def test_node_weight_values_are_departures_not_occupancy(tmp_path):
    """A -> B must give A=1, B=0 -- the values, not just the attribute name."""
    from oligomer_analysis.network import write_transition_gexf

    out = tmp_path / "t.gexf"
    write_transition_gexf(["A", "B"], np.array([[0, 1], [0, 0]], dtype=np.int64),
                          out, include_self=True)
    root = ET.fromstring(out.read_text(encoding="utf-8"))
    ns = {"g": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

    def nodes(r):
        return r.iter("node") if not ns else r.iter("{%s}node" % ns["g"])

    def attvalues(n):
        return n.iter("attvalue") if not ns else n.iter("{%s}attvalue" % ns["g"])

    got = {}
    for node in nodes(root):
        for av in attvalues(node):
            got[node.get("label")] = av.get("value")
            break                      # n_departures is the first declared attr
    assert got == {"A": "1", "B": "0"}, got


def test_the_added_residue_aliases_are_polar_not_merely_uncharged():
    """Asserting "not cation/anion" also passes for a name that fell through to
    `other`, so pin the actual class."""
    from oligomer_analysis.chemistry import RESIDUE_CLASS

    # every name below was checked against a published convention
    for neutral in ("ASPP", "ASPH", "ASH", "ASP0", "GLUP", "GLUH", "GLH", "GLU0",
                    "LYSN", "LSN", "LYN", "ARGN",
                    "HID", "HIE", "HSD", "HSE", "HISD", "HISE"):
        assert RESIDUE_CLASS.get(neutral) == "polar", (
            f"{neutral} is {RESIDUE_CLASS.get(neutral, 'other')!r}, not polar")
    for charged, want in (("LYP", "cation"), ("HIH", "cation"), ("CYM", "anion")):
        assert RESIDUE_CLASS.get(charged) == want, (
            f"{charged} is {RESIDUE_CLASS.get(charged, 'other')!r}, not {want}")
    for apolar in ("CYS", "CYX", "CYN"):
        assert RESIDUE_CLASS.get(apolar) == "apolar"
    # and names that could not be verified must NOT be invented into the table
    for unverified in ("LYS0", "ARG0", "ARN", "CYS2"):
        assert unverified not in RESIDUE_CLASS, (
            f"{unverified} is in the table but was never verified; an "
            "unrecognised name should fall to 'other' and warn instead")


def test_persistence_denominator_is_co_residency_not_time_since_first_contact():
    """Distinguishes the fix from a plausible wrong denominator.

    A pair that is co-resident for frames 0-9 but in contact only in 5-9 has
    persistence 5/10 = 0.5. Counting from first contact instead would give
    5/5 = 1.0, and the "every possible contact present" test cannot tell those
    apart because there the two denominators coincide.
    """
    from oligomer_analysis.models import ContactCache, OligomerSegment
    from oligomer_analysis.predictors import segment_contact_persistence

    n_fr = 10
    cmaps = np.zeros((n_fr, 2, 2), dtype=np.int32)
    for t in range(5, n_fr):
        cmaps[t, 0, 1] = cmaps[t, 1, 0] = 5
    cache = ContactCache(
        frames=np.arange(n_fr), cmaps=cmaps,
        nintra=np.zeros(n_fr), ninter=np.zeros(n_fr), ntotal=np.zeros(n_fr),
        n_free=np.zeros(n_fr), n_bound=np.zeros(n_fr),
        seg_centers=np.zeros((n_fr, 2, 3)), boxes=np.zeros((n_fr, 3)), meta={},
    )
    seg = OligomerSegment(
        segment_id="s", start_idx=0, end_idx=n_fr - 1,
        frames=list(range(n_fr)), sizes=[2] * n_fr,
        sets=[frozenset({0, 1})] * n_fr,
    )
    assert segment_contact_persistence(cache, seg, inter_thresh=1) == 0.5


def test_mechanism_shares_still_sum_to_one_with_the_new_label():
    """pre-commitment is registered, so it must not vanish from the summary
    while still counting toward the denominators."""
    from oligomer_analysis.mechanism import ResidenceEvent, mechanism_summary

    def ev(chain, label, duration):
        return ResidenceEvent(
            chain=chain, species="B", lineage_id="lin0",
            start_idx=0, end_idx=duration - 1, start_frame=0,
            end_frame=duration - 1, duration=duration,
            host_at_entry=2, host_at_exit=3, host_max_during=3,
            host_growth_during=1, host_growth_after=0.0,
            committed=True, commit_idx=5, present_at_commit=False,
            joined_after_commit=False, survives_to_lineage_end=False,
            lineage_end_idx=10, classification=label, reason="",
        )

    events = [ev(0, "pre-commitment", 4), ev(1, "incorporated", 6),
              ev(2, "bystander", 2)]
    summ = mechanism_summary(events).set_index("classification")
    assert "pre-commitment" in summ.index
    assert np.isclose(summ["visit_share"].sum(), 1.0)
    assert np.isclose(summ["frame_share"].sum(), 1.0)


def test_unrecognised_residues_are_named_not_silently_bucketed():
    """`other` contributes to no channel, so an unknown charged variant is an
    undercount the census cannot explain. The names must be surfaced."""
    from oligomer_analysis.chemistry import classify_residues

    with pytest.warns(UserWarning, match="does not recognise"):
        classify_residues(["ALA", "ASP", "MYSTERY", "MYSTERY", "ZZZ"])

    # solvent, ions and terminal caps belong in `other` and must stay quiet, or
    # the warning fires on every real system and gets tuned out
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        classify_residues(["ALA", "ASP", "W", "NA", "CL", "SOL", "ACE", "NH2"])


def test_summarize_cache_validates_last_fraction(growth):
    """An out-of-range last_fraction silently redefines "late": 0 averages the
    final frame alone and 5 averages the whole run, both looking legitimate."""
    _info, _cache, path = growth
    for bad in (0.0, -0.1, 1.5, 5.0):
        with pytest.raises(ValueError, match="last_fraction"):
            summarize_cache(str(path), last_fraction=bad)
    assert summarize_cache(str(path), last_fraction=1.0)["n_frames"] > 0


# ---------------------------------------------------------------------------
# #14 partition: K_sem must be corrected for autocorrelation
# ---------------------------------------------------------------------------

def test_K_sem_is_corrected_for_autocorrelation_and_naive_is_kept():
    """A naive sd/sqrt(n) over correlated frames understates the error badly --
    ~10x at tau=100. The corrected value must exceed the naive one, and the
    naive one must remain visible so the size of the correction can be judged.
    """
    n_frames, n_a, n_b = 400, 8, 4
    nseg = n_a + n_b
    smap = SpeciesMap(names=["A", "B"],
                      seg_species=np.array([0] * n_a + [1] * n_b, dtype=np.int16))

    # The cluster's B content is driven on a long timescale, so the K series
    # varies smoothly and is genuinely autocorrelated (tau ~ 6 frames) rather
    # than frame-to-frame noise. A fixed A core keeps the cluster well defined.
    t = np.arange(n_frames)
    nb_in = np.clip(np.round(2 + 1.8 * np.sin(2 * np.pi * t / 160)).astype(int),
                    0, n_b)
    na_in = 5
    cmaps = np.zeros((n_frames, nseg, nseg), dtype=np.int32)
    for f in range(n_frames):
        members = list(range(na_in)) + [n_a + k for k in range(int(nb_in[f]))]
        for i in members:
            for j in members:
                if i != j:
                    cmaps[f, i, j] = 4

    cache = ContactCache(
        frames=np.arange(n_frames), cmaps=cmaps,
        nintra=np.zeros(n_frames), ninter=np.zeros(n_frames),
        ntotal=np.zeros(n_frames), n_free=np.zeros(n_frames),
        n_bound=np.zeros(n_frames),
        seg_centers=np.zeros((n_frames, nseg, 3)),
        boxes=np.zeros((n_frames, 6)), meta={},
    )

    summ = partition_summary(cache, smap, inter_thresh=1, min_size=2,
                             last_fraction=0.5).set_index("species")
    for col in ("K_sem", "K_sem_naive", "K_tau", "K_n_eff", "K_n_dropped"):
        assert col in summ.columns, f"missing {col}"

    row = summ.loc["A"]
    assert np.isfinite(row["K_sem"]) and np.isfinite(row["K_sem_naive"])
    assert row["K_tau"] > 0.5, (
        f"tau={row['K_tau']}: the construction was meant to be correlated")
    assert row["K_sem"] > row["K_sem_naive"], (
        f"corrected {row['K_sem']:.5f} is not above naive "
        f"{row['K_sem_naive']:.5f} despite tau={row['K_tau']:.1f}")
    # n_eff must be the frames actually used, reduced by 2*tau
    assert row["K_n_eff"] < row["n_frames"]
