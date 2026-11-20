# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Per-oligomer contact chemistry: Stage A rows per chain pair, Stage B sums.

The two-species system of ``synthetic.two_species_trajectory`` is small enough
to count by hand, and the system-wide class matrix (tested in
``test_v34_features``) is the reference every per-pair sum must reproduce.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from synthetic import cache_from_cmaps, growth_trajectory, two_species_trajectory  # noqa: E402

from oligomer_analysis import (  # noqa: E402
    build_lineages_from_cache,
    load_cache,
    oligomer_chemistry_tables,
    oligomer_class_matrix,
    segment_chemistry_timeseries,
)
from oligomer_analysis.cache_io import (  # noqa: E402
    pack_pair_classes,
    save_contact_cache,
    subset_cache_by_indices,
    unpack_pair_classes,
)
from oligomer_analysis.coanapure import precompute_cache_pure  # noqa: E402
from oligomer_analysis.models import PrecomputeConfig, TargetStateConfig  # noqa: E402


def _precompute(gro, n_frames, spec, out):
    cfg = PrecomputeConfig(
        topology_file=None,
        traj_file=gro,
        frames=list(range(n_frames)),
        cutoff_contact=6.0,
        intra_exclude=2,
        species_spec=spec,
        contact_classes=True,
    )
    precompute_cache_pure(cfg, str(out), verbose=False)
    return load_cache(out)


@pytest.fixture(scope="module")
def two_species(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("chem_two_species")
    info = two_species_trajectory(tmp, n_a=3, n_b=2)
    return _precompute(info["gro"], info["n_frames"], info["species_spec"], tmp / "c.npz"), tmp / "c.npz"


@pytest.fixture(scope="module")
def growth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("chem_growth")
    info = growth_trajectory(tmp)
    return _precompute(info["gro"], info["n_frames"], info["species_spec"], tmp / "g.npz"), tmp / "g.npz"


def _rows_to_class_matrix(rows, n_classes):
    raw = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(raw, (rows[:, 2], rows[:, 3]), rows[:, 4])
    m = raw + raw.T
    np.fill_diagonal(m, np.diag(raw))
    return m


@pytest.mark.parametrize("which", ["two_species", "growth"])
def test_pair_rows_reproduce_system_matrix_and_cmaps(which, request):
    cache, _ = request.getfixturevalue(which)
    n_classes = len(cache.class_names)
    per_frame = unpack_pair_classes(cache.pair_class_rows, cache.pair_class_offsets)
    assert len(per_frame) == len(cache.frames)
    for f, rows in enumerate(per_frame):
        assert np.all(rows[:, 0] < rows[:, 1]) and np.all(rows[:, 2] <= rows[:, 3])
        assert np.array_equal(_rows_to_class_matrix(rows, n_classes), cache.class_cmaps[f])
        per_pair = np.zeros_like(cache.cmaps[f])
        np.add.at(per_pair, (rows[:, 0], rows[:, 1]), rows[:, 4])
        assert np.array_equal(per_pair, np.triu(cache.cmaps[f], 1)), f"frame {f}"


def test_oligomer_of_all_chains_equals_system_matrix(two_species):
    cache, _ = two_species
    everyone = range(cache.cmaps.shape[1])
    for f in range(len(cache.frames)):
        assert np.array_equal(oligomer_class_matrix(cache, f, everyone), cache.class_cmaps[f])


def test_oligomer_chemistry_by_hand(two_species):
    cache, _ = two_species
    names = cache.class_names
    sb = (names.index("cation"), names.index("anion"))
    # frame 1: each A-A pair has 16 residue contacts, 2 of them ASP-LYS salt bridges
    m = oligomer_class_matrix(cache, 1, {0, 1})
    assert int(np.triu(m).sum()) == 16
    assert int(m[sb]) == 2
    # the three A chains: 3 pairs
    m = oligomer_class_matrix(cache, 1, {0, 1, 2})
    assert int(np.triu(m).sum()) == 3 * 16 and int(m[sb]) == 6
    # a lone chain holds no inter-chain contacts
    assert int(np.triu(oligomer_class_matrix(cache, 1, {0})).sum()) == 0


def test_segment_tables_are_consistent(growth):
    cache, _ = growth
    result = build_lineages_from_cache(
        cache, inter_thresh=1, cluster_min_size=2,
        target_config=TargetStateConfig(min_size=4, min_duration=5),
    )
    segs, lins = oligomer_chemistry_tables(cache, result)
    assert len(segs) == len(result.segments) and len(lins) == len(result.lineages)
    for seg in result.segments:
        ts = segment_chemistry_timeseries(cache, seg)
        # contacts holding the oligomer together, recounted from the chain cmaps
        expect = [
            int(np.triu(cache.cmaps[seg.start_idx + k][np.ix_(sorted(m), sorted(m))], 1).sum())
            for k, m in enumerate(seg.sets)
        ]
        assert ts["contacts"].tolist() == expect
        row = segs.set_index("segment_id").loc[seg.segment_id]
        assert int(row["contacts"]) == sum(expect)
    frac = [c for c in segs.columns if c.endswith("_fraction")]
    sums = segs.loc[segs.contacts > 0, frac].sum(axis=1)
    assert np.allclose(sums, 1.0)
    assert int(lins["contacts"].sum()) == int(segs["contacts"].sum())


def test_round_trip_and_subset_keep_pair_rows(two_species, tmp_path):
    cache, _ = two_species
    out = tmp_path / "rt.npz"
    save_contact_cache(cache, out, save_csv_stats=False)
    back = load_cache(out)
    assert np.array_equal(back.pair_class_rows, cache.pair_class_rows)
    assert np.array_equal(back.pair_class_offsets, cache.pair_class_offsets)
    sub = subset_cache_by_indices(cache, [2, 1])
    per_frame = unpack_pair_classes(cache.pair_class_rows, cache.pair_class_offsets)
    sub_frames = unpack_pair_classes(sub.pair_class_rows, sub.pair_class_offsets)
    assert np.array_equal(sub_frames[0], per_frame[2]) and np.array_equal(sub_frames[1], per_frame[1])
    assert np.array_equal(oligomer_class_matrix(sub, 1, {0, 1}), oligomer_class_matrix(cache, 1, {0, 1}))


def test_pack_unpack_handles_empty_frames():
    frames = [np.empty((0, 5), np.int32), np.array([[0, 1, 2, 3, 4]], np.int32), np.empty((0, 5), np.int32)]
    rows, offsets = pack_pair_classes(frames)
    assert offsets.tolist() == [0, 0, 1, 1]
    back = unpack_pair_classes(rows, offsets)
    assert [len(r) for r in back] == [0, 1, 0]


def test_old_cache_gets_a_clear_error():
    cmaps = np.zeros((3, 4, 4), dtype=np.int32)
    cache = cache_from_cmaps(cmaps)
    with pytest.raises(ValueError, match="per-chain-pair contact classes"):
        oligomer_class_matrix(cache, 0, {0, 1})


def test_cli_writes_per_oligomer_tables(growth, tmp_path):
    _cache, path = growth
    seg_csv, lin_csv = tmp_path / "seg.csv", tmp_path / "lin.csv"
    proc = subprocess.run(
        [sys.executable, "-m", "oligomer_analysis", "chemistry", str(path),
         "--per-oligomer", str(seg_csv), "--per-lineage", str(lin_csv),
         "--min-size", "4", "--min-duration", "5"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    segs, lins = pd.read_csv(seg_csv), pd.read_csv(lin_csv)
    assert {"segment_id", "lineage_id", "contacts", "salt_bridge_fraction", "hydrophobic_fraction"} <= set(segs.columns)
    assert int(lins["contacts"].sum()) == int(segs["contacts"].sum())


def test_channel_names_cover_every_class_pair_once():
    from oligomer_analysis.chemistry import CONTACT_CLASSES, _channel_name

    assert _channel_name("polar", "polar") == "polar"
    assert _channel_name("polar", "apolar") == "mixed polar/apolar"
    assert _channel_name("aromatic", "polar") == "mixed polar/apolar"
    assert _channel_name("apolar", "aromatic") == "hydrophobic"
    assert _channel_name("cation", "anion") == "salt bridge"
    assert _channel_name("aromatic", "cation") == "cation-pi"
    assert _channel_name("anion", "anion") == "like-charge repulsive contact"
    names = {_channel_name(a, b) for a in CONTACT_CLASSES for b in CONTACT_CLASSES}
    assert "polar" in names


def test_cli_hotspot_blocks_on_a_boxed_trajectory(tmp_path):
    """Regression (3.8.0): the box probe reran frame 0 with --hotspot-blocks and failed."""
    from oligomer_analysis.cli import main

    info = two_species_trajectory(tmp_path, n_a=3, n_b=2)
    out = tmp_path / "h.npz"
    rc = main([
        "precompute", "--traj", str(info["gro"]), "--frames", "all", "--cutoff", "6.0",
        "--intra-exclude", "2", "--segment-mode", "atom_counts", "--species", info["species_spec"],
        "--contact-classes", "--hotspot-blocks", "2", "-o", str(out), "--quiet",
    ])
    assert rc == 0
    cache = load_cache(out)
    assert cache.meta["has_box"] is True
    assert cache.hotspot_maps.shape[0] == 2 and cache.pair_class_rows is not None


# ---------------------------------------------------------------------------
# periodic search: exact below half the smallest box height, warned above it
# ---------------------------------------------------------------------------


def _brute_pairs(pos, cutoff, mat, reach=2):
    """Every i<j pair with some periodic image within cutoff, over (2*reach+1)^3 images."""
    rng = range(-reach, reach + 1)
    shifts = np.array([(i, j, k) for i in rng for j in rng for k in rng], dtype=float) @ mat
    out = set()
    for i in range(len(pos)):
        d = pos[i + 1:, None, :] - pos[i] + shifts[None, :, :]
        hit = np.where((np.einsum("pkx,pkx->pk", d, d) <= cutoff * cutoff).any(axis=1))[0]
        out.update((i, i + 1 + int(h)) for h in hit)
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_triclinic_search_is_exact_below_half_the_box_height(seed):
    from oligomer_analysis.coanapure.neighbors import self_capped_pairs
    from oligomer_analysis.coanapure.pbc import box6_from_vectors, min_box_height

    rng = np.random.default_rng(seed)
    # GROMACS reduced form at its skew limit, plus random tilts inside it
    a = 40.0
    mat = np.array([[a, 0, 0],
                     [a / 2, 35.0, 0],
                     [rng.uniform(-a / 2, a / 2), rng.uniform(-17.5, 17.5), 30.0]])
    box6 = box6_from_vectors(mat)
    h = min_box_height(box6)
    cutoff = 0.45 * h
    pos = rng.uniform(0, 1, (120, 3)) @ mat
    got = {tuple(p) for p in self_capped_pairs(pos, cutoff, box6).tolist()}
    assert got == _brute_pairs(pos, cutoff, mat)


def test_min_box_height_and_cutoff_warning():
    from oligomer_analysis.coanapure.neighbors import self_capped_pairs
    from oligomer_analysis.coanapure.pbc import min_box_height

    assert np.isclose(min_box_height([30.0, 40.0, 50.0, 90, 90, 90]), 30.0)
    pos = np.random.default_rng(0).uniform(0, 20, (10, 3))
    with pytest.warns(RuntimeWarning, match="half the smallest box height"):
        self_capped_pairs(pos, 10.0, np.array([20.0, 20.0, 20.0, 90, 90, 90]))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        self_capped_pairs(pos, 9.9, np.array([20.0, 20.0, 20.0, 90, 90, 90]))


def test_guest_reasons_state_only_what_was_observed():
    from oligomer_analysis.mechanism import _classify_residence

    common = dict(committed=True, present_at_commit=True, joined_after_commit=False,
                  survives=False, growth_during=3, growth_min=1, commit_size=5)
    label, reason = _classify_residence(growth_after=float("nan"), **common)
    assert label == "catalytic" and "could not be followed" in reason
    label, reason = _classify_residence(growth_after=0.0, **common)
    assert label == "catalytic" and "at least as many host chains" in reason
    label, reason = _classify_residence(growth_after=-2.0, **common)
    assert label == "transient" and "fewer host chains" in reason
    for text in (reason,):
        assert "stable aggregate" not in text and "promotes" not in text


def test_cli_lineage_calls_roots_roots_not_nucleation_events(growth, tmp_path):
    """Roots include clusters already present in the first frame, so 'nucleation events' was wrong."""
    _cache, path = growth
    proc = subprocess.run(
        [sys.executable, "-m", "oligomer_analysis", "lineage", str(path), "--min-size", "4", "--min-duration", "5"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "directed fates of roots" in proc.stdout
    assert "nucleation event" not in proc.stdout
    help_text = subprocess.run([sys.executable, "-m", "oligomer_analysis", "lineage", "--help"],
                               capture_output=True, text=True, cwd=tmp_path).stdout
    assert "nucleation event" not in help_text
