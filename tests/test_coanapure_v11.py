# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests for the CoAnaPure 1.1 additions.

Covers: the non-periodic cell list and no-scipy routing, streaming Stage A
(constant memory), the model byte-offset index (random access), the threaded
frame loop, the dropped-leading-CRYST1 warning, and the GRO/TRR readers.
MDAnalysis-dependent cross-checks skip cleanly when it is not importable.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from oligomer_analysis.cache_io import load_cache
from oligomer_analysis.coanapure import (
    GROFrameStream,
    PDBFrameStream,
    PDBModelIndex,
    PrecomputeConfig,
    TRRFrameStream,
    build_model_index,
    precompute_cache_pure,
    read_gro_topology,
    read_pdb,
)
from oligomer_analysis.coanapure import neighbors as nb
from oligomer_analysis.coanapure.neighbors import _cell_list_pairs_nopbc


# --------------------------------------------------------------------------- #
# Synthetic trajectory builders
# --------------------------------------------------------------------------- #
def _write_pdb(path, n_frames=4, n_monomers=4, beads=2, leading_box=None):
    lines = []
    if leading_box is not None:
        lines.append(
            f"CRYST1{leading_box[0]:9.3f}{leading_box[1]:9.3f}"
            f"{leading_box[2]:9.3f}  90.00  90.00  90.00 P 1\n"
        )
    rng = np.random.RandomState(5)
    for fr in range(n_frames):
        lines.append(f"MODEL   {fr + 1:6d}\n")
        serial = 1
        for m in range(n_monomers):
            for b in range(beads):
                x = m * 10.0 + b + 0.1 * fr
                y, z = rng.rand() * 2, rng.rand() * 2
                lines.append(
                    f"ATOM  {serial:5d}  B{b:<2d} ALA {chr(65 + m)}{m + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
                )
                serial += 1
        lines.append("ENDMDL\n")
    with open(path, "w") as fh:
        fh.writelines(lines)


def _cfg(traj, frames, *, topology=None, n_jobs=1, n_monomers=4, beads=2):
    return PrecomputeConfig(
        topology_file=topology, traj_file=str(traj), frames=list(frames),
        cutoff_contact=1.8, intra_exclude=0, n_jobs=n_jobs,
        segment_mode="atom_blocks", atoms_per_monomer=beads, n_monomers=n_monomers,
    )


def _load_arrays(path):
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files if k != "meta_json"}


# --------------------------------------------------------------------------- #
# Item 1: non-periodic cell list + no-scipy routing
# --------------------------------------------------------------------------- #
def test_cell_list_nopbc_matches_bruteforce():
    rng = np.random.RandomState(1)
    for n, scale in [(400, 40.0), (1500, 90.0), (600, 8.0)]:
        pts = rng.rand(n, 3) * scale
        iu, ju = np.triu_indices(n, k=1)
        d = np.linalg.norm(pts[iu] - pts[ju], axis=1)
        ref = set(map(tuple, np.stack([iu[d <= 4.0], ju[d <= 4.0]], axis=1)))
        pairs, dist = _cell_list_pairs_nopbc(pts, 4.0)
        assert set(map(tuple, pairs)) == ref
        assert len(pairs) == len(ref)  # each pair exactly once
        assert np.allclose(np.sort(dist), np.sort(d[d <= 4.0]))


def test_cell_list_nopbc_degenerate_geometries():
    pts = np.zeros((500, 3))
    pts[:, 0] = np.linspace(0.0, 60.0, 500)  # collinear
    pairs, _ = _cell_list_pairs_nopbc(pts, 0.5)
    iu, ju = np.triu_indices(500, k=1)
    d = np.abs(pts[iu, 0] - pts[ju, 0])
    assert set(map(tuple, pairs)) == set(
        map(tuple, np.stack([iu[d <= 0.5], ju[d <= 0.5]], axis=1))
    )
    coincident = np.zeros((80, 3))
    pairs, _ = _cell_list_pairs_nopbc(coincident, 1.0)
    assert len(pairs) == 80 * 79 // 2


def test_no_scipy_routing_uses_cell_list(monkeypatch):
    rng = np.random.RandomState(2)
    pts = rng.rand(800, 3) * 60.0
    with_scipy = nb.self_capped_pairs(pts, 3.5, None)
    monkeypatch.setattr(nb, "_cKDTree", None)
    called = {}
    real = nb._cell_list_pairs_nopbc

    def spy(pos, cutoff):
        called["yes"] = True
        return real(pos, cutoff)

    monkeypatch.setattr(nb, "_cell_list_pairs_nopbc", spy)
    without = nb.self_capped_pairs(pts, 3.5, None)
    assert called.get("yes"), "no-scipy path must route through the free cell list"
    assert set(map(tuple, with_scipy)) == set(map(tuple, without))


# --------------------------------------------------------------------------- #
# Items 2 + 5: streaming Stage A and the threaded frame loop
# --------------------------------------------------------------------------- #
def test_streaming_precompute_matches_readpdb_compute(tmp_path):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=5)
    out = tmp_path / "c.npz"
    precompute_cache_pure(_cfg(p, range(5)), out_npz=str(out), verbose=False)
    cache = load_cache(str(out))
    assert cache.meta["engine"].startswith("coanapure-1.")
    assert len(cache.frames) == 5
    # Reference: in-memory struct positions must equal streamed decode.
    s = read_pdb(str(p))
    got = {k: pos for k, pos, _ in PDBFrameStream(str(p))}
    for k in range(5):
        assert np.array_equal(got[k], s.positions[k])


def test_threaded_equals_serial(tmp_path):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=8)
    a, b = tmp_path / "serial.npz", tmp_path / "threads.npz"
    precompute_cache_pure(_cfg(p, range(8), n_jobs=1), out_npz=str(a), verbose=False)
    precompute_cache_pure(_cfg(p, range(8), n_jobs=4), out_npz=str(b), verbose=False)
    da, db = _load_arrays(a), _load_arrays(b)
    for k, va in da.items():
        assert np.array_equal(va, db[k]), k


def test_sparse_subset_and_order(tmp_path):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=6)
    out = tmp_path / "c.npz"
    precompute_cache_pure(_cfg(p, [4, 1, 3]), out_npz=str(out), verbose=False)
    cache = load_cache(str(out))
    assert list(cache.frames) == [1, 3, 4]


def test_out_of_range_frame_raises(tmp_path):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=3)
    with pytest.raises(IndexError, match="out of range"):
        precompute_cache_pure(_cfg(p, [0, 7]), out_npz=str(tmp_path / "x.npz"), verbose=False)


# --------------------------------------------------------------------------- #
# Item 3: model byte-offset index
# --------------------------------------------------------------------------- #
def test_index_random_access_matches_sequential(tmp_path):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=6, leading_box=(50, 50, 50))
    idx_file = tmp_path / "t.idx.npz"
    idx = build_model_index(str(p), str(idx_file))
    assert idx.n_models == 6 and os.path.exists(idx_file)

    a, b = tmp_path / "seq.npz", tmp_path / "idx.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precompute_cache_pure(_cfg(p, [1, 5]), out_npz=str(a), verbose=False)
        precompute_cache_pure(
            _cfg(p, [1, 5]), out_npz=str(b), verbose=False, index_file=str(idx_file)
        )
    da, db = _load_arrays(a), _load_arrays(b)
    for k, va in da.items():
        assert np.array_equal(va, db[k]), k


def test_index_staleness_detected(tmp_path):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=3)
    idx_file = tmp_path / "t.idx.npz"
    build_model_index(str(p), str(idx_file))
    with open(p, "a") as fh:
        fh.write("REMARK appended\n")
    with pytest.raises(ValueError, match="stale"):
        PDBModelIndex.load(str(idx_file), str(p))


# --------------------------------------------------------------------------- #
# Item 6: dropped leading-CRYST1 warning
# --------------------------------------------------------------------------- #
def test_dropped_box_warns_and_broadcast_silences(tmp_path):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=3, leading_box=(50, 50, 50))
    with pytest.warns(UserWarning, match="WITHOUT periodic"):
        read_pdb(str(p))
    with pytest.warns(UserWarning, match="WITHOUT periodic"):
        list(PDBFrameStream(str(p)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        read_pdb(str(p), broadcast_box=True)
        list(PDBFrameStream(str(p), broadcast_box=True))
        list(PDBFrameStream(str(p), warn_dropped_box=False))


def test_no_warning_without_leading_box(tmp_path):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=3)  # no CRYST1 at all
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        read_pdb(str(p))
        list(PDBFrameStream(str(p)))


# --------------------------------------------------------------------------- #
# Item 4: GRO and TRR readers
# --------------------------------------------------------------------------- #
def _write_gro(path, n_frames=3, n_monomers=3, beads=2, box_nm=(8.0, 8.0, 8.0)):
    rng = np.random.RandomState(9)
    with open(path, "w") as fh:
        for fr in range(n_frames):
            fh.write(f"synthetic frame t= {fr:.1f}\n")
            fh.write(f"{n_monomers * beads:5d}\n")
            atom = 1
            for m in range(n_monomers):
                for b in range(beads):
                    x = (m * 1.2 + b * 0.1 + 0.01 * fr)
                    y, z = rng.rand() * 0.2, rng.rand() * 0.2
                    fh.write(
                        f"{m + 1:5d}{'ALA':<5s}{f'B{b}':>5s}{atom:5d}"
                        f"{x:8.3f}{y:8.3f}{z:8.3f}\n"
                    )
                    atom += 1
            fh.write(f"{box_nm[0]:10.5f}{box_nm[1]:10.5f}{box_nm[2]:10.5f}\n")


def test_gro_stream_and_topology(tmp_path):
    p = tmp_path / "t.gro"
    _write_gro(p, n_frames=3)
    topo = read_gro_topology(str(p))
    assert topo.n_atoms == 6 and topo.n_residues == 3
    frames = {k: (pos, box) for k, pos, box in GROFrameStream(str(p))}
    assert sorted(frames) == [0, 1, 2]
    pos0, box0 = frames[0]
    assert pos0.shape == (6, 3)
    assert np.allclose(box0[:3], 80.0) and np.allclose(box0[3:], 90.0)
    # nm -> angstrom: first atom x is 0.000 nm -> 0 A; second monomer at 1.2 nm -> 12 A
    assert abs(pos0[2, 0] - 12.0) < 0.51
    sub = {k for k, _, _ in GROFrameStream(str(p), wanted={2})}
    assert sub == {2}


def test_gro_precompute_runs(tmp_path):
    p = tmp_path / "t.gro"
    _write_gro(p, n_frames=3)
    out = tmp_path / "c.npz"
    cfg = PrecomputeConfig(
        topology_file=None, traj_file=str(p), frames=[0, 1, 2],
        cutoff_contact=6.0, intra_exclude=0,
        segment_mode="atom_blocks", atoms_per_monomer=2, n_monomers=3,
    )
    precompute_cache_pure(cfg, out_npz=str(out), verbose=False)
    cache = load_cache(str(out))
    assert cache.cmaps.shape == (3, 3, 3)
    assert cache.meta["traj_format"] == "gro"


def test_trr_requires_topology(tmp_path):
    p = tmp_path / "t.trr"
    p.write_bytes(b"")
    cfg = PrecomputeConfig(
        topology_file=None, traj_file=str(p), frames=[0],
        cutoff_contact=6.0, intra_exclude=0,
        segment_mode="atom_blocks", atoms_per_monomer=2, n_monomers=3,
    )
    with pytest.raises(ValueError, match="topology"):
        precompute_cache_pure(cfg, out_npz=str(tmp_path / "x.npz"), verbose=False)


def test_xtc_requires_topology(tmp_path):
    # Since CoAnaPure 1.2, .xtc is read natively -- but like .trr it carries no
    # residue identity, so a topology file is mandatory.
    p = tmp_path / "t.xtc"
    p.write_bytes(b"")
    cfg = PrecomputeConfig(
        topology_file=None, traj_file=str(p), frames=[0],
        cutoff_contact=6.0, intra_exclude=0,
        segment_mode="atom_blocks", atoms_per_monomer=2, n_monomers=3,
    )
    with pytest.raises(ValueError, match="topology"):
        precompute_cache_pure(cfg, out_npz=str(tmp_path / "x.npz"), verbose=False)


# --------------------------------------------------------------------------- #
# Cross-checks against MDAnalysis (skip when not installed)
# --------------------------------------------------------------------------- #
def test_trr_reader_matches_mdanalysis(tmp_path):
    mda = pytest.importorskip("MDAnalysis")
    n_mono, beads, n_frames = 5, 3, 4
    n_atoms = n_mono * beads
    rng = np.random.RandomState(21)
    u = mda.Universe.empty(
        n_atoms, n_residues=n_mono,
        atom_resindex=np.repeat(np.arange(n_mono), beads),
        residue_segindex=np.zeros(n_mono, dtype=int), trajectory=True,
    )
    u.add_TopologyAttr("names", [f"B{i % beads}" for i in range(n_atoms)])
    u.add_TopologyAttr("resnames", ["ALA"] * n_mono)
    u.add_TopologyAttr("resids", np.arange(1, n_mono + 1))

    trr = str(tmp_path / "t.trr")
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mda.Writer(trr, n_atoms=n_atoms) as w:
            for fr in range(n_frames):
                pos = (rng.rand(n_atoms, 3) * 30 + fr).astype(np.float32)
                frames.append(pos)
                u.atoms.positions = pos
                u.dimensions = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
                w.write(u.atoms)

    ours = {k: (pos, box) for k, pos, box in TRRFrameStream(trr)}
    assert sorted(ours) == list(range(n_frames))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ut = mda.Universe.empty(n_atoms, trajectory=True)
        ut.load_new(trr)
        for k, ts in enumerate(ut.trajectory):
            pos, box = ours[k]
            assert np.allclose(pos, ts.positions, atol=1e-3)
            assert np.allclose(box, ts.dimensions, atol=1e-4)
    sub = {k for k, _, _ in TRRFrameStream(trr, wanted={1, 3})}
    assert sub == {1, 3}
