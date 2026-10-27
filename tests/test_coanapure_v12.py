# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Tests for the CoAnaPure 1.2 additions.

Covers: atom selections (equivalence with a physically filtered file),
unequal-monomer segmentation (atom_counts), checkpoint/resume (interrupted run
resumes to a byte-identical cache), process-parallel execution, the vectorised
PDB fast path on ragged files, the native XTC reader (cross-checked against an
MDAnalysis-written file when MDAnalysis is available), and progress reporting.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from oligomer_analysis.cache_io import load_cache
from oligomer_analysis.coanapure import (
    PrecomputeConfig,
    XTCFrameStream,
    apply_selection,
    atom_mask_from_selection,
    count_xtc_frames,
    parse_selection,
    precompute_cache_pure,
    read_pdb,
)
from oligomer_analysis.coanapure import precompute_pure as pp
from oligomer_analysis.coanapure.pdb_io import ResidueTopology


# --------------------------------------------------------------------------- #
# Synthetic builders
# --------------------------------------------------------------------------- #
def _write_pdb(
    path,
    n_frames=4,
    n_monomers=4,
    beads=2,
    solvent=0,
    monomer_beads=None,
    leading_box=None,
):
    """Multi-model PDB: ``n_monomers`` protein monomers (``beads`` beads each, or
    ``monomer_beads`` per-monomer counts) plus ``solvent`` far-away W beads."""
    per_monomer = monomer_beads if monomer_beads is not None else [beads] * n_monomers
    lines = []
    if leading_box is not None:
        lines.append(
            f"CRYST1{leading_box[0]:9.3f}{leading_box[1]:9.3f}"
            f"{leading_box[2]:9.3f}  90.00  90.00  90.00 P 1\n"
        )
    rng = np.random.RandomState(5)
    coords = {}
    for fr in range(n_frames):
        lines.append(f"MODEL   {fr + 1:6d}\n")
        serial = 1
        for m, nb in enumerate(per_monomer):
            for b in range(nb):
                key = (m, b)
                if key not in coords:
                    coords[key] = (rng.rand() * 2, rng.rand() * 2)
                y0, z0 = coords[key]
                x = m * 10.0 + b + 0.1 * fr
                lines.append(
                    f"ATOM  {serial:5d}  B{b:<2d} ALA {chr(65 + (m % 26))}{m + 1:4d}    "
                    f"{x:8.3f}{y0:8.3f}{z0:8.3f}  1.00  0.00\n"
                )
                serial += 1
        for w in range(solvent):
            lines.append(
                f"ATOM  {serial:5d}  W   W   Z{900 + w:4d}    "
                f"{1000.0 + 5 * w:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00\n"
            )
            serial += 1
        lines.append("ENDMDL\n")
    with open(path, "w") as fh:
        fh.writelines(lines)


def _cfg(traj, frames, **kw):
    base = dict(
        topology_file=None, traj_file=str(traj), frames=list(frames),
        cutoff_contact=1.8, intra_exclude=0, n_jobs=1,
        segment_mode="atom_blocks", atoms_per_monomer=2, n_monomers=4,
    )
    base.update(kw)
    return PrecomputeConfig(**base)


def _load_arrays(path):
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files if k != "meta_json"}


def _assert_same_arrays(path_a, path_b):
    da, db = _load_arrays(path_a), _load_arrays(path_b)
    assert set(da) == set(db)
    for k, va in da.items():
        assert np.array_equal(va, db[k]), k


# --------------------------------------------------------------------------- #
# Selection: parsing
# --------------------------------------------------------------------------- #
def test_parse_selection_grammar():
    assert parse_selection("all") == (False, "all", [])
    assert parse_selection("resname W ION") == (False, "resname", ["W", "ION"])
    assert parse_selection("not resname W") == (True, "resname", ["W"])
    assert parse_selection("index 0-5 9") == (False, "index", ["0-5", "9"])
    for bad in ("", "not", "banana A", "resname", "all extra"):
        with pytest.raises(ValueError):
            parse_selection(bad)


def test_selection_masks(tmp_path):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=1, n_monomers=4, beads=2, solvent=3)
    topo = ResidueTopology.from_pdb_file(str(p))
    assert topo.n_atoms == 11

    m = atom_mask_from_selection(topo, "not resname W")
    assert m.sum() == 8 and m[:8].all() and not m[8:].any()
    m2 = atom_mask_from_selection(topo, "resname ALA")
    assert np.array_equal(m, m2)
    m3 = atom_mask_from_selection(topo, "index 0-7")
    assert np.array_equal(m, m3)
    m4 = atom_mask_from_selection(topo, "not chain Z")
    assert np.array_equal(m, m4)
    m5 = atom_mask_from_selection(topo, "resid 900-902")
    assert m5.sum() == 3 and m5[8:].all()

    with pytest.raises(ValueError, match="no atoms"):
        atom_mask_from_selection(topo, "resname XYZ")
    with pytest.raises(ValueError, match="outside"):
        atom_mask_from_selection(topo, "index 0-99")

    mask, sub = apply_selection(topo, "not resname W")
    assert sub.n_atoms == 8 and sub.n_residues == 4
    assert mask.sum() == 8
    none_mask, same = apply_selection(topo, None)
    assert none_mask is None and same is topo
    all_mask, same2 = apply_selection(topo, "all")
    assert all_mask is None and same2 is topo


def test_selection_equals_filtered_file(tmp_path):
    """Precompute with selection == precompute on a file with atoms removed."""
    full, filt = tmp_path / "full.pdb", tmp_path / "filtered.pdb"
    _write_pdb(full, n_frames=4, n_monomers=4, beads=2, solvent=5)
    _write_pdb(filt, n_frames=4, n_monomers=4, beads=2, solvent=0)

    a, b = tmp_path / "sel.npz", tmp_path / "ref.npz"
    precompute_cache_pure(
        _cfg(full, range(4), selection="not resname W"), out_npz=str(a), verbose=False
    )
    precompute_cache_pure(_cfg(filt, range(4)), out_npz=str(b), verbose=False)
    _assert_same_arrays(a, b)

    cache = load_cache(str(a))
    assert cache.meta["selection"] == "not resname W"
    assert cache.meta["selection_n_atoms"] == 8
    assert cache.meta["nseg"] == 4


def test_selection_with_separate_topology(tmp_path):
    full, filt = tmp_path / "full.pdb", tmp_path / "filtered.pdb"
    _write_pdb(full, n_frames=3, n_monomers=4, beads=2, solvent=4)
    _write_pdb(filt, n_frames=3, n_monomers=4, beads=2, solvent=0)
    a, b = tmp_path / "sel.npz", tmp_path / "ref.npz"
    precompute_cache_pure(
        _cfg(full, range(3), topology_file=str(full), selection="index 0-7"),
        out_npz=str(a), verbose=False,
    )
    precompute_cache_pure(_cfg(filt, range(3)), out_npz=str(b), verbose=False)
    _assert_same_arrays(a, b)


# --------------------------------------------------------------------------- #
# atom_counts segmentation (unequal monomers)
# --------------------------------------------------------------------------- #
def test_atom_counts_equal_blocks_match(tmp_path):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=3, n_monomers=4, beads=2)
    a, b = tmp_path / "counts.npz", tmp_path / "blocks.npz"
    precompute_cache_pure(
        _cfg(p, range(3), segment_mode="atom_counts", atoms_per_monomer=None,
             n_monomers=None, monomer_atom_counts=[2, 2, 2, 2]),
        out_npz=str(a), verbose=False,
    )
    precompute_cache_pure(_cfg(p, range(3)), out_npz=str(b), verbose=False)
    _assert_same_arrays(a, b)
    assert load_cache(str(a)).meta["segment_mode"] == "atom_counts"


def test_atom_counts_unequal(tmp_path):
    p = tmp_path / "t.pdb"
    # monomers of 2, 4 and 2 beads; monomers spaced 10 A apart, cutoff 1.8 A
    _write_pdb(p, n_frames=2, monomer_beads=[2, 4, 2])
    out = tmp_path / "c.npz"
    precompute_cache_pure(
        _cfg(p, range(2), segment_mode="atom_counts", atoms_per_monomer=None,
             n_monomers=3, monomer_atom_counts=[2, 4, 2]),
        out_npz=str(out), verbose=False,
    )
    cache = load_cache(str(out))
    assert cache.cmaps.shape == (2, 3, 3)
    assert cache.meta["monomer_atom_counts"] == [2, 4, 2]


def test_atom_counts_validation(tmp_path):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=1, monomer_beads=[2, 4, 2])
    with pytest.raises(ValueError, match="sum"):
        precompute_cache_pure(
            _cfg(p, [0], segment_mode="atom_counts", atoms_per_monomer=None,
                 n_monomers=None, monomer_atom_counts=[2, 4, 4]),
            out_npz=str(tmp_path / "x.npz"), verbose=False,
        )
    with pytest.raises(ValueError, match="monomer_atom_counts"):
        precompute_cache_pure(
            _cfg(p, [0], segment_mode="atom_counts", atoms_per_monomer=None,
                 n_monomers=None, monomer_atom_counts=None),
            out_npz=str(tmp_path / "x.npz"), verbose=False,
        )


# --------------------------------------------------------------------------- #
# Checkpoint / resume
# --------------------------------------------------------------------------- #
def test_checkpoint_resume_after_interrupt(tmp_path, monkeypatch):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=8)
    ref = tmp_path / "ref.npz"
    precompute_cache_pure(_cfg(p, range(8)), out_npz=str(ref), verbose=False)

    out = tmp_path / "resumed.npz"
    real = pp._compute_frame_arrays
    calls = {"n": 0}

    def dying(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 5:
            raise RuntimeError("simulated walltime kill")
        return real(*args, **kwargs)

    monkeypatch.setattr(pp, "_compute_frame_arrays", dying)
    with pytest.raises(RuntimeError, match="walltime"):
        precompute_cache_pure(
            _cfg(p, range(8)), out_npz=str(out), verbose=False, checkpoint_every=2
        )
    monkeypatch.setattr(pp, "_compute_frame_arrays", real)

    ckpt = str(out) + ".checkpoint.npz"
    assert os.path.exists(ckpt), "interrupted run must leave a checkpoint"
    with np.load(ckpt, allow_pickle=True) as d:
        n_done = len(d["frames"])
    assert 1 <= n_done < 8

    precompute_cache_pure(
        _cfg(p, range(8)), out_npz=str(out), verbose=False,
        checkpoint_every=2, resume=True,
    )
    _assert_same_arrays(ref, out)
    assert not os.path.exists(ckpt), "checkpoint must be removed after success"


def test_checkpoint_fingerprint_mismatch_ignored(tmp_path):
    from oligomer_analysis.models import FrameContacts

    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=4)
    out = tmp_path / "c.npz"
    # Manufacture a checkpoint for a DIFFERENT cutoff, then resume with another.
    cfg_a = _cfg(p, range(4), cutoff_contact=1.0)
    fp_a = pp._config_fingerprint(cfg_a, False)
    bogus = FrameContacts(
        frame=0, nintra=99, ninter=99, ntotal=198, n_free=0, n_bound=4,
        cmap=np.full((4, 4), 9, dtype=np.int32),
        seg_centers=np.zeros((4, 3), dtype=np.float32),
        box=np.zeros(6, dtype=np.float32),
    )
    pp._write_checkpoint(
        str(out) + ".checkpoint.npz",
        [bogus],
        fp_a,
        {"nseg": 4, "segids": [], "segment_mode": "atom_blocks",
         "topology_n_atoms": 8, "topology_n_residues": 4, "traj_format": "pdb"},
    )
    ref = tmp_path / "ref.npz"
    cfg_b = _cfg(p, range(4))  # cutoff 1.8 != 1.0
    precompute_cache_pure(cfg_b, out_npz=str(ref), verbose=False)
    precompute_cache_pure(cfg_b, out_npz=str(out), verbose=False, resume=True)
    _assert_same_arrays(ref, out)


def test_full_resume_without_recompute(tmp_path, monkeypatch):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=4)
    ref = tmp_path / "ref.npz"
    cfg = _cfg(p, range(4))
    precompute_cache_pure(cfg, out_npz=str(ref), verbose=False)

    # Build a complete checkpoint, then resume: no frame may be recomputed.
    out = tmp_path / "full.npz"
    ck = str(out) + ".checkpoint.npz"
    ref_cache = load_cache(str(ref))
    from oligomer_analysis.models import FrameContacts

    results = [
        FrameContacts(
            frame=int(ref_cache.frames[i]), nintra=int(ref_cache.nintra[i]),
            ninter=int(ref_cache.ninter[i]), ntotal=int(ref_cache.ntotal[i]),
            n_free=int(ref_cache.n_free[i]), n_bound=int(ref_cache.n_bound[i]),
            cmap=ref_cache.cmaps[i], seg_centers=ref_cache.seg_centers[i],
            box=ref_cache.boxes[i],
        )
        for i in range(4)
    ]
    topo_meta = {k: ref_cache.meta[k] for k in
                 ("nseg", "segids", "segment_mode", "atoms_per_monomer",
                  "topology_n_atoms", "topology_n_residues", "traj_format")}
    pp._write_checkpoint(ck, results, pp._config_fingerprint(cfg, False), topo_meta)

    def boom(*a, **k):
        raise AssertionError("full resume must not recompute any frame")

    monkeypatch.setattr(pp, "_compute_frame_arrays", boom)
    precompute_cache_pure(cfg, out_npz=str(out), verbose=False, resume=True)
    _assert_same_arrays(ref, out)


# --------------------------------------------------------------------------- #
# Process-parallel mode
# --------------------------------------------------------------------------- #
def test_process_parallel_equals_serial(tmp_path):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=8)
    a, b = tmp_path / "serial.npz", tmp_path / "procs.npz"
    precompute_cache_pure(_cfg(p, range(8)), out_npz=str(a), verbose=False)
    precompute_cache_pure(
        _cfg(p, range(8), n_jobs=2), out_npz=str(b), verbose=False, parallel="process"
    )
    _assert_same_arrays(a, b)
    cache = load_cache(str(b))
    assert cache.meta["parallel"] == "process"
    assert cache.meta["traj_format"] == "pdb+index"
    # the index sidecar was built automatically
    assert os.path.exists(str(p) + ".coanapure-index.npz")


def test_process_parallel_falls_back_for_gro(tmp_path):
    gro = tmp_path / "t.gro"
    with open(gro, "w") as fh:
        for fr in range(2):
            fh.write(f"frame {fr}\n    4\n")
            for m in range(2):
                for b in range(2):
                    x = m * 1.2 + b * 0.1
                    fh.write(f"{m + 1:5d}{'ALA':<5s}{f'B{b}':>5s}{m * 2 + b + 1:5d}{x:8.3f}{0.0:8.3f}{0.0:8.3f}\n")
            fh.write("   8.00000   8.00000   8.00000\n")
    cfg = _cfg(gro, range(2), n_jobs=2, n_monomers=2)
    with pytest.warns(UserWarning, match="falling back to threads"):
        precompute_cache_pure(
            cfg, out_npz=str(tmp_path / "c.npz"), verbose=False, parallel="process"
        )


def test_parallel_arg_validated(tmp_path):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=1)
    with pytest.raises(ValueError, match="parallel"):
        precompute_cache_pure(
            _cfg(p, [0]), out_npz=str(tmp_path / "c.npz"), verbose=False,
            parallel="banana",
        )


# --------------------------------------------------------------------------- #
# Vectorised PDB fast path
# --------------------------------------------------------------------------- #
def test_ragged_lines_match_uniform(tmp_path):
    uniform, ragged = tmp_path / "u.pdb", tmp_path / "r.pdb"
    _write_pdb(uniform, n_frames=3)
    text = uniform.read_text()
    # Strip trailing columns from every second ATOM line: coordinates intact,
    # but line lengths become ragged -> exercises the fallback path.
    out_lines = []
    for i, line in enumerate(text.splitlines(keepends=True)):
        if line.startswith("ATOM") and i % 2 == 0:
            out_lines.append(line[:54].rstrip() + "\n")
        else:
            out_lines.append(line)
    ragged.write_text("".join(out_lines))

    su, sr = read_pdb(str(uniform)), read_pdb(str(ragged))
    assert np.array_equal(su.positions, sr.positions)

    a, b = tmp_path / "u.npz", tmp_path / "r.npz"
    precompute_cache_pure(_cfg(uniform, range(3)), out_npz=str(a), verbose=False)
    precompute_cache_pure(_cfg(ragged, range(3)), out_npz=str(b), verbose=False)
    _assert_same_arrays(a, b)


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #
def test_progress_line_printed(tmp_path, capsys):
    p = tmp_path / "t.pdb"
    _write_pdb(p, n_frames=3)
    precompute_cache_pure(
        _cfg(p, range(3)), out_npz=str(tmp_path / "c.npz"),
        verbose=True, progress=True,
    )
    captured = capsys.readouterr().out
    assert "[3/3]" in captured and "frames/s" in captured and "ETA" in captured


# --------------------------------------------------------------------------- #
# Native XTC reader (cross-checked against MDAnalysis-written files)
# --------------------------------------------------------------------------- #
def _write_xtc_with_mda(tmp_path, n_atoms, n_frames, *, with_box=True, seed=7):
    mda = pytest.importorskip("MDAnalysis")
    rng = np.random.RandomState(seed)
    u = mda.Universe.empty(n_atoms, trajectory=True)
    path = str(tmp_path / "t.xtc")
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mda.Writer(path, n_atoms=n_atoms) as w:
            for fr in range(n_frames):
                pos = (rng.rand(n_atoms, 3) * 30 + fr).astype(np.float32)
                frames.append(pos)
                u.atoms.positions = pos
                if with_box:
                    u.dimensions = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
                w.write(u.atoms)
    return path, frames


def test_xtc_reader_matches_mdanalysis(tmp_path):
    mda = pytest.importorskip("MDAnalysis")
    path, _ = _write_xtc_with_mda(tmp_path, n_atoms=57, n_frames=5)

    ours = {k: (pos, box) for k, pos, box in XTCFrameStream(path)}
    assert sorted(ours) == list(range(5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ut = mda.Universe.empty(57, trajectory=True)
        ut.load_new(path)
        for k, ts in enumerate(ut.trajectory):
            pos, box = ours[k]
            assert np.allclose(pos, ts.positions, atol=2e-3), f"frame {k}"
            assert np.allclose(box, ts.dimensions, atol=1e-3)

    assert count_xtc_frames(path) == 5
    sub = {k for k, _, _ in XTCFrameStream(path, wanted={1, 3})}
    assert sub == {1, 3}
    with pytest.raises(IndexError, match="out of range"):
        list(XTCFrameStream(path, wanted={99}))


def test_xtc_small_system_uncompressed_path(tmp_path):
    mda = pytest.importorskip("MDAnalysis")
    path, frames = _write_xtc_with_mda(tmp_path, n_atoms=6, n_frames=3)
    ours = {k: pos for k, pos, _ in XTCFrameStream(path)}
    for k in range(3):
        assert np.allclose(ours[k], frames[k], atol=1e-3)


def test_xtc_precompute_runs(tmp_path):
    mda = pytest.importorskip("MDAnalysis")
    n_mono, beads = 4, 3
    n_atoms = n_mono * beads
    rng = np.random.RandomState(3)
    base = np.repeat(np.arange(n_mono)[:, None] * 20.0, beads, axis=0)
    base = np.concatenate([base, np.zeros((n_atoms, 2))], axis=1)

    topo_pdb = tmp_path / "topo.pdb"
    with open(topo_pdb, "w") as fh:
        serial = 1
        for m in range(n_mono):
            for b in range(beads):
                fh.write(
                    f"ATOM  {serial:5d}  B{b:<2d} ALA {chr(65 + m)}{m + 1:4d}    "
                    f"{base[serial - 1, 0]:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00\n"
                )
                serial += 1

    u = mda.Universe.empty(n_atoms, trajectory=True)
    xtc = str(tmp_path / "t.xtc")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mda.Writer(xtc, n_atoms=n_atoms) as w:
            for fr in range(4):
                jitter = rng.rand(n_atoms, 3) * 0.5
                u.atoms.positions = (base + jitter + fr * 0.01).astype(np.float32)
                w.write(u.atoms)

    out = tmp_path / "c.npz"
    cfg = PrecomputeConfig(
        topology_file=str(topo_pdb), traj_file=xtc, frames=[0, 1, 2, 3],
        cutoff_contact=2.5, intra_exclude=0,
        segment_mode="atom_blocks", atoms_per_monomer=beads, n_monomers=n_mono,
    )
    precompute_cache_pure(cfg, out_npz=str(out), verbose=False)
    cache = load_cache(str(out))
    assert cache.meta["traj_format"] == "xtc"
    assert cache.cmaps.shape == (4, n_mono, n_mono)
    # monomers 20 A apart, cutoff 2.5 A -> nothing binds
    assert int(cache.n_free.min()) == n_mono
