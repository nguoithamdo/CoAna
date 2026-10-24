"""Unit tests for CoAnaPure 1.0 (pure NumPy Stage A).

These tests do not require MDAnalysis.  A subset of cross-checks against
MDAnalysis runs only when it is importable.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from oligomer_analysis.coanapure import (
    minimize_vectors,
    self_capped_pairs,
    read_pdb,
    precompute_cache_pure,
    PrecomputeConfig,
)
from oligomer_analysis.cache_io import load_cache


# --------------------------------------------------------------------------- #
# PBC minimum image
# --------------------------------------------------------------------------- #
def test_minimize_vectors_orthorhombic():
    box = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    v = np.array([[6.0, 0.0, 0.0], [-6.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    out = minimize_vectors(v, box)
    # 6 -> -4, -6 -> 4, 4 stays, 5 -> -5 or 5 (ties allowed)
    assert np.isclose(out[0, 0], -4.0, atol=1e-4)
    assert np.isclose(out[1, 0], 4.0, atol=1e-4)
    assert np.isclose(out[1, 1], 4.0, atol=1e-4)
    assert np.isclose(abs(out[2, 2]), 5.0, atol=1e-4)


def test_minimize_vectors_no_shift_within_half_box():
    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
    v = np.array([[3.0, -4.0, 12.0]])
    out = minimize_vectors(v, box)
    assert np.allclose(out, v, atol=1e-4)


# --------------------------------------------------------------------------- #
# Neighbour search
# --------------------------------------------------------------------------- #
def test_self_capped_pairs_simple():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    pairs = self_capped_pairs(pts, cutoff=2.0, box=None)
    s = set(map(tuple, pairs))
    assert s == {(0, 1)}


def test_self_capped_pairs_pbc_wrap():
    # Two points across a periodic boundary should be neighbours.
    box = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    pts = np.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])
    pairs_pbc = self_capped_pairs(pts, cutoff=1.5, box=box)
    pairs_free = self_capped_pairs(pts, cutoff=1.5, box=None)
    assert set(map(tuple, pairs_pbc)) == {(0, 1)}
    assert len(pairs_free) == 0


def test_self_capped_pairs_matches_bruteforce_random():
    rng = np.random.RandomState(1)
    box = np.array([30.0, 30.0, 30.0, 90.0, 90.0, 90.0])
    pts = rng.rand(400, 3) * 30.0
    cutoff = 3.0
    grid = set(map(tuple, self_capped_pairs(pts, cutoff, box)))
    # brute force
    n = len(pts)
    iu, ju = np.triu_indices(n, k=1)
    diff = minimize_vectors(pts[iu] - pts[ju], box).astype(float)
    d = np.linalg.norm(diff, axis=1)
    brute = set(map(tuple, np.stack([iu[d <= cutoff], ju[d <= cutoff]], axis=1)))
    assert grid == brute


# --------------------------------------------------------------------------- #
# PDB reader + Stage A on a synthetic in-memory PDB
# --------------------------------------------------------------------------- #
def _write_synthetic_pdb(path, n_monomers=4, beads=2, n_frames=3, box=(50, 50, 50)):
    """Two-bead monomers laid out so a known subset are in contact."""
    lines = [f"CRYST1{box[0]:9.3f}{box[1]:9.3f}{box[2]:9.3f}  90.00  90.00  90.00 P 1\n"]
    rng = np.random.RandomState(0)
    base = np.array([[i * 10.0, 0.0, 0.0] for i in range(n_monomers)])
    for fr in range(n_frames):
        lines.append(f"MODEL   {fr + 1:6d}\n")
        serial = 1
        for m in range(n_monomers):
            for b in range(beads):
                pos = base[m] + np.array([b * 1.0, fr * 0.0, 0.0])
                lines.append(
                    f"ATOM  {serial:5d}  B{b:<2d} ALA {chr(65 + m % 26)}{m + 1:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00\n"
                )
                serial += 1
        lines.append("ENDMDL\n")
    with open(path, "w") as fh:
        fh.writelines(lines)


def test_read_pdb_topology(tmp_path):
    p = tmp_path / "syn.pdb"
    _write_synthetic_pdb(str(p), n_monomers=4, beads=2, n_frames=3)
    s = read_pdb(str(p))
    assert s.n_atoms == 8
    assert s.n_frames == 3
    assert s.n_residues == 4          # one residue per monomer here
    assert s.positions.shape == (3, 8, 3)
    # Multi-model + leading CRYST1 -> MDAnalysis drops the box; we mirror that.
    assert not s.has_box.any()


def test_read_pdb_broadcast_box(tmp_path):
    p = tmp_path / "syn.pdb"
    _write_synthetic_pdb(str(p), n_monomers=4, beads=2, n_frames=3)
    s = read_pdb(str(p), broadcast_box=True)
    assert s.has_box.all()
    assert np.allclose(s.boxes[:, :3], [50, 50, 50])


def test_precompute_pure_runs(tmp_path):
    p = tmp_path / "syn.pdb"
    _write_synthetic_pdb(str(p), n_monomers=4, beads=2, n_frames=3)
    out = tmp_path / "cache.npz"
    cfg = PrecomputeConfig(
        topology_file=None, traj_file=str(p), frames=[0, 1, 2],
        cutoff_contact=1.5, intra_exclude=0,
        segment_mode="residue_blocks", residues_per_monomer=1, n_monomers=4,
    )
    precompute_cache_pure(cfg, out_npz=str(out), verbose=False)
    cache = load_cache(str(out))
    assert len(cache.frames) == 3
    assert cache.cmaps.shape == (3, 4, 4)
    assert cache.meta["nseg"] == 4


# --------------------------------------------------------------------------- #
# Optional cross-check against MDAnalysis
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.path.exists(r"E:\LinuxData\project\People\Thu_huongdan_2025\NganKHSK\500frame.pdb"),
    reason="reference Martini PDB not available",
)
def test_matches_mdanalysis_single_frame():
    mda = pytest.importorskip("MDAnalysis")
    from oligomer_analysis.precompute import precompute_cache

    ref = r"E:\LinuxData\project\People\Thu_huongdan_2025\NganKHSK\500frame.pdb"
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        common = dict(
            topology_file=None, traj_file=ref, frames=[0],
            cutoff_contact=6.5, intra_exclude=2,
            segment_mode="atom_blocks", atoms_per_monomer=83, n_monomers=128,
        )
        mda_npz = os.path.join(d, "mda.npz")
        pure_npz = os.path.join(d, "pure.npz")
        precompute_cache(PrecomputeConfig(n_jobs=1, chunk_size=1, **common), out_npz=mda_npz)
        precompute_cache_pure(PrecomputeConfig(**common), out_npz=pure_npz, verbose=False)

        a = load_cache(mda_npz)
        b = load_cache(pure_npz)
        assert np.array_equal(a.cmaps, b.cmaps)
        assert np.array_equal(a.nintra, b.nintra)
        assert np.array_equal(a.ninter, b.ninter)
        assert np.allclose(a.seg_centers, b.seg_centers, atol=1e-2)
