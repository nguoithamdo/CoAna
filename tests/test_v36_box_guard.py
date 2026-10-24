"""3.6.0: a missing periodic box is never silent.

What must hold:
  * the CLI refuses a trajectory without a box unless --allow-no-box is given;
  * the Python API warns (NoBoxWarning) and records meta["has_box"];
  * a PDB in GROMACS layout (CRYST1 before every MODEL) keeps frame 0's box
    on every read path -- whole-file read, model index, and the streaming
    reader stopping after frame 0;
  * the MDAnalysis rule is unchanged where it is the only information: one
    leading CRYST1 in a multi-model file gives no box without broadcast_box;
  * ``coana benchmark --help`` prints help.

The system: two 2-bead chains 1 A apart across the x face of a 20 A box, so
the contact exists only under periodic boundaries.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from oligomer_analysis.cache_io import load_cache
from oligomer_analysis.cli import main
from oligomer_analysis.coanapure import precompute_cache_pure
from oligomer_analysis.coanapure.pdb_io import PDBModelIndex, read_pdb
from oligomer_analysis.coanapure.precompute_pure import NoBoxWarning
from oligomer_analysis.models import PrecomputeConfig

L = 20.0
CRYST1 = f"CRYST1{L:9.3f}{L:9.3f}{L:9.3f}  90.00  90.00  90.00 P 1           1\n"
# chain A at x = 0.3/0.5, chain B at x = 19.5/19.7 -> 0.6 A apart by minimum image
XS = [(0.3, 0.5), (19.7, 19.5)]


def _atoms():
    out, serial = [], 1
    for m, xs in enumerate(XS):
        for b, x in enumerate(xs):
            out.append(f"ATOM  {serial:5d}  B{b:<2d} ALA {chr(65 + m)}{m + 1:4d}    "
                       f"{x:8.3f}{10.0:8.3f}{10.0:8.3f}  1.00  0.00\n")
            serial += 1
    return out


def _write(path, n_frames=3, layout="gromacs"):
    """layout: 'gromacs' (CRYST1 before each MODEL), 'leading' (one CRYST1), 'none'."""
    lines = []
    if layout == "leading":
        lines.append(CRYST1)
    for fr in range(n_frames):
        if layout == "gromacs":
            lines.append(CRYST1)
        lines.append(f"MODEL   {fr + 1:6d}\n")
        lines += _atoms()
        lines.append("ENDMDL\n")
    path.write_text("".join(lines))
    return path


def _cfg(traj, frames):
    return PrecomputeConfig(
        topology_file=None, traj_file=str(traj), frames=list(frames),
        cutoff_contact=1.0, intra_exclude=0, n_jobs=1,
        segment_mode="atom_blocks", atoms_per_monomer=2, n_monomers=2,
    )


# ---- frame 0 keeps its box in GROMACS layout --------------------------------

def test_read_pdb_keeps_frame0_box(tmp_path):
    s = read_pdb(str(_write(tmp_path / "g.pdb")))
    assert s.has_box.all()
    assert np.allclose(np.asarray(s.boxes)[:, :3], L)


def test_model_index_keeps_frame0_box(tmp_path):
    idx = PDBModelIndex.build(str(_write(tmp_path / "g.pdb")))
    assert idx.has_box.all()


@pytest.mark.parametrize("frames", [[0], [0, 1, 2]])
def test_stage_a_sees_the_contact_across_the_face(tmp_path, frames):
    out = tmp_path / "c.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("error", NoBoxWarning)
        precompute_cache_pure(_cfg(_write(tmp_path / "g.pdb"), frames),
                              out_npz=str(out), verbose=False)
    c = load_cache(str(out))
    assert c.meta["has_box"] is True
    assert (c.cmaps[:, 0, 1] > 0).all(), "periodic contact lost"


def test_single_leading_box_still_follows_mdanalysis(tmp_path):
    with pytest.warns(UserWarning):
        s = read_pdb(str(_write(tmp_path / "l.pdb", layout="leading")))
    assert not s.has_box.any()
    s = read_pdb(str(_write(tmp_path / "l2.pdb", layout="leading")), broadcast_box=True)
    assert s.has_box.all()


# ---- no box: API warns, CLI refuses ----------------------------------------

def test_api_warns_and_records_missing_box(tmp_path):
    out = tmp_path / "c.npz"
    with pytest.warns(NoBoxWarning):
        precompute_cache_pure(_cfg(_write(tmp_path / "n.pdb", layout="none"), [0, 1]),
                              out_npz=str(out), verbose=False)
    c = load_cache(str(out))
    assert c.meta["has_box"] is False
    assert (c.cmaps[:, 0, 1] == 0).all()


def _cli(pdb, out, *extra):
    return main(["precompute", "--traj", str(pdb), "--cutoff", "1.0",
                 "--intra-exclude", "0", "--segment-mode", "atom_blocks",
                 "--atoms-per-monomer", "2", "--n-monomers", "2", "-o", str(out),
                 "--quiet", *extra])


def test_cli_refuses_a_trajectory_without_a_box(tmp_path, capsys):
    out = tmp_path / "c.npz"
    assert _cli(_write(tmp_path / "n.pdb", layout="none"), out) == 2
    assert not out.exists()
    assert "--allow-no-box" in capsys.readouterr().err


def test_cli_allow_no_box_runs_and_records_it(tmp_path):
    out = tmp_path / "c.npz"
    with pytest.warns(NoBoxWarning):
        assert _cli(_write(tmp_path / "n.pdb", layout="none"), out, "--allow-no-box") == 0
    assert load_cache(str(out)).meta["has_box"] is False


def test_cli_accepts_a_boxed_trajectory(tmp_path):
    out = tmp_path / "c.npz"
    assert _cli(_write(tmp_path / "g.pdb"), out) == 0
    assert load_cache(str(out)).meta["has_box"] is True


def test_cli_broadcast_box_satisfies_the_guard(tmp_path):
    out = tmp_path / "c.npz"
    assert _cli(_write(tmp_path / "l.pdb", layout="leading"), out, "--broadcast-box") == 0


def test_benchmark_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["benchmark", "--help"])
    assert exc.value.code == 0
    assert "--traj" in capsys.readouterr().out
