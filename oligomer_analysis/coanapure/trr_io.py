"""GROMACS .trr reader for CoAnaPure (pure NumPy, no MDAnalysis).

TRR is an XDR (big-endian) container.  Each frame is::

    int32 magic (1993)
    XDR string: format tag ("GMX_trn_file")
    int32 ir_size, e_size, box_size, vir_size, pres_size, top_size, sym_size,
          x_size, v_size, f_size, natoms, step, nre
    real  t, lambda            (float32 or float64, see below)
    payload: box (box_size bytes), vir, pres, x (x_size), v, f

The floating-point precision of a frame is not declared explicitly; it is
inferred from ``box_size / 9`` (or ``x_size / (3 * natoms)`` when the frame has
no box), exactly as GROMACS and MDAnalysis do.  Coordinates and box vectors are
stored in nm and converted to angstrom here (x10) for unit compatibility with
the PDB reader.

TRR carries no residue/atom identity, so Stage A additionally needs a topology
file (``.pdb`` or ``.gro``).

``.xtc`` is read natively by :mod:`~oligomer_analysis.coanapure.xtc_io` (pure
Python 3dfcoord decompressor); TRR remains the faster of the two readers, so
for repeated heavy use of a large XTC consider converting once::

    gmx trjconv -f traj.xtc -s topol.tpr -o traj.trr
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator, Optional, Set, Tuple

import numpy as np

from .pbc import box6_from_vectors

__all__ = ["TRRFrameStream"]

_NM_TO_ANGSTROM = 10.0
_TRR_MAGIC = 1993


def _read_exact(fh, n: int, what: str) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise ValueError(f"Truncated .trr file while reading {what}.")
    return data


def _read_i32(fh, count: int, what: str) -> np.ndarray:
    return np.frombuffer(_read_exact(fh, 4 * count, what), dtype=">i4").astype(np.int64)


class TRRFrameStream:
    """Iterate ``(frame_index, positions_angstrom, box6)`` over a ``.trr`` file.

    Mirrors :class:`~oligomer_analysis.coanapure.pdb_io.PDBFrameStream`:
    constant-memory streaming, payload seeking (not just decode skipping) for
    frames outside ``wanted``, and early stop after the last wanted frame.
    Frames without coordinates (force/velocity-only frames) are skipped and do
    not consume a frame index, matching MDAnalysis.
    """

    def __init__(
        self,
        path: str,
        *,
        wanted: Optional[Iterable[int]] = None,
        n_atoms_expected: Optional[int] = None,
    ) -> None:
        self.path = path
        self.wanted: Optional[Set[int]] = (
            None if wanted is None else {int(f) for f in wanted}
        )
        if self.wanted is not None and any(f < 0 for f in self.wanted):
            raise IndexError("Negative frame indices not supported.")
        self.n_atoms_expected = n_atoms_expected
        self.n_frames_seen = 0

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, Optional[np.ndarray]]]:
        wanted = self.wanted
        stop_after = max(wanted) if wanted else None
        n_atoms_ref = self.n_atoms_expected
        k = 0
        file_size = os.path.getsize(self.path)

        with open(self.path, "rb") as fh:
            while fh.tell() < file_size:
                magic = _read_i32(fh, 1, "frame magic")[0]
                if magic != _TRR_MAGIC:
                    raise ValueError(
                        f"Bad .trr magic {int(magic)} at frame {k} "
                        f"(expected {_TRR_MAGIC}); file corrupt or not TRR."
                    )
                # XDR string: int32 length, bytes padded to a 4-byte boundary.
                slen = int(_read_i32(fh, 1, "format tag length")[0])
                if not (0 <= slen <= 4096):
                    raise ValueError(f"Implausible .trr format-tag length {slen}.")
                _read_exact(fh, (slen + 3) // 4 * 4, "format tag")

                (
                    ir_size, e_size, box_size, vir_size, pres_size,
                    top_size, sym_size, x_size, v_size, f_size,
                    natoms, step, nre,
                ) = (int(v) for v in _read_i32(fh, 13, "frame header"))

                if ir_size or e_size or top_size or sym_size or nre:
                    raise ValueError(
                        f"Frame {k}: legacy TRR blocks (ir/e/top/sym/nre) are "
                        "not supported."
                    )
                if natoms <= 0:
                    raise ValueError(f"Frame {k}: non-positive atom count {natoms}.")
                if n_atoms_ref is None:
                    n_atoms_ref = natoms
                elif natoms != n_atoms_ref:
                    raise ValueError(
                        f"Frame {k} has {natoms} atoms but {n_atoms_ref} were "
                        "expected; trajectory atom count is not constant."
                    )

                if box_size:
                    fsize = box_size // 9
                elif x_size:
                    fsize = x_size // (3 * natoms)
                elif v_size:
                    fsize = v_size // (3 * natoms)
                elif f_size:
                    fsize = f_size // (3 * natoms)
                else:
                    fsize = 4
                if fsize not in (4, 8):
                    raise ValueError(f"Frame {k}: cannot infer TRR precision.")
                fdtype = ">f8" if fsize == 8 else ">f4"

                _read_exact(fh, 2 * fsize, "t and lambda")

                has_coords = x_size > 0
                decode = has_coords and (wanted is None or k in wanted)

                if decode and box_size:
                    m = np.frombuffer(
                        _read_exact(fh, box_size, "box"), dtype=fdtype
                    ).astype(np.float64).reshape(3, 3)
                    box6 = box6_from_vectors(m * _NM_TO_ANGSTROM)
                else:
                    fh.seek(box_size, 1)
                    box6 = None
                fh.seek(vir_size + pres_size, 1)

                if decode:
                    if x_size != 3 * natoms * fsize:
                        raise ValueError(
                            f"Frame {k}: coordinate block size {x_size} does not "
                            f"match {natoms} atoms at {fsize}-byte precision."
                        )
                    x = np.frombuffer(
                        _read_exact(fh, x_size, "coordinates"), dtype=fdtype
                    ).astype(np.float64).reshape(natoms, 3)
                    pos = (x * _NM_TO_ANGSTROM).astype(np.float32)
                else:
                    fh.seek(x_size, 1)
                fh.seek(v_size + f_size, 1)

                if has_coords:
                    self.n_frames_seen = k + 1
                    if decode:
                        yield k, pos, box6
                    if stop_after is not None and k >= stop_after:
                        return
                    k += 1

        if k == 0:
            raise ValueError("No coordinate frames found in .trr file.")
        if wanted is not None:
            missing = sorted(f for f in wanted if f >= k)
            if missing:
                raise IndexError(f"Requested frame {missing[0]} out of range 0..{k - 1}")
