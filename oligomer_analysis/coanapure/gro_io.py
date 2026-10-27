# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""GROMACS .gro reader for CoAnaPure (pure NumPy, no MDAnalysis).

A ``.gro`` file stores one or more concatenated frames::

    title line (may contain "t= <time>")
    number of atoms
    <n_atoms> fixed-column atom lines
    box line (3 or 9 free-format floats, nm)

Atom line columns: resid [0:5], resname [5:10], atom name [10:15],
atom number [15:20], x/y/z [20:28]/[28:36]/[36:44] (nm, 3 decimals); optional
velocities follow and are ignored.  Coordinates and boxes are converted to
angstrom (x10) so caches are unit-compatible with the PDB reader.

Residue segmentation matches MDAnalysis' GRO reader: a new residue starts
whenever ``(resid, resname)`` changes from the previous atom, and the file has
no chain concept (every residue gets chain ``""``, i.e. one segment under
``segment_mode="mdanalysis"`` -- use ``atom_blocks``/``residue_blocks`` for
multi-monomer systems, exactly as with PDB input).
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np

from .pbc import box6_from_vectors
from .pdb_io import ResidueTopology

__all__ = ["read_gro_topology", "GROFrameStream"]

_NM_TO_ANGSTROM = 10.0


def _parse_gro_box(line: str) -> Optional[np.ndarray]:
    """Free-format GROMACS box line (nm) -> 6-vector box in angstrom, or None."""
    try:
        vals = [float(x) for x in line.split()]
    except ValueError:
        return None
    if len(vals) >= 9:
        # v1x v2y v3z v1y v1z v2x v2z v3x v3y
        m = np.array(
            [
                [vals[0], vals[3], vals[4]],
                [vals[5], vals[1], vals[6]],
                [vals[7], vals[8], vals[2]],
            ],
            dtype=np.float64,
        )
    elif len(vals) >= 3:
        m = np.diag(vals[:3]).astype(np.float64)
    else:
        return None
    box6 = box6_from_vectors(m * _NM_TO_ANGSTROM)
    return box6


def _parse_gro_coords(lines: List[str]) -> np.ndarray:
    """Vectorised float parse of the fixed 24-char coordinate columns (nm)."""
    buf = "".join(line[20:44].ljust(24) for line in lines).encode("ascii", "replace")
    flat = np.frombuffer(buf, dtype="S8").astype(np.float64)
    return (flat.reshape(len(lines), 3) * _NM_TO_ANGSTROM).astype(np.float32)


def _topology_from_gro_lines(atom_lines: List[str]) -> ResidueTopology:
    residue_of_atom: List[int] = []
    res_first_atom: List[int] = []
    res_atom_counts: List[int] = []
    res_resid: List[int] = []
    res_resname: List[str] = []
    res_chain: List[str] = []

    prev_key = None
    for atom_idx, line in enumerate(atom_lines):
        resid = line[0:5].strip()
        resname = line[5:10].strip()
        key = (resid, resname)
        if key != prev_key:
            res_first_atom.append(atom_idx)
            res_atom_counts.append(0)
            try:
                res_resid.append(int(resid))
            except ValueError:
                res_resid.append(-1)
            res_resname.append(resname)
            res_chain.append("")
            prev_key = key
        residue_of_atom.append(len(res_first_atom) - 1)
        res_atom_counts[-1] += 1

    if not residue_of_atom:
        raise ValueError("No atom records found in .gro file.")
    return ResidueTopology(
        n_atoms=len(residue_of_atom),
        n_residues=len(res_first_atom),
        residue_of_atom=np.asarray(residue_of_atom, dtype=np.int32),
        res_first_atom=np.asarray(res_first_atom, dtype=np.int64),
        res_atom_counts=np.asarray(res_atom_counts, dtype=np.int32),
        res_resid=np.asarray(res_resid, dtype=np.int32),
        res_resname=res_resname,
        res_chain=res_chain,
    )


def read_gro_topology(path: str) -> ResidueTopology:
    """Residue segmentation tables from the first frame of a ``.gro`` file."""
    with open(path, "r", errors="replace") as fh:
        fh.readline()  # title
        n_line = fh.readline()
        try:
            n_atoms = int(n_line.strip())
        except ValueError as exc:
            raise ValueError(f"Malformed .gro atom-count line: {n_line!r}") from exc
        atom_lines = [fh.readline() for _ in range(n_atoms)]
    if not atom_lines or atom_lines[-1] == "":
        raise ValueError("Truncated .gro file: fewer atom lines than declared.")
    return _topology_from_gro_lines(atom_lines)


class GROFrameStream:
    """Iterate ``(frame_index, positions_angstrom, box6)`` over a ``.gro`` file.

    Mirrors :class:`~oligomer_analysis.coanapure.pdb_io.PDBFrameStream`:
    constant-memory streaming, decode-skipping for frames outside ``wanted``,
    early stop after the last wanted frame, and ``self.topology`` captured from
    the first frame (unless ``capture_topology=False``).
    """

    def __init__(
        self,
        path: str,
        *,
        wanted: Optional[Iterable[int]] = None,
        capture_topology: bool = True,
        n_atoms_expected: Optional[int] = None,
    ) -> None:
        self.path = path
        self.wanted: Optional[Set[int]] = (
            None if wanted is None else {int(f) for f in wanted}
        )
        if self.wanted is not None and any(f < 0 for f in self.wanted):
            raise IndexError("Negative frame indices not supported.")
        self.capture_topology = bool(capture_topology)
        self.n_atoms_expected = n_atoms_expected
        self.topology: Optional[ResidueTopology] = None
        self.n_frames_seen = 0

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, Optional[np.ndarray]]]:
        wanted = self.wanted
        stop_after = max(wanted) if wanted else None
        n_atoms_ref = self.n_atoms_expected
        k = 0
        with open(self.path, "r", errors="replace") as fh:
            while True:
                title = fh.readline()
                if title == "":
                    break  # clean EOF between frames
                n_line = fh.readline()
                if n_line == "":
                    raise ValueError(f"Truncated .gro frame {k}: missing atom count.")
                try:
                    n_atoms = int(n_line.strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed .gro atom-count line in frame {k}: {n_line!r}"
                    ) from exc
                if n_atoms_ref is None:
                    n_atoms_ref = n_atoms
                elif n_atoms != n_atoms_ref:
                    raise ValueError(
                        f"Frame {k} has {n_atoms} atoms but {n_atoms_ref} were "
                        "expected; trajectory atom count is not constant."
                    )

                decode = wanted is None or k in wanted
                need_lines = decode or (self.capture_topology and k == 0)
                if need_lines:
                    atom_lines = [fh.readline() for _ in range(n_atoms)]
                    if atom_lines and atom_lines[-1] == "":
                        raise ValueError(f"Truncated .gro frame {k}.")
                else:
                    atom_lines = None
                    for _ in range(n_atoms):
                        fh.readline()
                box_line = fh.readline()
                if box_line == "":
                    raise ValueError(f"Truncated .gro frame {k}: missing box line.")

                if self.capture_topology and k == 0 and self.topology is None:
                    self.topology = _topology_from_gro_lines(atom_lines)

                self.n_frames_seen = k + 1
                if decode:
                    yield k, _parse_gro_coords(atom_lines), _parse_gro_box(box_line)
                if stop_after is not None and k >= stop_after:
                    return
                k += 1

        if k == 0:
            raise ValueError("No frames found in .gro file.")
        if wanted is not None:
            missing = sorted(f for f in wanted if f >= k)
            if missing:
                raise IndexError(f"Requested frame {missing[0]} out of range 0..{k - 1}")
