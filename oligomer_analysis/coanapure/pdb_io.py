# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Pure-Python PDB reader for CoAnaPure (no MDAnalysis).

Stage A only needs a small slice of what ``MDAnalysis.Universe`` provides:

* per-atom residue identity, so atoms can be grouped into residues exactly the
  way MDAnalysis groups them;
* per-frame Cartesian coordinates;
* the periodic box *as MDAnalysis attaches it per model*.

Residue segmentation rule (matches MDAnalysis): a new residue starts whenever any
of ``(resSeq, iCode, resName, chainID)`` changes from the previous atom.

Per-frame box rule (reverse-engineered to match the MDAnalysis PDB reader,
verified against MDAnalysis 2.9):

* a ``CRYST1`` inside a ``MODEL``/``ENDMDL`` block is attached to that model;
* a *leading* ``CRYST1`` (before the first ``MODEL``) is attached to frame 0
  **only when the file has at most one model**; in a multi-model trajectory a
  leading ``CRYST1`` is ignored on every frame.

That last case is the important one: a Martini trajectory PDB with 250 models and
a single top-of-file ``CRYST1`` ends up with **no periodic box on any frame** in
MDAnalysis, so the original :mod:`oligomer_analysis.precompute` computes those
trajectories without PBC.  CoAnaPure reproduces this exactly by default so its
cache is bit-comparable with the original pipeline.

Set ``broadcast_box=True`` in :func:`read_pdb` to instead apply the sole
``CRYST1`` box to *every* frame -- the physically correct behaviour for a
constant-volume run that only writes ``CRYST1`` once.  This deviates from
MDAnalysis on purpose and is the recommended setting for new analyses.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np

__all__ = [
    "PDBStructure",
    "ResidueTopology",
    "PDBModelIndex",
    "PDBFrameStream",
    "read_pdb",
    "build_model_index",
]

_DROPPED_BOX_MSG = (
    "This multi-model PDB trajectory carries a leading CRYST1 box that MDAnalysis "
    "semantics drop from every frame, so the analysis will run WITHOUT periodic "
    "boundaries. Pass broadcast_box=True to apply the box to all frames "
    "(physically correct for a constant-volume run), or ignore this warning to "
    "stay bit-compatible with the original pipeline."
)


@dataclass(slots=True)
class PDBStructure:
    """Parsed PDB topology + trajectory.

    Attributes
    ----------
    positions : ndarray (n_frames, n_atoms, 3) float32
    boxes : ndarray (n_frames, 6) float64
        Per-frame box; a row of zeros means "no box for this frame".
    has_box : ndarray (n_frames,) bool
        Whether a valid box is attached to each frame.
    residue_of_atom, res_first_atom, res_atom_counts : residue lookup tables.
    res_resid, res_resname, res_chain : per-residue identity metadata.
    """

    n_atoms: int
    n_frames: int
    n_residues: int
    positions: np.ndarray
    boxes: np.ndarray
    has_box: np.ndarray
    residue_of_atom: np.ndarray
    res_first_atom: np.ndarray
    res_atom_counts: np.ndarray
    res_resid: np.ndarray
    res_resname: List[str]
    res_chain: List[str]

    @property
    def box(self) -> Optional[np.ndarray]:
        """First valid box in the trajectory, or ``None`` if none present."""
        idx = np.flatnonzero(self.has_box)
        return self.boxes[idx[0]].copy() if idx.size else None

    def frame_box(self, frame_index: int) -> Optional[np.ndarray]:
        """Box attached to ``frame_index`` (``None`` if that frame has no box)."""
        if self.has_box[frame_index]:
            return self.boxes[frame_index].copy()
        return None


@dataclass(slots=True)
class ResidueTopology:
    """Format-agnostic residue segmentation tables.

    The minimum topology Stage A needs: which residue each atom belongs to and
    per-residue identity metadata.  :class:`PDBStructure` exposes the same
    attribute set, so either object can drive
    :func:`~oligomer_analysis.coanapure.precompute_pure.build_topology_cache_pure`.
    """

    n_atoms: int
    n_residues: int
    residue_of_atom: np.ndarray
    res_first_atom: np.ndarray
    res_atom_counts: np.ndarray
    res_resid: np.ndarray
    res_resname: List[str]
    res_chain: List[str]

    @classmethod
    def from_pdb_lines(cls, lines: List[str]) -> "ResidueTopology":
        (
            residue_of_atom,
            res_first_atom,
            res_atom_counts,
            res_resid,
            res_resname,
            res_chain,
        ) = _read_topology(lines)
        if residue_of_atom.shape[0] == 0:
            raise ValueError("No ATOM/HETATM records found for topology.")
        return cls(
            n_atoms=int(residue_of_atom.shape[0]),
            n_residues=int(res_first_atom.shape[0]),
            residue_of_atom=residue_of_atom,
            res_first_atom=res_first_atom,
            res_atom_counts=res_atom_counts,
            res_resid=res_resid,
            res_resname=res_resname,
            res_chain=res_chain,
        )

    @classmethod
    def from_pdb_file(cls, path: str) -> "ResidueTopology":
        with open(path, "r", errors="replace") as fh:
            return cls.from_pdb_lines(fh.readlines())


def _parse_cryst1(line: str) -> Optional[np.ndarray]:
    try:
        a = float(line[6:15]); b = float(line[15:24]); c = float(line[24:33])
        alpha = float(line[33:40]); beta = float(line[40:47]); gamma = float(line[47:54])
    except (ValueError, IndexError):
        return None
    box = np.array([a, b, c, alpha, beta, gamma], dtype=np.float64)
    if np.any(box[:3] <= 0):
        return None
    return box


def _atom_identity(line: str):
    resname = line[17:20].strip()
    chain = line[21:22]
    resseq = line[22:26].strip()
    icode = line[26:27]
    return (resseq, icode, resname, chain)


def _read_topology(lines: List[str]):
    """Build residue segmentation from the atom records of the first model."""
    residue_of_atom: List[int] = []
    res_first_atom: List[int] = []
    res_atom_counts: List[int] = []
    res_resid: List[int] = []
    res_resname: List[str] = []
    res_chain: List[str] = []

    prev_key = None
    atom_idx = 0
    started = False
    for line in lines:
        if line.startswith("MODEL"):
            if res_first_atom:  # topology defined by the first model only
                break
            started = True
            continue
        if line.startswith("ENDMDL"):
            if res_first_atom:
                break
            continue
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue

        key = _atom_identity(line)
        if key != prev_key:
            res_first_atom.append(atom_idx)
            res_atom_counts.append(0)
            resseq, icode, resname, chain = key
            try:
                res_resid.append(int(resseq))
            except ValueError:
                res_resid.append(-1)
            res_resname.append(resname)
            res_chain.append(chain)
            prev_key = key
        residue_of_atom.append(len(res_first_atom) - 1)
        res_atom_counts[-1] += 1
        atom_idx += 1

    return (
        np.asarray(residue_of_atom, dtype=np.int32),
        np.asarray(res_first_atom, dtype=np.int64),
        np.asarray(res_atom_counts, dtype=np.int32),
        np.asarray(res_resid, dtype=np.int32),
        res_resname,
        res_chain,
    )


def _parse_frame_slices(slices: List[str], n_atoms: int) -> np.ndarray:
    """Vectorised float parse of one model's 24-char coordinate fields."""
    buf = "".join(slices).encode("ascii", "replace")
    flat = np.frombuffer(buf, dtype="S8").astype(np.float64)
    return flat.reshape(n_atoms, 3).astype(np.float32)


def _parse_frame_lines(lines: List[str], n_atoms: int) -> np.ndarray:
    """Parse one model's raw ATOM/HETATM lines into an ``(n_atoms, 3)`` array.

    Fast path: machine-written trajectories have constant line length, so the
    coordinate columns (30..54) of the whole model can be sliced with a single
    byte-matrix reshape instead of a per-line Python loop.  Ragged or short
    lines fall back to the per-line slice (identical output).
    """
    if lines:
        width = len(lines[0])
        if width >= 55:  # 54 coordinate columns + at least the newline
            uniform = True
            for s in lines:
                if len(s) != width:
                    uniform = False
                    break
            if uniform:
                buf = "".join(lines).encode("ascii", "replace")
                mat = np.frombuffer(buf, dtype=np.uint8).reshape(n_atoms, width)
                flat = np.frombuffer(mat[:, 30:54].tobytes(), dtype="S8").astype(np.float64)
                return flat.reshape(n_atoms, 3).astype(np.float32)
    return _parse_frame_slices([ln[30:54].ljust(24) for ln in lines], n_atoms)


def _parse_frame_lines_bytes(lines: List[bytes], n_atoms: int) -> np.ndarray:
    """Binary-mode counterpart of :func:`_parse_frame_lines` (no newline in
    ``lines``, e.g. from ``bytes.splitlines``)."""
    if lines:
        width = len(lines[0])
        if width >= 54:
            uniform = True
            for s in lines:
                if len(s) != width:
                    uniform = False
                    break
            if uniform:
                mat = np.frombuffer(b"".join(lines), dtype=np.uint8).reshape(n_atoms, width)
                flat = np.frombuffer(mat[:, 30:54].tobytes(), dtype="S8").astype(np.float64)
                return flat.reshape(n_atoms, 3).astype(np.float32)
    return _parse_frame_slices(
        [ln.decode("ascii", "replace")[30:54].ljust(24) for ln in lines], n_atoms
    )


def _read_frames(
    line_iter,
    n_atoms_expected: Optional[int],
    *,
    capture_topology: bool,
    max_frames: Optional[int] = None,
):
    """Stream a PDB, parsing coordinates one model at a time.

    Iterating the file (rather than reading it whole) keeps memory bounded for
    multi-GB trajectories: only the current model's coordinate slices and the
    list of already-parsed ``(n_atoms, 3)`` frame arrays are held.

    Box rule: only a ``CRYST1`` *inside* a model block is attached to that frame.
    A leading ``CRYST1`` (before the first ``MODEL``) is applied to frame 0 only
    when the file has <= 1 model -- matching the MDAnalysis PDB reader.

    When ``capture_topology`` is True the first model's atom records are collected
    so the caller can derive residue segmentation from the trajectory itself.
    """
    frame_arrays: List[np.ndarray] = []
    boxes: List[Optional[np.ndarray]] = []
    leading_box: Optional[np.ndarray] = None
    saw_model = False
    in_block_box: Optional[np.ndarray] = None

    cur_slices: List[str] = []
    n_atoms: Optional[int] = None
    topo_lines: List[str] = []
    capturing_topo = capture_topology

    def flush():
        nonlocal in_block_box, n_atoms
        if cur_slices:
            if n_atoms is None:
                n_atoms = len(cur_slices)
            elif len(cur_slices) != n_atoms:
                raise ValueError(
                    f"Model {len(frame_arrays)} has {len(cur_slices)} atoms but "
                    f"model 0 has {n_atoms}; trajectory atom count is not constant."
                )
            frame_arrays.append(_parse_frame_lines(cur_slices, n_atoms))
            boxes.append(in_block_box)
            cur_slices.clear()
            in_block_box = None

    for line in line_iter:
        rec = line[:6]
        if rec == "CRYST1":
            if saw_model:
                in_block_box = _parse_cryst1(line)
            else:
                leading_box = _parse_cryst1(line)
            continue
        if rec[:5] == "MODEL":
            flush()
            saw_model = True
            if frame_arrays:
                capturing_topo = False  # only the first model defines topology
            if max_frames is not None and len(frame_arrays) >= max_frames:
                break  # enough models read (files that omit ENDMDL)
            continue
        if rec == "ENDMDL":
            flush()
            capturing_topo = False
            if max_frames is not None and len(frame_arrays) >= max_frames:
                break  # stop early once enough models have been read
            continue
        if rec[:4] == "ATOM" or rec == "HETATM":
            cur_slices.append(line)
            if capturing_topo:
                topo_lines.append(line)
    flush()

    if not frame_arrays:
        raise ValueError("No ATOM/HETATM coordinate records found in PDB.")
    if n_atoms_expected is not None and n_atoms != n_atoms_expected:
        raise ValueError(
            f"Coordinate atom count {n_atoms} does not match topology atom count "
            f"{n_atoms_expected}."
        )

    n_frames = len(frame_arrays)
    if leading_box is not None and boxes[0] is None and (
        n_frames == 1 or any(b is not None for b in boxes[1:])
    ):
        # Single model, or GROMACS layout (CRYST1 written before every MODEL):
        # the leading CRYST1 is frame 0's own box.
        boxes[0] = leading_box

    positions = np.stack(frame_arrays, axis=0)
    box_arr = np.zeros((n_frames, 6), dtype=np.float64)
    has_box = np.zeros(n_frames, dtype=bool)
    for k, bx in enumerate(boxes):
        if bx is not None:
            box_arr[k] = bx
            has_box[k] = True
    return positions, box_arr, has_box, leading_box, topo_lines


def read_pdb(
    traj_file: str,
    topology_file: Optional[str] = None,
    *,
    broadcast_box: bool = False,
    max_frames: Optional[int] = None,
) -> PDBStructure:
    """Read a PDB trajectory (and optional separate topology) into arrays.

    Parameters
    ----------
    traj_file : str
        PDB file providing coordinates (one or many ``MODEL`` blocks).
    topology_file : str, optional
        PDB file providing residue/atom identity.  When omitted, the topology is
        taken from the first model of ``traj_file`` (mirrors
        ``MDAnalysis.Universe(traj_file)``).
    broadcast_box : bool
        When True, apply the first valid box found to *every* frame instead of
        only the model that carried the ``CRYST1``.  Default False reproduces the
        MDAnalysis per-model behaviour used by the original pipeline.
    max_frames : int, optional
        Stop reading after this many models.  Lets a caller that only needs the
        first N frames avoid streaming the rest of a multi-GB trajectory.
    """
    # A separate topology file (small) is read whole; otherwise the trajectory's
    # first model provides the topology while it is streamed for coordinates.
    topo_lines_sep: Optional[List[str]] = None
    n_atoms_expected: Optional[int] = None
    if topology_file is not None:
        with open(topology_file, "r", errors="replace") as fh:
            topo_lines_sep = fh.readlines()

    if topo_lines_sep is not None:
        topo = _read_topology(topo_lines_sep)
        n_atoms_expected = int(topo[0].shape[0])

    with open(traj_file, "r", errors="replace") as fh:
        positions, boxes, has_box, leading_box, topo_lines_traj = _read_frames(
            fh, n_atoms_expected, capture_topology=(topo_lines_sep is None),
            max_frames=max_frames,
        )

    if topo_lines_sep is None:
        topo = _read_topology(topo_lines_traj)

    (
        residue_of_atom,
        res_first_atom,
        res_atom_counts,
        res_resid,
        res_resname,
        res_chain,
    ) = topo

    n_atoms = int(residue_of_atom.shape[0])
    if positions.shape[1] != n_atoms:
        raise ValueError(
            f"Topology has {n_atoms} atoms but trajectory has {positions.shape[1]}."
        )

    if broadcast_box:
        # Prefer any per-frame box; otherwise fall back to a leading CRYST1 that
        # MDAnalysis would have dropped for a multi-model file.
        source = None
        if has_box.any():
            source = boxes[np.flatnonzero(has_box)[0]]
        elif leading_box is not None:
            source = leading_box
        if source is not None:
            boxes = np.tile(source, (positions.shape[0], 1))
            has_box = np.ones(positions.shape[0], dtype=bool)
    elif positions.shape[0] > 1 and leading_box is not None and not has_box.any():
        warnings.warn(_DROPPED_BOX_MSG, UserWarning, stacklevel=2)

    return PDBStructure(
        n_atoms=n_atoms,
        n_frames=int(positions.shape[0]),
        n_residues=int(res_first_atom.shape[0]),
        positions=positions,
        boxes=boxes,
        has_box=has_box,
        residue_of_atom=residue_of_atom,
        res_first_atom=res_first_atom,
        res_atom_counts=res_atom_counts,
        res_resid=res_resid,
        res_resname=res_resname,
        res_chain=res_chain,
    )


# --------------------------------------------------------------------------- #
# Streaming access (constant memory) and random access (model byte index)
# --------------------------------------------------------------------------- #

_INDEX_VERSION = 1


def default_index_path(traj_file: str) -> str:
    """Sidecar path where :func:`build_model_index` stores the index by default."""
    return traj_file + ".coanapure-index.npz"


@dataclass(slots=True)
class PDBModelIndex:
    """Byte-offset index over the models of a PDB trajectory.

    Built with one fast binary scan (no float parsing), the index records where
    each model's atom records live in the file plus the per-model box under the
    exact MDAnalysis semantics (leading ``CRYST1`` honoured only for single-model
    files).  With an index, any subset of frames can be read by seeking straight
    to those models instead of streaming the whole file prefix.
    """

    file_size: int
    mtime_ns: int
    n_models: int
    n_atoms: int
    atoms_start: np.ndarray   # (n_models,) int64: byte offset of first atom line
    atoms_end: np.ndarray     # (n_models,) int64: byte offset just past last atom line
    boxes: np.ndarray         # (n_models, 6) float64
    has_box: np.ndarray       # (n_models,) bool
    leading_box: Optional[np.ndarray]

    @classmethod
    def build(cls, traj_file: str) -> "PDBModelIndex":
        st = os.stat(traj_file)
        atoms_start: List[int] = []
        atoms_end: List[int] = []
        boxes: List[Optional[np.ndarray]] = []
        counts: List[int] = []

        leading_box: Optional[np.ndarray] = None
        saw_model = False
        in_block_box: Optional[np.ndarray] = None
        cur_start = -1
        cur_end = -1
        cur_count = 0
        offset = 0

        def flush() -> None:
            nonlocal in_block_box, cur_start, cur_end, cur_count
            if cur_count:
                atoms_start.append(cur_start)
                atoms_end.append(cur_end)
                boxes.append(in_block_box)
                counts.append(cur_count)
                cur_start = -1
                cur_end = -1
                cur_count = 0
                in_block_box = None

        with open(traj_file, "rb") as fh:
            for raw in fh:
                rec = raw[:6]
                if rec == b"CRYST1":
                    parsed = _parse_cryst1(raw.decode("ascii", "replace"))
                    if saw_model:
                        in_block_box = parsed
                    else:
                        leading_box = parsed
                elif rec[:5] == b"MODEL":
                    flush()
                    saw_model = True
                elif rec == b"ENDMDL":
                    flush()
                elif rec[:4] == b"ATOM" or rec == b"HETATM":
                    if cur_count == 0:
                        cur_start = offset
                    cur_end = offset + len(raw)
                    cur_count += 1
                offset += len(raw)
        flush()

        if not counts:
            raise ValueError("No ATOM/HETATM coordinate records found in PDB.")
        n_atoms = counts[0]
        for k, c in enumerate(counts):
            if c != n_atoms:
                raise ValueError(
                    f"Model {k} has {c} atoms but model 0 has {n_atoms}; "
                    "trajectory atom count is not constant."
                )

        n_models = len(counts)
        box_arr = np.zeros((n_models, 6), dtype=np.float64)
        hb = np.zeros(n_models, dtype=bool)
        for k, bx in enumerate(boxes):
            if bx is not None:
                box_arr[k] = bx
                hb[k] = True
        # MDAnalysis rule: a leading CRYST1 applies only to a single-model file.
        # GROMACS writes CRYST1 before every MODEL; then the leading one is
        # frame 0's own box (MDAnalysis drops it, leaving frame 0 unboxed).
        if leading_box is not None and not hb[0] and (n_models == 1 or hb[1:].any()):
            box_arr[0] = leading_box
            hb[0] = True

        return cls(
            file_size=int(st.st_size),
            mtime_ns=int(st.st_mtime_ns),
            n_models=n_models,
            n_atoms=int(n_atoms),
            atoms_start=np.asarray(atoms_start, dtype=np.int64),
            atoms_end=np.asarray(atoms_end, dtype=np.int64),
            boxes=box_arr,
            has_box=hb,
            leading_box=leading_box,
        )

    def save(self, index_file: str) -> None:
        np.savez_compressed(
            index_file,
            version=np.int64(_INDEX_VERSION),
            file_size=np.int64(self.file_size),
            mtime_ns=np.int64(self.mtime_ns),
            n_models=np.int64(self.n_models),
            n_atoms=np.int64(self.n_atoms),
            atoms_start=self.atoms_start,
            atoms_end=self.atoms_end,
            boxes=self.boxes,
            has_box=self.has_box,
            leading_present=np.bool_(self.leading_box is not None),
            leading_box=(
                self.leading_box
                if self.leading_box is not None
                else np.zeros(6, dtype=np.float64)
            ),
        )

    @classmethod
    def load(cls, index_file: str, traj_file: Optional[str] = None) -> "PDBModelIndex":
        with np.load(index_file) as d:
            if int(d["version"]) != _INDEX_VERSION:
                raise ValueError(f"Unsupported index version in {index_file!r}.")
            idx = cls(
                file_size=int(d["file_size"]),
                mtime_ns=int(d["mtime_ns"]),
                n_models=int(d["n_models"]),
                n_atoms=int(d["n_atoms"]),
                atoms_start=d["atoms_start"].astype(np.int64),
                atoms_end=d["atoms_end"].astype(np.int64),
                boxes=d["boxes"].astype(np.float64),
                has_box=d["has_box"].astype(bool),
                leading_box=(
                    d["leading_box"].astype(np.float64)
                    if bool(d["leading_present"])
                    else None
                ),
            )
        if traj_file is not None:
            st = os.stat(traj_file)
            if int(st.st_size) != idx.file_size or int(st.st_mtime_ns) != idx.mtime_ns:
                raise ValueError(
                    f"Index {index_file!r} is stale for {traj_file!r} "
                    "(file size or mtime changed); rebuild it with build_model_index()."
                )
        return idx


def build_model_index(traj_file: str, index_file: Optional[str] = None) -> PDBModelIndex:
    """Build a model byte-offset index for ``traj_file`` and save it.

    The index enables true random access: requesting frames ``[1900, 1950]`` of a
    multi-GB trajectory then seeks directly to those two models instead of
    parsing frames 0..1899 first.  Saved to ``default_index_path(traj_file)``
    unless ``index_file`` is given.
    """
    idx = PDBModelIndex.build(traj_file)
    idx.save(index_file if index_file is not None else default_index_path(traj_file))
    return idx


class PDBFrameStream:
    """Iterate ``(frame_index, positions, box)`` over a PDB trajectory.

    This is the constant-memory core of CoAnaPure Stage A: frames are yielded
    one at a time, so peak memory is one frame plus file buffers no matter how
    long the trajectory is.  Two access modes:

    * **sequential** (default): one pass over the file; models outside ``wanted``
      are scanned but not decoded, and reading stops after the last wanted model.
    * **indexed** (``index=`` a :class:`PDBModelIndex`): seeks straight to each
      wanted model -- true random access for sparse frame subsets.

    Box semantics match :func:`read_pdb` (and MDAnalysis): only an in-model
    ``CRYST1`` attaches to a frame; a leading ``CRYST1`` applies only to a
    single-model file.  ``broadcast_box=True`` instead applies the first valid
    box (leading, else first in-model) to every yielded frame.  When a leading
    box is being dropped under the default semantics, a :class:`UserWarning` is
    emitted (silence it with ``warn_dropped_box=False``).

    After the first model has been read, ``self.topology`` holds the
    :class:`ResidueTopology` derived from it (unless ``capture_topology=False``).
    """

    def __init__(
        self,
        traj_file: str,
        *,
        wanted: Optional[Iterable[int]] = None,
        broadcast_box: bool = False,
        index: Optional[PDBModelIndex] = None,
        capture_topology: bool = True,
        n_atoms_expected: Optional[int] = None,
        warn_dropped_box: bool = True,
    ) -> None:
        self.traj_file = traj_file
        self.wanted: Optional[Set[int]] = (
            None if wanted is None else {int(f) for f in wanted}
        )
        if self.wanted is not None:
            bad = [f for f in self.wanted if f < 0]
            if bad:
                raise IndexError(f"Negative frame indices not supported: {sorted(bad)}")
        self.broadcast_box = bool(broadcast_box)
        self.index = index
        self.capture_topology = bool(capture_topology)
        self.n_atoms_expected = n_atoms_expected
        self.warn_dropped_box = bool(warn_dropped_box)
        self.topology: Optional[ResidueTopology] = None
        self.n_models_seen = 0

    # -- helpers ---------------------------------------------------------- #

    def _want(self, k: int) -> bool:
        return self.wanted is None or k in self.wanted

    def _check_count(self, k: int, count: int, expected: Optional[int]) -> int:
        if expected is not None and count != expected:
            raise ValueError(
                f"Model {k} has {count} atoms but {expected} were expected; "
                "trajectory atom count is not constant."
            )
        return count

    def _warn_dropped(self) -> None:
        if self.warn_dropped_box:
            warnings.warn(_DROPPED_BOX_MSG, UserWarning, stacklevel=3)

    # -- indexed access --------------------------------------------------- #

    def _decode_block(self, fh, k: int, idx: "PDBModelIndex") -> np.ndarray:
        fh.seek(int(idx.atoms_start[k]))
        blob = fh.read(int(idx.atoms_end[k]) - int(idx.atoms_start[k]))
        lines: List[bytes] = []
        for raw in blob.splitlines():
            rec = raw[:6]
            if rec[:4] == b"ATOM" or rec == b"HETATM":
                lines.append(raw)
        self._check_count(k, len(lines), idx.n_atoms)
        return _parse_frame_lines_bytes(lines, len(lines))

    def _iter_indexed(self) -> Iterator[Tuple[int, np.ndarray, Optional[np.ndarray]]]:
        idx = self.index
        assert idx is not None
        self.n_models_seen = idx.n_models
        if self.n_atoms_expected is not None and idx.n_atoms != self.n_atoms_expected:
            raise ValueError(
                f"Coordinate atom count {idx.n_atoms} does not match topology "
                f"atom count {self.n_atoms_expected}."
            )
        frames = (
            sorted(self.wanted) if self.wanted is not None else list(range(idx.n_models))
        )
        out_of_range = [f for f in frames if f >= idx.n_models]
        if out_of_range:
            raise IndexError(
                f"Requested frame {out_of_range[0]} out of range 0..{idx.n_models - 1}"
            )

        source: Optional[np.ndarray] = None
        if self.broadcast_box:
            if idx.has_box.any():
                source = idx.boxes[int(np.flatnonzero(idx.has_box)[0])].copy()
            elif idx.leading_box is not None:
                source = idx.leading_box.copy()
        elif idx.n_models > 1 and idx.leading_box is not None and not idx.has_box.any():
            self._warn_dropped()

        with open(self.traj_file, "rb") as fh:
            if self.capture_topology and self.topology is None:
                fh.seek(int(idx.atoms_start[0]))
                blob = fh.read(int(idx.atoms_end[0]) - int(idx.atoms_start[0]))
                lines = [
                    raw.decode("ascii", "replace")
                    for raw in blob.splitlines()
                    if raw[:4] == b"ATOM" or raw[:6] == b"HETATM"
                ]
                self.topology = ResidueTopology.from_pdb_lines(lines)
            for k in frames:
                pos = self._decode_block(fh, k, idx)
                if self.broadcast_box:
                    box = source.copy() if source is not None else None
                else:
                    box = idx.boxes[k].copy() if idx.has_box[k] else None
                yield k, pos, box

    # -- sequential access ------------------------------------------------ #

    def _iter_sequential(self) -> Iterator[Tuple[int, np.ndarray, Optional[np.ndarray]]]:
        wanted = self.wanted
        stop_after = max(wanted) if wanted else None
        state = {
            "n_atoms": self.n_atoms_expected,
            "in_block_box": None,
            "cur_count": 0,
            "frames_done": 0,
            "capturing_topo": self.capture_topology,
            "pending": None,   # frame 0 held until single/multi-model is known
            "in_block_seen": False,
            "warned": False,
            "source": None,    # broadcast box source
        }
        leading_box: Optional[np.ndarray] = None
        saw_model = False
        cur_slices: List[str] = []
        topo_lines: List[str] = []

        def flush() -> List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]]:
            ready: List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]] = []
            if state["cur_count"] == 0:
                return ready
            k = state["frames_done"]
            state["n_atoms"] = self._check_count(k, state["cur_count"], state["n_atoms"])
            if state["capturing_topo"]:
                self.topology = ResidueTopology.from_pdb_lines(topo_lines)
                state["capturing_topo"] = False
            pos = (
                _parse_frame_lines(cur_slices, state["cur_count"])
                if cur_slices
                else None
            )
            box = state["in_block_box"]
            if box is not None:
                state["in_block_seen"] = True
                if state["source"] is None:
                    state["source"] = box

            if state["pending"] is not None:
                # A second frame exists, so the leading box does NOT apply to
                # frame 0 (multi-model MDAnalysis rule).
                pk, ppos, _ = state["pending"]
                state["pending"] = None
                if box is not None:
                    # Per-model CRYST1 layout: the leading box is frame 0's own.
                    ready.append((pk, ppos, leading_box))
                else:
                    if not self.broadcast_box and not state["in_block_seen"] and not state["warned"]:
                        self._warn_dropped()
                        state["warned"] = True
                    ready.append((pk, ppos, None))

            if (
                k == 0
                and box is None
                and leading_box is not None
                and not self.broadcast_box
            ):
                state["pending"] = (k, pos, None)
            else:
                ready.append((k, pos, box))

            cur_slices.clear()
            state["cur_count"] = 0
            state["in_block_box"] = None
            state["frames_done"] += 1
            return ready

        def emit(items):
            for k, pos, box in items:
                self.n_models_seen = max(self.n_models_seen, k + 1)
                if pos is None:  # frame not wanted -> not decoded
                    continue
                if self.broadcast_box and box is None:
                    box = state["source"]
                yield k, pos, (None if box is None else np.array(box, dtype=np.float64))

        with open(self.traj_file, "r", errors="replace") as fh:
            for line in fh:
                rec = line[:6]
                if rec == "CRYST1":
                    parsed = _parse_cryst1(line)
                    if saw_model:
                        state["in_block_box"] = parsed
                    else:
                        leading_box = parsed
                        if self.broadcast_box and parsed is not None and state["source"] is None:
                            state["source"] = parsed
                    continue
                if rec[:5] == "MODEL":
                    for item in emit(flush()):
                        yield item
                    saw_model = True
                    if (
                        stop_after is not None
                        and state["frames_done"] > stop_after
                        and state["pending"] is None
                    ):
                        # A further model exists and nothing is pending -> done.
                        self.n_models_seen = max(
                            self.n_models_seen, state["frames_done"] + 1
                        )
                        return
                    continue
                if rec == "ENDMDL":
                    for item in emit(flush()):
                        yield item
                    if stop_after is not None and state["frames_done"] > stop_after:
                        if state["pending"] is None:
                            # Peek ahead for another MODEL record so early-stop
                            # cannot mislabel a multi-model file.
                            for line2 in fh:
                                if line2[:5] == "MODEL":
                                    self.n_models_seen = max(
                                        self.n_models_seen, state["frames_done"] + 1
                                    )
                                    break
                            return
                        # Frame 0 is pending: scan on until the single/multi
                        # question is decided by the next MODEL record or EOF.
                        next_has_box = False
                        for line2 in fh:
                            if line2[:6] == "CRYST1":
                                next_has_box = True
                            if line2[:5] == "MODEL":
                                for item in emit(
                                    [(state["pending"][0], state["pending"][1],
                                      leading_box if next_has_box else None)]
                                ):
                                    yield item
                                if not next_has_box and not self.broadcast_box and not state["warned"]:
                                    self._warn_dropped()
                                    state["warned"] = True
                                state["pending"] = None
                                self.n_models_seen = max(
                                    self.n_models_seen, state["frames_done"] + 1
                                )
                                return
                        pk, ppos, _ = state["pending"]
                        state["pending"] = None
                        for item in emit([(pk, ppos, leading_box)]):
                            yield item
                        return
                    continue
                if rec[:4] == "ATOM" or rec == "HETATM":
                    state["cur_count"] += 1
                    if self._want(state["frames_done"]):
                        cur_slices.append(line)
                    if state["capturing_topo"]:
                        topo_lines.append(line)
                    continue
            # EOF
            for item in emit(flush()):
                yield item
            if state["pending"] is not None:
                # Single-model file: the leading CRYST1 applies to frame 0.
                pk, ppos, _ = state["pending"]
                state["pending"] = None
                for item in emit([(pk, ppos, leading_box)]):
                    yield item

        if state["frames_done"] == 0:
            raise ValueError("No ATOM/HETATM coordinate records found in PDB.")
        if wanted is not None:
            missing = sorted(f for f in wanted if f >= state["frames_done"])
            if missing:
                raise IndexError(
                    f"Requested frame {missing[0]} out of range "
                    f"0..{state['frames_done'] - 1}"
                )

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, Optional[np.ndarray]]]:
        if self.index is not None:
            return self._iter_indexed()
        return self._iter_sequential()
