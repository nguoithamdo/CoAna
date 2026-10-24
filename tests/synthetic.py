"""Synthetic trajectories and caches shared by the test modules.

Every fixture here is small enough that the expected answer can be worked out by
hand, which is the point: a test that only checks a pipeline runs would pass on
a decomposition that silently mislabels half its contacts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from oligomer_analysis.models import ContactCache

__all__ = [
    "write_gro",
    "two_species_trajectory",
    "growth_trajectory",
    "GROWTH_SCHEDULE",
    "cache_from_cmaps",
    "chain_positions",
]


def write_gro(
    path,
    *,
    frames: Sequence[np.ndarray],
    resnames: Sequence[str],
    atom_names: Sequence[str],
    resids: Sequence[int],
    box: Tuple[float, float, float] = (10.0, 10.0, 10.0),
) -> None:
    """Write a (possibly multi-frame) GROMACS ``.gro`` file.

    Coordinates are in nanometres, as the format requires; the readers convert
    to the angstrom used everywhere downstream.
    """
    n_atoms = len(resnames)
    for k, pos in enumerate(frames):
        if pos.shape != (n_atoms, 3):
            raise ValueError(f"frame {k} has shape {pos.shape}, expected {(n_atoms, 3)}")

    lines: List[str] = []
    for k, pos in enumerate(frames):
        lines.append(f"synthetic frame {k}")
        lines.append(f"{n_atoms:5d}")
        for i in range(n_atoms):
            lines.append(
                f"{resids[i] % 100000:5d}{resnames[i]:<5s}{atom_names[i]:>5s}"
                f"{(i + 1) % 100000:5d}"
                f"{pos[i, 0]:8.3f}{pos[i, 1]:8.3f}{pos[i, 2]:8.3f}"
            )
        lines.append(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}")
    path = str(path)
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def chain_positions(
    n_residues: int,
    origin: Tuple[float, float, float],
    *,
    spacing: float = 0.05,
    beads_per_residue: int = 1,
) -> np.ndarray:
    """A straight chain of residues laid along x from ``origin`` (nanometres)."""
    pts = []
    for r in range(n_residues):
        for b in range(beads_per_residue):
            pts.append(
                (
                    origin[0] + r * spacing + b * spacing * 0.1,
                    origin[1],
                    origin[2],
                )
            )
    return np.asarray(pts, dtype=np.float64)


def two_species_trajectory(
    tmp_path,
    *,
    n_a: int = 3,
    n_b: int = 2,
    res_a: Sequence[str] = ("ALA", "ASP", "LYS", "PHE"),
    res_b: Sequence[str] = ("GLU", "LEU"),
    separation: float = 3.0,
    contact_gap: float = 0.04,
) -> Dict[str, object]:
    """A two-species box whose contacts are arranged by construction.

    Frame 0 keeps every chain far apart (no inter-chain contacts).  Frame 1 walks
    all chains onto one line spaced by ``contact_gap``, so neighbouring chains
    touch and the whole box becomes a single cluster.  Frame 2 restores the
    separation for chains of species B only, leaving species A clustered.

    Returns the ``.gro`` path together with the species spec and the residue
    lists, so a test can predict the class matrix from the residue names alone.
    """
    res_a = list(res_a)
    res_b = list(res_b)
    resnames: List[str] = []
    resids: List[int] = []
    chain_lengths: List[int] = []
    for _ in range(n_a):
        resnames.extend(res_a)
        resids.extend(range(1, len(res_a) + 1))
        chain_lengths.append(len(res_a))
    for _ in range(n_b):
        resnames.extend(res_b)
        resids.extend(range(1, len(res_b) + 1))
        chain_lengths.append(len(res_b))
    atom_names = ["BB"] * len(resnames)

    def build(gaps: Sequence[float]) -> np.ndarray:
        out = []
        y = 0.0
        for length, gap in zip(chain_lengths, gaps):
            out.append(chain_positions(length, (0.0, y, 0.0), spacing=0.02))
            y += gap
        return np.concatenate(out, axis=0)

    far = [separation] * len(chain_lengths)
    near = [contact_gap] * len(chain_lengths)
    # Gaps apply *after* each chain, so the last A chain must carry the
    # separation or the first B chain would land on the A line.
    mixed = [contact_gap] * (n_a - 1) + [separation] * (n_b + 1)

    frames = [build(far), build(near), build(mixed)]
    gro = tmp_path / "two_species.gro"
    write_gro(
        gro,
        frames=frames,
        resnames=resnames,
        atom_names=atom_names,
        resids=resids,
        box=(50.0, 50.0, 50.0),
    )
    return {
        "gro": str(gro),
        "n_frames": len(frames),
        "species_spec": f"A:{len(res_a)}x{n_a},B:{len(res_b)}x{n_b}",
        "monomer_atom_counts": [len(res_a)] * n_a + [len(res_b)] * n_b,
        "resnames": resnames,
        "res_a": res_a,
        "res_b": res_b,
        "n_a": n_a,
        "n_b": n_b,
    }


# Frame at which each membership set takes effect; chains 0..5 are species A,
# chains 6 and 7 are species B.  Read downwards: an A dimer nucleates, grows to
# a trimer, B0 joins at host size 3 and never leaves, the fourth A chain and B1
# arrive together (the assembly *commits* at four host chains here), B1 leaves
# five frames later while the assembly holds, and the last two A chains join.
GROWTH_SCHEDULE: List[Tuple[int, Tuple[int, ...]]] = [
    (0, ()),
    (5, (0, 1)),
    (10, (0, 1, 2)),
    (15, (0, 1, 2, 6)),
    (20, (0, 1, 2, 3, 6, 7)),
    (25, (0, 1, 2, 3, 6)),
    (30, (0, 1, 2, 3, 4, 5, 6)),
]


def growth_trajectory(
    tmp_path,
    *,
    n_a: int = 6,
    n_b: int = 2,
    n_frames: int = 40,
    schedule: Optional[Sequence[Tuple[int, Tuple[int, ...]]]] = None,
    res_a: Sequence[str] = ("ALA", "ASP", "LYS", "PHE"),
    res_b: Sequence[str] = ("GLU", "LEU"),
    separation: float = 3.0,
    contact_gap: float = 0.04,
) -> Dict[str, object]:
    """A two-species box in which one assembly nucleates and grows on a schedule.

    ``schedule`` lists ``(start_frame, members)`` pairs; from ``start_frame``
    onwards exactly the chains in ``members`` sit on one contact line and every
    other chain is isolated.  The default :data:`GROWTH_SCHEDULE` gives, with a
    commitment size of four host chains, one B chain that is *incorporated* and
    one that is *catalytic* -- the two mechanistic outcomes a residence analysis
    has to be able to tell apart.

    Returns the ``.gro`` path, the species spec, the schedule actually used and
    ``members_at`` (frame -> member tuple) so a test can derive any expected
    count from the construction rather than from the code under test.
    """
    schedule = list(GROWTH_SCHEDULE if schedule is None else schedule)
    res_a = list(res_a)
    res_b = list(res_b)
    chain_lengths = [len(res_a)] * n_a + [len(res_b)] * n_b
    n_chains = len(chain_lengths)
    for _start, members in schedule:
        for c in members:
            if not 0 <= c < n_chains:
                raise ValueError(f"schedule names chain {c}; only {n_chains} chains exist")

    resnames: List[str] = []
    resids: List[int] = []
    for _ in range(n_a):
        resnames.extend(res_a)
        resids.extend(range(1, len(res_a) + 1))
    for _ in range(n_b):
        resnames.extend(res_b)
        resids.extend(range(1, len(res_b) + 1))
    atom_names = ["BB"] * len(resnames)

    def members_at(frame: int) -> Tuple[int, ...]:
        current: Tuple[int, ...] = ()
        for start, members in schedule:
            if frame >= start:
                current = tuple(sorted(members))
        return current

    def build(frame: int) -> np.ndarray:
        members = members_at(frame)
        slot = {c: k for k, c in enumerate(members)}
        out = []
        for c, length in enumerate(chain_lengths):
            if c in slot:
                y = slot[c] * contact_gap
            else:
                # Isolated chains sit on a second line far from the assembly
                # and from each other.
                y = 10.0 + c * separation
            out.append(chain_positions(length, (0.0, y, 0.0), spacing=0.02))
        return np.concatenate(out, axis=0)

    frames = [build(f) for f in range(n_frames)]
    gro = tmp_path / "growth.gro"
    write_gro(
        gro,
        frames=frames,
        resnames=resnames,
        atom_names=atom_names,
        resids=resids,
        box=(50.0, 50.0, 50.0),
    )
    return {
        "gro": str(gro),
        "n_frames": n_frames,
        "species_spec": f"A:{len(res_a)}x{n_a},B:{len(res_b)}x{n_b}",
        "monomer_atom_counts": chain_lengths,
        "resnames": resnames,
        "res_a": res_a,
        "res_b": res_b,
        "n_a": n_a,
        "n_b": n_b,
        "schedule": schedule,
        "members_at": {f: members_at(f) for f in range(n_frames)},
    }


def cache_from_cmaps(
    cmaps: Sequence[np.ndarray],
    *,
    frames: Optional[Sequence[int]] = None,
    meta: Optional[Dict] = None,
    class_cmaps: Optional[np.ndarray] = None,
    hotspot_maps: Optional[np.ndarray] = None,
) -> ContactCache:
    """A minimal in-memory cache built from hand-written contact maps."""
    cmaps_arr = np.asarray(cmaps, dtype=np.int32)
    n_frames, nseg = cmaps_arr.shape[0], cmaps_arr.shape[1]
    frames_arr = np.asarray(
        list(range(n_frames)) if frames is None else list(frames), dtype=np.int32
    )
    adj = cmaps_arr >= 1
    diag = np.arange(nseg)
    adj[:, diag, diag] = False
    bound = adj.any(axis=2).sum(axis=1).astype(np.int32)
    ninter = np.triu(cmaps_arr, 1).sum(axis=(1, 2)).astype(np.int32)
    base_meta = {"nseg": int(nseg), "segment_mode": "atom_counts"}
    base_meta.update(meta or {})
    return ContactCache(
        frames=frames_arr,
        cmaps=cmaps_arr,
        nintra=np.zeros(n_frames, dtype=np.int32),
        ninter=ninter,
        ntotal=ninter,
        n_free=(nseg - bound).astype(np.int32),
        n_bound=bound,
        seg_centers=np.zeros((n_frames, nseg, 3), dtype=np.float32),
        boxes=np.tile(
            np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float32),
            (n_frames, 1),
        ),
        meta=base_meta,
        class_cmaps=(None if class_cmaps is None else np.asarray(class_cmaps)),
        hotspot_maps=(None if hotspot_maps is None else np.asarray(hotspot_maps)),
    )
