# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later AND BSD-2-Clause
#
# The XTC decompressor in this file is a port of xdr3dfcoord (Frans van Hoesel)
# from the xdrfile library, used under the following license:
#
# Copyright (c) 2009-2014, Erik Lindahl & David van der Spoel
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""GROMACS .xtc reader for CoAnaPure (pure Python/NumPy, no MDAnalysis).

XTC is an XDR (big-endian) container whose coordinates are compressed with the
lossy *3dfcoord* scheme (coordinates quantised to ``1/precision`` nm, typically
0.001 nm = 0.01 A, then bit-packed with an adaptive integer coder).  Each frame
is::

    int32  magic (1995)
    int32  natoms
    int32  step
    float  time
    9 x float  box vectors (3x3, nm, row-major)
    int32  lsize (== natoms)
    if lsize <= 9:                      # tiny systems are stored uncompressed
        3*lsize floats
    else:
        float  precision
        int32  minint[3], maxint[3]
        int32  smallidx
        int32  nbytes  +  nbytes opaque bytes (padded to a 4-byte boundary)

The decompressor below is a faithful port of ``xdr3dfcoord`` from Frans van
Hoesel's xdrfile library (the reference implementation used by GROMACS and
MDAnalysis), so decoded integers match those readers exactly; the only
difference downstream is float rounding at the last decimal place.

Because every frame header stores its own compressed byte count, frames outside
``wanted`` are *skipped by seeking* without decompression -- sparse frame
subsets of a long .xtc are cheap.  Decompression itself runs in Python and is
the slowest of CoAnaPure's readers (~10-100 ms/frame depending on system size);
for repeated heavy use of a large trajectory, ``.trr`` stays the faster choice
(``gmx trjconv -f traj.xtc -o traj.trr``).

XTC carries no residue/atom identity, so Stage A additionally needs a topology
file (``.pdb`` or ``.gro``).  Coordinates and box are converted nm -> angstrom
(x10) for unit compatibility with the PDB reader.
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator, Optional, Set, Tuple

import numpy as np

from .pbc import box6_from_vectors

__all__ = ["XTCFrameStream", "count_xtc_frames"]

_NM_TO_ANGSTROM = 10.0
_XTC_MAGIC = 1995

# Quantisation table from xdrfile.c; index 9 (FIRSTIDX) is the smallest usable size.
_MAGICINTS = (
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
    80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812, 1024, 1290,
    1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003,
    16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031,
    131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561,
    832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
    4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216,
)
_FIRSTIDX = 9


def _read_exact(fh, n: int, what: str) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise ValueError(f"Truncated .xtc file while reading {what}.")
    return data


def _sizeofint(size: int) -> int:
    """Smallest bit count n with 2**n > size (xdrfile ``sizeofint``)."""
    num = 1
    nbits = 0
    while size >= num and nbits < 32:
        nbits += 1
        num <<= 1
    return nbits


def _sizeofints(sizes) -> int:
    """Bits needed for the mixed-radix triple (xdrfile ``sizeofints``)."""
    prod = 1
    for s in sizes:
        prod *= int(s)
    return prod.bit_length()


def _decompress_frame(
    data: bytes,
    lsize: int,
    minint: Tuple[int, int, int],
    maxint: Tuple[int, int, int],
    smallidx: int,
) -> np.ndarray:
    """Decode the packed integer coordinates of one frame.

    Faithful port of the read path of ``xdr3dfcoord`` (xdrfile).  Returns the
    raw quantised integers as an ``(lsize, 3)`` int64 array; the caller divides
    by ``precision``.  All decoder state lives in locals for speed -- this loop
    runs once per atom.
    """
    sizeint = (
        maxint[0] - minint[0] + 1,
        maxint[1] - minint[1] + 1,
        maxint[2] - minint[2] + 1,
    )
    if (sizeint[0] | sizeint[1] | sizeint[2]) > 0xFFFFFF:
        bitsizeint = (_sizeofint(sizeint[0]), _sizeofint(sizeint[1]), _sizeofint(sizeint[2]))
        bitsize = 0
    else:
        bitsizeint = (0, 0, 0)
        bitsize = _sizeofints(sizeint)

    if smallidx >= len(_MAGICINTS) or _MAGICINTS[smallidx] == 0:
        raise ValueError(f"Invalid .xtc smallidx {smallidx}.")
    smaller = _MAGICINTS[max(_FIRSTIDX, smallidx - 1)] // 2
    smallnum = _MAGICINTS[smallidx] // 2
    sizesmall = _MAGICINTS[smallidx]

    # Bit reader state (decodebits in xdrfile): a 32-bit shift register.
    cnt = 0
    lastbits = 0
    lastbyte = 0

    def decodebits(nbits: int) -> int:
        nonlocal cnt, lastbits, lastbyte
        mask = (1 << nbits) - 1
        num = 0
        while nbits >= 8:
            lastbyte = ((lastbyte << 8) | data[cnt]) & 0xFFFFFFFF
            cnt += 1
            num |= (lastbyte >> lastbits) << (nbits - 8)
            nbits -= 8
        if nbits > 0:
            if lastbits < nbits:
                lastbits += 8
                lastbyte = ((lastbyte << 8) | data[cnt]) & 0xFFFFFFFF
                cnt += 1
            lastbits -= nbits
            num |= (lastbyte >> lastbits) & ((1 << nbits) - 1)
        return num & mask

    def decodeints(nbits: int, sz0: int, sz1: int, sz2: int) -> Tuple[int, int, int]:
        # Read nbits as little-endian 8-bit digits, then peel z and y off the
        # mixed-radix value (multi-precision divmod in xdrfile's decodeints).
        value = 0
        shift = 0
        while nbits > 8:
            value |= decodebits(8) << shift
            shift += 8
            nbits -= 8
        if nbits > 0:
            value |= decodebits(nbits) << shift
        value, z = divmod(value, sz2)
        x, y = divmod(value, sz1)
        return x, y, z

    mnx, mny, mnz = minint
    out = np.empty((lsize, 3), dtype=np.int64)
    i = 0
    run = 0
    while i < lsize:
        if bitsize == 0:
            tx = decodebits(bitsizeint[0])
            ty = decodebits(bitsizeint[1])
            tz = decodebits(bitsizeint[2])
        else:
            tx, ty, tz = decodeints(bitsize, sizeint[0], sizeint[1], sizeint[2])
        tx += mnx
        ty += mny
        tz += mnz
        px, py, pz = tx, ty, tz  # prevcoord

        flag = decodebits(1)
        is_smaller = 0
        if flag:
            run = decodebits(5)
            is_smaller = run % 3
            run -= is_smaller
            is_smaller -= 1

        if run > 0:
            slot = i  # the big atom's output row; may be swapped with the next
            i += 1
            for k in range(0, run, 3):
                dx, dy, dz = decodeints(smallidx, sizesmall, sizesmall, sizesmall)
                dx += px - smallnum
                dy += py - smallnum
                dz += pz - smallnum
                if k == 0:
                    # Writer swapped atoms 1 and 2 (water optimisation): the
                    # delta-coded atom is emitted first, the big atom second.
                    out[slot, 0] = dx
                    out[slot, 1] = dy
                    out[slot, 2] = dz
                    out[i, 0] = px
                    out[i, 1] = py
                    out[i, 2] = pz
                    # prevcoord stays at the delta-coded atom (px..pz swap).
                    px, py, pz = dx, dy, dz
                else:
                    px, py, pz = dx, dy, dz
                    out[i, 0] = dx
                    out[i, 1] = dy
                    out[i, 2] = dz
                i += 1
        else:
            out[i, 0] = tx
            out[i, 1] = ty
            out[i, 2] = tz
            i += 1

        smallidx += is_smaller
        if is_smaller < 0:
            smallnum = smaller
            if smallidx > _FIRSTIDX:
                smaller = _MAGICINTS[smallidx - 1] // 2
            else:
                smaller = 0
        elif is_smaller > 0:
            smaller = smallnum
            smallnum = _MAGICINTS[smallidx] // 2
        sizesmall = _MAGICINTS[smallidx]
        if sizesmall == 0:
            raise ValueError("Invalid size found while decompressing .xtc frame.")

    return out


def count_xtc_frames(path: str) -> int:
    """Number of frames in an .xtc file (fast header scan, no decompression)."""
    return sum(1 for _ in _iter_headers(path))


def _iter_headers(path: str):
    """Yield ``(frame_index, natoms, box6_or_None, payload)`` per frame.

    ``payload`` is a callable returning the decoded positions (angstrom,
    float32) -- reading/seeking of the compressed block happens lazily so
    unwanted frames cost only a seek.
    """
    file_size = os.path.getsize(path)
    with open(path, "rb") as fh:
        k = 0
        while fh.tell() < file_size:
            head = np.frombuffer(_read_exact(fh, 4 * 3, "frame header"), dtype=">i4")
            magic, natoms = int(head[0]), int(head[1])
            if magic != _XTC_MAGIC:
                raise ValueError(
                    f"Bad .xtc magic {magic} at frame {k} (expected {_XTC_MAGIC}); "
                    "file corrupt or not XTC."
                )
            if natoms <= 0:
                raise ValueError(f"Frame {k}: non-positive atom count {natoms}.")
            _read_exact(fh, 4, "time")
            boxm = (
                np.frombuffer(_read_exact(fh, 36, "box"), dtype=">f4")
                .astype(np.float64)
                .reshape(3, 3)
            )
            box6 = None
            if np.any(boxm.diagonal() > 0):
                box6 = box6_from_vectors(boxm * _NM_TO_ANGSTROM)

            lsize = int(np.frombuffer(_read_exact(fh, 4, "lsize"), dtype=">i4")[0])
            if lsize != natoms:
                raise ValueError(
                    f"Frame {k}: lsize {lsize} does not match natoms {natoms}."
                )

            if lsize <= 9:
                nfloats = 3 * lsize
                blob = _read_exact(fh, 4 * nfloats, "small-system coordinates")

                def payload(blob=blob, lsize=lsize) -> np.ndarray:
                    x = np.frombuffer(blob, dtype=">f4").astype(np.float64)
                    return (x.reshape(lsize, 3) * _NM_TO_ANGSTROM).astype(np.float32)

            else:
                sub = np.frombuffer(_read_exact(fh, 4 * 9, "compression header"), dtype=">i4")
                precision = float(np.frombuffer(sub[:1].tobytes(), dtype=">f4")[0])
                minint = (int(sub[1]), int(sub[2]), int(sub[3]))
                maxint = (int(sub[4]), int(sub[5]), int(sub[6]))
                smallidx = int(sub[7])
                nbytes = int(sub[8])
                if not (0 < nbytes <= file_size):
                    raise ValueError(f"Frame {k}: implausible compressed size {nbytes}.")
                if precision <= 0:
                    raise ValueError(f"Frame {k}: non-positive precision {precision}.")
                padded = (nbytes + 3) // 4 * 4
                start = fh.tell()

                def payload(
                    start=start, padded=padded, nbytes=nbytes, lsize=lsize,
                    minint=minint, maxint=maxint, smallidx=smallidx,
                    precision=precision, fh=fh,
                ) -> np.ndarray:
                    here = fh.tell()
                    fh.seek(start)
                    data = _read_exact(fh, padded, "compressed coordinates")[:nbytes]
                    fh.seek(here)
                    ints = _decompress_frame(data, lsize, minint, maxint, smallidx)
                    coords = ints.astype(np.float64) / precision
                    return (coords * _NM_TO_ANGSTROM).astype(np.float32)

                fh.seek(padded, 1)

            yield k, natoms, box6, payload
            k += 1


class XTCFrameStream:
    """Iterate ``(frame_index, positions_angstrom, box6)`` over an ``.xtc`` file.

    Mirrors :class:`~oligomer_analysis.coanapure.trr_io.TRRFrameStream`:
    constant-memory streaming, seek-based skipping of frames outside ``wanted``
    (only the fixed-size headers of skipped frames are read), and early stop
    after the last wanted frame.
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
        k = -1
        for k, natoms, box6, payload in _iter_headers(self.path):
            if n_atoms_ref is None:
                n_atoms_ref = natoms
            elif natoms != n_atoms_ref:
                raise ValueError(
                    f"Frame {k} has {natoms} atoms but {n_atoms_ref} were "
                    "expected; trajectory atom count is not constant."
                )
            self.n_frames_seen = k + 1
            if wanted is None or k in wanted:
                yield k, payload(), box6
            if stop_after is not None and k >= stop_after:
                break

        if k < 0:
            raise ValueError("No coordinate frames found in .xtc file.")
        if wanted is not None:
            missing = sorted(f for f in wanted if f > k)
            if missing:
                raise IndexError(f"Requested frame {missing[0]} out of range 0..{k}")
