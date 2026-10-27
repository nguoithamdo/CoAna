# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .models import ContactCache, FrameContacts


def validate_cache(cache: ContactCache) -> None:
    """Basic shape checks for a loaded cache."""
    n_frames = len(cache.frames)
    if cache.cmaps.shape[0] != n_frames:
        raise ValueError("cache.cmaps first dimension must match len(cache.frames)")
    for name in ("nintra", "ninter", "ntotal", "n_free", "n_bound"):
        arr = getattr(cache, name)
        if len(arr) != n_frames:
            raise ValueError(f"cache.{name} length must match len(cache.frames)")
    if cache.seg_centers.shape[0] != n_frames:
        raise ValueError("cache.seg_centers first dimension must match len(cache.frames)")
    if cache.boxes.shape[0] != n_frames:
        raise ValueError("cache.boxes first dimension must match len(cache.frames)")
    if cache.class_cmaps is not None:
        if cache.class_cmaps.shape[0] != n_frames:
            raise ValueError("cache.class_cmaps first dimension must match len(cache.frames)")
        n_classes = len(cache.meta.get("contact_classes") or [])
        if n_classes and cache.class_cmaps.shape[1:] != (n_classes, n_classes):
            raise ValueError(
                f"cache.class_cmaps is {cache.class_cmaps.shape[1:]} but metadata "
                f"names {n_classes} contact classes"
            )
    if cache.hotspot_maps is not None and cache.hotspot_maps.ndim != 5:
        raise ValueError(
            "cache.hotspot_maps must be (n_blocks, n_species, n_species, n_res, n_res)"
        )


class _MetaUnpickler(pickle.Unpickler):
    """Unpickler that can rebuild a NumPy object array and nothing else.

    Caches written before 3.7.0 store ``meta_json`` as a 0-d object array,
    which ``np.save`` pickles. A full unpickler would run whatever a crafted
    file asks it to, so only the three globals that array needs are allowed.
    """

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"cache metadata refers to {module}.{name}; a CoAna cache never does, "
            "so the file is not trusted"
        )


def _read_meta_json(z) -> Dict:
    """Return the cache metadata without unpickling anything unsafe."""
    try:
        raw = z["meta_json"]
    except ValueError:
        # Object array from a pre-3.7.0 cache: read it through the allowlist.
        with z.zip.open("meta_json.npy") as fh:
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                np.lib.format.read_array_header_1_0(fh)
            else:
                np.lib.format.read_array_header_2_0(fh)
            raw = _MetaUnpickler(io.BytesIO(fh.read())).load()
    if not isinstance(raw, np.ndarray) or raw.size != 1:
        raise ValueError("cache meta_json is not a single string")
    text = raw.item()
    if not isinstance(text, str):
        raise ValueError("cache meta_json is not a string")
    meta = json.loads(text)
    if not isinstance(meta, dict):
        raise ValueError("cache meta_json is not a JSON object")
    return meta


def load_cache(npz_path: str | Path) -> ContactCache:
    """Load a Stage A cache from disk (or from an open binary file).

    Nothing in the file is unpickled except, for caches written before
    3.7.0, the metadata string, and that through a strict allowlist. A cache
    from an untrusted source therefore cannot run code when it is loaded.
    """
    z = np.load(npz_path, allow_pickle=False)
    meta = _read_meta_json(z)
    keys = set(z.files)
    cache = ContactCache(
        frames=z["frames"],
        cmaps=z["cmaps"],
        nintra=z["nintra"],
        ninter=z["ninter"],
        ntotal=z["ntotal"],
        n_free=z["n_free"],
        n_bound=z["n_bound"],
        seg_centers=z["seg_centers"],
        boxes=z["boxes"],
        meta=meta,
        # Optional Stage A extras. Absent from every cache written before they
        # existed, so they are read by presence, never assumed.
        class_cmaps=(z["class_cmaps"] if "class_cmaps" in keys else None),
        hotspot_maps=(z["hotspot_maps"] if "hotspot_maps" in keys else None),
    )
    validate_cache(cache)
    return cache


def save_cache(
    out_npz: str | Path,
    *,
    results: Iterable[FrameContacts],
    meta: Dict,
    save_csv_stats: bool = True,
    hotspot_maps: Optional[np.ndarray] = None,
) -> None:
    """Save per-frame results into the compressed cache format.

    ``hotspot_maps`` is the already-reduced ``(n_blocks, n_species, n_species,
    n_res, n_res)`` residue contact tensor; per-frame contact-class matrices are
    taken from the results themselves when every frame carries one.
    """
    results = sorted(results, key=lambda x: x.frame)
    out_npz = Path(out_npz)

    frames = np.array([r.frame for r in results], dtype=np.int32)
    nintra = np.array([r.nintra for r in results], dtype=np.int32)
    ninter = np.array([r.ninter for r in results], dtype=np.int32)
    ntotal = np.array([r.ntotal for r in results], dtype=np.int32)
    n_free = np.array([r.n_free for r in results], dtype=np.int32)
    n_bound = np.array([r.n_bound for r in results], dtype=np.int32)
    cmaps = np.stack([r.cmap for r in results]).astype(np.int32, copy=False)
    seg_centers = np.stack([r.seg_centers for r in results]).astype(np.float32, copy=False)
    boxes = np.stack([r.box for r in results]).astype(np.float32, copy=False)

    extra = {}
    if results and all(r.class_cmap is not None for r in results):
        extra["class_cmaps"] = np.stack([r.class_cmap for r in results]).astype(
            np.int64, copy=False
        )
    if hotspot_maps is not None:
        extra["hotspot_maps"] = np.asarray(hotspot_maps, dtype=np.int64)

    np.savez_compressed(
        out_npz,
        frames=frames,
        cmaps=cmaps,
        nintra=nintra,
        ninter=ninter,
        ntotal=ntotal,
        n_free=n_free,
        n_bound=n_bound,
        seg_centers=seg_centers,
        boxes=boxes,
        meta_json=np.array(json.dumps(meta)),
        **extra,
    )

    if save_csv_stats:
        pd.DataFrame(
            {
                "frame": frames,
                "nintra": nintra,
                "ninter": ninter,
                "ntotal": ntotal,
                "n_free": n_free,
                "n_bound": n_bound,
            }
        ).to_csv(out_npz.with_name(out_npz.stem + "_contact_stats.csv"), index=False)


def save_contact_cache(cache: ContactCache, out_npz: str | Path, *, save_csv_stats: bool = True) -> None:
    """Save an existing :class:`ContactCache` object to ``.npz`` format.

    This is useful after subsetting a large cache to selected frames.
    """
    out_npz = Path(out_npz)
    meta = dict(cache.meta)
    meta["n_frames"] = int(len(cache.frames))
    meta["frames_min"] = int(np.min(cache.frames)) if len(cache.frames) else None
    meta["frames_max"] = int(np.max(cache.frames)) if len(cache.frames) else None
    extra = {}
    if cache.class_cmaps is not None:
        extra["class_cmaps"] = cache.class_cmaps
    if cache.hotspot_maps is not None:
        extra["hotspot_maps"] = cache.hotspot_maps
    np.savez_compressed(
        out_npz,
        frames=cache.frames,
        cmaps=cache.cmaps,
        nintra=cache.nintra,
        ninter=cache.ninter,
        ntotal=cache.ntotal,
        n_free=cache.n_free,
        n_bound=cache.n_bound,
        seg_centers=cache.seg_centers,
        boxes=cache.boxes,
        meta_json=np.array(json.dumps(meta)),
        **extra,
    )
    if save_csv_stats:
        pd.DataFrame({
            "frame": cache.frames,
            "nintra": cache.nintra,
            "ninter": cache.ninter,
            "ntotal": cache.ntotal,
            "n_free": cache.n_free,
            "n_bound": cache.n_bound,
        }).to_csv(out_npz.with_name(out_npz.stem + "_contact_stats.csv"), index=False)


def subset_cache_by_indices(cache: ContactCache, indices) -> ContactCache:
    """Return a smaller cache containing selected cache-array indices.

    ``indices`` are zero-based positions in ``cache.frames``. To subset by actual
    frame numbers, use :func:`subset_cache_by_frames`.
    """
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if len(indices) == 0:
        raise ValueError("indices cannot be empty")
    if indices.min() < 0 or indices.max() >= len(cache.frames):
        raise IndexError("indices contain values outside cache frame range")
    meta = dict(cache.meta)
    meta["n_frames"] = int(len(indices))
    meta["frames_min"] = int(cache.frames[indices].min())
    meta["frames_max"] = int(cache.frames[indices].max())
    meta["subset_from_cache"] = True
    # Hotspot maps are block sums over the *original* frame set, so they cannot
    # be sliced to a subset. Dropping them is the only correct answer; keeping
    # them would silently pair a subset cache with whole-trajectory hotspots.
    if cache.hotspot_maps is not None:
        meta.pop("hotspot_blocks", None)
        meta["hotspot_dropped_on_subset"] = True
    return ContactCache(
        frames=cache.frames[indices],
        cmaps=cache.cmaps[indices],
        nintra=cache.nintra[indices],
        ninter=cache.ninter[indices],
        ntotal=cache.ntotal[indices],
        n_free=cache.n_free[indices],
        n_bound=cache.n_bound[indices],
        seg_centers=cache.seg_centers[indices],
        boxes=cache.boxes[indices],
        meta=meta,
        class_cmaps=(None if cache.class_cmaps is None else cache.class_cmaps[indices]),
        hotspot_maps=None,
    )


def subset_cache_by_frames(cache: ContactCache, frames, *, strict: bool = False) -> ContactCache:
    """Return a smaller cache containing selected actual frame numbers.

    Parameters
    ----------
    strict
        If True, raise an error when any requested frame number is missing.
        If False, silently keep only frames present in the cache.
    """
    frame_to_index = {int(fr): i for i, fr in enumerate(cache.frames)}
    missing = [int(fr) for fr in frames if int(fr) not in frame_to_index]
    if missing and strict:
        raise ValueError(f"Requested frames not present in cache: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    indices = [frame_to_index[int(fr)] for fr in frames if int(fr) in frame_to_index]
    if not indices:
        raise ValueError("No requested frames were present in cache")
    return subset_cache_by_indices(cache, indices)


__all__ = [
    "load_cache",
    "validate_cache",
    "save_cache",
    "save_contact_cache",
    "subset_cache_by_indices",
    "subset_cache_by_frames",
]
