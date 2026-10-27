# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""3.7.0: loading a cache must never run code from the file.

``load_cache`` used ``np.load(..., allow_pickle=True)``, so a crafted ``.npz``
executed arbitrary code the moment it was opened, which matters as soon as
caches are uploaded to a web app or shared between groups. Every test here
fails on the 3.6.2 loader.
"""
from __future__ import annotations

import io
import json
import os
import pickle

import numpy as np
import pytest

from oligomer_analysis.cache_io import load_cache, save_contact_cache
from oligomer_analysis.models import ContactCache

T, N = 4, 5


def _arrays():
    return dict(
        frames=np.arange(T, dtype=np.int32),
        cmaps=np.zeros((T, N, N), dtype=np.int32),
        nintra=np.zeros(T, dtype=np.int32),
        ninter=np.zeros(T, dtype=np.int32),
        ntotal=np.zeros(T, dtype=np.int32),
        n_free=np.full(T, N, dtype=np.int32),
        n_bound=np.zeros(T, dtype=np.int32),
        seg_centers=np.zeros((T, N, 3), dtype=np.float32),
        boxes=np.tile(np.array([90, 90, 90, 90, 90, 90], np.float32), (T, 1)),
    )


META = {"nseg": N, "cutoff_contact": 6.5, "engine": "test"}


def test_pre_370_cache_with_object_metadata_still_loads(tmp_path):
    """Every cache written so far stores meta_json as a pickled object array."""
    path = tmp_path / "old.npz"
    np.savez_compressed(path, meta_json=np.array(json.dumps(META), dtype=object), **_arrays())
    cache = load_cache(path)
    assert cache.meta == META
    assert cache.cmaps.shape == (T, N, N)


def test_new_caches_store_metadata_without_pickle(tmp_path):
    path = tmp_path / "new.npz"
    cache = ContactCache(meta=dict(META), **_arrays())
    save_contact_cache(cache, path, save_csv_stats=False)
    with np.load(path, allow_pickle=False) as z:
        assert json.loads(str(z["meta_json"].item()))["cutoff_contact"] == 6.5
    assert load_cache(path).meta["engine"] == "test"


def test_cache_loads_from_an_open_binary_file(tmp_path):
    """The web app hands the loader an in-memory upload, not a path."""
    path = tmp_path / "old.npz"
    np.savez_compressed(path, meta_json=np.array(json.dumps(META), dtype=object), **_arrays())
    cache = load_cache(io.BytesIO(path.read_bytes()))
    assert cache.meta == META


class _Exploit:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def _write_member(path, extra):
    import zipfile

    with zipfile.ZipFile(path, "a") as zf:
        for name, payload in extra.items():
            zf.writestr(name, payload)


def _npy_with_pickle(obj):
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buf, {"descr": "|O", "fortran_order": False, "shape": ()}
    )
    buf.write(pickle.dumps(obj))
    return buf.getvalue()


def test_crafted_metadata_is_refused_without_running(tmp_path, monkeypatch):
    payload = _npy_with_pickle(_Exploit())
    calls = []
    monkeypatch.setattr(os, "system", lambda cmd: calls.append(cmd) or 0)
    path = tmp_path / "evil.npz"
    np.savez_compressed(path, **_arrays())
    _write_member(path, {"meta_json.npy": payload})
    with pytest.raises(pickle.UnpicklingError):
        load_cache(path)
    assert calls == []


def test_crafted_data_array_is_refused(tmp_path, monkeypatch):
    payload = _npy_with_pickle(_Exploit())
    calls = []
    monkeypatch.setattr(os, "system", lambda cmd: calls.append(cmd) or 0)
    arrays = _arrays()
    del arrays["cmaps"]
    path = tmp_path / "evil2.npz"
    np.savez_compressed(path, meta_json=np.array(json.dumps(META)), **arrays)
    _write_member(path, {"cmaps.npy": payload})
    with pytest.raises(ValueError):
        load_cache(path)
    assert calls == []


def test_metadata_must_be_a_json_object(tmp_path):
    path = tmp_path / "list.npz"
    np.savez_compressed(path, meta_json=np.array("[1, 2]"), **_arrays())
    with pytest.raises(ValueError):
        load_cache(path)
