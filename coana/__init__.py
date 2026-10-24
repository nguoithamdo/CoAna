"""CoAna -- which oligomers mattered, and why.

``import coana`` is the canonical name from 3.6.0. The implementation lives in
the ``oligomer_analysis`` package, which keeps working unchanged, so existing
scripts and notebooks need no edits.

The two names refer to the *same* module objects, not copies:
``coana.lineage is oligomer_analysis.lineage`` holds, so a ``ContactCache`` made
through one name passes ``isinstance`` checks written against the other. A plain
re-export package would load every module a second time and break exactly that.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

import oligomer_analysis as _impl
from oligomer_analysis import *  # noqa: F401,F403 -- the public API, unchanged

__version__ = _impl.__version__
__all__ = list(getattr(_impl, "__all__", []))

_ALIAS = __name__
_TARGET = _impl.__name__


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``coana.<sub>`` to the already-loaded ``oligomer_analysis.<sub>``."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(_ALIAS + "."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        try:
            return importlib.import_module(_TARGET + spec.name[len(_ALIAS):])
        except ModuleNotFoundError as exc:
            # Report the name the user typed, not the internal one.
            raise ModuleNotFoundError(
                f"No module named {spec.name!r}", name=spec.name) from exc

    def exec_module(self, module):
        # The target module is already executed; executing again would
        # duplicate its state.
        return None


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
