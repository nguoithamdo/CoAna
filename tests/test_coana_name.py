# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""3.6.0: the package is ``coana``; ``oligomer_analysis`` keeps working.

What must hold:
  * ``import coana`` gives the same public API and version;
  * ``coana.X`` and ``oligomer_analysis.X`` are the SAME module objects --
    copies would make ``isinstance`` fail across the two names;
  * nested subpackages resolve the same way;
  * a typo reports the name the user typed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_import_coana_gives_the_api_and_version():
    import coana
    import oligomer_analysis

    assert coana.__version__ == oligomer_analysis.__version__
    for name in ("load_cache", "build_lineages_from_cache", "directed_fates",
                 "residence_events", "calibrate_dense_phase"):
        assert getattr(coana, name) is getattr(oligomer_analysis, name), name


def test_submodules_are_the_same_objects_not_copies():
    import coana.lineage
    import coana.models
    import oligomer_analysis.lineage
    import oligomer_analysis.models

    assert coana.lineage is oligomer_analysis.lineage
    assert coana.models.ContactCache is oligomer_analysis.models.ContactCache


def test_an_object_made_under_one_name_is_recognised_under_the_other():
    import numpy as np
    from coana.models import ContactCache as A
    from oligomer_analysis.models import ContactCache as B

    c = A(frames=np.arange(1), cmaps=np.zeros((1, 2, 2), dtype=np.int32),
          nintra=np.zeros(1), ninter=np.zeros(1), ntotal=np.zeros(1),
          n_free=np.zeros(1), n_bound=np.zeros(1),
          seg_centers=np.zeros((1, 2, 3)), boxes=np.zeros((1, 6)), meta={})
    assert isinstance(c, B)


def test_nested_subpackages_resolve():
    import coana.coanapure.pdb_io
    import oligomer_analysis.coanapure.pdb_io

    assert coana.coanapure.pdb_io is oligomer_analysis.coanapure.pdb_io


def test_a_typo_reports_the_name_the_user_typed():
    with pytest.raises(ModuleNotFoundError) as exc:
        import coana.does_not_exist  # noqa: F401
    assert exc.value.name == "coana.does_not_exist"


def test_coana_can_be_imported_first_in_a_fresh_interpreter():
    """Import order must not matter: `import coana` before anything else."""
    code = ("import coana, coana.lineage, oligomer_analysis.lineage as l;"
            "assert coana.lineage is l; print(coana.__version__)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().startswith("3.")
