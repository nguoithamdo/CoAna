# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""The citation command and the licensing metadata stay consistent."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from oligomer_analysis import __version__
from oligomer_analysis import citations
from oligomer_analysis.cli import main

ROOT = Path(__file__).resolve().parents[1]
DOI = re.compile(r"^10\.\d{4,9}/\S+$")


def test_citations_command_prints_version_doi_and_acknowledgment(capsys):
    assert main(["citations"]) == 0
    out = capsys.readouterr().out
    assert f"CoAna {__version__}" in out
    assert f"https://doi.org/{citations.SOFTWARE_DOI}" in out
    assert citations.ACKNOWLEDGMENT in out
    assert "not a license condition" in out
    for ref in citations.METHOD_REFERENCES:
        assert ref.doi in out


def test_unpublished_methodology_paper_is_a_placeholder_not_a_reference(capsys):
    assert citations.METHODOLOGY_PUBLICATION is None
    main(["citations"])
    assert citations.METHODOLOGY_PLACEHOLDER in capsys.readouterr().out


def test_version_doi_is_used_when_the_release_has_one():
    assert citations.VERSION_DOIS["3.6.1"] in citations.software_reference("3.6.1")
    assert citations.SOFTWARE_DOI in citations.software_reference("0.0.0")


def test_bibtex_entries_are_balanced_and_unique(capsys):
    assert main(["citations", "--bibtex"]) == 0
    out = capsys.readouterr().out
    assert out.count("{") == out.count("}")
    keys = re.findall(r"^@\w+\{([^,]+),", out, flags=re.M)
    assert "coana" in keys and len(keys) == len(set(keys)) == 1 + len(citations.METHOD_REFERENCES)
    assert f"version = {{{__version__}}}" in out


def test_references_have_wellformed_dois():
    dois = [citations.SOFTWARE_DOI, *citations.VERSION_DOIS.values()]
    dois += [ref.doi for ref in citations.METHOD_REFERENCES]
    assert all(DOI.match(d) for d in dois)
    assert len({ref.key for ref in citations.METHOD_REFERENCES}) == len(citations.METHOD_REFERENCES)


def test_coana_alias_exposes_the_same_module():
    import coana.citations

    assert coana.citations is citations


def test_console_script_entry_point_runs():
    proc = subprocess.run([sys.executable, "-m", "oligomer_analysis.cli", "citations"],
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0 and "Suggested acknowledgment" in proc.stdout


# --------------------------------------------------------------- licensing


def test_license_file_is_the_lgpl_2_1():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.lstrip().startswith("GNU LESSER GENERAL PUBLIC LICENSE")
    assert "Version 2.1, February 1999" in text
    assert "END OF TERMS AND CONDITIONS" in text


def test_metadata_declares_the_same_license_everywhere():
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "LGPL-2.1-or-later"
    assert {"LICENSE", "NOTICE", "AUTHORS"} <= set(project["license-files"])
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"^license: LGPL-2\.1-or-later$", cff, flags=re.M)
    assert f"doi: {citations.SOFTWARE_DOI}" in cff
    assert citations.SOFTWARE_DOI in (ROOT / "CITATION.md").read_text(encoding="utf-8")


def test_every_python_file_has_an_spdx_header():
    missing = []
    for path in sorted(ROOT.glob("**/*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith((".", "build/", "dist/")) or "/." in rel:
            continue
        head = path.read_text(encoding="utf-8").splitlines()[:3]
        if not any(line.startswith("# SPDX-License-Identifier: LGPL-2.1-or-later") for line in head):
            missing.append(rel)
    assert not missing, missing


def test_ported_xdrfile_code_keeps_its_bsd_notice():
    head = (ROOT / "oligomer_analysis" / "coanapure" / "xtc_io.py").read_text(encoding="utf-8")[:3000]
    assert "SPDX-License-Identifier: LGPL-2.1-or-later AND BSD-2-Clause" in head
    assert "Copyright (c) 2009-2014, Erik Lindahl & David van der Spoel" in head
    assert "Erik Lindahl & David van der Spoel" in (ROOT / "NOTICE").read_text(encoding="utf-8")
