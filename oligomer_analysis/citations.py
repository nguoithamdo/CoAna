# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""What to cite when CoAna contributes to published work (``coana citations``).

Citing CoAna is a scholarly request, not a condition of its license (the LGPL
imposes none); see CITATION.md. Every reference below was checked against its
DOI record. The CoAna methodology publication is not out yet, so it is shown as
a placeholder rather than an invented reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = [
    "SOFTWARE_DOI",
    "VERSION_DOIS",
    "METHODOLOGY_PUBLICATION",
    "ACKNOWLEDGMENT",
    "Reference",
    "METHOD_REFERENCES",
    "software_reference",
    "citation_text",
    "citation_bibtex",
]

#: Zenodo concept DOI: resolves to the newest archived version.
SOFTWARE_DOI = "10.5281/zenodo.23623975"

#: Version-specific Zenodo DOIs, for releases that have one.
VERSION_DOIS: Dict[str, str] = {
    "3.6.1": "10.5281/zenodo.23623976",
    "3.7.0": "10.5281/zenodo.23690737",
}

#: The original CoAna methodology paper. None until it is published.
METHODOLOGY_PUBLICATION: Optional[str] = None
METHODOLOGY_PLACEHOLDER = (
    "[CoAna methodology publication: in preparation. Until it appears, cite the "
    "software DOI below.]"
)

ACKNOWLEDGMENT = (
    "Molecular dynamics trajectory analysis was performed using CoAna, a "
    "lineage-aware molecular assembly analysis framework developed by "
    "Nguyen Truong Co (Truong Co Group)."
)

_TITLE = "CoAna: lineage-aware analysis of protein aggregation in molecular dynamics"


@dataclass(frozen=True)
class Reference:
    key: str
    used_for: str
    text: str
    doi: str
    bibtex: str


METHOD_REFERENCES: List[Reference] = [
    Reference(
        "sokal1997",
        "Integrated autocorrelation time with the automatic window "
        "(convergence diagnostics, coana converged, autocorrelation-corrected errors)",
        "Sokal, A. (1997). Monte Carlo Methods in Statistical Mechanics: "
        "Foundations and New Algorithms. NATO ASI Series, 131-192.",
        "10.1007/978-1-4899-0319-8_6",
        "@incollection{sokal1997,\n"
        "  author = {Sokal, A.},\n"
        "  title = {Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms},\n"
        "  booktitle = {NATO ASI Series},\n"
        "  pages = {131--192},\n"
        "  year = {1997},\n"
        "  doi = {10.1007/978-1-4899-0319-8_6}\n"
        "}",
    ),
    Reference(
        "humphries2008",
        "Small-world-ness S = (C/Cr)/(L/Lr) of contact networks "
        "(small_worldness_S, small-world panels)",
        "Humphries, M. D. & Gurney, K. (2008). Network 'small-world-ness': a "
        "quantitative method for determining canonical network equivalence. "
        "PLoS ONE 3, e0002051.",
        "10.1371/journal.pone.0002051",
        "@article{humphries2008,\n"
        "  author = {Humphries, Mark D. and Gurney, Kevin},\n"
        "  title = {Network `Small-World-Ness': A Quantitative Method for Determining "
        "Canonical Network Equivalence},\n"
        "  journal = {PLoS ONE},\n"
        "  volume = {3},\n"
        "  pages = {e0002051},\n"
        "  year = {2008},\n"
        "  doi = {10.1371/journal.pone.0002051}\n"
        "}",
    ),
    Reference(
        "virtanen2020",
        "SciPy, used for the periodic neighbour search of CoAnaPure Stage A (coana precompute)",
        "Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for "
        "scientific computing in Python. Nature Methods 17, 261-272.",
        "10.1038/s41592-019-0686-2",
        "@article{virtanen2020,\n"
        "  author = {Virtanen, Pauli and others},\n"
        "  title = {{SciPy} 1.0: fundamental algorithms for scientific computing in {Python}},\n"
        "  journal = {Nature Methods},\n"
        "  volume = {17},\n"
        "  pages = {261--272},\n"
        "  year = {2020},\n"
        "  doi = {10.1038/s41592-019-0686-2}\n"
        "}",
    ),
    Reference(
        "harris2020",
        "NumPy, used throughout",
        "Harris, C. R. et al. (2020). Array programming with NumPy. Nature 585, 357-362.",
        "10.1038/s41586-020-2649-2",
        "@article{harris2020,\n"
        "  author = {Harris, Charles R. and others},\n"
        "  title = {Array programming with {NumPy}},\n"
        "  journal = {Nature},\n"
        "  volume = {585},\n"
        "  pages = {357--362},\n"
        "  year = {2020},\n"
        "  doi = {10.1038/s41586-020-2649-2}\n"
        "}",
    ),
    Reference(
        "michaudagrawal2011",
        "MDAnalysis, only if you used the optional MDAnalysis engine "
        "(precompute_cache, coana benchmark)",
        "Michaud-Agrawal, N., Denning, E. J., Woolf, T. B. & Beckstein, O. (2011). "
        "MDAnalysis: A toolkit for the analysis of molecular dynamics simulations. "
        "J. Comput. Chem. 32, 2319-2327.",
        "10.1002/jcc.21787",
        "@article{michaudagrawal2011,\n"
        "  author = {Michaud-Agrawal, Naveen and Denning, Elizabeth J. and Woolf, Thomas B. "
        "and Beckstein, Oliver},\n"
        "  title = {{MDAnalysis}: A toolkit for the analysis of molecular dynamics simulations},\n"
        "  journal = {Journal of Computational Chemistry},\n"
        "  volume = {32},\n"
        "  number = {10},\n"
        "  pages = {2319--2327},\n"
        "  year = {2011},\n"
        "  doi = {10.1002/jcc.21787}\n"
        "}",
    ),
    Reference(
        "gowers2016",
        "MDAnalysis, only if you used the optional MDAnalysis engine",
        "Gowers, R. et al. (2016). MDAnalysis: A Python Package for the Rapid "
        "Analysis of Molecular Dynamics Simulations. Proceedings of the Python in "
        "Science Conference, 98-105.",
        "10.25080/Majora-629e541a-00e",
        "@inproceedings{gowers2016,\n"
        "  author = {Gowers, Richard and others},\n"
        "  title = {{MDAnalysis}: A {Python} Package for the Rapid Analysis of Molecular "
        "Dynamics Simulations},\n"
        "  booktitle = {Proceedings of the Python in Science Conference},\n"
        "  pages = {98--105},\n"
        "  year = {2016},\n"
        "  doi = {10.25080/Majora-629e541a-00e}\n"
        "}",
    ),
]


def _version() -> str:
    from . import __version__

    return __version__


def software_reference(version: Optional[str] = None) -> str:
    """The citation of the CoAna software itself, with the best DOI available."""
    version = version or _version()
    doi = VERSION_DOIS.get(version, SOFTWARE_DOI)
    return f"Nguyen, T. C. {_TITLE} (version {version}). Zenodo. https://doi.org/{doi}"


def citation_text(version: Optional[str] = None) -> str:
    """Everything ``coana citations`` prints, as plain text."""
    version = version or _version()
    lines = [
        f"CoAna {version}",
        "",
        "If you use CoAna to generate, process, analyze, or interpret results included in a",
        "scientific publication, conference paper, thesis, technical report, or published",
        "commercial research, please cite the original CoAna publication and acknowledge",
        "the software. (A request, not a license condition: CoAna is LGPL-2.1-or-later.)",
        "",
        "1. Original CoAna publication",
        "   " + (METHODOLOGY_PUBLICATION or METHODOLOGY_PLACEHOLDER),
        "",
        "2. The software, with the exact version you used",
        "   " + software_reference(version),
    ]
    if version not in VERSION_DOIS:
        lines.append(f"   (no version-specific DOI for {version}: the DOI above resolves to the "
                     "newest archived version, so state the version in your text)")
    lines += [
        "",
        "3. Methods and libraries, where you used them",
    ]
    for ref in METHOD_REFERENCES:
        lines += [f"   - {ref.used_for}:", f"     {ref.text} https://doi.org/{ref.doi}"]
    lines += [
        "",
        "Suggested acknowledgment",
        "   " + ACKNOWLEDGMENT,
        "",
        "BibTeX: coana citations --bibtex. Full policy: CITATION.md.",
    ]
    return "\n".join(lines)


def citation_bibtex(version: Optional[str] = None) -> str:
    """BibTeX entries for the software and the method references."""
    version = version or _version()
    doi = VERSION_DOIS.get(version, SOFTWARE_DOI)
    software = (
        "@software{coana,\n"
        "  author = {Nguyen, Truong Co},\n"
        f"  title = {{{{CoAna}}: lineage-aware analysis of protein aggregation in molecular dynamics}},\n"
        f"  version = {{{version}}},\n"
        "  publisher = {Zenodo},\n"
        f"  doi = {{{doi}}},\n"
        "  url = {https://github.com/nguoithamdo/CoAna}\n"
        "}"
    )
    placeholder = "% " + METHODOLOGY_PLACEHOLDER if METHODOLOGY_PUBLICATION is None else ""
    entries = [e for e in (placeholder, software) if e]
    entries += [ref.bibtex for ref in METHOD_REFERENCES]
    return "\n\n".join(entries) + "\n"
