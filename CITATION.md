# Citing CoAna

CoAna is research software. Its development is only visible, and only
continues to be supported, when work that uses it says so. Like GROMACS and
MDAnalysis, CoAna therefore asks to be cited.

## Policy

If you use CoAna to generate, process, analyze, or interpret results included
in a scientific publication, conference paper, thesis, technical report, or
published commercial research, please cite the original CoAna publication and
acknowledge the software.

This applies equally to academic and commercial users, and to results produced
with a modified version of CoAna.

**This is a scholarly request, not a license condition.** CoAna is licensed
under the GNU Lesser General Public License, version 2.1 or later
(LGPL-2.1-or-later), and the LGPL places no citation requirement on its users.
Nothing in this file adds a restriction to the license or limits any right it
grants. It describes what good scientific practice expects when software
contributes to published results.

`coana citations` prints everything below for the version you have installed;
`coana citations --bibtex` prints it as BibTeX.

## What to cite

### 1. The original CoAna publication

> [CoAna methodology publication: in preparation. The reference will be added
> here and to `CITATION.cff` when it is published.]

Until it appears, cite the software (2.).

### 2. The software, with the version you used

> Nguyen, T. C. CoAna: lineage-aware analysis of protein aggregation in
> molecular dynamics (version X.Y.Z). Zenodo.
> https://doi.org/10.5281/zenodo.23623975

Give the exact version (`coana version`). The DOI above is the Zenodo concept
DOI and always resolves to the newest archived version. Where a release has its
own DOI, cite that one instead:

| Version | DOI |
|---|---|
| 3.6.1 | [10.5281/zenodo.23623976](https://doi.org/10.5281/zenodo.23623976) |

GitHub's "Cite this repository" button gives the same reference from
`CITATION.cff`.

### 3. Methods and libraries, where you used them

Some analyses implement published methods, and some rest on other packages.
Cite these when the corresponding result appears in your work.

| Used for | Reference |
|---|---|
| Integrated autocorrelation time with the automatic window: convergence diagnostics, `coana converged`, autocorrelation-corrected errors | Sokal, A. (1997). Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms. NATO ASI Series, 131–192. [doi:10.1007/978-1-4899-0319-8_6](https://doi.org/10.1007/978-1-4899-0319-8_6) |
| Small-world-ness S = (C/Cr)/(L/Lr) of contact networks: `small_worldness_S`, the small-world panels | Humphries, M. D. & Gurney, K. (2008). Network 'small-world-ness': a quantitative method for determining canonical network equivalence. *PLoS ONE* 3, e0002051. [doi:10.1371/journal.pone.0002051](https://doi.org/10.1371/journal.pone.0002051) |
| SciPy, for the periodic neighbour search of CoAnaPure Stage A (`coana precompute`) | Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods* 17, 261–272. [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2) |
| NumPy, used throughout | Harris, C. R. et al. (2020). Array programming with NumPy. *Nature* 585, 357–362. [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2) |
| MDAnalysis, only with the optional MDAnalysis engine (`precompute_cache`, `coana benchmark`) | Michaud-Agrawal, N., Denning, E. J., Woolf, T. B. & Beckstein, O. (2011). MDAnalysis: A toolkit for the analysis of molecular dynamics simulations. *J. Comput. Chem.* 32, 2319–2327. [doi:10.1002/jcc.21787](https://doi.org/10.1002/jcc.21787) — and Gowers, R. et al. (2016). MDAnalysis: A Python Package for the Rapid Analysis of Molecular Dynamics Simulations. *Proceedings of the Python in Science Conference*, 98–105. [doi:10.25080/Majora-629e541a-00e](https://doi.org/10.25080/Majora-629e541a-00e) |

The simulation engine that produced your trajectories (GROMACS, for example)
has its own citation requirements; follow those as well.

## Suggested acknowledgment

> Molecular dynamics trajectory analysis was performed using CoAna, a
> lineage-aware molecular assembly analysis framework developed by Nguyen
> Truong Co (Truong Co Group).

Put it in the methods section or the acknowledgments, together with the
references above.

## BibTeX

```bibtex
@software{coana,
  author = {Nguyen, Truong Co},
  title = {{CoAna}: lineage-aware analysis of protein aggregation in molecular dynamics},
  version = {X.Y.Z},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.23623975},
  url = {https://github.com/nguoithamdo/CoAna}
}
```

`coana citations --bibtex` prints this entry for your installed version,
together with the method references.

## Questions

For anything about citing CoAna, open an issue at
https://github.com/nguoithamdo/CoAna/issues.
