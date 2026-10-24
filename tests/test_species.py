"""Tests for the species layer on mixed systems (e.g. aSyn + Abeta40)."""

from __future__ import annotations

import numpy as np
import pytest

from oligomer_analysis.species import (
    SpeciesMap,
    cluster_compositions,
    composition_timeseries,
    species_bound_free,
    species_contact_matrix,
    validate_segmentation,
)


class _Cache:
    """Minimal stand-in for ContactCache (only what the species layer reads)."""

    def __init__(self, cmaps, meta=None, frames=None):
        self.cmaps = np.asarray(cmaps, dtype=np.int32)
        self.meta = meta or {}
        self.frames = (
            np.arange(self.cmaps.shape[0]) if frames is None else np.asarray(frames)
        )


def _counts(n_a=3, n_b=2, size_a=295, size_b=92):
    return [size_a] * n_a + [size_b] * n_b


# ---------------------------------------------------------------- SpeciesMap


def test_from_spec_builds_contiguous_blocks():
    smap = SpeciesMap.from_spec("aSyn:295x3,Ab40:92x2")
    assert smap.names == ["aSyn", "Ab40"]
    assert smap.nseg == 5
    assert list(smap.seg_species) == [0, 0, 0, 1, 1]
    assert list(smap.counts()) == [3, 2]
    assert list(smap.mask("Ab40")) == [False, False, False, True, True]


def test_from_spec_validates_against_monomer_atom_counts():
    # correct spec passes
    SpeciesMap.from_spec("aSyn:295x3,Ab40:92x2", monomer_atom_counts=_counts())
    # a wrong bead count is caught rather than silently mislabelling chains
    with pytest.raises(ValueError, match="does not match"):
        SpeciesMap.from_spec("aSyn:295x3,Ab40:96x2", monomer_atom_counts=_counts())
    # a wrong chain count is caught too
    with pytest.raises(ValueError, match="does not match"):
        SpeciesMap.from_spec("aSyn:295x4,Ab40:92x1", monomer_atom_counts=_counts())


def test_from_spec_rejects_malformed_and_duplicate_names():
    with pytest.raises(ValueError):
        SpeciesMap.from_spec("aSyn295x3")
    with pytest.raises(ValueError):
        SpeciesMap.from_spec("aSyn:295x3,aSyn:92x2")


def test_auto_from_counts_groups_consecutive_equal_sizes():
    smap = SpeciesMap.auto_from_counts(_counts())
    assert smap.names == ["sp295", "sp92"]
    assert list(smap.seg_species) == [0, 0, 0, 1, 1]


def test_auto_from_counts_keeps_separated_blocks_distinct():
    # same size, not contiguous -> two species, not one
    smap = SpeciesMap.auto_from_counts([295, 92, 295])
    assert smap.n_species == 3
    assert smap.names == ["sp295", "sp92", "sp295_2"]


def test_from_cache_uniform_gives_single_species():
    cache = _Cache(np.zeros((2, 4, 4)), meta={"atoms_per_monomer": 295})
    smap = SpeciesMap.from_cache(cache)
    assert smap.n_species == 1
    assert smap.nseg == 4


def test_from_cache_uses_monomer_atom_counts():
    cache = _Cache(np.zeros((2, 5, 5)), meta={"monomer_atom_counts": _counts()})
    smap = SpeciesMap.from_cache(cache, spec="aSyn:295x3,Ab40:92x2")
    assert smap.names == ["aSyn", "Ab40"]
    assert smap.describe() == "aSyn x3, Ab40 x2".replace(" x", "x")


def test_from_cache_rejects_mismatched_segment_count():
    cache = _Cache(np.zeros((2, 4, 4)), meta={"monomer_atom_counts": _counts()})
    with pytest.raises(ValueError, match="segments"):
        SpeciesMap.from_cache(cache)


# ------------------------------------------------------- contact decomposition


def _hand_built_frame():
    """5 chains: 0,1,2 = aSyn; 3,4 = Ab40.

    Contacts: 0-1 (aSyn-aSyn), 1-3 and 2-3 (aSyn-Ab40), 3-4 (Ab40-Ab40).
    Chain 0 also carries an intra-monomer count on the diagonal, which must be
    ignored.
    """
    c = np.zeros((5, 5), dtype=np.int32)
    for i, j in [(0, 1), (1, 3), (2, 3), (3, 4)]:
        c[i, j] = c[j, i] = 1
    c[0, 0] = 17  # intra-monomer contacts, not a pair contact
    return c


def test_species_contact_matrix_counts_each_pair_once():
    cache = _Cache(_hand_built_frame()[None, ...])
    smap = SpeciesMap.from_spec("aSyn:295x3,Ab40:92x2")
    m = species_contact_matrix(cache, smap, unique=True)
    assert m.shape == (1, 2, 2)
    assert m[0, 0, 0] == pytest.approx(1.0)  # 0-1
    assert m[0, 0, 1] == pytest.approx(2.0)  # 1-3, 2-3
    assert m[0, 1, 0] == pytest.approx(2.0)  # symmetric
    assert m[0, 1, 1] == pytest.approx(1.0)  # 3-4


def test_species_contact_matrix_ignores_the_diagonal():
    frame = _hand_built_frame()
    cache = _Cache(frame[None, ...])
    smap = SpeciesMap.from_spec("aSyn:295x3,Ab40:92x2")
    before = species_contact_matrix(cache, smap)
    frame[0, 0] = 9999
    after = species_contact_matrix(_Cache(frame[None, ...]), smap)
    np.testing.assert_allclose(before, after)


def test_species_contact_matrix_does_not_mutate_the_cache():
    frame = _hand_built_frame()
    cache = _Cache(frame[None, ...])
    original = cache.cmaps.copy()
    species_contact_matrix(cache, SpeciesMap.from_spec("aSyn:295x3,Ab40:92x2"))
    np.testing.assert_array_equal(cache.cmaps, original)


def test_species_contact_matrix_rejects_wrong_sized_map():
    cache = _Cache(np.zeros((1, 5, 5)))
    with pytest.raises(ValueError, match="covers"):
        species_contact_matrix(cache, SpeciesMap.from_spec("aSyn:295x2"))


def test_single_species_matrix_matches_total_pair_count():
    cache = _Cache(_hand_built_frame()[None, ...])
    smap = SpeciesMap.from_spec("all:1x5")
    m = species_contact_matrix(cache, smap)
    assert m[0, 0, 0] == pytest.approx(4.0)  # four distinct contacting pairs


# ------------------------------------------------------------- bound and free


def test_species_bound_free_is_reported_per_species():
    cache = _Cache(_hand_built_frame()[None, ...])
    smap = SpeciesMap.from_spec("aSyn:295x3,Ab40:92x2")
    n_bound, n_free, n_total = species_bound_free(cache, smap)
    # aSyn 0,1,2 all bound; both Ab40 bound
    assert list(n_bound[0]) == [3, 2]
    assert list(n_free[0]) == [0, 0]
    assert list(n_total) == [3, 2]


def test_species_bound_free_counts_an_isolated_chain_as_free():
    c = np.zeros((5, 5), dtype=np.int32)
    c[0, 1] = c[1, 0] = 1
    c[4, 4] = 12  # only intra-monomer contacts -> still free
    cache = _Cache(c[None, ...])
    smap = SpeciesMap.from_spec("aSyn:295x3,Ab40:92x2")
    n_bound, n_free, _ = species_bound_free(cache, smap)
    assert list(n_bound[0]) == [2, 0]
    assert list(n_free[0]) == [1, 2]


# ------------------------------------------------------------- compositions


def test_cluster_compositions_reports_membership_by_species():
    cache = _Cache(_hand_built_frame()[None, ...])
    smap = SpeciesMap.from_spec("aSyn:295x3,Ab40:92x2")
    per_frame = cluster_compositions(cache, smap, min_size=2)
    assert len(per_frame) == 1
    (members, comp), = per_frame[0]
    assert members == frozenset({0, 1, 2, 3, 4})
    assert list(comp) == [3, 2]


def test_cluster_compositions_separates_pure_and_mixed_clusters():
    # 0-1 pure aSyn ; 2 alone ; 3-4 pure Ab40
    c = np.zeros((5, 5), dtype=np.int32)
    for i, j in [(0, 1), (3, 4)]:
        c[i, j] = c[j, i] = 1
    cache = _Cache(c[None, ...])
    smap = SpeciesMap.from_spec("aSyn:295x3,Ab40:92x2")
    comps = [tuple(comp) for _m, comp in cluster_compositions(cache, smap, min_size=2)[0]]
    assert sorted(comps) == [(0, 2), (2, 0)]


def test_composition_timeseries_columns_and_values():
    cache = _Cache(_hand_built_frame()[None, ...], frames=[7])
    smap = SpeciesMap.from_spec("aSyn:295x3,Ab40:92x2")
    df = composition_timeseries(cache, smap)
    row = df.iloc[0]
    assert row["frame"] == 7
    assert row["n_clusters"] == 1
    assert row["n_hetero_clusters"] == 1
    assert row["largest_size"] == 5
    assert row["largest_aSyn"] == 3
    assert row["largest_Ab40"] == 2
    assert row["n_bound_aSyn"] == 3
    assert row["n_Ab40_in_hetero"] == 2
    assert row["contacts_aSyn_Ab40"] == pytest.approx(2.0)
    assert row["contacts_aSyn_aSyn"] == pytest.approx(1.0)


def test_size_species_ranks_clusters_by_one_species_only():
    # cluster A: 1 aSyn + 2 Ab40 (size 3) ; cluster B: 2 aSyn (size 2)
    c = np.zeros((5, 5), dtype=np.int32)
    for i, j in [(0, 3), (3, 4), (1, 2)]:
        c[i, j] = c[j, i] = 1
    cache = _Cache(c[None, ...])
    smap = SpeciesMap.from_spec("aSyn:295x3,Ab40:92x2")

    # counting every chain, the mixed cluster is "largest"
    plain = composition_timeseries(cache, smap).iloc[0]
    assert plain["largest_size"] == 3
    assert plain["largest_aSyn"] == 1

    # counting only aSyn, the genuinely larger aSyn cluster wins
    weighted = composition_timeseries(cache, smap, size_species=["aSyn"]).iloc[0]
    assert weighted["largest_aSyn"] == 2
    assert weighted["largest_size"] == 2


# ------------------------------------------------------------- validation


def _write_gro(path, chain_lengths, beads_per_res=2):
    """Write a minimal .gro whose residue numbering restarts per chain."""
    lines = []
    n_atoms = sum(chain_lengths)
    lines.append("test system")
    lines.append(f"{n_atoms}")
    idx = 0
    for length in chain_lengths:
        for a in range(length):
            resid = a // beads_per_res + 1
            idx += 1
            lines.append(
                f"{resid:5d}{'ALA':>5s}{'BB':>5s}{idx:5d}"
                f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}"
            )
    lines.append("  5.00000  5.00000  5.00000")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_validate_segmentation_accepts_a_correct_block_list(tmp_path):
    gro = _write_gro(tmp_path / "ok.gro", [10, 10, 4, 4])
    rep = validate_segmentation([10, 10, 4, 4], str(gro))
    assert rep.ok
    assert rep.straddling == []
    assert rep.n_atoms_topology == 28


def test_validate_segmentation_catches_boundaries_inside_a_chain(tmp_path):
    gro = _write_gro(tmp_path / "bad.gro", [10, 10, 4, 4])
    # same total, wrong split: monomers straddle real chains
    rep = validate_segmentation([12, 8, 4, 4], str(gro))
    assert not rep.ok
    assert rep.straddling
    assert "inside a real chain" in " ".join(rep.messages)


def test_validate_segmentation_catches_atom_count_mismatch(tmp_path):
    gro = _write_gro(tmp_path / "n.gro", [10, 10])
    rep = validate_segmentation([10, 10, 4], str(gro))
    assert not rep.ok
    assert "atom count mismatch" in " ".join(rep.messages)


def test_validate_segmentation_without_gro_reports_but_does_not_fail():
    rep = validate_segmentation([295] * 50 + [92] * 10)
    assert rep.ok
    assert rep.n_atoms_expected == 50 * 295 + 10 * 92
    assert "not checked" in " ".join(rep.messages)
