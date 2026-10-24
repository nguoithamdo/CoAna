"""Compare CoAnaPure 1.1 (pure NumPy/SciPy) against the MDAnalysis Stage A engine.

This reproduces, then benchmarks, the expensive trajectory-facing precomputation
without MDAnalysis.  Both engines run on the same PDB with the same config; the
two caches are compared array-by-array and timings are reported.

Run:
    python examples/run_coanapure_benchmark.py
"""

from oligomer_analysis.coanapure.benchmark import run_benchmark

# A Martini aggregation trajectory: 128 monomers x 83 beads, reused PDB chain IDs,
# so monomers must be defined by fixed atom blocks (segment_mode="atom_blocks").
TOPOLOGY = "initial1.pdb"
TRAJECTORY = "md1_centered_filtered_1956frames.pdb"

if __name__ == "__main__":
    # NOTE: MDAnalysis reads this 1.66 GB / 1956-model PDB by random access
    # (~1.6 s/frame), so the comparison is run on a subset here.  CoAnaPure reads
    # the file once; on the *full* trajectory it is far faster (see README).
    run_benchmark(
        traj_file=TRAJECTORY,
        topology_file=TOPOLOGY,
        frames=range(0, 50),
        cutoff_contact=6.5,
        intra_exclude=2,
        segment_mode="atom_blocks",
        atoms_per_monomer=83,
        n_monomers=128,
        out_prefix="coanapure_bench",
    )
