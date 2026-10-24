from oligomer_analysis import PrecomputeConfig, precompute_cache

cfg = PrecomputeConfig(
    topology_file="initial.pdb",
    traj_file="dynamic_700frames_330K.pdb",
    frames=list(range(700)),
    cutoff_contact=6.5,
    intra_exclude=2,
    n_jobs=-1,
    chunk_size=50,
    in_memory=False,
    fallback_box=None,
)

precompute_cache(cfg, out_npz="cache_cut6.5_compactness.npz")
