from oligomer_analysis import load_cache, analyze_from_cache, plot_small_world_range

cache = load_cache("cache_cut6.5_compactness.npz")
res = analyze_from_cache(
    cache,
    inter_thresh=1,
    size_thresh=3,
    continue_min_size=1,
    overlap_min=1,
    shrink_tolerance=3,
    dur_store_min=5,
)

finished = res["finished"]

plot_small_world_range(
    cache=cache,
    finished=finished,
    target_label="oli8_16_20_25_26",
    i=0,
    j=23,
    inter_thresh=1,
    by="pathway_index",
    n_rand=50,
    ncols=8,
)
