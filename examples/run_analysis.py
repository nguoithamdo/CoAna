from oligomer_analysis import analyze_from_cache, load_cache

cache = load_cache("cache_cut6.5_compactness.npz")
res = analyze_from_cache(
    cache,
    inter_thresh=1,
    size_thresh=3,
    continue_min_size=1,
    overlap_min=1,
    shrink_tolerance=3,
    dur_store_min=5,
    plot_dur_min=15,
    do_plots=False,
)
print(f"Tracked {len(res['finished'])} pathways")
