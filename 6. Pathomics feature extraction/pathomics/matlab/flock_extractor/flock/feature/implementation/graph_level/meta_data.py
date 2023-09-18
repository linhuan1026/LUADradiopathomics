"""
Feature names that are used for graph-features
"""

class _FeatName:
    VOR_AREA_STD: str = 'voronoi_area_std'
    VOR_AREA_MEAN: str = 'voronoi_area_mean'
    VOR_AREA_MIN_MAX: str = 'voronoi_area_min_max'
    VOR_AREA_DISORDER: str = 'voronoi_area_disorder'

    VOR_PERI_STD: str = 'voronoi_peri_std'
    VOR_PERI_MEAN: str = 'voronoi_peri_mean'
    VOR_PERI_MIN_MAX: str = 'voronoi_peri_min_max'
    VOR_PERI_DISORDER: str = 'voronoi_peri_disorder'

    VOR_CHORD_STD: str = 'voronoi_chord_std'
    VOR_CHORD_MEAN: str = 'voronoi_chord_mean'
    VOR_CHORD_MIN_MAX: str = 'voronoi_chord_min_max'
    VOR_CHORD_DISORDER: str = 'voronoi_chord_disorder'

    DEL_AREA_STD: str = 'delaunay_area_std'
    DEL_AREA_MEAN: str = 'delaunay_area_mean'
    DEL_AREA_MIN_MAX: str = 'delaunay_area_min_max'
    DEL_AREA_DISORDER: str = 'delaunay_area_disorder'

    DEL_PERI_STD: str = 'delaunay_peri_std'
    DEL_PERI_MEAN: str = 'delaunay_peri_mean'
    DEL_PERI_MIN_MAX: str = 'delaunay_peri_min_max'
    DEL_PERI_DISORDER: str = 'delaunay_peri_disorder'

    MST_EDGE_STD: str = 'mst_edge_std'
    MST_EDGE_MEAN: str = 'mst_edge_mean'
    MST_EDGE_MIN_MAX: str = 'mst_edge_min_max'
    MST_EDGE_DISORDER: str = 'mst_edge_disorder'

    NUC_SUM: str = 'nuc_sum'
    NUC_VOR_AREA: str = 'nuc_vor_area'
    NUC_DENSITY: str = 'nuc_density'

    KNN_MEAN: str = 'knn_mean_'
    KNN_STD: str = 'knn_std_'
    KNN_DISORDER: str = 'knn_disorder_'

    RR_MEAN: str = 'rr_mean_'
    RR_STD: str = 'rr_std_'
    RR_DISORDER: str = 'rr_disorder_'
