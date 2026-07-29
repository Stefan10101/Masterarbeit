# shared.clustering package
try:
    from .features import FEATURE_FUNCS, get_feature_func
    from .clustering_core import (
        build_feature_matrix,
        run_clustering_for_variable,
    )
except ImportError:
    from features import FEATURE_FUNCS, get_feature_func
    from clustering_core import (
        build_feature_matrix,
        run_clustering_for_variable,
    )
