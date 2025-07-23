from . import _C

def knn_dist_self(points, K):
    return _C.knn_dist_self(points, K)

def knn_dist_query_ref(queries, refs, K):
    return _C.knn_dist_query_ref(queries, refs, K)
