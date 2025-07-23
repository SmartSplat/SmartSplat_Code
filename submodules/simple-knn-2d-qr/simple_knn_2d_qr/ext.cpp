#include <torch/extension.h>
#include "simple_knn_2d_qr.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_dist_self", &knn_dist_self, "Self KNN Distance");
    m.def("knn_dist_query_ref", &knn_dist_query_ref, "Query-Ref KNN Distance");
}
