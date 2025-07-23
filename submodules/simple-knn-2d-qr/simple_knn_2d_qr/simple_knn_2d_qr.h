#pragma once
#include <torch/extension.h>

torch::Tensor knn_dist_self(const torch::Tensor& points, int K);
torch::Tensor knn_dist_query_ref(const torch::Tensor& queries, const torch::Tensor& refs, int K);
