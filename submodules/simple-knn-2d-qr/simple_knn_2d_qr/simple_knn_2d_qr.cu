#include "simple_knn_2d_qr.h"
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <cfloat>
#include <cuda_runtime.h>
#include <algorithm>

#define BLOCK_SIZE 256
#define BOX_SIZE 1024

struct MinMax2D {
    float2 min;
    float2 max;
};

struct Float2Min {
    __host__ __device__ float2 operator()(const float2& a, const float2& b) const {
        return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
    }
};

struct Float2Max {
    __host__ __device__ float2 operator()(const float2& a, const float2& b) const {
        return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
    }
};

__host__ __device__ uint32_t prepMorton(uint32_t x) {
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}

__host__ __device__ uint32_t coord2Morton2D(float2 coord, float2 minv, float2 maxv) {
    uint32_t x = prepMorton(static_cast<uint32_t>(((coord.x - minv.x) / (maxv.x - minv.x)) * ((1 << 16) - 1)));
    uint32_t y = prepMorton(static_cast<uint32_t>(((coord.y - minv.y) / (maxv.y - minv.y)) * ((1 << 16) - 1)));
    return x | (y << 1);
}

__global__ void computeMortonCodes(int P, const float2* points, float2 minv, float2 maxv, uint32_t* codes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < P)
        codes[idx] = coord2Morton2D(points[idx], minv, maxv);
}

__device__ __host__ float distBoxPoint2D(const MinMax2D& box, const float2& p) {
    float dx = 0.f, dy = 0.f;
    if (p.x < box.min.x) dx = box.min.x - p.x;
    else if (p.x > box.max.x) dx = p.x - box.max.x;

    if (p.y < box.min.y) dy = box.min.y - p.y;
    else if (p.y > box.max.y) dy = p.y - box.max.y;

    return dx * dx + dy * dy;
}

__device__ void updateKBestDynamic(const float2& ref, const float2& point, float* knn, int K) {
    float dx = point.x - ref.x;
    float dy = point.y - ref.y;
    float dist = dx * dx + dy * dy;

    int maxIdx = 0;
    float maxDist = knn[0];
    for (int i = 1; i < K; i++) {
        if (knn[i] > maxDist) {
            maxDist = knn[i];
            maxIdx = i;
        }
    }

    if (dist < maxDist) {
        knn[maxIdx] = dist;
    }
}

__global__ void boxMeanDistDynamic(int P, const float2* points, const uint32_t* indices, const MinMax2D* boxes, float* dists, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    float2 point = points[indices[idx]];

    extern __shared__ float shared_knn[];
    float* best = &shared_knn[threadIdx.x * K];

    for (int i = 0; i < K; i++) {
        best[i] = FLT_MAX;
    }

    int start = max(0, idx - K);
    int end = min(P - 1, idx + K);

    for (int i = start; i <= end; ++i) {
        if (i == idx) continue;
        updateKBestDynamic(point, points[indices[i]], best, K);
    }

    float reject = FLT_MAX;
    for (int i = 0; i < K; i++) {
        if (best[i] < reject) reject = best[i];
    }

    int nBoxes = (P + BOX_SIZE - 1) / BOX_SIZE;
    for (int b = 0; b < nBoxes; ++b) {
        MinMax2D box = boxes[b];
        float distToBox = distBoxPoint2D(box, point);
        if (distToBox > reject) continue;

        int box_start = b * BOX_SIZE;
        int box_end = min(P, (b + 1) * BOX_SIZE);

        for (int i = box_start; i < box_end; ++i) {
            if (i == idx) continue;
            updateKBestDynamic(point, points[indices[i]], best, K);
        }
    }

    float sum = 0.f;
    for (int i = 0; i < K; i++) sum += best[i];
    dists[indices[idx]] = sum / K;
}

__global__ void knnQueryToRefDynamic(
    int Q, int P,
    const float2* queries,
    const float2* refs,
    const uint32_t* ref_sorted_indices,
    const MinMax2D* ref_boxes,
    float* out_dists,
    int K)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= Q) return;

    float2 query = queries[qid];

    extern __shared__ float shared_knn[];
    float* best = &shared_knn[threadIdx.x * K];

    for (int i = 0; i < K; ++i) best[i] = FLT_MAX;

    int nBoxes = (P + BOX_SIZE - 1) / BOX_SIZE;
    for (int b = 0; b < nBoxes; ++b) {
        MinMax2D box = ref_boxes[b];
        float distToBox = distBoxPoint2D(box, query);

        float maxAccept = FLT_MAX;
        for (int i = 0; i < K; ++i) {
            maxAccept = fminf(maxAccept, best[i]);
        }
        if (distToBox > maxAccept) continue;

        int box_start = b * BOX_SIZE;
        int box_end = min(P, (b + 1) * BOX_SIZE);

        for (int i = box_start; i < box_end; ++i) {
            float2 pt = refs[ref_sorted_indices[i]];
            updateKBestDynamic(query, pt, best, K);
        }
    }

    float sum = 0.f;
    for (int i = 0; i < K; i++) sum += best[i];
    out_dists[qid] = sum / K;
}

__global__ void boxMinMax2D(int P, const float2* points, const uint32_t* indices, MinMax2D* boxes) {
    __shared__ MinMax2D shared[BOX_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    MinMax2D mm;
    if (gid < P) {
        mm.min = mm.max = points[indices[gid]];
    } else {
        mm.min = make_float2(FLT_MAX, FLT_MAX);
        mm.max = make_float2(-FLT_MAX, -FLT_MAX);
    }
    shared[tid] = mm;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset && (gid + offset) < P) {
            shared[tid].min.x = fminf(shared[tid].min.x, shared[tid + offset].min.x);
            shared[tid].min.y = fminf(shared[tid].min.y, shared[tid + offset].min.y);
            shared[tid].max.x = fmaxf(shared[tid].max.x, shared[tid + offset].max.x);
            shared[tid].max.y = fmaxf(shared[tid].max.y, shared[tid + offset].max.y);
        }
        __syncthreads();
    }

    if (tid == 0) {
        boxes[blockIdx.x] = shared[0];
    }
}

class SimpleKNN2D {
public:
    static void knn(int P, const float2* points, float* meanDists, int K) {
        thrust::device_vector<uint32_t> morton_codes(P);
        thrust::device_vector<uint32_t> indices(P);
        thrust::sequence(indices.begin(), indices.end());

        float2 h_min, h_max;

        auto points_begin = thrust::device_pointer_cast(points);
        auto points_end = points_begin + P;

        Float2Min f2min;
        Float2Max f2max;

        float2 init_min = make_float2(FLT_MAX, FLT_MAX);
        float2 init_max = make_float2(-FLT_MAX, -FLT_MAX);

        h_min = thrust::reduce(points_begin, points_end, init_min, f2min);
        h_max = thrust::reduce(points_begin, points_end, init_max, f2max);

        int blocks = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
        computeMortonCodes<<<blocks, BLOCK_SIZE>>>(P, points, h_min, h_max, thrust::raw_pointer_cast(morton_codes.data()));
        cudaDeviceSynchronize();

        thrust::sort_by_key(morton_codes.begin(), morton_codes.end(), indices.begin());

        int nBoxes = (P + BOX_SIZE - 1) / BOX_SIZE;
        thrust::device_vector<MinMax2D> boxes(nBoxes);
        boxMinMax2D<<<nBoxes, BOX_SIZE>>>(P, points, thrust::raw_pointer_cast(indices.data()), thrust::raw_pointer_cast(boxes.data()));
        cudaDeviceSynchronize();

        int smem_size = BLOCK_SIZE * K * sizeof(float);
        boxMeanDistDynamic<<<blocks, BLOCK_SIZE, smem_size>>>(P, points, thrust::raw_pointer_cast(indices.data()), thrust::raw_pointer_cast(boxes.data()), meanDists, K);
        cudaDeviceSynchronize();
    }

    static void knn_query_to_ref(int Q, int P, const float2* queries, const float2* refs, float* meanDists, int K) {
        thrust::device_vector<uint32_t> morton_codes(P);
        thrust::device_vector<uint32_t> indices(P);
        thrust::sequence(indices.begin(), indices.end());

        auto refs_begin = thrust::device_pointer_cast(refs);
        auto refs_end = refs_begin + P;

        Float2Min f2min;
        Float2Max f2max;

        float2 init_min = make_float2(FLT_MAX, FLT_MAX);
        float2 init_max = make_float2(-FLT_MAX, -FLT_MAX);

        float2 h_min = thrust::reduce(refs_begin, refs_end, init_min, f2min);
        float2 h_max = thrust::reduce(refs_begin, refs_end, init_max, f2max);

        int blocks_ref = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
        computeMortonCodes<<<blocks_ref, BLOCK_SIZE>>>(P, refs, h_min, h_max, thrust::raw_pointer_cast(morton_codes.data()));
        cudaDeviceSynchronize();

        thrust::sort_by_key(morton_codes.begin(), morton_codes.end(), indices.begin());

        int nBoxes = (P + BOX_SIZE - 1) / BOX_SIZE;
        thrust::device_vector<MinMax2D> boxes(nBoxes);
        boxMinMax2D<<<nBoxes, BOX_SIZE>>>(P, refs, thrust::raw_pointer_cast(indices.data()), thrust::raw_pointer_cast(boxes.data()));
        cudaDeviceSynchronize();

        int blocks_query = (Q + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int smem_size = BLOCK_SIZE * K * sizeof(float);

        knnQueryToRefDynamic<<<blocks_query, BLOCK_SIZE, smem_size>>>(Q, P, queries, refs, thrust::raw_pointer_cast(indices.data()), thrust::raw_pointer_cast(boxes.data()), meanDists, K);
        cudaDeviceSynchronize();
    }
};

torch::Tensor knn_dist_self(const torch::Tensor& points, int K) {
    TORCH_CHECK(points.is_cuda(), "points must be CUDA tensor");
    TORCH_CHECK(points.dtype() == torch::kFloat32, "points must be float32");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 2, "points must be Nx2");

    int P = points.size(0);
    auto meanDists = torch::empty({P}, torch::device(points.device()).dtype(torch::kFloat32));
    SimpleKNN2D::knn(P, reinterpret_cast<const float2*>(points.data_ptr<float>()), meanDists.data_ptr<float>(), K);
    return meanDists;
}

torch::Tensor knn_dist_query_ref(const torch::Tensor& queries, const torch::Tensor& refs, int K) {
    TORCH_CHECK(queries.is_cuda() && refs.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(queries.dtype() == torch::kFloat32 && refs.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(queries.dim() == 2 && queries.size(1) == 2, "queries must be Qx2");
    TORCH_CHECK(refs.dim() == 2 && refs.size(1) == 2, "refs must be Px2");

    int Q = queries.size(0);
    int P = refs.size(0);
    auto meanDists = torch::empty({Q}, torch::device(queries.device()).dtype(torch::kFloat32));
    SimpleKNN2D::knn_query_to_ref(Q, P, reinterpret_cast<const float2*>(queries.data_ptr<float>()), reinterpret_cast<const float2*>(refs.data_ptr<float>()), meanDists.data_ptr<float>(), K);
    return meanDists;
}
