import torch
import time
from simple_knn_2d_qr import knn_dist_self, knn_dist_query_ref

def brute_force_knn_mean_distance(queries, refs, K):
    Q, _ = queries.shape
    P, _ = refs.shape
    dists = torch.cdist(queries, refs, p=2).pow(2)  # squared Euclidean distance
    knn_dists, _ = torch.topk(dists, K, largest=False)
    return knn_dists.mean(dim=1)

def test_knn_cuda():
    device = 'cuda'
    torch.manual_seed(42)

    Q = 10000
    P = 200000
    K = 3

    queries = torch.rand(Q, 2, device=device)
    refs = torch.rand(P, 2, device=device)

    # 暴力计算计时
    start = time.time()
    bf_dists = brute_force_knn_mean_distance(queries, refs, K)
    torch.cuda.synchronize()  # 确保 GPU 计算完成
    bf_time = time.time() - start

    # CUDA KNN 计时
    start = time.time()
    cuda_dists = knn_dist_query_ref(queries, refs, K)
    torch.cuda.synchronize()
    cuda_time = time.time() - start

    print(f"Brute force mean distance: {bf_dists.mean().item():.6f}")
    print(f"CUDA knn mean distance: {cuda_dists.mean().item():.6f}")

    print(f"Brute force time: {bf_time:.4f} seconds")
    print(f"CUDA KNN time: {cuda_time:.4f} seconds")

    max_diff = torch.abs(bf_dists - cuda_dists).max().item()
    mean_diff = torch.abs(bf_dists - cuda_dists).mean().item()

    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")

    assert max_diff < 3e-4, "Max difference too large! Possible bug."
    assert mean_diff < 1e-5, "Mean difference too large! Possible bug."

    print("Test passed! CUDA KNN matches brute-force results within tolerance.")

if __name__ == "__main__":
    test_knn_cuda()
