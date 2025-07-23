import torch
import numpy as np
from typing import Literal, Tuple, Optional
import faiss
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import torch.nn.functional as F

def compute_scale_from_inv_gradient(grad_at_pts: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """根据梯度值计算 scale，适用于 gradient importance 采样"""
    return 0.5 * (1.0 / (grad_at_pts + eps)).unsqueeze(-1)

def compute_scale_from_inv_gradient_clamped(
    grad_at_pts: torch.Tensor, min_scale=0.5, max_scale=30.0, eps: float = 1e-5
) -> torch.Tensor:
    """倒数后做上下限裁剪，防止scale过大或过小"""
    scale = 1.0 / (grad_at_pts + eps)
    scale = scale.clamp(min=min_scale, max=max_scale)
    return scale.unsqueeze(-1)


def compute_scale_from_inv_gradient_log(
    grad_at_pts: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """倒数取log，抑制极端值爆炸"""
    scale = torch.log1p(1.0 / (grad_at_pts + eps))  # log(1 + 1/g)
    return scale.unsqueeze(-1)

def compute_scale_exp(grad_at_pts, alpha=0.5, base_scale=10.0):
    scale = base_scale * torch.exp(-alpha * grad_at_pts)
    return scale.unsqueeze(-1)

def compute_scale_from_norm_inv_gradient_log(
    grad_at_pts: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """倒数取log，抑制极端值爆炸"""
    grad_min = grad_at_pts.min()
    grad_max = grad_at_pts.max()
    grad_norm = (grad_at_pts - grad_min) / (grad_max - grad_min + eps)
    scale = torch.log1p(10.0 / (grad_norm + eps))
    return scale.unsqueeze(-1)

def compute_scale_from_inv_gradient_sqrt(
    grad_at_pts: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """倒数开根号，缓和梯度小值引起的scale极端大"""
    scale = 1.0 / torch.sqrt(grad_at_pts + eps)
    return scale.unsqueeze(-1)

def compute_scale_from_mixed_norm(
    grad_at_pts: torch.Tensor, alpha=0.7, eps: float = 1e-5
) -> torch.Tensor:
    """结合归一化梯度的加权倒数，平衡scale范围"""
    grad_min = grad_at_pts.min()
    grad_max = grad_at_pts.max()
    grad_norm = (grad_at_pts - grad_min) / (grad_max - grad_min + eps)
    base_scale = 1.0 / (grad_at_pts + eps)
    scale = base_scale * (alpha + (1 - alpha) * (1 - grad_norm))
    return scale.unsqueeze(-1)


def compute_scale_from_smooth_floor(
    grad_at_pts: torch.Tensor, eps: float = 1e-2
) -> torch.Tensor:
    """梯度加软阈值，防止极小梯度导致scale爆炸"""
    grad_safe = torch.clamp(grad_at_pts, min=eps)
    scale = 1.0 / grad_safe
    return scale.unsqueeze(-1)

def compute_scale_from_norm_gradient(grad_at_pts: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    grad_norm = (grad_at_pts - grad_at_pts.min()) / (grad_at_pts.max() - grad_at_pts.min() + eps)
    alpha = 0.7  # 控制归一化权重贡献比例
    base_scale = 1.0 / (grad_at_pts + eps)
    scale = base_scale * (alpha + (1 - alpha) * (1 - grad_norm))
    # scale = 0.5 * (1.0 - grad_norm) * (1.0 / (grad_at_pts + eps))
    # 
    scale = scale.clamp(0.5, 15.0)
    return scale.unsqueeze(-1)

def compute_scale_from_weight(weight_at_pts: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """根据混合权重（梯度 + 方差）计算 scale，适用于变分重要性采样"""
    return 0.5 * (1.0 / (weight_at_pts + eps)).unsqueeze(-1)

# ===== 新增的重要性采样 scale 计算方法 =====

def compute_scale_from_adaptive_gradient(
    grad_at_pts: torch.Tensor,
    coords: torch.Tensor,
    image_bounds: Tuple[float, float, float, float],  # (xmin, xmax, ymin, ymax)
    adaptive_factor: float = 0.1,
    eps: float = 1e-5
) -> torch.Tensor:
    """自适应梯度 scale：考虑图像边界和局部密度"""
    # 基础梯度 scale
    base_scale = 0.5 * (1.0 / (grad_at_pts + eps))
    
    # 边界约束：靠近边界的点 scale 更小
    xmin, xmax, ymin, ymax = image_bounds
    x, y = coords[..., 0], coords[..., 1]
    
    dist_to_boundary = torch.min(torch.stack([
        x - xmin, xmax - x, y - ymin, ymax - y
    ]), dim=0)[0]
    
    boundary_factor = torch.clamp(dist_to_boundary / (adaptive_factor * min(xmax-xmin, ymax-ymin)), 0.1, 1.0)
    
    return (base_scale * boundary_factor).unsqueeze(-1)

def compute_scale_from_frequency_analysis(
    coords: torch.Tensor,
    image_tensor: torch.Tensor,  # (H, W) or (H, W, C)
    image_bounds: Tuple[float, float, float, float],
    window_size: int = 32,
    eps: float = 1e-5
) -> torch.Tensor:
    """基于局部频率分析的 scale 计算"""
    H, W = image_tensor.shape[:2]
    xmin, xmax, ymin, ymax = image_bounds
    
    # 将坐标映射到图像像素空间
    x_img = ((coords[..., 0] - xmin) / (xmax - xmin) * (W - 1)).long()
    y_img = ((coords[..., 1] - ymin) / (ymax - ymin) * (H - 1)).long()
    
    scales = []
    for i in range(len(coords)):
        # 提取局部窗口
        x_start = max(0, x_img[i] - window_size // 2)
        x_end = min(W, x_img[i] + window_size // 2)
        y_start = max(0, y_img[i] - window_size // 2)
        y_end = min(H, y_img[i] + window_size // 2)
        
        if len(image_tensor.shape) == 3:
            patch = image_tensor[y_start:y_end, x_start:x_end].mean(dim=-1)
        else:
            patch = image_tensor[y_start:y_end, x_start:x_end]
        
        # 计算局部高频成分
        patch_fft = torch.fft.fft2(patch.float())
        high_freq_energy = torch.abs(patch_fft).sum() - torch.abs(patch_fft[0, 0])
        
        # 高频区域 scale 更小
        scale = 1.0 / (high_freq_energy / patch.numel() + eps)
        scales.append(scale)
    
    return torch.stack(scales).unsqueeze(-1)

def compute_scale_from_perceptual_importance(
    coords: torch.Tensor,
    grad_at_pts: torch.Tensor,
    saliency_map: torch.Tensor,  # (H, W) 显著性图
    image_bounds: Tuple[float, float, float, float],
    eps: float = 1e-5
) -> torch.Tensor:
    """结合梯度和视觉显著性的 scale 计算"""
    H, W = saliency_map.shape
    xmin, xmax, ymin, ymax = image_bounds
    
    # 将坐标映射到显著性图
    x_img = ((coords[..., 0] - xmin) / (xmax - xmin) * (W - 1)).long()
    y_img = ((coords[..., 1] - ymin) / (ymax - ymin) * (H - 1)).long()
    
    # 获取显著性值
    saliency_values = saliency_map[y_img, x_img]
    
    # 结合梯度和显著性
    combined_importance = grad_at_pts * (1 + saliency_values)
    
    return 0.5 * (1.0 / (combined_importance + eps)).unsqueeze(-1)

def compute_scale_from_voronoi(
    coords_query: torch.Tensor,
    coords_ref: torch.Tensor,
    min_scale: float = 0.5,
    max_scale: float = 30.0
) -> torch.Tensor:
    """基于 Voronoi 图的 scale 计算：每个点的 scale 基于其 Voronoi 单元大小"""
    # 合并所有点进行 Voronoi 分割
    all_coords = torch.cat([coords_query, coords_ref], dim=0)
    coords_np = all_coords.cpu().numpy()
    
    # 计算 Voronoi 图
    vor = Voronoi(coords_np)
    
    scales = []
    for i in range(len(coords_query)):
        # 找到该点的 Voronoi 单元
        region_idx = vor.point_region[i]
        vertex_indices = vor.regions[region_idx]
        
        if len(vertex_indices) > 0 and -1 not in vertex_indices:
            # 计算 Voronoi 单元的面积
            vertices = vor.vertices[vertex_indices]
            area = polygon_area(vertices)
            scale = 0.5 * np.sqrt(area)  # 面积的平方根作为 scale
        else:
            # 如果是无界区域，使用 KNN 距离
            distances = cdist([coords_np[i]], coords_np)[0]
            distances = np.sort(distances)[1:4]  # 排除自己，取前3个
            scale = np.mean(distances) * 0.5
        
        scales.append(scale)
    
    scales = torch.tensor(scales, dtype=torch.float32, device=coords_query.device)
    return scales.unsqueeze(-1).clamp(min_scale, max_scale)

def compute_scale_from_density_estimation(
    coords_query: torch.Tensor,
    coords_ref: torch.Tensor,
    bandwidth: float = 1.0,
    min_scale: float = 0.5,
    max_scale: float = 30.0
) -> torch.Tensor:
    """基于核密度估计的 scale 计算"""
    # 使用 Gaussian kernel 估计密度
    coords_ref_expanded = coords_ref.unsqueeze(1)  # (M, 1, 2)
    coords_query_expanded = coords_query.unsqueeze(0)  # (1, N, 2)
    
    # 计算距离
    dist_sq = torch.sum((coords_ref_expanded - coords_query_expanded) ** 2, dim=2)
    
    # Gaussian 核密度估计
    density = torch.exp(-dist_sq / (2 * bandwidth ** 2)).mean(dim=0)
    
    # 密度高的地方 scale 小
    scale = 0.0001 * 1.0 / (density + 1e-6)
    
    # scale = torch.sqrt(scale)

    return scale.unsqueeze(-1).clamp(min_scale, max_scale)

def compute_scale_from_gap_filling(
    coords_query: torch.Tensor,
    coords_ref: torch.Tensor,
    image_bounds: Tuple[float, float, float, float],
    grid_resolution: int = 64,
    min_scale: float = 0.5,
    max_scale: float = 30.0
) -> torch.Tensor:
    """专门用于填补空白区域的 scale 计算"""
    xmin, xmax, ymin, ymax = image_bounds
    
    # 创建网格
    x_grid = torch.linspace(xmin, xmax, grid_resolution)
    y_grid = torch.linspace(ymin, ymax, grid_resolution)
    grid_x, grid_y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    grid_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(coords_query.device)
    
    # 计算网格点到重要性采样点的距离
    grid_coords_expanded = grid_coords.unsqueeze(1)  # (G, 1, 2)
    ref_coords_expanded = coords_ref.unsqueeze(0)    # (1, M, 2)
    
    dist_to_ref = torch.cdist(grid_coords_expanded.squeeze(1), ref_coords_expanded.squeeze(0))
    min_dist_to_ref = dist_to_ref.min(dim=1)[0]  # 每个网格点到最近重要性采样点的距离
    
    # 对于每个 query 点，找到最近的"空白"网格点
    query_expanded = coords_query.unsqueeze(1)  # (N, 1, 2)
    grid_expanded = grid_coords.unsqueeze(0)    # (1, G, 2)
    
    dist_to_grid = torch.cdist(query_expanded.squeeze(1), grid_expanded.squeeze(0))
    
    scales = []
    for i in range(len(coords_query)):
        # 找到该 query 点附近的网格点
        nearby_grid_idx = torch.argsort(dist_to_grid[i])[:10]  # 取最近的10个
        nearby_gaps = min_dist_to_ref[nearby_grid_idx]
        
        # scale 应该能够覆盖最大的空白区域
        scale = nearby_gaps.max() * 0.7  # 稍微保守一点
        scales.append(scale)
    
    scales = torch.stack(scales)
    return scales.unsqueeze(-1).clamp(min_scale, max_scale)

def compute_scale_delaunay_with_ref(
    coords_query: torch.Tensor,
    coords_ref: torch.Tensor,
    min_scale: float = 0.5,
    max_scale: float = 30.0
) -> torch.Tensor:
    from scipy.spatial import Delaunay
    all_coords = torch.cat([coords_query, coords_ref], dim=0).cpu().numpy()
    tri = Delaunay(all_coords)
    scales = []
    for i in range(len(coords_query)):  # 仅计算查询点的scale
        neighbors = tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i+1]]
        edge_lengths = np.linalg.norm(all_coords[neighbors] - all_coords[i], axis=1)
        scale = np.mean(edge_lengths) * 0.5
        scales.append(scale)
    return torch.tensor(scales, dtype=torch.float32, device=coords_query.device).clamp(min_scale, max_scale).unsqueeze(-1)
    # return torch.tensor(scales, dtype=torch.float32, device=coords_query.device).unsqueeze(-1)

def compute_scale_delaunay_with_boundary(
    coords_query: torch.Tensor,
    coords_ref: torch.Tensor,
    boundary_multiplier: float = 2.0,
    min_scale: float = 0.5,
    max_scale: float = 30.0
) -> torch.Tensor:
    from scipy.spatial import Delaunay
    from scipy.spatial import ConvexHull
    all_coords = torch.cat([coords_query, coords_ref], dim=0).cpu().numpy()
    tri = Delaunay(all_coords)
    hull = ConvexHull(all_coords)
    boundary_indices = set(hull.vertices)

    n_query = len(coords_query)
    scales = []
    for i in range(n_query):
        # 获取Delaunay邻接边长度
        neighbors = tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i+1]]
        edge_lengths = np.linalg.norm(all_coords[neighbors] - all_coords[i], axis=1)
        scale = np.mean(edge_lengths) * 0.5 if len(edge_lengths) > 0 else min_scale
        # 如果是边界点，放大scale
        if i in boundary_indices:
            scale = min(scale * boundary_multiplier, max_scale)
        scales.append(scale)

    return torch.tensor(scales, dtype=torch.float32, device=coords_query.device).clamp(min_scale, max_scale).unsqueeze(-1)

def compute_scale_from_multiscale_coverage(
    coords_query: torch.Tensor,
    coords_ref: torch.Tensor,
    scales: list = [0.5, 1.0, 2.0, 4.0],
    coverage_threshold: float = 0.8,
    min_scale: float = 0.5,
    max_scale: float = 30.0
) -> torch.Tensor:
    """多尺度覆盖分析：选择能够最好覆盖空白区域的 scale"""
    final_scales = []
    
    for query_pt in coords_query:
        best_scale = scales[0]
        best_coverage = 0
        
        for test_scale in scales:
            # 计算该尺度下的覆盖情况
            distances = torch.norm(coords_ref - query_pt.unsqueeze(0), dim=1)
            covered_points = (distances <= test_scale).sum().float()
            coverage = covered_points / len(coords_ref)
            
            # 寻找合适的覆盖度
            if coverage >= coverage_threshold:
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_scale = test_scale
                break
        
        final_scales.append(best_scale)
    
    scales_tensor = torch.tensor(final_scales, device=coords_query.device)
    return scales_tensor.unsqueeze(-1).clamp(min_scale, max_scale)

# ===== 辅助函数 =====

def polygon_area(vertices):
    """计算多边形面积 (Shoelace formula)"""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0



def compute_scale_delaunay(
    coords_query: torch.Tensor,
    min_scale: float = 0.5,
    max_scale: float = 30.0
) -> torch.Tensor:
    from scipy.spatial import Delaunay

    coords_np = coords_query.cpu().numpy()
    tri = Delaunay(coords_np)
    scales = []
    for i in range(len(coords_np)):
        # 找到包含该点的所有三角形边
        simplex_indices = tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i+1]]
        edge_lengths = np.linalg.norm(coords_np[simplex_indices] - coords_np[i], axis=1)
        scale = np.mean(edge_lengths) * 0.5  # 调节系数
        scales.append(scale)
    return torch.tensor(scales, device=coords_query.device).clamp(min_scale, max_scale).unsqueeze(-1)

# ===== 主函数更新 =====

def compute_importance_scales(
    values: torch.Tensor,
    coords: Optional[torch.Tensor] = None,
    mode: Literal["inv_gradient", "exp_decay", "norm_gradient" , "weight", "adaptive", "frequency", "perceptual"] = "inv_gradient",
    image_bounds: Optional[Tuple[float, float, float, float]] = None,
    image_tensor: Optional[torch.Tensor] = None,
    saliency_map: Optional[torch.Tensor] = None,
    min_scale: float = 0.1,
    max_scale: float = 30.0,
    base_scale: float = 5.0,
    eps: float = 1e-5
) -> torch.Tensor:
    """主函数：计算 importance 采样点的 scale"""
    if mode == "inv_gradient":
        return compute_scale_from_inv_gradient(values, eps)
    elif mode == "inv_gradient_clamped":
        return compute_scale_from_inv_gradient_clamped(values, min_scale, max_scale, eps)
    elif mode == "inv_gradient_log10":
        return compute_scale_from_inv_gradient_log(values, eps)
    elif mode == "norm_inv_gradient_log10":
        return compute_scale_from_norm_inv_gradient_log(values, eps)
    elif mode == "exp_decay":
        return compute_scale_exp(values, alpha=0.5, base_scale=base_scale)
    elif mode == "inv_gradient_sqrt":
        return compute_scale_from_inv_gradient_sqrt(values, eps)
    elif mode == "norm_inv_gradient":
        return compute_scale_from_mixed_norm(values, eps)
    elif mode == "smooth_floor":
        return compute_scale_from_smooth_floor(values, eps)
    if mode == "norm_inv_gradient":
        return compute_scale_from_norm_gradient(values, eps)
    elif mode == "weight":
        return compute_scale_from_weight(values, eps)
    elif mode == "adaptive":
        return compute_scale_from_adaptive_gradient(values, coords, image_bounds, eps=eps)
    elif mode == "frequency":
        return compute_scale_from_frequency_analysis(coords, image_tensor, image_bounds, eps=eps)
    elif mode == "perceptual":
        return compute_scale_from_perceptual_importance(coords, values, saliency_map, image_bounds, eps)
    else:
        raise ValueError(f"Unsupported importance mode: {mode}")

def compute_uniform_scales(
    coords_query: torch.Tensor,
    coords_ref: torch.Tensor,
    method: Literal["knn", "voronoi", "density", "gap_filling", "multiscale"] = "knn",
    image_bounds: Optional[Tuple[float, float, float, float]] = None,
    knn_K: int = 3,
    min_scale: float = 0.5,
    max_scale: float = 30.0,
    **kwargs
) -> torch.Tensor:
    """主函数：计算 uniform 采样点的 scale，支持多种策略"""
    if method == "knn":
        return compute_scale_from_knn(coords_query, coords_ref, knn_K, min_scale, max_scale)
    elif method == "voronoi":
        return compute_scale_from_voronoi(coords_query, coords_ref, min_scale, max_scale)
    elif method == "delaunay":
        return compute_scale_delaunay_with_ref(coords_query, coords_ref, min_scale, max_scale)
    elif method == "delaunay_with_boundary":
        return compute_scale_delaunay_with_boundary(coords_query, coords_ref, min_scale, max_scale)
    elif method == "density":
        return compute_scale_from_density_estimation(coords_query, coords_ref, 
                                                   kwargs.get('bandwidth', 1.0), min_scale, max_scale)
    elif method == "gap_filling":
        return compute_scale_from_gap_filling(coords_query, coords_ref, image_bounds, 
                                            kwargs.get('grid_resolution', 64), min_scale, max_scale)
    elif method == "multiscale":
        return compute_scale_from_multiscale_coverage(coords_query, coords_ref, 
                                                    kwargs.get('scales', [2.0, 4.0]),
                                                    kwargs.get('coverage_threshold', 0.8),
                                                    min_scale, max_scale)
    else:
        raise ValueError(f"Unsupported uniform scale method: {method}")

# 原有的 KNN 和相关函数保持不变
def compute_scale_from_knn(
    coords_query: torch.Tensor,
    coords_ref: torch.Tensor,
    knn_K: int = 3,
    min_scale: float = 0.5,
    max_scale: float = 30.0
) -> torch.Tensor:
    """通过 FAISS KNN 距离估计均匀采样点的 scale"""
    coords_ref_np = coords_ref.cpu().numpy().astype('float32')
    coords_query_np = coords_query.cpu().numpy().astype('float32')

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(2)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index.add(coords_ref_np)

    D, _ = gpu_index.search(coords_query_np, knn_K)
    dist_mean = D.mean(axis=1)
    scale = torch.sqrt(torch.clamp(torch.from_numpy(dist_mean), min=1e-6)).to(coords_query.device)
    # return scale.unsqueeze(-1).clamp(min_scale, max_scale)
    return scale.unsqueeze(-1)

def knn_dist_query_ref(coords_query: torch.Tensor, coords_ref: torch.Tensor, K: int = 3) -> torch.Tensor:
    """返回 query 到 reference 的 KNN 平均距离，用于分析密度或 scale 推断"""
    coords_ref_np = coords_ref.cpu().numpy().astype('float32')
    coords_query_np = coords_query.cpu().numpy().astype('float32')

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(2)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index.add(coords_ref_np)

    D, _ = gpu_index.search(coords_query_np, K)
    dist_mean = D.mean(axis=1)
    return torch.from_numpy(dist_mean).to(coords_query.device)