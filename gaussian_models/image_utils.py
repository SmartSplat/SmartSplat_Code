import torch
import torch.nn.functional as F
import simple_knn_2d._C as knn2d_cuda

def compute_gradient_magnitude(image: torch.Tensor):
    """
    image: (1, C=3, H, W), on CUDA
    return: (H, W) gradient magnitude map
    """
    assert image.dim() == 4 and image.shape[1] == 3, "Expect image shape (1, 3, H, W)"

    # RGB to grayscale
    gray = 0.2989 * image[:, 0] + 0.5870 * image[:, 1] + 0.1140 * image[:, 2]  # (1, H, W)
    gray = gray.unsqueeze(1)  # (1, 1, H, W)

    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze(0).squeeze(0)  # (H, W)
    return grad_mag

def sample_gradient_means_from_large_image(large_image, N, tile_size=256, tiles_per_batch=64):
    """
    large_image: (1, 3, H, W)
    return: (N, 2) sampled positions
    """
    _, _, H, W = large_image.shape
    device = large_image.device
    all_coords = []

    # 动态分配每个 tile 的采样数，使总和等于 N
    base = N // tiles_per_batch
    remainder = N % tiles_per_batch
    num_samples_per_tile = [base + 1 if i < remainder else base for i in range(tiles_per_batch)]

    for n_per_tile in num_samples_per_tile:
        top = torch.randint(0, H - tile_size, (1,)).item()
        left = torch.randint(0, W - tile_size, (1,)).item()

        patch = large_image[:, :, top:top+tile_size, left:left+tile_size]  # (1, 3, H, W)
        grad = compute_gradient_magnitude(patch)  # (H, W)

        flat_grad = grad.flatten()
        prob = flat_grad / (flat_grad.sum() + 1e-8)

        idx = torch.multinomial(prob, n_per_tile, replacement=False)
        y, x = torch.div(idx, tile_size, rounding_mode='floor'), idx % tile_size

        x_full = x + left
        y_full = y + top
        coords = torch.stack([x_full.float(), y_full.float()], dim=-1)  # (n_per_tile, 2)
        all_coords.append(coords)

    return torch.cat(all_coords, dim=0).to(device)  # shape: (N, 2)

def sample_mixed_gradient_uniform_by_tiles(large_image, N, ratio=0.6, tile_size=1024, tiles_per_batch=64):
    """
    large_image: (1,3,H,W) CUDA tensor, 归一化RGB
    N: 总采样点数
    ratio: 重要性采样占比
    tile_size: 每个tile大小
    tiles_per_batch: 采样tile数量

    return:
        coords: (N,2) 坐标 (x,y) 格式，大图坐标系
        scales: (N,1) scale大小
    """

    _, _, H, W = large_image.shape
    device = large_image.device

    n_imp = int(N * ratio)
    n_uni = N - n_imp

    # 把重要性采样点数均匀分配到每个tile
    base = n_imp // tiles_per_batch
    remainder = n_imp % tiles_per_batch
    samples_per_tile = [base + 1 if i < remainder else base for i in range(tiles_per_batch)]

    coords_imp_list = []
    scales_imp_list = []

    for n_samples in samples_per_tile:
        # 随机选tile左上角
        top = torch.randint(0, H - tile_size + 1, (1,), device=device).item()
        left = torch.randint(0, W - tile_size + 1, (1,), device=device).item()

        patch = large_image[:, :, top:top+tile_size, left:left+tile_size]
        grad = compute_gradient_magnitude(patch)  # (tile_size, tile_size)

        grad_flat = grad.flatten()
        prob = grad_flat / (grad_flat.sum() + 1e-8)

        # 采样 n_samples 点
        idx = torch.multinomial(prob, n_samples, replacement=False)
        ys = idx // tile_size
        xs = idx % tile_size

        xs_global = xs.float() + left
        ys_global = ys.float() + top

        coords_tile = torch.stack([xs_global, ys_global], dim=-1)  # (n_samples, 2)
        coords_imp_list.append(coords_tile)

        grad_at_pts = grad[ys, xs]
        scale_tile = 0.5 * (1.0 / (grad_at_pts + 1e-5))  # 梯度大 scale 小，防止除0
        scales_imp_list.append(scale_tile.unsqueeze(-1))

    coords_imp = torch.cat(coords_imp_list, dim=0)  # (n_imp, 2)
    scales_imp = torch.cat(scales_imp_list, dim=0)  # (n_imp, 1)

    # 全图均匀采样
    xs_uni = torch.rand(n_uni, device=device) * W
    ys_uni = torch.rand(n_uni, device=device) * H
    coords_uni = torch.stack([xs_uni, ys_uni], dim=-1)
    scales_uni = torch.ones(n_uni, device=device).unsqueeze(-1) * 3.0  # 均匀采样较大 scale

    coords = torch.cat([coords_imp, coords_uni], dim=0)
    scales = torch.cat([scales_imp, scales_uni], dim=0)

    return coords, scales

def sample_mixed_gradient_uniform_by_tiles_with_colors(large_image, N, ratio=0.6, tile_size=1024, tiles_per_batch=64):
    """
    large_image: (1,3,H,W) CUDA tensor, 归一化RGB
    N: 总采样点数
    ratio: 重要性采样占比
    tile_size: 每个tile大小
    tiles_per_batch: 采样tile数量

    return:
        coords: (N,2) 坐标 (x,y) 格式，大图坐标系
        scales: (N,1) scale大小
        colors: (N,3) 采样点颜色值
    """
    _, _, H, W = large_image.shape
    device = large_image.device

    n_imp = int(N * ratio)
    n_uni = N - n_imp

    base = n_imp // tiles_per_batch
    remainder = n_imp % tiles_per_batch
    samples_per_tile = [base + 1 if i < remainder else base for i in range(tiles_per_batch)]

    coords_imp_list = []
    scales_imp_list = []
    colors_imp_list = []

    for n_samples in samples_per_tile:
        top = torch.randint(0, H - tile_size + 1, (1,), device=device).item()
        left = torch.randint(0, W - tile_size + 1, (1,), device=device).item()

        patch = large_image[:, :, top:top+tile_size, left:left+tile_size]
        grad = compute_gradient_magnitude(patch)  # (tile_size, tile_size)

        grad_flat = grad.flatten()
        prob = grad_flat / (grad_flat.sum() + 1e-8)

        idx = torch.multinomial(prob, n_samples, replacement=False)
        ys = idx // tile_size
        xs = idx % tile_size

        xs_global = xs.float() + left
        ys_global = ys.float() + top

        coords_tile = torch.stack([xs_global, ys_global], dim=-1)  # (n_samples, 2)
        coords_imp_list.append(coords_tile)

        grad_at_pts = grad[ys, xs]
        scale_tile = 0.5 * (1.0 / (grad_at_pts + 1e-5))
        scales_imp_list.append(scale_tile.unsqueeze(-1))

        # 提取颜色值
        color_tile = F.grid_sample(
            large_image, 
            coords_tile.view(1, -1, 1, 2) / torch.tensor([W - 1, H - 1], device=device).view(1, 1, 1, 2) * 2 - 1, 
            align_corners=True, 
            mode='bilinear'
        ).squeeze(3).squeeze(0).permute(1, 0)  # (n_samples, 3)
        colors_imp_list.append(color_tile)

    coords_imp = torch.cat(coords_imp_list, dim=0)
    scales_imp = torch.cat(scales_imp_list, dim=0)
    colors_imp = torch.cat(colors_imp_list, dim=0)

    # 均匀采样
    xs_uni = torch.rand(n_uni, device=device) * W
    ys_uni = torch.rand(n_uni, device=device) * H
    coords_uni = torch.stack([xs_uni, ys_uni], dim=-1)
    scales_uni = torch.ones(n_uni, device=device).unsqueeze(-1) * 3.0

    # 均匀采样的颜色值
    color_uni = F.grid_sample(
        large_image, 
        coords_uni.view(1, -1, 1, 2) / torch.tensor([W - 1, H - 1], device=device).view(1, 1, 1, 2) * 2 - 1,
        align_corners=True, 
        mode='bilinear'
    ).squeeze(3).squeeze(0).permute(1, 0)  # (n_uni, 3)

    coords = torch.cat([coords_imp, coords_uni], dim=0)
    scales = torch.cat([scales_imp, scales_uni], dim=0)
    colors = torch.cat([colors_imp, color_uni], dim=0)

    return coords, scales, colors

"""
coords, scales = sample_mixed_gradient_uniform_by_tiles(image, N=5000)

# mean 模式
colors_mean = sample_color_by_scale_circle_batch(image, coords, scales, kernel_size=9, mode='mean')

# gaussian 模式
colors_gauss = sample_color_by_scale_circle_batch(image, coords, scales, kernel_size=9, mode='gaussian')
"""

def sample_color_by_scale_circle_batch(large_image, coords, scales, kernel_size=9, mode='mean'):
    """
    并行从图像中采样颜色，支持 scale 自适应区域。
    large_image: (1, 3, H, W)
    coords: (N, 2) float tensor, in image coordinate (x, y)
    scales: (N, 1) float tensor
    kernel_size: must be odd
    mode: 'mean' or 'gaussian'
    returns: (N, 3)
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    assert mode in ['mean', 'gaussian']
    
    N = coords.shape[0]
    device = coords.device
    _, _, H, W = large_image.shape

    half = kernel_size // 2

    # 1. 生成基础 grid
    grid_x, grid_y = torch.meshgrid(
        torch.arange(-half, half + 1, device=device),
        torch.arange(-half, half + 1, device=device),
        indexing='ij'
    )  # (k, k)
    offset_grid = torch.stack([grid_y, grid_x], dim=-1).float()  # (k, k, 2)

    # 2. 计算圆形 mask 和高斯权重模板
    dist_sq = offset_grid[..., 0]**2 + offset_grid[..., 1]**2  # (k, k)
    circle_mask = (dist_sq <= (half + 0.5) ** 2).float()  # (k, k)

    offset_grid = offset_grid.view(1, kernel_size * kernel_size, 2)  # (1, K², 2)

    # 3. expand 到每个点
    coords_exp = coords.view(N, 1, 2)  # (N, 1, 2)
    scales_exp = scales.view(N, 1, 1)  # (N, 1, 1)
    scale_factor = scales_exp / half  # normalize

    sample_coords = coords_exp + offset_grid * scale_factor  # (N, K², 2)

    # 4. 归一化到 [-1, 1] 区间供 grid_sample 使用
    norm_coords = sample_coords.clone()
    norm_coords[..., 0] = (norm_coords[..., 0] / (W - 1)) * 2 - 1  # x
    norm_coords[..., 1] = (norm_coords[..., 1] / (H - 1)) * 2 - 1  # y
    norm_coords = norm_coords.view(N, kernel_size * kernel_size, 1, 2)

    # 5. 批量 grid_sample，采样 RGB 值
    sampled = F.grid_sample(
        large_image.expand(N, -1, -1, -1),  # (N, 3, H, W)
        norm_coords,  # (N, K², 1, 2)
        mode='bilinear',
        align_corners=True
    ).squeeze(3)  # (N, 3, K²)

    # 6. 构建权重 mask
    mask_flat = circle_mask.flatten().view(1, 1, -1)  # (1, 1, K²)

    if mode == 'mean':
        weighted = sampled * mask_flat  # (N, 3, K²)
        avg_color = weighted.sum(dim=2) / (mask_flat.sum() + 1e-6)  # (N, 3)

    elif mode == 'gaussian':
        sigma = scales.view(N, 1, 1) + 1e-6
        dist_sq_batch = dist_sq.view(1, 1, -1)  # (1, 1, K²)
        gaussian_weights = torch.exp(-dist_sq_batch / (2 * (half ** 2))) * mask_flat  # (1,1,K²)
        weighted = sampled * gaussian_weights  # (N, 3, K²)
        avg_color = weighted.sum(dim=2) / (gaussian_weights.sum() + 1e-6)  # (N, 3)

    return avg_color.clamp(0.0, 1.0)  # (N, 3)

def knn_mean_sq_distance(query_pts, ref_pts, K=3):
    """
    query_pts: (N, 2) float tensor
    ref_pts: (M, 2) float tensor
    return: (N,) 每个 query 到 ref 的 K 近邻平均平方距离
    """
    N, _ = query_pts.shape
    M, _ = ref_pts.shape

    dists = torch.cdist(query_pts, ref_pts, p=2)  # (N, M)
    knn_dists, _ = torch.topk(dists, k=K, largest=False)  # (N, K)
    return (knn_dists ** 2).mean(dim=1)  # (N,)


def sample_mixed_gradient_uniform_by_tiles_with_mixed_colors(
    large_image, N, ratio=0.6, tile_size=1024, tiles_per_batch=64,
    kernel_size=9, color_mode='mean', knn_K=4
):
    """
    从图像中采样 N 个点，包含梯度重要性采样和均匀采样。
    所有点根据其 scale 自适应区域，进行颜色采样。

    Args:
        large_image: (1,3,H,W) 图像张量，已归一化到 [0,1]
        N: 采样总点数
        ratio: 重要性采样占比
        tile_size: tile 大小，用于局部重要性采样
        tiles_per_batch: tile 批次数（共 ratio*N 个点）
        kernel_size: 颜色采样核大小，必须为奇数
        color_mode: 'mean' 或 'gaussian'
        knn_K: KNN 近邻数，用于均匀采样点 scale 推断

    Returns:
        coords: (N,2) 图像坐标 (x,y)
        scales: (N,1) 每点的 scale
        colors: (N,3) RGB 颜色值
    """
    _, _, H, W = large_image.shape
    device = large_image.device

    n_imp = int(N * ratio)
    n_uni = N - n_imp

    # ---- 梯度重要性采样 ----
    base = n_imp // tiles_per_batch
    remainder = n_imp % tiles_per_batch
    samples_per_tile = [base + 1 if i < remainder else base for i in range(tiles_per_batch)]

    coords_imp_list = []
    scales_imp_list = []

    for n_samples in samples_per_tile:
        top = torch.randint(0, H - tile_size + 1, (1,), device=device).item()
        left = torch.randint(0, W - tile_size + 1, (1,), device=device).item()

        patch = large_image[:, :, top:top+tile_size, left:left+tile_size]
        grad = compute_gradient_magnitude(patch)  # (tile_size, tile_size)

        grad_flat = grad.flatten()
        prob = grad_flat / (grad_flat.sum() + 1e-8)

        idx = torch.multinomial(prob, n_samples, replacement=False)
        ys = idx // tile_size
        xs = idx % tile_size

        xs_global = xs.float() + left
        ys_global = ys.float() + top

        coords_tile = torch.stack([xs_global, ys_global], dim=-1)  # (n_samples, 2)
        coords_imp_list.append(coords_tile)

        grad_at_pts = grad[ys, xs]
        scale_tile = 0.5 * (1.0 / (grad_at_pts + 1e-5))
        scales_imp_list.append(scale_tile.unsqueeze(-1))

    coords_imp = torch.cat(coords_imp_list, dim=0)  # (n_imp, 2)
    scales_imp = torch.cat(scales_imp_list, dim=0)  # (n_imp, 1)
    print(f"[Init] Scales_imp: {len(scales_imp)}, Scales_imp ∈ ({scales_imp.min():.2f}, {scales_imp.max():.2f})")

    # ---- 均匀采样 + KNN 推断 scale ----
    xs_uni = torch.rand(n_uni, device=device) * W
    ys_uni = torch.rand(n_uni, device=device) * H
    coords_uni = torch.stack([xs_uni, ys_uni], dim=-1)  # (n_uni, 2)

    # 使用 KNN 推断 scale（参考全部采样点的密度）
    all_coords = torch.cat([coords_imp, coords_uni], dim=0)  # (N, 2)
    with torch.no_grad():
        # 只计算uni_coords中K最近邻距离
        # dist_sq_uni = knn2d_cuda.distCUDA2(coords_uni)  # (n_uni,)
        # scale_uni = torch.sqrt(torch.clamp(dist_sq_uni, min=1e-6)).unsqueeze(-1)  # (n_uni, 1)
        # scale_uni = scale_uni.clamp(1.0, 20.0)

        # 在所有点中找uni_coords的K最近邻距离(torch.cdist计算, 大量点时会OOM)
        # dist_sq_uni = knn_mean_sq_distance(coords_uni, all_coords, K=3)  # (n_uni,)
        # scale_uni = torch.sqrt(torch.clamp(dist_sq_uni, min=1e-6)).unsqueeze(-1)
        # scale_uni = scale_uni.clamp(0.5, 30.0)

        # 使用KNN-CUDA计算
        from simple_knn_2d_qr import knn_dist_self, knn_dist_query_ref
        dist_sq_uni = knn_dist_query_ref(coords_uni, all_coords, K=3)  # (n_uni,)
        scale_uni = torch.sqrt(torch.clamp(dist_sq_uni, min=1e-6)).unsqueeze(-1)

        print(f"[Init] Scale_uni: {len(scale_uni)}, Scale_uni ∈ ({scale_uni.min():.2f}, {scale_uni.max():.2f})")

    # 合并所有点
    coords = torch.cat([coords_imp, coords_uni], dim=0)  # (N, 2)
    scales = torch.cat([scales_imp, scale_uni], dim=0)   # (N, 1)

    # ---- 颜色采样 ----
    colors = sample_color_by_scale_circle_batch(
        large_image, coords, scales,
        kernel_size=kernel_size,
        mode=color_mode
    )  # (N, 3)

    return coords, scales, colors


"""
coords, scales = sample_mixed_gradient_uniform_by_tiles(image_16k, N=30000)
colors = sample_from_16k_image(image_16k, coords, scales, batch_size=8192, kernel_size=9, mode='gaussian')
"""
def sample_from_16k_image(large_image, all_coords, all_scales, batch_size=8192, kernel_size=9, mode='mean'):
    """
    针对16K图像的颜色采样策略：将全部点按batch处理，避免OOM。
    large_image: (1, 3, H, W)
    all_coords: (N, 2)
    all_scales: (N, 1)
    return: (N, 3)
    """
    N = all_coords.shape[0]
    results = []

    for i in range(0, N, batch_size):
        coords_batch = all_coords[i:i+batch_size]
        scales_batch = all_scales[i:i+batch_size]
        color_batch = sample_color_by_scale_circle_batch(
            large_image, coords_batch, scales_batch,
            kernel_size=kernel_size, mode=mode
        )
        results.append(color_batch)

    return torch.cat(results, dim=0)  # (N, 3)


