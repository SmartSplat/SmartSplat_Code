import torch
import torch.nn.functional as F
from torch.quasirandom import SobolEngine

@torch.no_grad()
def sample_color_by_scale_circle_batch(large_image, coords, scales, kernel_size=9, mode='mean'):
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    assert mode in ['center', "gaussian_median", 'mean', 'gaussian', 'median', 'bilateral', 'maxgrad', 'sobol', 'sobel_dir']

    N = coords.shape[0]
    device = coords.device
    _, _, H, W = large_image.shape
    half = kernel_size // 2

    if mode == 'center':
        norm_coords = coords.clone()
        norm_coords[..., 0] = (norm_coords[..., 0] / (W - 1)) * 2 - 1
        norm_coords[..., 1] = (norm_coords[..., 1] / (H - 1)) * 2 - 1
        norm_coords = norm_coords.view(N, 1, 1, 2)
        center_sample = F.grid_sample(
            large_image.expand(N, -1, -1, -1),
            norm_coords,
            mode='bilinear',
            align_corners=True
        ).squeeze(-1).squeeze(-1)
        return center_sample.clamp(0.0, 1.0)

    # 通用 patch grid
    grid_x, grid_y = torch.meshgrid(
        torch.arange(-half, half + 1, device=device),
        torch.arange(-half, half + 1, device=device),
        indexing='ij'
    )
    offset_grid = torch.stack([grid_y, grid_x], dim=-1).float()
    dist_sq = offset_grid[..., 0]**2 + offset_grid[..., 1]**2
    circle_mask = (dist_sq <= (half + 0.5)**2).float()
    circle_mask_flat = circle_mask.flatten().view(1, 1, -1)
    dist_sq_flat = dist_sq.flatten().view(1, 1, -1)

    offset_grid = offset_grid.view(1, -1, 2)
    coords_exp = coords.view(N, 1, 2)
    scales_exp = scales.view(N, 1, 1)
    scale_factor = scales_exp / half
    sample_coords = coords_exp + offset_grid * scale_factor

    norm_coords = sample_coords.clone()
    norm_coords[..., 0] = (norm_coords[..., 0] / (W - 1)) * 2 - 1
    norm_coords[..., 1] = (norm_coords[..., 1] / (H - 1)) * 2 - 1
    norm_coords = norm_coords.view(N, kernel_size * kernel_size, 1, 2)

    sampled = F.grid_sample(
        large_image.expand(N, -1, -1, -1),
        norm_coords,
        mode='bilinear',
        align_corners=True
    ).squeeze(3)

    if mode == 'mean':
        weighted = sampled * circle_mask_flat
        avg_color = weighted.sum(dim=2) / (circle_mask_flat.sum() + 1e-6)

    elif mode == 'gaussian':
        sigma = scale_factor.view(N, 1, 1) * half + 1e-6
        weights = torch.exp(-dist_sq_flat / (2 * sigma ** 2)) * circle_mask_flat
        weighted = sampled * weights
        avg_color = weighted.sum(dim=2) / (weights.sum(dim=2) + 1e-6)

    elif mode == 'median':
        masked = sampled * circle_mask_flat
        sorted_vals, _ = masked.sort(dim=2)
        median_idx = int(circle_mask_flat.sum().item() // 2)
        avg_color = sorted_vals[:, :, median_idx]

    elif mode == 'gaussian_median':
        # 使用 Gaussian 权重，或设为全1实现普通 median
        sigma = scale_factor.view(N, 1, 1) * half + 1e-6
        weights = torch.exp(-dist_sq_flat / (2 * sigma ** 2)) * circle_mask_flat  # (1, 1, K)
        # 退化为不同median
        # weights = circle_mask_flat  # 所有像素等权
        weights = weights.expand(N, 1, -1)  # 广播到每个 batch
        sampled_masked = sampled * circle_mask_flat
        

        # 按像素值升序排序，同时打乱权重的顺序
        sorted_vals, sort_idx = sampled_masked.sort(dim=2)
        sorted_weights = torch.gather(weights.expand(-1, 3, -1), 2, sort_idx)
        # sorted_weights = torch.gather(weights, 2, sort_idx)

        # 归一化权重
        sorted_weights = sorted_weights / (sorted_weights.sum(dim=2, keepdim=True) + 1e-6)

        # 累加权重并寻找第一个 >= 0.5 的索引
        cum_weights = sorted_weights.cumsum(dim=2)
        median_mask = (cum_weights >= 0.5).float()
        median_idx = median_mask.argmax(dim=2, keepdim=True)  # (N, 1)

        # 提取对应颜色
        avg_color = torch.gather(sorted_vals, 2, median_idx.expand(-1, sampled.shape[1], -1)).squeeze(2)


    elif mode == 'bilateral':
        sigma_spatial = scale_factor.view(N, 1, 1) * half + 1e-6
        spatial_weights = torch.exp(-dist_sq_flat / (2 * sigma_spatial ** 2)) * circle_mask_flat

        center_coords = coords.clone()
        center_coords[..., 0] = (center_coords[..., 0] / (W - 1)) * 2 - 1
        center_coords[..., 1] = (center_coords[..., 1] / (H - 1)) * 2 - 1
        center_coords = center_coords.view(N, 1, 1, 2)
        center_rgb = F.grid_sample(
            large_image.expand(N, -1, -1, -1),
            center_coords,
            mode='bilinear',
            align_corners=True
        ).squeeze(-1).squeeze(-1).unsqueeze(-1)

        sigma_color = 0.2
        color_diff = sampled - center_rgb
        color_dist_sq = (color_diff ** 2).sum(dim=1, keepdim=True)
        color_weights = torch.exp(-color_dist_sq / (2 * sigma_color ** 2))

        bilateral_weights = spatial_weights * color_weights
        weighted = sampled * bilateral_weights
        avg_color = weighted.sum(dim=2) / (bilateral_weights.sum(dim=2) + 1e-6)

    elif mode == 'maxgrad':
        gray = (0.2989 * sampled[:, 0] + 0.5870 * sampled[:, 1] + 0.1140 * sampled[:, 2])
        gray_masked = gray * circle_mask_flat.squeeze()
        gray_reshaped = gray_masked.view(N, kernel_size, kernel_size)

        gx = F.pad(gray_reshaped, (1, 1, 1, 1), mode='replicate')[:, 1:-1, 2:] - \
             F.pad(gray_reshaped, (1, 1, 1, 1), mode='replicate')[:, 1:-1, :-2]
        gy = F.pad(gray_reshaped, (1, 1, 1, 1), mode='replicate')[:, 2:, 1:-1] - \
             F.pad(gray_reshaped, (1, 1, 1, 1), mode='replicate')[:, :-2, 1:-1]
        grad_mag = (gx ** 2 + gy ** 2).view(N, -1)
        grad_masked = grad_mag * circle_mask_flat.squeeze()

        max_idx = grad_masked.argmax(dim=1)
        avg_color = torch.gather(sampled, dim=2, index=max_idx[:, None, None].expand(-1, 3, 1)).squeeze(2)

    elif mode == 'sobol':
        S = 32
        sobol_engine = SobolEngine(dimension=2, scramble=True)
        sobol_points = sobol_engine.draw(S).to(device)
        sobol_points = (sobol_points - 0.5) * 2.0 * half
        sobol_offsets = sobol_points.unsqueeze(0).expand(N, -1, -1) * scale_factor.view(N, 1, 1)
        sobol_coords = coords.unsqueeze(1) + sobol_offsets

        norm_sobol = sobol_coords.clone()
        norm_sobol[..., 0] = (norm_sobol[..., 0] / (W - 1)) * 2 - 1
        norm_sobol[..., 1] = (norm_sobol[..., 1] / (H - 1)) * 2 - 1
        norm_sobol = norm_sobol.view(N, S, 1, 2)

        sobol_sampled = F.grid_sample(
            large_image.expand(N, -1, -1, -1),
            norm_sobol,
            mode='bilinear',
            align_corners=True
        ).squeeze(3)

        avg_color = sobol_sampled.mean(dim=2)

    elif mode == 'sobel_dir':
        gray = (0.2989 * sampled[:, 0] + 0.5870 * sampled[:, 1] + 0.1140 * sampled[:, 2])
        gray_reshaped = (gray * circle_mask_flat.squeeze()).view(N, kernel_size, kernel_size)

        gx = F.pad(gray_reshaped, (1, 1, 1, 1), mode='replicate')[:, 1:-1, 2:] - \
             F.pad(gray_reshaped, (1, 1, 1, 1), mode='replicate')[:, 1:-1, :-2]
        gy = F.pad(gray_reshaped, (1, 1, 1, 1), mode='replicate')[:, 2:, 1:-1] - \
             F.pad(gray_reshaped, (1, 1, 1, 1), mode='replicate')[:, :-2, 1:-1]

        gx_center = gx[:, half, half]
        gy_center = gy[:, half, half]
        theta = torch.atan2(gy_center, gx_center)
        direction = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

        M = 5
        step_sizes = torch.linspace(0.2, 1.0, M, device=device).view(1, M, 1)
        offset_vec = direction.view(N, 1, 2) * scales.view(N, 1, 1) * step_sizes
        direction_coords = coords.view(N, 1, 2) + offset_vec

        norm_coords = direction_coords.clone()
        norm_coords[..., 0] = (norm_coords[..., 0] / (W - 1)) * 2 - 1
        norm_coords[..., 1] = (norm_coords[..., 1] / (H - 1)) * 2 - 1
        norm_coords = norm_coords.view(N, M, 1, 2)

        color_sampled = F.grid_sample(
            large_image.expand(N, -1, -1, -1),
            norm_coords,
            mode='bilinear',
            align_corners=True
        ).squeeze(3)

        avg_color = color_sampled.mean(dim=2)

    return avg_color.clamp(0.0, 1.0)


@torch.no_grad()
def safe_sample_color_by_scale_circle(
    large_image,
    coords,
    scales,
    kernel_size=9,
    mode='mean',
    batch_size=1024,
    verbose=False
):
    N = coords.shape[0]
    results = []

    for i in range(0, N, batch_size):
        if verbose:
            print(f"[sample_color] batch {i}/{N}")
        coords_batch = coords[i:i + batch_size]
        scales_batch = scales[i:i + batch_size]
        colors_batch = sample_color_by_scale_circle_batch(
            large_image, coords_batch, scales_batch,
            kernel_size=kernel_size, mode=mode
        )
        results.append(colors_batch)

    return torch.cat(results, dim=0)

"""
from color_sampler import safe_sample_color_by_scale_circle

colors = safe_sample_color_by_scale_circle(
    large_image,  # (1,3,H,W)
    coords,       # (N,2)
    scales,       # (N,1)
    kernel_size=9,
    mode='sobel_dir',
    batch_size=2048,
    verbose=True
)
"""