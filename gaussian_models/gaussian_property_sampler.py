import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from PIL import Image
import torchvision.transforms as transforms
import faiss
from typing import Tuple, List
import warnings
import os
warnings.filterwarnings('ignore')

Image.MAX_IMAGE_PIXELS = 10000000000

from  gaussian_models.scale_sampler import compute_importance_scales, compute_uniform_scales
from gaussian_models.color_sampler import safe_sample_color_by_scale_circle

def image_path_to_tensor(image_path: str):
        """Load image and convert to tensor"""
        img = Image.open(image_path)
        transform = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
        return img_tensor.to("cuda")

class GPUVariationalSamplingVisualizer:
    def __init__(self, N=500, tile_size=1024, tiles_per_batch=16, ratio=0.7, 
                 lambda_g=0.5, lambda_v=0.5, knn_K=3, kernel_size=9, device='cuda'):
        self.N = N
        self.tile_size = tile_size
        self.tiles_per_batch = tiles_per_batch
        self.ratio = ratio
        self.lambda_g = lambda_g
        self.lambda_v = lambda_v
        self.knn_K = knn_K
        self.kernel_size = kernel_size
        self.device = device
        
        # Initialize FAISS GPU resources
        self.faiss_res = faiss.StandardGpuResources()
        
    def compute_gradient_magnitude_gpu(self, image):
        """GPU-optimized gradient magnitude computation"""
        # Convert to grayscale
        if image.dim() == 4:  # (B, C, H, W)
            gray = image.mean(dim=1, keepdim=True)
        else:  # (C, H, W)
            gray = image.mean(dim=0, keepdim=True).unsqueeze(0)
        
        # Sobel operators on GPU
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=image.dtype, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=image.dtype, device=self.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        return magnitude.squeeze()
    
    def compute_color_variance_gpu(self, image):
        """GPU-optimized color variance computation using unfold"""
        if image.dim() == 4:
            B, C, H, W = image.shape
        else:
            C, H, W = image.shape
            image = image.unsqueeze(0)
            B = 1
            
        kernel_size = 5
        pad = kernel_size // 2
        
        # GPU-efficient unfold operation
        patches = F.unfold(image, kernel_size, padding=pad)  # [B, C*k*k, H*W]
        patches = patches.view(B, C, kernel_size*kernel_size, H*W)
        
        # Compute variance across spatial dimensions
        variance = patches.var(dim=2)  # [B, C, H*W]
        variance = variance.mean(dim=1)  # [B, H*W] - average across channels
        variance = variance.view(H, W)
        
        return variance
    
    def sample_uniform_with_faiss_kdtree_gpu(self, coords_imp, exclude_radius, n_uni, H, W, 
                                            max_trials=50, batch_size=10000):
        """GPU-accelerated uniform sampling using FAISS with exclusion"""
        if len(coords_imp) == 0:
            # Direct random sampling when no exclusion points
            coords = torch.rand(n_uni, 2, device=self.device)
            coords[:, 0] *= W
            coords[:, 1] *= H
            return coords
        
        # Convert importance coordinates to numpy for FAISS
        coords_imp_np = coords_imp.cpu().numpy().astype('float32')
        
        # Create GPU FAISS index
        index_flat = faiss.IndexFlatL2(2)
        gpu_index = faiss.index_cpu_to_gpu(self.faiss_res, 0, index_flat)
        gpu_index.add(coords_imp_np)
        
        coords_uni_list = []
        trials = 0
        exclude_radius_sq = exclude_radius ** 2
        
        print(f"Starting uniform sampling with exclusion radius: {exclude_radius:.2f}")
        
        while len(coords_uni_list) < n_uni and trials < max_trials:
            # Generate candidate points on GPU
            remaining = n_uni - len(coords_uni_list)
            current_batch_size = min(batch_size, remaining * 5)  # Generate more candidates
            
            candidates = torch.rand(current_batch_size, 2, device=self.device)
            candidates[:, 0] *= W
            candidates[:, 1] *= H
            
            # Convert to numpy for FAISS search
            candidates_np = candidates.cpu().numpy().astype('float32')
            
            # Search nearest neighbors on GPU
            D, I = gpu_index.search(candidates_np, 1)
            
            # Filter points that are far enough (on CPU for now, but fast)
            valid_mask = D[:, 0] > exclude_radius_sq
            valid_candidates = candidates[valid_mask]
            
            if len(valid_candidates) > 0:
                # Take only what we need
                take_num = min(len(valid_candidates), remaining)
                selected = valid_candidates[:take_num]
                coords_uni_list.append(selected)
                
                # Add selected points to the index for future exclusion
                if take_num > 0:
                    selected_np = selected.cpu().numpy().astype('float32')
                    gpu_index.add(selected_np)
            
            trials += 1
            
            # if trials % 10 == 0:
                # current_count = sum(len(batch) for batch in coords_uni_list)
                # print(f"Trial {trials}: collected {current_count}/{n_uni} uniform points")
        
        # Concatenate all collected points
        if coords_uni_list:
            coords_uni = torch.cat(coords_uni_list, dim=0)[:n_uni]
        else:
            print(f"Warning: Failed to sample enough uniform points, generating random fallback")
            coords_uni = torch.rand(n_uni, 2, device=self.device)
            coords_uni[:, 0] *= W
            coords_uni[:, 1] *= H
        
        print(f"Uniform sampling completed: {len(coords_uni)} points")
        return coords_uni
    
    def gpu_knn_distances(self, query_coords, ref_coords, K):
        """GPU-accelerated KNN distance computation using FAISS"""
        if len(ref_coords) < K:
            # Fallback for small datasets
            distances = torch.cdist(query_coords, ref_coords)
            k_actual = min(K, distances.shape[1])
            knn_distances, _ = torch.topk(distances, k_actual, dim=1, largest=False)
            return knn_distances.mean(dim=1)
        
        # Convert to numpy for FAISS
        query_np = query_coords.cpu().numpy().astype('float32')
        ref_np = ref_coords.cpu().numpy().astype('float32')
        
        # Create GPU FAISS index
        index_flat = faiss.IndexFlatL2(2)
        gpu_index = faiss.index_cpu_to_gpu(self.faiss_res, 0, index_flat)
        gpu_index.add(ref_np)
        
        # Search K+1 nearest neighbors (including self)
        D, I = gpu_index.search(query_np, min(K+1, len(ref_coords)))
        
        # Convert back to torch and compute mean distances
        distances = torch.from_numpy(np.sqrt(D)).to(self.device)
        
        # Exclude self-distances (first column if exists)
        if distances.shape[1] > 1:
            # Exclude the closest point (likely self with distance ~0)
            valid_distances = distances[:, 1:]
        else:
            valid_distances = distances
            
        mean_distances = valid_distances.mean(dim=1)
        return mean_distances
    
    def sample_color_by_scale_gpu(self, image, coords, scales, mode='mean'):
        """GPU-optimized batch color sampling"""
        B, C, H, W = image.shape
        n_points = len(coords)
        colors = torch.zeros(n_points, C, device=self.device)
        
        # Process in smaller batches to manage memory
        batch_size = min(500, n_points)
        
        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            batch_coords = coords[i:end_idx]
            batch_scales = scales[i:end_idx]
            
            for j, (coord, scale) in enumerate(zip(batch_coords, batch_scales)):
                x, y = coord[0].item(), coord[1].item()
                radius = max(1, int(scale.item()))
                
                # Clamp coordinates to image bounds
                x_min = max(0, int(x - radius))
                x_max = min(W, int(x + radius) + 1)
                y_min = max(0, int(y - radius))
                y_max = min(H, int(y + radius) + 1)
                
                if x_max > x_min and y_max > y_min:
                    # Extract region (already on GPU)
                    region = image[0, :, y_min:y_max, x_min:x_max]
                    
                    if mode == 'mean':
                        colors[i + j] = region.mean(dim=[1, 2])
                    elif mode == 'center':
                        center_y = min(int(y) - y_min, region.shape[1] - 1)
                        center_x = min(int(x) - x_min, region.shape[2] - 1)
                        colors[i + j] = region[:, center_y, center_x]
        
        return colors
    
    def sample_variational_importance_gpu(self, large_image):
        """GPU-accelerated main variational importance sampling function"""
        _, _, H, W = large_image.shape
        n_imp = int(self.N * self.ratio)
        n_uni = self.N - n_imp
        
        coords_imp_list = []
        scales_imp_list = []
        
        print(f"Starting GPU variational importance sampling: {n_imp} importance + {n_uni} uniform")
        print(f"Image size: {H}x{W}, Tile size: {self.tile_size}")
        
        random_sample =  False
        if random_sample:
            # Importance sampling with random tiles
            n_samples_per_tile = n_imp // self.tiles_per_batch
            residual = n_imp % self.tiles_per_batch
            
            for t in range(self.tiles_per_batch):
                # Random tile location
                top = torch.randint(0, max(1, H - self.tile_size + 1), (1,), device=self.device).item()
                left = torch.randint(0, max(1, W - self.tile_size + 1), (1,), device=self.device).item()
                
                # Extract patch (stays on GPU)
                actual_tile_h = min(self.tile_size, H - top)
                actual_tile_w = min(self.tile_size, W - left)
                patch = large_image[:, :, top:top+actual_tile_h, left:left+actual_tile_w]
                
                # Compute features on GPU
                grad = self.compute_gradient_magnitude_gpu(patch)
                color_var = self.compute_color_variance_gpu(patch)
                
                # Normalize features
                grad_norm = grad / (grad.max() + 1e-8)
                color_var_norm = color_var / (color_var.max() + 1e-8)
                
                # Combine weights
                weight = self.lambda_g * grad_norm + self.lambda_v * color_var_norm
                weight = weight.clamp(min=1e-8)  # Avoid zero weights
                
                # Multinomial sampling on GPU
                weight_flat = weight.flatten()
                prob = weight_flat / weight_flat.sum()
                
                n_samples_tile = n_samples_per_tile + (1 if t < residual else 0)
                if n_samples_tile > 0:
                    idx = torch.multinomial(prob, min(n_samples_tile, len(prob)), replacement=False)
                    
                    # Convert indices to coordinates
                    ys = idx // actual_tile_w
                    xs = idx % actual_tile_w
                    
                    # Global coordinates
                    xs_global = xs.float() + left
                    ys_global = ys.float() + top
                    
                    coords_tile = torch.stack([xs_global, ys_global], dim=-1)
                    coords_imp_list.append(coords_tile)
                    
                    # Compute scales based on weights
                    weight_at_pts = weight[ys, xs]
                    # scale_tile = torch.clamp(1.0 / (weight_at_pts + 1e-5), 1.0, 30.0)
                    # scale_tile = 1.0 / (weight_at_pts + 1e-5)
                    scale_tile = compute_importance_scales(weight_at_pts, coords_tile, mode="norm_inv_gradient_log10")
                    
                    scales_imp_list.append(scale_tile.unsqueeze(-1))
                
                if (t + 1) % 5 == 0:
                    print(f"Processed tile {t+1}/{self.tiles_per_batch}")
        else:
            # 使用自适应步长确保均匀覆盖
            # 计算理想的tile数量
            h_tiles = max(1, (H + self.tile_size - 1) // self.tile_size)  # 向上取整
            w_tiles = max(1, (W + self.tile_size - 1) // self.tile_size)  # 向上取整
            
            # 计算自适应步长，确保均匀分布
            if h_tiles == 1:
                h_stride = 0
                h_positions = [max(0, (H - self.tile_size) // 2)]  # 居中
            else:
                h_stride = (H - self.tile_size) // (h_tiles - 1)
                h_positions = [min(i * h_stride, H - self.tile_size) for i in range(h_tiles)]
            
            if w_tiles == 1:
                w_stride = 0
                w_positions = [max(0, (W - self.tile_size) // 2)]  # 居中
            else:
                w_stride = (W - self.tile_size) // (w_tiles - 1)
                w_positions = [min(j * w_stride, W - self.tile_size) for j in range(w_tiles)]
            
            total_tiles = len(h_positions) * len(w_positions)
            
            # 每个tile分配的采样点数
            n_samples_per_tile = n_imp // total_tiles
            residual = n_imp % total_tiles
            base_scale = np.sqrt((H * W) / (self.N * np.pi)) / 3.0
            
            tile_idx = 0
            for i, top in enumerate(h_positions):
                for j, left in enumerate(w_positions):
                    # 确保tile不越界（在自适应步长下应该不会发生）
                    actual_tile_h = min(self.tile_size, H - top)
                    actual_tile_w = min(self.tile_size, W - left)
                    
                    patch = large_image[:, :, top:top+actual_tile_h, left:left+actual_tile_w]
                    
                    # 计算特征（梯度+颜色方差）
                    grad = self.compute_gradient_magnitude_gpu(patch)
                    color_var = self.compute_color_variance_gpu(patch)
                    
                    # 归一化并融合权重
                    grad_norm = grad / (grad.max() + 1e-8)
                    color_var_norm = color_var / (color_var.max() + 1e-8)
                    weight = self.lambda_g * grad_norm + self.lambda_v * color_var_norm
                    weight = weight.clamp(min=1e-8)
                    
                    # 重要性采样
                    weight_flat = weight.flatten()
                    prob = weight_flat / weight_flat.sum()
                    
                    n_samples_tile = n_samples_per_tile + (1 if tile_idx < residual else 0)
                    
                    if n_samples_tile > 0 and len(prob) > 0:
                        idx = torch.multinomial(prob, min(n_samples_tile, len(prob)), replacement=False)
                        ys = idx // actual_tile_w
                        xs = idx % actual_tile_w
                        
                        # 转换到全局坐标
                        xs_global = xs.float() + left
                        ys_global = ys.float() + top
                        coords_tile = torch.stack([xs_global, ys_global], dim=-1)
                        coords_imp_list.append(coords_tile)
                        
                        # 计算scale
                        weight_at_pts = weight[ys, xs]
                        scale_tile = compute_importance_scales(weight_at_pts,
                                                            coords_tile,
                                                            mode="exp_decay",
                                                            base_scale=base_scale)
                        scales_imp_list.append(scale_tile.unsqueeze(-1))
                    
                    tile_idx += 1
                    # print(f"Processed tile {tile_idx}/{total_tiles}")
                    # print(f"Tile[{i},{j}]: top={top}, left={left}, size=({actual_tile_h},{actual_tile_w}), samples={n_samples_tile}")
                    # print(f"Weight shape: {weight.shape}, prob length: {len(prob)}")
        
        # Concatenate importance sampling results
        if coords_imp_list:
            coords_imp = torch.cat(coords_imp_list, dim=0)
            scales_imp = torch.cat(scales_imp_list, dim=0)
        else:
            coords_imp = torch.empty(0, 2, device=self.device)
            scales_imp = torch.empty(0, 1, device=self.device)

        print(f"Importance sampling completed: {len(coords_imp)} points")
        
        # Uniform sampling with exclusion
        if n_uni > 0:
            min_radius = np.sqrt((H * W) / (self.N * np.pi)) / 3.0
            scales_median = scales_imp.median().item()
            print(f"Min Radius: {min_radius}, Scales_imp.median: {scales_median}")
            exclude_radius = max(min_radius, scales_median)
            
            coords_uni = self.sample_uniform_with_faiss_kdtree_gpu(
                coords_imp, exclude_radius, n_uni, H, W
            )
        else:
            coords_uni = torch.empty(0, 2, device=self.device)
        
        # Combine all coordinates
        all_coords = torch.cat([coords_imp, coords_uni], dim=0)
        
        # Compute scales for uniform points using GPU KNN
        if len(coords_uni) > 0:
            # knn_distances = self.gpu_knn_distances(coords_uni, all_coords, self.knn_K)
            # # scale_uni = torch.clamp(knn_distances.unsqueeze(-1), 2.0, 30.0)
            # scale_uni = knn_distances.unsqueeze(-1)
            # Literal["knn", "voronoi", "density", "gap_filling", "multiscale"] = "knn",
            scale_uni = compute_uniform_scales(coords_uni, all_coords, method="knn").unsqueeze(-1)
        else:
            scale_uni = torch.empty(0, 1, device=self.device)
        
        
        # Final coordinates and scales
        coords_final = all_coords
        scales_final = torch.cat([scales_imp, scale_uni], dim=0)
        
        # Sample colors on GPU
        print("Sampling colors on GPU...")
        # ['center', 'mean', 'gaussian', 'median', 'bilateral', 'maxgrad', 'sobol', 'sobel_dir']
        # gaussian_median
        colors = safe_sample_color_by_scale_circle(large_image,
                                                   coords_final,
                                                   scales_final,
                                                   mode="gaussian_median")
        # colors = self.sample_color_by_scale_gpu(large_image, coords_final, scales_final)
        
        print(f"GPU sampling completed! Total points: {len(coords_final)}")
        
        return {
            'coords_final': coords_final,
            'scales_final': scales_final,
            'colors': colors,
            'coords_imp': coords_imp,
            'scales_imp': scales_imp,
            'coords_uni': coords_uni,
            'scale_uni': scale_uni
        }
    
    def create_step_visualization_data_gpu(self, large_image):
        """Create step-by-step visualization data on GPU"""
        print("Creating GPU visualization steps data...")
        _, _, H, W = large_image.shape
        
        # Use a demo tile for step visualization
        demo_tile_size = min(1024, min(H, W) // 3)
        top = H // 4
        left = W // 4
        base_scale = np.sqrt((H * W) / (self.N * np.pi))

        # Extract demo patch (stays on GPU)
        demo_patch = large_image[:, :, top:top+demo_tile_size, left:left+demo_tile_size]
        
        steps_data = {}
        
        # Step 1: Original image
        steps_data['original'] = {
            'image': demo_patch,
            'title': 'Original Image Patch',
            'coords_global_offset': (left, top)
        }
        
        # Step 2: Gradient magnitude (computed on GPU)
        grad = self.compute_gradient_magnitude_gpu(demo_patch)
        steps_data['gradient'] = {
            'image': grad,
            'title': 'Gradient Magnitude',
            'colormap': 'hot'
        }
        
        # Step 3: Color variance (computed on GPU)  
        color_var = self.compute_color_variance_gpu(demo_patch)
        steps_data['color_variance'] = {
            'image': color_var,
            'title': 'Color Variance',
            'colormap': 'viridis'
        }
        
        # Step 4: Combined weight map
        grad_norm = grad / (grad.max() + 1e-8)
        color_var_norm = color_var / (color_var.max() + 1e-8)
        combined_weight = self.lambda_g * grad_norm + self.lambda_v * color_var_norm
        
        steps_data['combined_weight'] = {
            'image': combined_weight,
            'title': f'Combined Weight (λg={self.lambda_g}, λv={self.lambda_v})',
            'colormap': 'plasma'
        }
        
        # Step 5: Gradient-only sampling
        n_demo_samples = 500
        grad_weight_flat = grad_norm.flatten()
        prob_grad = grad_weight_flat / (grad_weight_flat.sum() + 1e-8)
        idx_grad = torch.multinomial(prob_grad, n_demo_samples, replacement=False)
        
        ys_grad = idx_grad // demo_tile_size
        xs_grad = idx_grad % demo_tile_size
        coords_grad = torch.stack([xs_grad.float(), ys_grad.float()], dim=-1)

        # scales_grad = 1.0 / (grad_norm[ys_grad, xs_grad] + 1e-5)
        # scales_grad = torch.clamp(scales_grad, 2.0, 20.0)
        scales_grad_at_pts = grad_norm[ys_grad, xs_grad]
        scales_grad = compute_importance_scales(scales_grad_at_pts,
                                              coords_grad,
                                              mode="exp_decay",
                                              base_scale=base_scale)
        
        
        steps_data['gradient_sampling'] = {
            'image': demo_patch,
            'coords': coords_grad,
            'scales': scales_grad,
            'title': 'Gradient-only Sampling',
            'color': 'red'
        }
        
        # Step 6: Combined sampling
        combined_weight_flat = combined_weight.flatten()
        prob_combined = combined_weight_flat / (combined_weight_flat.sum() + 1e-8)
        idx_combined = torch.multinomial(prob_combined, n_demo_samples, replacement=False)
        
        ys_comb = idx_combined // demo_tile_size
        xs_comb = idx_combined % demo_tile_size
        coords_comb = torch.stack([xs_comb.float(), ys_comb.float()], dim=-1)
        scales_comb = 1.0 / (combined_weight[ys_comb, xs_comb] + 1e-5)
        # scales_comb = torch.clamp(scales_comb, 2.0, 20.0)
        scales_comb_at_pts = combined_weight[ys_grad, xs_grad]
        scales_comb = compute_importance_scales(scales_comb_at_pts,
                                                coords_comb,
                                                mode="exp_decay",
                                                base_scale=base_scale)
        
        steps_data['combined_sampling'] = {
            'image': demo_patch,
            'coords': coords_comb,
            'scales': scales_comb,  
            'title': 'Gradient + Variance Sampling',
            'color': 'blue'
        }
        
        return steps_data
    
    def save_step_visualizations_gpu(self, large_image, full_results, save_dir="gpu_sampling_steps"):
        """Save individual visualization steps with GPU acceleration"""
        print("Saving GPU step visualizations...")
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Get step-by-step data
        steps_data = self.create_step_visualization_data_gpu(large_image)
        
        # Prepare display image
        _, _, H, W = large_image.shape
        downsample_factor = max(1, max(H, W) // 2000)
        
        if downsample_factor > 1:
            small_image = F.interpolate(large_image,
                                      size=(H//downsample_factor, W//downsample_factor),
                                      mode='bilinear', align_corners=False)
            display_image = small_image[0].permute(1, 2, 0).cpu().numpy()
            coord_scale = 1.0 / downsample_factor
        else:
            display_image = large_image[0].permute(1, 2, 0).cpu().numpy()
            coord_scale = 1.0
        
        # Step 1: Original patch
        plt.figure(figsize=(12, 10))
        original_patch = steps_data['original']['image'][0].permute(1, 2, 0).cpu().numpy()
        plt.imshow(original_patch)
        plt.axis('off')
        plt.title('Step 1: Original Image Patch', fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/step1_original_patch.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Step 2: Gradient magnitude  
        plt.figure(figsize=(12, 10))
        grad_data = steps_data['gradient']['image'].cpu().numpy()
        plt.imshow(grad_data, cmap='hot')
        plt.colorbar(label='Gradient Magnitude', shrink=0.8)
        plt.axis('off')
        plt.title('Step 2: Gradient Magnitude', fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/step2_gradient_magnitude.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Step 3: Color variance
        plt.figure(figsize=(12, 10))
        var_data = steps_data['color_variance']['image'].cpu().numpy()
        plt.imshow(var_data, cmap='viridis')
        plt.colorbar(label='Color Variance', shrink=0.8)
        plt.axis('off')
        plt.title('Step 3: Color Variance', fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/step3_color_variance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Step 4: Combined weight map
        plt.figure(figsize=(12, 10))
        weight_data = steps_data['combined_weight']['image'].cpu().numpy()
        plt.imshow(weight_data, cmap='plasma')
        plt.colorbar(label='Combined Weight', shrink=0.8)
        plt.axis('off')
        plt.title(f'Step 4: Combined Weight (λg={self.lambda_g}, λv={self.lambda_v})', fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/step4_combined_weight.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Step 5: Gradient-only sampling
        plt.figure(figsize=(12, 10))
        plt.imshow(original_patch)
        
        coords_grad = steps_data['gradient_sampling']['coords'].cpu().numpy()
        scales_grad = steps_data['gradient_sampling']['scales'].cpu().numpy()
        
        for coord, scale in zip(coords_grad, scales_grad):
            circle = Circle((coord[0], coord[1]), scale, 
                          fill=False, color='red', linewidth=1.5, alpha=0.8)
            plt.gca().add_patch(circle)
        
        plt.axis('off')
        plt.title('Step 5: Gradient-only Importance Sampling', fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/step5_gradient_sampling.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Step 6: Combined sampling  
        plt.figure(figsize=(12, 10))
        plt.imshow(original_patch)
        
        coords_comb = steps_data['combined_sampling']['coords'].cpu().numpy()
        scales_comb = steps_data['combined_sampling']['scales'].cpu().numpy()
        
        for coord, scale in zip(coords_comb, scales_comb):
            circle = Circle((coord[0], coord[1]), scale,
                          fill=False, color='blue', linewidth=1.5, alpha=0.8)
            plt.gca().add_patch(circle)
        
        plt.axis('off')
        plt.title('Step 6: Gradient + Variance Importance Sampling', fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/step6_combined_sampling.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Step 7: Full image importance sampling
        plt.figure(figsize=(16, 12))
        plt.imshow(display_image)
        
        coords_imp = full_results['coords_imp'].cpu().numpy() * coord_scale
        scales_imp = full_results['scales_imp'].cpu().numpy() * coord_scale
        
        # Show subset to avoid overcrowding
        # n_show_imp = min(800, len(coords_imp))
        n_show_imp = len(coords_imp)
        if n_show_imp > 0:
            indices = np.random.choice(len(coords_imp), n_show_imp, replace=False)
            
            for i in indices:
                coord, scale = coords_imp[i], scales_imp[i, 0]
                circle = Circle((coord[0], coord[1]), scale,
                              fill=False, color='red', linewidth=0.8, alpha=0.6)
                plt.gca().add_patch(circle)
        
        plt.axis('off')
        plt.title(f'Step 7: Full Image Importance Sampling ({len(coords_imp)} points)', 
                 fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/step7_full_importance_sampling.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Step 8: Uniform sampling with exclusion
        plt.figure(figsize=(16, 12))
        plt.imshow(display_image)
        
        coords_uni = full_results['coords_uni'].cpu().numpy() * coord_scale
        scale_uni = full_results['scale_uni'].cpu().numpy() * coord_scale
        
        # n_show_uni = min(500, len(coords_uni))
        n_show_uni = len(coords_uni)
        if n_show_uni > 0:
            indices = np.random.choice(len(coords_uni), n_show_uni, replace=False)
            
            for i in indices:
                coord, scale = coords_uni[i], scale_uni[i, 0]  
                circle = Circle((coord[0], coord[1]), scale,
                              fill=False, color='green', linewidth=0.8, alpha=0.7)
                plt.gca().add_patch(circle)
        
        plt.axis('off')
        plt.title(f'Step 8: Uniform Sampling with Exclusion ({len(coords_uni)} points)',
                 fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/step8_uniform_sampling.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Step 9: Combined sampling (importance + uniform)
        plt.figure(figsize=(16, 12))
        plt.imshow(display_image)
        
        # Show importance points (red)
        n_show_imp = min(400, len(coords_imp))
        if n_show_imp > 0:
            indices_imp = np.random.choice(len(coords_imp), n_show_imp, replace=False)
            for i in indices_imp:
                coord, scale = coords_imp[i], scales_imp[i, 0]
                circle = Circle((coord[0], coord[1]), scale,
                              fill=False, color='red', linewidth=0.7, alpha=0.6)
                plt.gca().add_patch(circle)
        
        # Show uniform points (green)  
        n_show_uni = min(300, len(coords_uni))
        if n_show_uni > 0:
            indices_uni = np.random.choice(len(coords_uni), n_show_uni, replace=False)
            for i in indices_uni:
                coord, scale = coords_uni[i], scale_uni[i, 0]
                circle = Circle((coord[0], coord[1]), scale,
                              fill=False, color='green', linewidth=0.7, alpha=0.6)
                plt.gca().add_patch(circle)
        
        plt.axis('off')
        plt.title(f'Step 9: Combined Sampling (Red: Importance, Green: Uniform)', 
                 fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/step9_combined_sampling.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Step 10: Final result with colors
        plt.figure(figsize=(16, 12))
        plt.imshow(display_image)
        
        coords_all = full_results['coords_final'].cpu().numpy() * coord_scale
        colors_all = full_results['colors'].cpu().numpy()
        scales_all = full_results['scales_final'].cpu().numpy() * coord_scale
        
        n_show = min(1000, len(coords_all))
        indices = np.random.choice(len(coords_all), n_show, replace=False)
        
        for i in indices:
            coord = coords_all[i]
            color = colors_all[i]
            scale = scales_all[i, 0]
            
            # Ensure colors are in [0,1] range
            color_normalized = np.clip(color, 0, 1)
            
            circle = Circle((coord[0], coord[1]), scale,
                          fill=True, facecolor=color_normalized,
                          edgecolor='white', linewidth=0.3, alpha=0.9)
            plt.gca().add_patch(circle)
        
        plt.axis('off')
        plt.title(f'Step 10: Final Result with Sampled Colors ({len(coords_all)} points)',
                 fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/step10_final_with_colors.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"All GPU visualization steps saved to {save_dir}/")
    
    def visualize_complete_process_gpu(self, large_image, save_dir="gpu_sampling_steps"):
        """Complete GPU-accelerated visualization process"""
        print("Starting complete GPU sampling and visualization process...")
        
        # Execute full sampling with timing
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        full_results = self.sample_variational_importance_gpu(large_image)
        end_time.record()
        
        torch.cuda.synchronize()
        sampling_time = start_time.elapsed_time(end_time) / 1000.0
        
        # Save step-by-step visualizations
        # TODO 
        # self.save_step_visualizations_gpu(large_image, full_results, save_dir)
        
        print(f"Complete process finished in {sampling_time:.2f} seconds")
        return full_results, sampling_time

# Main usage function
def main():
    print("Initializing GPU Variational Sampling...")
    
    # Configuration
    # Load and process image
    # path = "../inr_dataset/tokyo/02.jpg"  # Update path as needed
    path = "../inr_dataset/DIV8K/0001.png"
    print(f"Loading image from {path}")
    gt_image = image_path_to_tensor(path)
    print(f"Image loaded successfully: {gt_image.shape}")
    H, W = gt_image.shape[-2:]  # Large image dimensions
    compression_ratio = 500
    num_points = int((H * W * 3) / (7 * compression_ratio))
    
    # Initialize GPU-accelerated visualizer
    visualizer = GPUVariationalSamplingVisualizer(
        N=num_points,
        tile_size=1024,
        tiles_per_batch=20,
        ratio=0.7,
        lambda_g=0.9,
        lambda_v=0.1,
        knn_K=3,
        device='cuda'
    )
    
    try:
        # Execute complete GPU-accelerated process with step visualization
        results, sampling_time = visualizer.visualize_complete_process_gpu(
            gt_image, 
            save_dir="logs/gpu_tokyo_sampling_steps"
        )
        
        print(f"\nComplete process finished!")
        print(f"Sampling time: {sampling_time:.2f} seconds")
        print(f"Total sampling points: {len(results['coords_final'])}")
        print(f"Importance sampling points: {len(results['coords_imp'])}")
        print(f"Uniform sampling points: {len(results['coords_uni'])}")
        
        # Save comprehensive results
        torch.save({
            'coords': results['coords_final'].cpu(),
            'scales': results['scales_final'].cpu(),
            'colors': results['colors'].cpu(),
            'coords_imp': results['coords_imp'].cpu(),
            'scales_imp': results['scales_imp'].cpu(),
            'coords_uni': results['coords_uni'].cpu(),
            'scale_uni': results['scale_uni'].cpu(),
            'image_shape': gt_image.shape,
            'sampling_time': sampling_time,
            'config': {
                'N': visualizer.N,
                'ratio': visualizer.ratio,
                'lambda_g': visualizer.lambda_g,
                'lambda_v': visualizer.lambda_v,
                'tile_size': visualizer.tile_size,
                'tiles_per_batch': visualizer.tiles_per_batch
            }
        }, 'gpu_sampling_complete_results.pt')
        
        print("Complete results saved to gpu_sampling_complete_results.pt")
        print("Step-by-step visualizations saved to logs/gpu_tokyo_sampling_steps/")
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Utility function for quick step visualization only
def visualize_steps_only(image_path, save_dir="quick_steps"):
    """Quick function to generate only step visualizations"""
    visualizer = GPUVariationalSamplingVisualizer(
        N=5000,  # Smaller for quick demo
        tile_size=512,
        tiles_per_batch=8,
        ratio=0.7,
        lambda_g=0.6,
        lambda_v=0.4,
        device='cuda'
    )
    
    gt_image = image_path_to_tensor(image_path)
    print(f"Quick step visualization for: {gt_image.shape}")
    
    # Generate step data and save
    results = visualizer.sample_variational_importance_gpu(gt_image)
    visualizer.save_step_visualizations_gpu(gt_image, results, save_dir)
    
    return results

if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available! This implementation requires GPU.")
        exit(1)
    
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    results = main()