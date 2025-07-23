from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum
from gaussian_models.utils import *
import torch
import torch.nn as nn
import math
from optimizer import Adan
import time
from gaussian_models.gaussian_property_sampler import GPUVariationalSamplingVisualizer

class SmartSplat(nn.Module):
    def __init__(self, loss_type="L1-SSIM", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = kwargs["device"]
        
        init_image = kwargs.get("init_image", None)

        if init_image is not None:
            with torch.no_grad():
                sampler = GPUVariationalSamplingVisualizer(
                            N=self.init_num_points,
                            tile_size=1024,
                            tiles_per_batch=20,
                            ratio=0.7,
                            lambda_g=0.9,
                            lambda_v=0.1,
                            knn_K=3,
                            device='cuda'
                        )
                
                results = sampler.sample_variational_importance_gpu(init_image)
                means_init = results["coords_final"]
                scale_init = results["scales_final"].squeeze(-1)
                colors_init = results["colors"]
                means_init = means_init.to(self.device)
                scale_init = scale_init.repeat(1,2).to(self.device)
                colors_init = colors_init.to(self.device)

                del init_image
                
        print(f"[Init] Points: {len(means_init)}, Means ∈ ({means_init[:,0].min():.1f}, {means_init[:,0].max():.1f}) × ({means_init[:,1].min():.1f}, {means_init[:,1].max():.1f})")
        print(f"[Init] Scales: {len(scale_init)}, Scale ∈ ({scale_init[:,0].min():.1f}, {scale_init[:,0].max():.1f}) × ({scale_init[:,1].min():.1f}, {scale_init[:,1].max():.1f})")
        print(f"[Init] Colors: {len(colors_init)}, Color ∈ ({colors_init[:,0].min():.1f}, {colors_init[:,0].max():.1f}) × ({colors_init[:,1].min():.1f}, {colors_init[:,1].max():.1f})")

        normalized_xy = torch.stack([
            (means_init[:, 0]) / (self.W),
            (means_init[:, 1]) / (self.H)
        ], dim=-1)  # ∈ [0,1]

        xyz_init = normalized_xy * 2. - 1.  # ∈ (-1, 1)
        self._xyz = nn.Parameter(xyz_init)

        self._scaling = nn.Parameter(scale_init)
        self._features_dc = nn.Parameter(colors_init)
        
        sample_points = len(xyz_init)
        if self.init_num_points != sample_points:
            print("Expect Points: ", self.init_num_points)
            self.init_num_points = sample_points
            print("Sampling Points: ", self.init_num_points)

        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self._rotation = nn.Parameter(torch.zeros(self.init_num_points, 1))

        self.last_size = (self.H, self.W)
        self.background = torch.ones(3, device=self.device)
        self.rotation_activation = torch.sigmoid

        self.lr_xyz = 1e-4
        self.lr_color = 5e-3
        self.lr_scale = 5e-2
        self.lr_rot = 1e-3

        param_groups = [
            {'params': [self._xyz], 'lr': self.lr_xyz, "name": "xyz"},
            {'params': [self._features_dc], 'lr': self.lr_color, "name": "f_dc"},
            {'params': [self._scaling], 'lr': self.lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': self.lr_rot, "name": "rotation"},
        ]

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        else:
            self.optimizer = Adan(param_groups)

    @property
    def get_scaling(self):
        return torch.abs(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)*2*math.pi
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self._opacity
    
    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(self.get_xyz, self.get_scaling, self.get_rotation, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]

        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.8)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        return loss, psnr

