import os
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
from fused_ssim import fused_ssim
import torch
import numpy as np

def compress_matrix_flatten_categorical(indices):
    """
    Compress a list of categorical indices using histogram-based encoding.
    
    Args:
        indices (list): List of integer indices (e.g., from a quantizer).
    
    Returns:
        compressed (np.ndarray): Compressed representation (e.g., run-length encoded).
        histogram_table (np.ndarray): Histogram of index frequencies.
        unique (np.ndarray): Unique indices in the input.
    """
    indices = np.array(indices, dtype=np.int32)
    unique, counts = np.unique(indices, return_counts=True)
    histogram_table = counts.astype(np.float64)
    
    # Simple run-length encoding (RLE) as a placeholder
    compressed = []
    current_idx = indices[0]
    count = 1
    for idx in indices[1:]:
        if idx == current_idx:
            count += 1
        else:
            compressed.append((current_idx, count))
            current_idx = idx
            count = 1
    compressed.append((current_idx, count))
    compressed = np.array(compressed, dtype=[('index', np.int32), ('count', np.int32)])
    
    return compressed, histogram_table, unique

def decompress_matrix_flatten_categorical(compressed, histogram_table, unique):
    """
    Decompress a list of categorical indices from compressed data.
    
    Args:
        compressed (np.ndarray): Compressed representation (e.g., RLE).
        histogram_table (np.ndarray): Histogram of index frequencies (unused in RLE).
        unique (np.ndarray): Unique indices (unused in RLE).
    
    Returns:
        indices (list): Reconstructed list of indices.
    """
    indices = []
    for idx, count in compressed:
        indices.extend([idx] * count)
    return indices

def get_np_size(arr):
    """
    Estimate the size of a numpy array in bytes.
    
    Args:
        arr (np.ndarray or list): Input array or list.
    
    Returns:
        int: Size in bytes.
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    return arr.nbytes

class LogWriter:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        self.training_file_path = os.path.join(file_path, "train.txt")
        self.testing_file_path = os.path.join(file_path, "test.txt")

    def write(self, text, train=False):
        # 打印到控制台
        if train:
            with open(self.training_file_path, 'a') as file:
                file.write(text + '\n')
        else:
            print(text)
            with open(self.testing_file_path, 'a') as file:
                file.write(text + '\n')
        


def loss_fn(pred: torch.Tensor, target: torch.Tensor, loss_type='L2', lambda_value=0.8):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target)
    if loss_type == 'RelL2':
        relative_l2_error = (pred - target)**2 / (pred**2 + 0.01)
        loss = relative_l2_error.mean()
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'L2-SSIM':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - fused_ssim(pred, target))
    elif loss_type == 'L1-SSIM':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - fused_ssim(pred, target, train=True))
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - fused_ssim(pred, target))
    elif loss_type == 'Fusion2':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - fused_ssim(pred, target))
    elif loss_type == 'Fusion3':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * F.l1_loss(pred, target)
    elif loss_type == 'Fusion4':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - fused_ssim(pred, target))
    elif loss_type == 'Fusion_hinerv':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value)  * (1 - fused_ssim(pred, target))
    return loss

def strip_lowerdiag(L):
    if L.shape[1] == 3:
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]

    elif L.shape[1] == 2:
        uncertainty = torch.zeros((L.shape[0], 3), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 1, 1]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_rotation_2d(r):
    '''
    Build rotation matrix in 2D.
    '''
    R = torch.zeros((r.size(0), 2, 2), device='cuda')
    R[:, 0, 0] = torch.cos(r)[:, 0]
    R[:, 0, 1] = -torch.sin(r)[:, 0]
    R[:, 1, 0] = torch.sin(r)[:, 0]
    R[:, 1, 1] = torch.cos(r)[:, 0]
    return R

def build_scaling_rotation_2d(s, r, device):
    L = torch.zeros((s.shape[0], 2, 2), dtype=torch.float, device='cuda')
    R = build_rotation_2d(r, device)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L = R @ L
    return L
    
def build_covariance_from_scaling_rotation_2d(scaling, scaling_modifier, rotation, device):
    '''
    Build covariance metrix from rotation and scale matricies.
    '''
    L = build_scaling_rotation_2d(scaling_modifier * scaling, rotation, device)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def build_triangular(r):
    R = torch.zeros((r.size(0), 2, 2), device=r.device)
    R[:, 0, 0] = r[:, 0]
    R[:, 1, 0] = r[:, 1]
    R[:, 1, 1] = r[:, 2]
    return R