import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
# from tqdm.notebook import tqdm
from functools import partial
import math, os, copy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from prettytable import PrettyTable
import scipy.io as sio
import imgvision as iv
import csv
import datetime
IN_CH_HSI = 48
IN_CH_MSI = 4
from dataset import *
from Our_network import *
from metrix import *

from show_img import show_img

# 消融1 添加PAA模块

def get_free_gpu():
    """
    自动检测并返回最空闲的GPU设备
    Returns:
        torch.device: 最空闲的GPU设备，如果没有GPU则返回CPU
    """
    if not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        return torch.device('cpu')
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("未检测到GPU，使用CPU")
        return torch.device('cpu')
    
    print(f"检测到 {num_gpus} 个GPU")
    
    # 检查每个GPU的内存使用情况
    gpu_memory_usage = []
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        cached_memory = torch.cuda.memory_reserved(i)
        free_memory = total_memory - max(allocated_memory, cached_memory)
        usage_ratio = (total_memory - free_memory) / total_memory
        
        gpu_memory_usage.append({
            'gpu_id': i,
            'total_memory': total_memory / 1024**3,  # GB
            'free_memory': free_memory / 1024**3,    # GB
            'usage_ratio': usage_ratio,
            'gpu_name': torch.cuda.get_device_properties(i).name
        })
        
        print(f"GPU {i} ({torch.cuda.get_device_properties(i).name}): "
              f"总内存 {total_memory/1024**3:.1f}GB, "
              f"空闲内存 {free_memory/1024**3:.1f}GB, "
              f"使用率 {usage_ratio*100:.1f}%")
    
    # 选择使用率最低的GPU
    best_gpu = min(gpu_memory_usage, key=lambda x: x['usage_ratio'])
    selected_gpu_id = best_gpu['gpu_id']
    
    print(f"选择GPU {selected_gpu_id} ({best_gpu['gpu_name']}) - "
          f"空闲内存: {best_gpu['free_memory']:.1f}GB")
    
    return torch.device(f'cuda:{selected_gpu_id}')


class CSVLogger:
    """CSV日志记录器，用于保存训练过程中的指标数据"""
    
    def __init__(self, filename="SAM-PSRF-train_PaviaC_FFT.csv", log_dir="./logs"):
        """
        初始化CSV日志记录器
        Args:
            filename: CSV文件名
            log_dir: 日志保存目录
        """
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        self.filepath = os.path.join(log_dir, filename)
        self.fieldnames = [
            'epoch', 'timestamp', 
            'PSNR', 'SAM', 'SSIM', 'MSE', 'ERGAS',
            'gpu_memory_used', 'is_best_psnr', 'is_best_sam'
        ]
        
        # 创建CSV文件并写入表头
        self._init_csv()
        print(f"CSV日志文件已创建: {self.filepath}")
    
    def _init_csv(self):
        """初始化CSV文件，写入表头"""
        with open(self.filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def log_metrics(self, epoch, metrics, is_best_psnr=False, is_best_sam=False):
        """
        记录评估指标到CSV文件（仅在模型评估时调用）
        Args:
            epoch: 当前训练轮次
            metrics: 评价指标字典 {'PSNR': value, 'SAM': value, ...}
            is_best_psnr: 是否为最佳PSNR
            is_best_sam: 是否为最佳SAM
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 获取GPU内存使用情况
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        # 准备记录数据
        log_data = {
            'epoch': epoch,
            'timestamp': timestamp,
            'gpu_memory_used': f"{gpu_memory:.2f}GB",
            'is_best_psnr': is_best_psnr,
            'is_best_sam': is_best_sam
        }
        
        # 添加评价指标
        # for key in ['PSNR', 'SAM', 'SSIM', 'MSE', 'ERGAS']:
        for key in ['PSNR', 'SAM', 'ERGAS']:

            log_data[key] = metrics.get(key, "N/A")
        
        # 写入CSV文件
        with open(self.filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(log_data)
        
        print(f"评估指标已保存到CSV文件 - Epoch {epoch}")


"""
    Define U-net Architecture:
    Approximate reverse diffusion process by using U-net
    U-net of SR3 : U-net backbone + Positional Encoding of time + Multihead Self-Attention
"""

# 三维空间噪声
def generate_spatial_noise(gtHS):
    """
    标准空间高斯噪声（全图）
    """
    return torch.randn_like(gtHS)  # shape: [B, C, H, W]

# ================== 生成空间噪声mask ==================
def generate_spatial_mask_time_scaled(gtHS, sqrt_1m_alpha_t, base_ratio=0.001, max_ratio=0.5):
    """
    生成空间噪声 mask，噪点数量随时间因子变化
    参数:
        gtHS: [B, C, H, W]
        sqrt_1m_alpha_t: 时间因子，对应扩散噪声强度
        base_ratio: 最小噪点比例
        max_ratio: 最大噪点比例
    返回:
        mask: [B, 1, H, W]
    """
    B, C, H, W = gtHS.shape
    device = gtHS.device

    # 根据时间因子确定噪点比例
    # 时间因子越大，噪点比例越大
    noise_ratio = base_ratio + (max_ratio - base_ratio) * sqrt_1m_alpha_t
    noise_ratio = noise_ratio.clamp(max=1.0)  # 防止超过 1

    # 随机生成 mask
    rand_vals = torch.rand(B, 1, H, W, device=device)
    mask = (rand_vals < noise_ratio).float()
    return mask

def psnr(output, groundtruth):
    if isinstance(output, torch.Tensor) and isinstance(groundtruth, torch.Tensor):
        mse = F.mse_loss(output, groundtruth)
        psnr_value = 10 * np.log10(1 / mse.item())

    else:
        output = np.asarray(output)
        groundtruth = np.asarray(groundtruth)
        mse = np.mean((output - groundtruth) ** 2)
        psnr_value = 10 * np.log10(1 / np.sqrt(mse))
    return psnr_value


def generate_time_scaled_spectral_noise_logspace(reflectance, t, sqrt_1m_alpha_t, 
                                        spectral_kernel_size=5, sigma=1.0, eps=1e-6):
    """
    在 -log(r) 空间中生成光谱噪声，但最终返回的是噪声项（可与 spatial_noise 相加）。
    
    参数:
        reflectance: 输入反射率张量 [B, C, H, W], 取值范围 (0,1)
        t: 扩散时间步
        sqrt_1m_alpha_t: 时间缩放因子，形状 [B, 1, 1, 1] 或标量
        spectral_kernel_size: 谱维高斯平滑卷积核大小
        sigma: 高斯核标准差
        eps: 防止 log(0) 的数值稳定项
    
    返回:
        spectral_noise: 光谱噪声项 [B, C, H, W]，可直接与 spatial_noise 相加
    """
    B, C, H, W = reflectance.shape

    # 1. 转换到 -log(r) 空间
    neg_log_r = -torch.log(reflectance.clamp(min=eps))

    # 2. 基于反射率强度生成 mask（空间扰动区域）
    spatial_intensity = reflectance.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
    threshold = 0.8
    mask = (spatial_intensity > threshold).float()

    # 3. 生成原始随机谱噪声
    raw_spec_noise = torch.randn(B, C, H, W, device=reflectance.device)

    # 4. 沿光谱维做 1D 高斯平滑
    reshaped = raw_spec_noise.permute(0, 2, 3, 1).reshape(-1, 1, C)  # [B*H*W, 1, C]
    coords = torch.arange(spectral_kernel_size, device=reflectance.device) - spectral_kernel_size // 2
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, spectral_kernel_size)
    smooth_spec = F.conv1d(reshaped, kernel, padding=spectral_kernel_size // 2)
    smooth_spec = smooth_spec.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B,C,H,W]

    # 5. 在 -log(r) 空间中的扰动：delta(-log r) = smooth_spec * mask * sqrt(1-alpha_t)
    delta_neglog = smooth_spec * mask * (sqrt_1m_alpha_t / 10.0)

    # 6. 将 delta(-log r) 映射回反射率空间的等效噪声
    #   r' = exp(-( -log(r) + delta)) = r * exp(-delta)
    #   noise = r' - r = r * (exp(-delta) - 1)
    spectral_noise = reflectance * (torch.exp(-delta_neglog) - 1.0)

    return spectral_noise


def generate_time_scaled_spectral_noise(spatial_noise, t, sqrt_1m_alpha_t, spectral_kernel_size=5, sigma=1.0): 
    """ 生成“一维平滑的谱维噪声”，仅在空间扰动掩膜(mask)处保留，并乘以 sqrt(1 - alpha_t)。s' = s + η_log """ 
    B, C, H, W = spatial_noise.shape # 获取扰动强度（按像素取最大值） 
    spatial_intensity = spatial_noise.abs().mean(dim=1, keepdim=True) # shape: [B, 1, H, W] 
    threshold = 0.8 # 可调节：扰动越大位置越可靠 
    mask = (spatial_intensity > threshold).float() # [B, 1, H, W] # 对每个位置生成独立谱向曲线 
    raw_spec_noise = torch.randn(B, C, H, W, device=spatial_noise.device) # 做谱维平滑（1D Gaussian） 
    reshaped = raw_spec_noise.permute(0, 2, 3, 1).reshape(-1, 1, C) 
    coords = torch.arange(spectral_kernel_size, device=spatial_noise.device) - spectral_kernel_size // 2 
    kernel = torch.exp(-0.5 * (coords / sigma)**2) 
    kernel /= kernel.sum() 
    kernel = kernel.view(1, 1, spectral_kernel_size) 
    smooth_spec = F.conv1d(reshaped, kernel, padding=spectral_kernel_size // 2) 
    smooth_spec = smooth_spec.reshape(B, H, W, C).permute(0, 3, 1, 2) # back to [B, C, H, W] # 仅保留空间扰动位置的谱噪声，并乘以时间因子 
    spec_noise = smooth_spec * mask * sqrt_1m_alpha_t # broadcast: [B, 1, 1, 1] 
    return spec_noise

def generate_spectral_noise(gtHS, spatial_mask, spectral_kernel_size=5, sigma=1.0, scale=0.1):
    """
    生成谱噪声，并在谱维度上进行平滑
    """
    B, C, H, W = gtHS.shape

    # 使用空间噪声强度作为 mask
    mask = spatial_mask

    # 原始谱噪声
    raw_spec_noise = torch.randn(B, C, H, W, device=gtHS.device)

    # ===== 谱维度平滑 =====
    # 先 reshape 为 [B*H*W, 1, C] 方便 conv1d
    reshaped = raw_spec_noise.permute(0, 2, 3, 1).reshape(-1, 1, C)

    # 高斯卷积核
    coords = torch.arange(spectral_kernel_size, device=gtHS.device) - spectral_kernel_size // 2
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, spectral_kernel_size)

    # 1D 卷积平滑
    smooth_spec = F.conv1d(reshaped, kernel, padding=spectral_kernel_size//2)
    
    # reshape 回原始形状 [B, C, H, W]
    smooth_spec = smooth_spec.reshape(B, H, W, C).permute(0, 3, 1, 2)

    # 标准化卷积后的谱噪声
    smooth_spec = (smooth_spec - smooth_spec.mean()) / (smooth_spec.std() + 1e-8)

    # 再施加 mask + scale
    spec_noise = smooth_spec * mask * scale

    return spec_noise

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def calculate_sam(target_data, reference_data):
    # 归一化目标数据和参考数据
    b, c, h, w = target_data.shape
    target_data = target_data.reshape(b, c, h*w).permute(0,2,1)
    reference_data = reference_data.reshape(b, c, h * w).permute(0, 2, 1)
    target_data_norm = torch.nn.functional.normalize(target_data, dim=2)
    reference_data_norm = torch.nn.functional.normalize(reference_data, dim=2)

    # 计算点积
    dot_product = torch.einsum('bnc,bnc->bn', target_data_norm, reference_data_norm)

    # 计算长度乘积
    length_product = torch.norm(target_data_norm, dim=2) * torch.norm(reference_data_norm, dim=2)

    # 计算SAM光谱角
    sam = torch.acos(dot_product / length_product)
    sam_mean = torch.mean(torch.mean(sam, dim=1))
    return sam_mean


def extract(a, t, x_shape):
    """
    从给定的张量a中检索特定的元素。t是一个包含要检索的索引的张量，
    这些索引对应于a张量中的元素。这个函数的输出是一个张量，
    包含了t张量中每个索引对应的a张量中的元素
    :param a:
    :param t:
    :param x_shape:
    :return:
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        # Input : tensor of value of coefficient alpha at specific step of diffusion process e.g. torch.Tensor([0.03])
        # Transform level of noise into representation of given desired dimension
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels * (1 + self.use_affine_level)))

    def forward(self, x, noise_embed):
        noise = self.noise_func(noise_embed).view(x.shape[0], -1, 1, 1)
        if self.use_affine_level:
            gamma, beta = noise.chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + noise
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


# Linear Multi-head Self-attention
class SelfAtt(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32, att_num=0):
        super(SelfAtt, self).__init__()
        self.groupnorm = nn.GroupNorm(norm_groups, channel_dim)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)
        self.att = att_num

    def forward(self, x):
        x_org = x
        b, c, h, w = x.size()
        x = self.groupnorm(x)
        qkv = rearrange(self.qkv(x), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        keys = F.softmax(keys, dim=-1)
        att = torch.einsum('bhdn,bhen->bhde', keys, values)
        out = torch.einsum('bhde,bhdn->bhen', att, queries)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

        return x_org+self.att*self.proj(out)


class Cross_Att(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32, att_num=0):
        super(Cross_Att, self).__init__()
        self.att = att_num
        self.groupnorm_1 = nn.GroupNorm(norm_groups, channel_dim)
        self.groupnorm_2 = nn.GroupNorm(norm_groups, channel_dim)
        self.num_heads = num_heads
        self.qkv_1 = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.qkv_2 = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)

        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)

        self.downsample = nn.Sequential(nn.Conv2d(channel_dim, 2 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(2 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(2 * channel_dim, 4 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(4 * channel_dim, 4 * channel_dim, 3, 2, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),
                                        )

        self.upsample = nn.Sequential(nn.Conv2d(2 * channel_dim, 1 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(2 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(4 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(4 * channel_dim, 4 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),
                                      )

    def forward(self, x, y, mode):
        b, c, h, w = x.size()
        x_org = x
        if mode == 'spe':
            b, c, h, w = x.size()
            x = self.groupnorm_1(x)
            y = self.groupnorm_1(y)
            qkv_1 = rearrange(self.qkv_1(x), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads, qkv=3)
            queries_1, keys_1, values_1 = qkv_1[0], qkv_1[1], qkv_1[2]
            qkv_2 = rearrange(self.qkv_2(y), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads, qkv=3)
            queries_2, keys_2, values_2 = qkv_2[0], qkv_2[1], qkv_2[2]
            keys_1 = F.softmax(keys_1, dim=-1)
            keys_2 = F.softmax(keys_2, dim=-1)
            att = torch.einsum('bhdn,bhen->bhde', keys_1, values_2)
            out = torch.einsum('bhde,bhdn->bhen', att, queries_1)
            out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)
        else:
            x = self.groupnorm_2(x)
            y = self.groupnorm_2(y)
            if h == 512:
                times = h/64
            else:
                times = h/20
            n = np.log(times)/np.log(2)
            for i in range(int(n)):
                x = self.downsample[2 * i](x)
                x = self.downsample[2 * i+1](x)

            for i in range(int(n)):
                y = self.downsample[2 * i](y)
                y = self.downsample[2 * i+1](y)

            b, c, h, w = x.size()

            x = x.reshape(b, c, h * w).repeat(1, 1, 3)
            y = y.reshape(b, c, h * w).repeat(1, 1, 3)
            qkv_1 = rearrange(x, "b c (qkv heads h) -> (qkv) b heads h c", heads=self.num_heads, qkv=3)
            queries_1, keys_1, values_1 = qkv_1[0], qkv_1[1], qkv_1[2]
            qkv_2 = rearrange(y, "b c (qkv heads h) -> (qkv) b heads h c", heads=self.num_heads, qkv=3)
            queries_2, keys_2, values_2 = qkv_2[0], qkv_2[1], qkv_2[2]

            keys_1 = F.softmax(keys_1, dim=-1)
            keys_2 = F.softmax(keys_2, dim=-1)
            att = torch.einsum('bhdn,bhen->bhde', keys_1, values_2)
            out = torch.einsum('bhde,bhdn->bhen', att, queries_1)
            out = rearrange(out, 'b heads (h w) c -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

            for i in range(int(n)):
                l = int(n)-1-i
                out = self.upsample[2 * l](out)
                out = self.upsample[2 * l+1](out)

        return x_org+self.att*self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=False):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        y = self.block1(x)
        y = self.noise_func(y, time_emb)
        y = self.block2(y)
        x = y + self.res_conv(x)
        return x


class ResBlock_skip(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=True):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):

        y = self.block1(x)

        return y+self.res_conv(x)


class SGPD(nn.Module):
    def __init__(self, in_channel=37, out_channel=34, skip_input=102, inner_channel=64, norm_groups=32,
                 channel_mults=[1, 2, 4, 8, 8], res_blocks=3, dropout=0, img_size=128):
        super().__init__()

        self_att = []
        cros_att = []
        dim_out = [inner_channel, inner_channel * 2, inner_channel * 2]
        for i in reversed(range(len(dim_out))):
            self_att.append(SelfAtt(dim_out[i], num_heads=1, norm_groups=norm_groups))

        self.self_att = nn.ModuleList(self_att)

        for j in reversed(range(len(dim_out))):
            cros_att.append(Cross_Att(dim_out[j], num_heads=1, norm_groups=norm_groups))

        self.cros_att = nn.ModuleList(cros_att)

        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        pre_channel_skip = inner_channel
        feat_channels = [pre_channel]
        feat_channels_skips = [pre_channel]

        now_res = img_size

        # Downsampling stage of SGPD
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResBlock(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout),
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout, att=False)
        ])

        # Skip stage of SGPD
        skip_downs = [nn.Conv2d(skip_input, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                skip_downs.append(ResBlock_skip(
                    pre_channel_skip, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, att=False))
                pre_channel_skip = channel_mult
            if not is_last:
                feat_channels_skips.append(channel_mult)
                skip_downs.append(Downsample(pre_channel_skip))
                now_res = now_res // 2
        self.skip_downs = nn.ModuleList(skip_downs)

        # Upsampling stage of SGPD
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]

            for i in range(0, res_blocks + 1):
                ups.append(ResBlock(
                    pre_channel + feat_channels.pop()*2, channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult

            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

    def forward(self, x, skip_input, noise_level, mode = None):
        # Embedding of time step with noise coefficient alpha
        t = self.noise_level_mlp(noise_level)

        feats_skip = []
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        k = 0
        for i, layer in enumerate(self.skip_downs):

            # skip_input =
            skip_input = layer(skip_input)
            feats_skip.append(skip_input)

        for layer in self.mid:
            x = layer(x, t)
        z = 0
        for i, layer in enumerate(self.ups):
            if isinstance(layer, ResBlock):
                if i == 0:
                    x = layer(torch.cat([x, feats.pop(), feats_skip.pop()], dim=1), t)
                elif isinstance(self.ups[i - 1], Upsample):
                    temp_feats_skip = feats_skip.pop()
                    temp_feats = feats.pop()
                    x = layer(torch.cat([x, self.self_att[z](temp_feats_skip), self.cros_att[z](temp_feats, temp_feats_skip, mode=mode)], dim=1), t)
                    z = z + 1
                else:
                    x = layer(torch.cat([x, feats.pop(), feats_skip.pop()], dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


"""
    Define Diffusion process framework to train desired model:
    Forward Diffusion process:
        Given original image x_0, apply Gaussian noise ε_t for each time step t
        After proper length of time step, image x_T reachs to pure Gaussian noise
    Objective of model f :
        model f is trained to predict actual added noise ε_t for each time step t
"""


class ReconstructionSAMLoss(nn.Module):
    def __init__(self, reduction='mean', dim=1, eps=1e-8):
        """
        Args:
            reduction (str): 指定输出的归约方式: 'none' | 'mean' | 'sum'. 默认: 'mean'.
            dim (int): 指定在哪个维度计算余弦相似度. 对于图像 [B, C, H, W], 通常 dim=1.
            eps (float): 防止除零的小数.
        """
        super(ReconstructionSAMLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Supported: 'none', 'mean', 'sum'.")
        
        self.reduction = reduction
        self.dim = dim
        self.eps = eps

    def forward(self, input, target):
        # 1. 计算余弦相似度
        # input/target: [B, C, H, W] -> sim: [B, H, W]
        cosine_sim = F.cosine_similarity(input, target, dim=self.dim, eps=self.eps)
        
        # 2. 转换为 Loss (范围 0~2, 0表示完全一致)
        loss = 1.0 - cosine_sim

        # 3. 根据 reduction 参数处理输出
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss

class MaxGradientLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        MaxGradientLoss: 计算两张图最大梯度响应的差异。
        注意：不需要在 init 中传入 device，请在外部使用 .to(device) 或 .cuda()
        """
        super(MaxGradientLoss, self).__init__()
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

        # 1. 定义 Sobel 算子 (Float类型)
        # 注意这里直接在 CPU 上创建，稍后随模型移动
        kernel_x_data = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y_data = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        
        # 2. 注册为 Buffer
        # 这样 kernel_x 会自动变成 self.kernel_x，并出现在 model.state_dict() 中
        self.register_buffer('kernel_x', kernel_x_data)
        self.register_buffer('kernel_y', kernel_y_data)

    def get_gradient_map(self, x):
        b, c, h, w = x.shape
        # Group Conv
        x_reshape = x.view(b * c, 1, h, w)

        kx = self.kernel_x.to(device=x.device, dtype=x.dtype)
        ky = self.kernel_y.to(device=x.device, dtype=x.dtype)
        
        # 关键点：F.conv2d 会自动检查 input 和 kernel 是否在同一个 device
        # 如果你忘了在外部调用 .cuda()，这里报错会提示 device mismatch
        gx = F.conv2d(x_reshape, kx, padding=1)
        gy = F.conv2d(x_reshape, ky, padding=1)
        
        return torch.abs(gx) + torch.abs(gy)

    def forward(self, hsi_pred, msi_target):
        # 1. 自动转换类型 (如果输入是 fp16，把 kernel 也转为 fp16，增强鲁棒性)
        if hsi_pred.dtype != self.kernel_x.dtype:
            self.kernel_x = self.kernel_x.to(dtype=hsi_pred.dtype)
            self.kernel_y = self.kernel_y.to(dtype=hsi_pred.dtype)

        g_hsi = self.get_gradient_map(hsi_pred).view(hsi_pred.shape)
        g_msi = self.get_gradient_map(msi_target).view(msi_target.shape)
        
        g_hsi_max, _ = torch.max(g_hsi, dim=1, keepdim=True)
        g_msi_max, _ = torch.max(g_msi, dim=1, keepdim=True)
        
        loss = torch.abs(g_hsi_max - g_msi_max)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class Diffusion(nn.Module):
    def __init__(self, model, device, img_size, LR_size, channels=3):
        super().__init__()
        self.channels = channels
        self.model = model.to(device)
        self.img_size = img_size
        self.LR_size = LR_size
        self.device = device
        # 粗配准
        self.CRN = CoarseRegistrationNetwork(patch_size=img_size//4, dim=256, num_heads=4, in_ch_msi=4,
                                             in_ch_hsi=self.channels).to(device)

        # self.upSample = nn.Upsample(scale_factor=4, mode='bicubic')
        self.downSample = nn.Upsample(scale_factor=0.25, mode='bicubic')
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')

        # complementary fusion block
        self.fuse = nn.Sequential(
            nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels*2, self.channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
        ).to(device)

    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
            self.sam_loss = ReconstructionSAMLoss()
            self.loss_mode = 'l1'
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
            self.sam_loss = None
            self.loss_mode = 'l2'
        elif loss_type == 'l1+SAMloss':
            self.loss_func = nn.L1Loss(reduction='sum')
            self.sam_loss = ReconstructionSAMLoss(reduction='sum')
            self.loss_mode = 'l1+SAMloss'
            self.sam_weight = 1e-2
        elif loss_type == 'l1+SAMloss+MaxGradient':
            self.loss_func = nn.L1Loss(reduction='mean')
            self.sam_loss = ReconstructionSAMLoss(reduction='mean')
            self.max_gradient_loss = MaxGradientLoss(reduction='mean')
            self.loss_mode = 'l1+SAMloss+MaxGradient'
            self.sam_weight = 1e-2
            self.max_gradient_weight = 1e-2
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            warmup_frac = 0.1
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end']
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        self.num_timesteps = int(len(betas))
        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_hrMS=None, condition_lrHS=None, HSI_Patch=None, MSI_Patch2=None):
        batch_size, c = x.shape[0], condition_lrHS.shape[1]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        # x_recon = self.predict_start(x, t, noise=self.model(torch.cat([condition_x, x], dim=1), noise_level))
        lrHS_reg = self.CRN(x, condition_lrHS, self.downSample(condition_hrMS), condition_hrMS)
        x_start = self.model(torch.cat([lrHS_reg, x], dim=1), condition_hrMS, HSI_Patch[0], MSI_Patch2[0], condition_lrHS)

        posterior_mean = (
                self.posterior_mean_coef1[t] * x_start.clamp(-1, 1) +
                self.posterior_mean_coef2[t] * x
        )

        posterior_variance = self.posterior_log_variance_clipped[t]

        mean, posterior_log_variance = posterior_mean, posterior_variance
        return mean, posterior_log_variance

    # Progress single step of reverse diffusion process
    # Given mean and log variance of posterior, sample reverse diffusion result from the posterior
    @torch.no_grad()
    def p_sample(self, img_noise, t, clip_denoised=True, condition_hrMS=None, condition_lrHS=None, HSI_Patch=None, MSI_Patch2=None):

        mean, log_variance = self.p_mean_variance(x=img_noise, t=t, clip_denoised=clip_denoised, condition_hrMS=condition_hrMS, condition_lrHS=condition_lrHS, HSI_Patch=HSI_Patch, MSI_Patch2=MSI_Patch2)
        noise = torch.randn_like(img_noise) if t > 0 else torch.zeros_like(img_noise)
        return mean + noise * (0.5 * log_variance).exp()

    # Progress whole reverse diffusion process
    @torch.no_grad()
    def super_resolution(self, gtHS, hrMS, lrHS, HSI_Patch, MSI_Patch2):
        img_noise = torch.rand_like(gtHS, device=gtHS.device)
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img_noise, i, condition_hrMS=hrMS, condition_lrHS=lrHS, HSI_Patch=HSI_Patch, MSI_Patch2=MSI_Patch2)
        return img

    def net(self, gtHS, hrMS, lrHS_reg, HSI_Patch, MSI_Patch2, lrHS):

        gtHS = gtHS
        hrMS = hrMS
        lrHS_reg = lrHS_reg
        lrHS=lrHS

        b, c, h, w = gtHS.shape
        t = torch.randint(1, schedule_opt['n_timestep'], size=(b,))
        sqrt_alpha_cumprod_t = extract(torch.from_numpy(self.sqrt_alphas_cumprod_prev), t, gtHS.shape)
        sqrt_alpha = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1).type(torch.float32).to(gtHS.device)
        sqrt_1m_alpha_t = (1 - sqrt_alpha ** 2).sqrt() 
        # # --------------------- 原始扩散模型噪声添加 ---------------------------------- #
        # noise = torch.randn_like(gtHS).to(gtHS.device)
        # # Perturbed image obtained by forward diffusion process at random time step t
        # x_noisy = sqrt_alpha * gtHS + (1 - sqrt_alpha ** 2).sqrt() * noise
        # # The bilateral model predict actual x0 added at time step t
        # # --------------------- 原始扩散模型噪声添加 ---------------------------------- #


        # --------------------- 通过空间掩膜，添加光谱噪声 -------------------------------------- #
        # step1: 空间掩膜
        mask = generate_spatial_mask_time_scaled(gtHS, sqrt_1m_alpha_t)

        # step2: 谱维噪声
        spectral_noise = generate_spectral_noise(gtHS, mask)

        # 修改成-logr下的乘性噪声
        eps=1e-6
        A = -torch.log(gtHS.clamp(min=eps))
        A_noisy = sqrt_alpha * A + (1 - sqrt_alpha ** 2).sqrt() * spectral_noise
        x_noisy = torch.exp(-A_noisy)
        # ----------------------------------------------------------------------- # 

        outputs = self.model(torch.cat([lrHS_reg, x_noisy], 1), hrMS, HSI_Patch, MSI_Patch2, lrHS)

        # complementary fusion
        if hasattr(self, 'loss_mode') and self.loss_mode == 'l1+SAMloss':
            l1 = self.loss_func(outputs, gtHS)
            l1 = l1# /(gtHS.shape[0]*gtHS.shape[1]*gtHS.shape[2]*gtHS.shape[3])
            sam = self.sam_loss(outputs, gtHS)# /(gtHS.shape[0]*gtHS.shape[1]*gtHS.shape[2]*gtHS.shape[3])
            Loss = l1 + self.sam_weight * sam
        elif hasattr(self, 'loss_mode') and self.loss_mode == 'l1+SAMloss+MaxGradient':
            l1 = self.loss_func(outputs, gtHS)
            l1 = l1  # /(gtHS.shape[0]*gtHS.shape[1]*gtHS.shape[2]*gtHS.shape[3])
            sam = self.sam_loss(outputs, gtHS)  # /(gtHS.shape[0]*gtHS.shape[1]*gtHS.shape[2]*gtHS.shape[3])
            max_gradient = self.max_gradient_loss(outputs, hrMS)  # /(gtHS.shape[0]*gtHS.shape[1]*gtHS.shape[2]*gtHS.shape[3])
            Loss = l1 + self.sam_weight * sam + self.max_gradient_weight * max_gradient
        else:
            Loss = self.loss_func(outputs, gtHS)
            l1 = Loss/(gtHS.shape[0]*gtHS.shape[1]*gtHS.shape[2]*gtHS.shape[3])
            sam = self.sam_loss(outputs, gtHS)/(gtHS.shape[0]*gtHS.shape[1]*gtHS.shape[2]*gtHS.shape[3])
        return Loss, l1, sam

    def forward(self, gtHS, hrMS, lrHS, HSI_Patch, MSI_Patch2, *args, **kwargs):
        x = lrHS
        y = hrMS
        x = self.upsample(x)
        # 粗配准
        self.lrHS_reg = self.CRN(x, lrHS, self.downSample(y), y)
        return self.net(gtHS, hrMS, self.lrHS_reg, HSI_Patch, MSI_Patch2, *args, **kwargs)


# Class to train & test desired model
class SR3():
    def __init__(self, device, img_size, LR_size, loss_type, dataloader, testloader, csv_name, img_path,
                 schedule_opt, save_path, load_path=None, load=True,
                 in_channel=62, out_channel=102, inner_channel=64, norm_groups=8,
                 channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0, lr=1e-3, distributed=False):
        super(SR3, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.device = device
        self.save_path = save_path
        self.img_size = img_size
        self.LR_size = LR_size
        self.csv_name = csv_name
        self.img_path = img_path
        self.out_channel = out_channel

        model = CCFnet(patch_size=img_size, in_ch_msi=4, in_ch_hsi=self.out_channel).to(device=device)

        self.sr3 = Diffusion(model, device, img_size, LR_size, self.out_channel)
        # Apply weight initialization & set loss & set noise schedule
        # self.sr3.apply(self.weights_init_orthogonal)
        self.sr3.set_loss(loss_type)
        self.sr3.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            self.sr3 = nn.DataParallel(self.sr3)

        self.optimizer = torch.optim.Adam(self.sr3.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,
        #     T_max=EPOCH,    # 从起始 LR 退火到 eta_min 的总步数（epoch 或 step 都可）
        #     eta_min=1e-6  # 最小学习率
        # )

        params = sum(p.numel() for p in self.sr3.parameters())
        print(f"Number of model parameters : {params}")
        
        # 初始化CSV日志记录器
        self.csv_logger = CSVLogger(filename=csv_name)
        
        # 初始化最佳指标跟踪
        self.best_psnr = 0.0
        self.best_sam = float('inf')
        
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)

        if load:
            self.load(load_path)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    def train(self, epoch, verbose):

        train = True
        for i in range(epoch):
            i = i
            train_loss = 0
            l=0
            s=0
            self.sr3.train()
            randn1 = np.random.randint(0, 100)

            if train:
                for step, [gtHS, hrMS, lrHS, HSI_Patch, MSI_Patch2] in enumerate(tqdm(self.dataloader)):
                    # 高光谱和全色图像
                    gtHS = gtHS.type(torch.float32).to(DEVICE)
                    hrMS = hrMS.type(torch.float32).to(DEVICE)
                    lrHS = lrHS.type(torch.float32).to(DEVICE)
                    HSI_Patch = HSI_Patch.type(torch.float32).to(DEVICE)
                    MSI_Patch2 = MSI_Patch2.type(torch.float32).to(DEVICE)

                    self.optimizer.zero_grad()
                    loss, l1, sam = self.sr3(gtHS, hrMS, lrHS, HSI_Patch, MSI_Patch2, lrHS)
                    loss.backward()
                    self.optimizer.step()
                    # self.scheduler.step()

                    train_loss += loss.item()
                    l += l1.item()
                    s += sam.item()

                print('epoch: {}'.format(i))
                print('损失函数:')
                x = PrettyTable()
                x.add_column("loss", ['value'])
                x.add_column("loss_all", [train_loss / float(len(self.dataloader))])
                x.add_column("l1", [l / float(len(self.dataloader))])
                x.add_column("sam", [s / float(len(self.dataloader))])
                print(x)

            if (i + 1) % verbose == 0:
                self.sr3.eval()
                all_metrics = {'PSNR': [], 'SAM': [], 'SSIM': [], 'RMSE': [], 'ERGAS': []}

                # 遍历整个验证集
                for idx, (gtHS, hrMS, lrHS, HSI_Patch, MSI_Patch2) in enumerate(tqdm(self.testloader)):
                    gtHS = gtHS.type(torch.float32).to(DEVICE)
                    hrMS = hrMS.type(torch.float32).to(DEVICE)
                    lrHS = lrHS.type(torch.float32).to(DEVICE)
                    HSI_Patch = HSI_Patch.type(torch.float32).to(DEVICE)
                    MSI_Patch2 = MSI_Patch2.type(torch.float32).to(DEVICE)

                    fuse_result = self.test(gtHS, lrHS, hrMS, HSI_Patch, MSI_Patch2)

                    # ---- 每张图计算 ----
                    Metric = iv.spectra_metric(
                        gtHS[0].detach().cpu().permute(1, 2, 0).numpy(),
                        fuse_result[0].detach().cpu().permute(1, 2, 0).numpy(),
                        max_v=1, 
                        scale=4, 
                    )
                    PSNR = Metric.PSNR()
                    SAM_out = Metric.SAM('map')
                    ERGAS = Metric.ERGAS()

                    # 判断 SAM_out 是不是 tuple/list
                    if isinstance(SAM_out, (tuple, list)):
                        SAM_val = SAM_out[0]
                        SAM_map = SAM_out[1].reshape(gtHS.shape[2], gtHS.shape[3])
                    else:
                        SAM_val = SAM_out
                        SAM_map = None
                    all_metrics['PSNR'].append(PSNR)
                    all_metrics['SAM'].append(SAM_val)
                    all_metrics['ERGAS'].append(ERGAS)

                    # ---- 汇总平均 ----
                    avg_metrics = {}
                    for k, v in all_metrics.items():
                        if len(v) > 0:
                            tensor_v = torch.tensor(
                                [float(x) if not isinstance(x, torch.Tensor) else x.item() for x in v],
                                dtype=torch.float32,
                                device='cpu'
                            )
                            avg_metrics[k] = float(tensor_v.mean().item())
                        else:
                            avg_metrics[k] = float('nan')

                print("验证集平均指标:")
                table = PrettyTable()
                table.field_names = ["Index", "PSNR", "SAM", "ERGAS"]
                table.add_row(['Average', avg_metrics['PSNR'], avg_metrics['SAM'], # avg_metrics['SSIM'],
                            # avg_metrics['RMSE'], 
                            avg_metrics['ERGAS']])
                print(table)

                # 检查是否需要保存最佳模型
                is_best_psnr = avg_metrics['PSNR'] > self.best_psnr
                is_best_sam = avg_metrics['SAM'] < self.best_sam
                

                # 保存平均指标到 CSV
                self.csv_logger.log_metrics(i, avg_metrics, is_best_psnr, is_best_sam)
                
                if is_best_psnr:
                    self.best_psnr = avg_metrics['PSNR']
                    self.save_best_model(self.save_path, 'PSNR_best')
                    print(f"保存最佳PSNR模型: {self.best_psnr:.4f}")

                if is_best_sam:
                    self.best_sam = avg_metrics['SAM']
                    self.save_best_model(self.save_path, 'SAM_best')
                    print(f"保存最佳SAM模型: {self.best_sam:.4f}")

                show_img(i, gtHS, lrHS, hrMS, fuse_result, self.img_path, SAM_map, bandlist=[30,20,10])


    def test(self, gtHS, lrHS, hrMS, HSI_Patch, MSI_Patch2):
        lrHS = lrHS
        hrMS = hrMS
        gtHS = gtHS
        self.sr3.eval()
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                result_SR = self.sr3.module.super_resolution(gtHS, hrMS, lrHS, HSI_Patch, MSI_Patch2)
            else:
                result_SR = self.sr3.super_resolution(gtHS, hrMS, lrHS, HSI_Patch, MSI_Patch2)
        self.sr3.train()
        return result_SR

    def save(self, save_path, i):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path+'SR3_model_epoch-{}.pt'.format(i))

    def save_best_model(self, save_path, model_type):
        """保存最佳模型"""
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path + f'SR3_model_epoch-{model_type}.pt')

    def load(self, load_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path))
        print("Model loaded successfully")

def set_seed(seed):
    """
    设置随机种子以确保结果稳定
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed = 42
    set_seed(seed)  # 设置随机种子

    batch_size = 100
    LR_size = 32
    img_size = 128

    # 超参数
    EPOCH = 2000
    BATCHSIZE = 8
    # 自动选择最空闲的GPU
    DEVICE = get_free_gpu()
    print(f"使用设备: {DEVICE}")
    
    PATCH_SIZE = 16
    IN_CH_HSI = 48
    IN_CH_MSI = 4
    # 实际训练
    path = "./dataHou2/"
    # # 做测试
    # path = "./dataPaC_copy/"
    train_datasat = Datasat('train', 128, path, IN_CH_HSI=IN_CH_HSI, IN_CH_MSI=IN_CH_MSI)
    train_loader = DataLoader(train_datasat, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

    test_datasat = Datasat('test', 128, path, IN_CH_HSI=IN_CH_HSI, IN_CH_MSI=IN_CH_MSI)
    test_loader = DataLoader(test_datasat, batch_size=1, shuffle=False, num_workers=0)

    cuda = torch.cuda.is_available()
    # 使用自动选择的设备
    device = DEVICE
    schedule_opt = {'schedule': 'cosine', 'n_timestep': 2000, 'linear_start': 1e-4, 'linear_end': 0.002}

    sr3 = SR3(device, img_size=img_size, LR_size=LR_size, loss_type='l1+SAMloss+MaxGradient', # 'l1'
              dataloader=train_loader, testloader=test_loader, schedule_opt=schedule_opt,
              # save_path='./model/LRHS_Elastic1000_PaviaC_FrPAA_Multi_Fusion7_Grad/',
              # save_path='./model/Our_Houston2/', 
              save_path='./model/visual_houston/', 
              load_path='./model/Our_Houston2/SR3_model_epoch-1099.pt', 
              csv_name='Our_Houston2.csv',
              img_path='./img/Our_Houston2/',
              load=False,
              in_channel=IN_CH_HSI+IN_CH_MSI+IN_CH_HSI, out_channel=IN_CH_HSI,
              inner_channel=64,
              # norm_groups=16, channel_mults=(1, 2, 2, 2), dropout=0, res_blocks=2, lr=1e-4, distributed=False)
              norm_groups=16, channel_mults=(1, 2, 2, 2), dropout=0, res_blocks=2, lr=1e-3, distributed=False)
    print("-------------------Noisy + PAA + Fusion----------------------")

    sr3.train(epoch=EPOCH, verbose=100)