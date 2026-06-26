import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from KAN.kan import KANLinear
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math


# ==========================
# v2.5 删除F3Module，替换成FrFT


from FrFT_2d import FrFT2DModule
import einops

frft_module_45 = FrFT2DModule(order=0.5, log_output=False)
frft_module_90 = FrFT2DModule(order=1.0, log_output=False)


# 每次对一个输入特征进行处理
class FrFTConvLayer(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, 3,1,1)
        self.act = nn.ReLU(inplace=False)
        self.norm1_x = nn.LayerNorm(embed_dim)

        self.norm_pre_attn = nn.LayerNorm(embed_dim)

        p = embed_dim // 3
        r = embed_dim - 2 * p
        self.split_sizes = (p, p, r)

        self.conv_alpha0 = nn.Sequential(
            nn.Conv2d(p, p, 3, 1, 1),
            nn.ReLU(inplace=False)
        )
        self.conv45 = nn.Sequential(
            nn.Conv2d(2 * p, 2 * p, 1),
            nn.ReLU(inplace=False)
        )
        self.conv90 = nn.Sequential(
            nn.Conv2d(2 * r, 2 * r, 1),
            nn.ReLU(inplace=False)
        )
        self.conv_log = nn.Sequential(
            nn.Conv2d(r, r, 1),
            nn.ReLU(inplace=False)
        )
        self.conv_fuse90 = nn.Sequential(
            nn.Conv2d(2 * r, r, 1),
            nn.ReLU(inplace=False)
        )

        self.frft45 = frft_module_45
        self.frft90 = frft_module_90

    @staticmethod
    def _stack_real_imag(z: torch.Tensor) -> torch.Tensor:
        real_imag = torch.view_as_real(z)
        real_imag = real_imag.permute(0, 4, 1, 2, 3).contiguous()
        return real_imag.reshape(z.size(0), 2 * z.size(1), *z.shape[2:])
    @staticmethod
    def _unstk_to_complex(pair: torch.Tensor) -> torch.Tensor:
        real, imag = pair.chunk(2, dim=1)
        return torch.complex(real, imag)

    def _to_channels_first(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.shape}")
        if x.shape[1] == self.norm1_x.normalized_shape[0]:
            return x
        if x.shape[-1] == self.norm1_x.normalized_shape[0]:
            return x.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Cannot determine channel axis for shape {x.shape}")

    def _compute_reduced_features(self, x):
        x5 = self._to_channels_first(x)
        p, _, r = self.split_sizes

        x0 = x5[:, :p]
        x45 = x5[:, p:2 * p]
        x90_in = x5[:, 2 * p:]

        out0 = self.conv_alpha0(x0)

        z45 = self.frft45.FrFT2D(x45)
        z45 = self._stack_real_imag(z45)
        c45 = self.conv45(z45)
        out45 = self.frft45.IFrFT2D(self._unstk_to_complex(c45)).real.clone()

        z90 = self.frft90.FrFT2D(x90_in)
        z90_hid = self._stack_real_imag(z90)
        c90 = self.conv90(z90_hid)
        out90 = self.frft90.IFrFT2D(self._unstk_to_complex(c90)).real.clone()

        log_mag = self.conv_log(torch.log1p(torch.abs(z90)))
        mag = torch.expm1(log_mag)
        z_log = torch.polar(mag, torch.angle(z90))
        out_log = self.frft90.IFrFT2D(z_log).real.clone()

        fused = self.conv_fuse90(torch.cat([out90, out_log], dim=1))
        out = torch.cat([out0, out45, fused], dim=1)
        #out.add_(x5)
        out = out + x5
        return out

    def forward(self, x):
        x = self.act(self.conv(x))

        B,C,H,W = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()

        x_norm = self.norm1_x(x)
        rx = self._compute_reduced_features(x_norm)
        tx = einops.rearrange(rx, 'b c h w -> b (h w) c')
        tx = self.norm_pre_attn(tx)
        tx = tx.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        # tx = tx.permute(0, 3, 1, 2).contiguous()
        return tx


try:
    from DSv2.ds_v2 import DSMv3
except ImportError:
    raise ImportError('Failed to import Deformable Sampling Module.')

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv 
        self.partial_conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)
    
    def forward(self, x):
        # x.shape = (b, c, h, w)
        x1, x2,= torch.split(x, [self.dim_conv,self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        x = torch.cat((x1, x2), 1)
        return x

class to_channel_first(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x.shape = (b, h, w, c)
        return x.permute(0, 3, 1, 2)

class to_channel_last(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x.shape = (b, c, h, w)
        return x.permute(0, 2, 3, 1)


# ##########################################################################
# ## Refinement Feed-forward Network (FRFN)
class Mlp(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim*2),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv 
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        # bs x h x w x c
        bs, h, w, c = x.size()
        # hh = int(math.sqrt(hw))


        # spatial restore
        x = x.permute(0, 3, 1, 2) # (b, c, h, w)

        x1, x2,= torch.split(x, [self.dim_conv,self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = x.permute(0, 2, 3, 1) # (b, h, w, c)

        x = self.linear1(x) # (b, h, w, 2c)
        #gate mechanism
        x_1,x_2 = x.chunk(2,dim=-1)

        x_1 = x_1.permute(0, 3, 1, 2) # (b, c, h, w)
        x_1 = self.dwconv(x_1).permute(0, 2, 3, 1) # (b, h, w, c)
        x = x_1 * x_2
        
        x = self.linear2(x)
        # x = self.eca(x)

        return x


class MSDSNv3(nn.Module):
    def __init__(self, inc=3, kernel_size=3, padding=1, stride=1,groups=4):
        """mutli-scale deformable sampling network, different head different scale.

        Args:
            inc (int, optional): _description_. Defaults to 3.
            kernel_size (int, optional): _description_. Defaults to 3.
            padding (int, optional): _description_. Defaults to 1.
            stride (int, optional): _description_. Defaults to 1.
            groups (int, optional): _description_. Defaults to 4.
        """
        super(MSDSNv3, self).__init__()
        self.kernel_size = kernel_size
        self.groups = groups
        self.conv_offset_list = nn.ModuleList([
            nn.Sequential(PConv(inc),
                          nn.Conv2d(inc, 2 * kernel_size ** 2, 1))
            for _ in range(groups)])
        self.conv_mask = nn.Sequential(PConv(inc),
                                       nn.Conv2d(inc, kernel_size ** 2, 1))
        self.ds_list = nn.ModuleList([
            DSMv3(inc // groups, kernel_size=kernel_size, stride=stride, padding=(2*i+1) * padding, dilation=2*i+1)
            for i in range(groups)])
        self.init_offset()

    def init_offset(self):
        for op in self.conv_offset_list:
            for m in op.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for m in self.conv_mask.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            y1, y2 = y.chunk(2, dim=0)
            y1_list = y1.chunk(self.groups, dim=1)
            y2_list = y2.chunk(self.groups, dim=1)

            y1_new_list = list()
            y2_new_list = list()

            mask = self.conv_mask(x)
            # print("x.dtype is {}".format(x.dtype))
            # print("mask.dtype is {}".format(mask.dtype))

            for y1_old, y2_old, op1, op2 in zip(y1_list, y2_list, self.conv_offset_list, self.ds_list):
                offset = op1.forward(x)
                # print("offset.dtype is {}".format(offset.dtype))
                y1_new_list.append(op2.forward(y1_old.float().contiguous(), offset, mask).view(b, c//self.groups, -1, h, w))
                y2_new_list.append(op2.forward(y2_old.float().contiguous(), offset, mask).view(b, c//self.groups, -1, h, w)) 

            y1 = torch.cat(y1_new_list, dim=1)
            y2 = torch.cat(y2_new_list, dim=1)
            y = torch.cat([y1, y2], dim=0) # (2 * b, c, N, h, w)
        return y



class DeformCenterAttention(nn.Module):
    """
    """
    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 stride=1,
                 padding=True,
                 kernel_size=3):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.k_size = kernel_size  # kernel size
        self.stride = stride  # stride
        # self.pat_size = patch_size  # patch size

        self.in_channels = dim  # origin channel is 3, patch channel is in_channel
        self.num_heads = num_heads
        self.head_channel = dim // num_heads
        # self.dim = dim # patch embedding dim
        # it seems that padding must be true to make unfolded dim matchs query dim h*w*ks*ks
        self.pad_size = kernel_size // 2 if padding is True else 0  # padding size
        self.pad = nn.ZeroPad2d(self.pad_size)  # padding around the input
        self.scale = qk_scale or (dim // num_heads)**-0.5
        # self.unfold = nn.Unfold(kernel_size=self.k_size, stride=self.stride, padding=0, dilation=1)
        # 改用q来决定kv要采样哪些位置
        self.dsn = MSDSNv3(inc=dim, kernel_size=kernel_size, padding=self.pad_size, stride=1, groups=num_heads)

        self.qkv_bias = qkv_bias
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_dwconv = DWConv(dim=dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        # x = x.reshape(B, H, W, C)
        assert C == self.in_channels

        self.num_patch = H * W # 切出来的patch的数量

        qkv = self.qkv_proj(x)
        # (B, H, W, 3 * C)
        qkv = self.qkv_dwconv(qkv)
        q = qkv[:, :, :, :C]
        kv = qkv[:, :, :, C:]

        

        # # (2, B, NumHeads, HeadsC, H, W)
        kv = kv.reshape(B, H, W, 2, self.num_heads, self.head_channel).permute(3, 0, 4, 5, 1, 2)

        # kv = self.pad(kv)  # (2, B, NumH, HeadC, H, W)
        # H, W = H + self.pad_size * 2, W + self.pad_size * 2

        # unfold plays role of conv2d to get patch data
        kv = kv.reshape(2 * B, -1, H, W) # (2 * B, C, pad_H, pad_W)
        kv = self.dsn(q.permute(0, 3, 1, 2), kv) # (2 * B, C, N, h, w)
        # kv = self.unfold(kv)

        # # (B, NumHeads, H, W, HeadC)
        q = q.reshape(B, H, W, self.num_heads, self.head_channel).permute(0, 3, 1, 2, 4)
        # # q = self.pad(q).permute(0, 1, 3, 4, 2)  # (B, NumH, H, W, HeadC)
        # # query need to be copied by (self.k_size*self.k_size) times
        q = q.unsqueeze(dim=4)
        q = q * self.scale
        # # if stride is not 1, q should be masked to match ks*ks*patch
        # # ...

        kv = kv.reshape(2, B, self.num_heads, self.head_channel, self.k_size**2,
                        self.num_patch)  # (2, B, NumH, HC, ks*ks, NumPatch)
        kv = kv.permute(0, 1, 2, 5, 4, 3)  # (2, B, NumH, NumPatch, ks*ks, HC)
        k, v = kv[0], kv[1]

        # (B, NumH, NumPatch, 1, HeadC)
        q = q.reshape(B, self.num_heads, self.num_patch, 1, self.head_channel)
        attn = (q @ k.transpose(-2, -1))  # (B, NumH, NumPatch, ks*ks, ks*ks)
        attn = self.softmax(attn)  # softmax last dim
        attn = self.attn_drop(attn)

        out = (attn @ v).squeeze(3)  # (B, NumH, NumPatch, HeadC)
        out = out.permute(0, 2, 1, 3).reshape(B, H, W, C)  # (B, Ph, Pw, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        # out = out.reshape(B, -1, C)
        return out

class SADAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DeformCenterAttention(
            dim=dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
           
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim=dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DeformCenterFFTCrossAttention(nn.Module):
    """
    Frequency-enhanced Deformable Cross Attention (with F3Module)
    - HSI 和 MSI 特征都先经过 F3Module 的频域增强
    - 在增强后的频域表示上执行 Deformable Cross Attention
    - 输出再通过 IFFT 返回空间域
    """
    def __init__(self, dim, num_heads=4, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, local_size=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        # ===========================
        # 变形采样网络
        # ===========================
        self.dsn = MSDSNv3(
            inc=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
            groups=num_heads
        )

        # ===========================
        # 注意力线性层
        # ===========================
        self.q_proj = nn.Linear(dim*2, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim*2, dim * 2, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim*2)
        self.proj_drop = nn.Dropout(proj_drop)

        self.hsi_frft = FrFTConvLayer(in_channels=dim, embed_dim=dim)
        self.msi_frft = FrFTConvLayer(in_channels=dim, embed_dim=dim)

    def forward(self, x_hsi, x_msi):
        """
        Args:
            x_hsi: (B, N, C)
            x_msi: (B, N, C)
        Returns:
            out: (B, N, C)
        """
        B, N, C = x_hsi.shape
        H = W = int(N ** 0.5)

        # ---------------------------------------------------------------
        # Step 1. FrFT空间特征增强
        # ---------------------------------------------------------------
        x_hsi_2d = x_hsi.view(B, H, W, C).permute(0, 3, 1, 2)
        x_msi_2d = x_msi.view(B, H, W, C).permute(0, 3, 1, 2)

        x_hsi_2d = self.hsi_frft(x_hsi_2d)
        x_msi_2d = self.msi_frft(x_msi_2d)

        # ---------------------------------------------------------------
        # Step 2. 转换为频域特征进行交互（重新做FFT，仅用于注意力匹配）
        # ---------------------------------------------------------------
        freq_hsi = torch.fft.fft2(x_hsi_2d, dim=(-2, -1))
        freq_msi = torch.fft.fft2(x_msi_2d, dim=(-2, -1))

        # 拆分实虚 -> 拼接为通道
        freq_hsi_cat = torch.cat([freq_hsi.real, freq_hsi.imag], dim=1)  # (B, 2C, H, W)
        freq_msi_cat = torch.cat([freq_msi.real, freq_msi.imag], dim=1)

        # 转换为 (B, N, 2C)
        freq_hsi_flat = freq_hsi_cat.permute(0, 2, 3, 1).reshape(B, H*W, 2*C)
        freq_msi_flat = freq_msi_cat.permute(0, 2, 3, 1).reshape(B, H*W, 2*C)

        # ---------------------------------------------------------------
        # Step 3. Q/K/V计算
        # ---------------------------------------------------------------
        q = self.q_proj(freq_hsi_flat) # [b,n,c]
        kv = self.kv_proj(freq_msi_flat) # [b,n,2c]
        k, v = kv.chunk(2, dim=-1)

        # ---------------------------------------------------------------
        # Step 4. 频域可变形采样
        # ---------------------------------------------------------------
        q_2d = q.view(B, H, W, C)
        kv_2d = torch.cat([k, v], dim=-1).view(2*B, H, W, C)
        kv_2d = kv_2d.permute(0, 3, 1, 2).contiguous()
        kv_2d = self.dsn(q_2d.permute(0, 3, 1, 2), kv_2d) # [2 * b, c, N, h, w]

        # -------------------------------------------------
        # 聚合采样特征：
        kv_deformed = kv_2d.sum(dim=2)
        kv_deformed = kv_deformed.view(2, B, C, H, W)
        k_new = kv_deformed[0] # [B, C, H, W]
        v_new = kv_deformed[1] # [B, C, H, W]
        # 转换回序列格式用于 Attention
        # [B, C, H, W] -> [B, N_tokens, C]
        k = k_new.flatten(2).transpose(1, 2)
        v = v_new.flatten(2).transpose(1, 2)
        # ---------------------------------------------------------------
        # Step 5. 频域注意力
        # ---------------------------------------------------------------
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) 
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        freq_out = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        freq_out = self.proj(freq_out)
        freq_out = self.proj_drop(freq_out)

        # ---------------------------------------------------------------
        # Step 6. IFFT恢复空间域
        # ---------------------------------------------------------------
        freq_out = freq_out.view(B, H, W, 2*C).permute(0, 3, 1, 2) # (B, 2C, H, W)
        #   拆分实虚部
        freq_real, freq_imag = freq_out.chunk(2, dim=1)
        
        # 组合复数
        freq_complex = torch.complex(freq_real, freq_imag)
        # freq_complex = torch.complex(
        #     freq_out.permute(0, 3, 1, 2),  # 实部（假定虚部为0）
        #     torch.zeros_like(freq_out.permute(0, 3, 1, 2))
        # )
        spatial_out = torch.fft.ifft2(freq_complex, dim=(-2, -1)).real
        spatial_out = spatial_out.permute(0, 2, 3, 1).reshape(B, N, C)

        return spatial_out

class DeformCenterCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.dsn = MSDSNv3(inc=dim, kernel_size=kernel_size, padding=kernel_size // 2, stride=1, groups=num_heads)

        # projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_hsi, x_msi):
        B, N, C = x_hsi.shape
        q = self.q_proj(x_hsi) # 这里 可能需要通过 x_hsi的k和x_msi的v 然后通过x_hsi的q去查找
        kv = self.kv_proj(x_msi)
        k, v = kv.chunk(2, dim=-1)

        # deformable dynamic sampling
        q_2d = q.view(B, int(N**0.5), int(N**0.5), C)
        kv_2d = torch.cat([k, v], dim=-1).view(2*B, int(N**0.5), int(N**0.5), C)
        kv_2d = kv_2d.permute(0, 3, 1, 2).contiguous()
        kv_2d = self.dsn(q_2d.permute(0, 3, 1, 2), kv_2d)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.proj_drop(out)

# class SADCrossAttentionBlock(nn.Module):
#     def __init__(self, dim, num_heads=4, drop=0.1, attn_drop=0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = DeformCenterFFTCrossAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim*4),
#             nn.GELU(),
#             nn.Linear(dim*4, dim)
#         )
#         self.drop_path = DropPath(drop)

#     def forward(self, x_hsi, x_msi):
#         x = x_hsi + self.drop_path(self.attn(self.norm1(x_hsi), self.norm1(x_msi)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


class StructureTensor(nn.Module):
    def __init__(self,dim, patch_size=160, img_size=160):
        super(StructureTensor, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).repeat(dim,dim,1,1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).repeat(dim,dim,1,1)

        self.sobel_x_cov = torch.nn.Conv2d(dim, dim, (3, 3), stride=1, padding=1, bias=False)
        self.sobel_x_cov.weight.data = sobel_x
        self.sobel_y_cov = torch.nn.Conv2d(dim, dim, (3, 3), stride=1, padding=1, bias=False)
        self.sobel_y_cov.weight.data = sobel_y

        self.sobel_x_cov.weight.requires_grad = False
        self.sobel_y_cov.weight.requires_grad = False
        self.batchnorm = nn.BatchNorm2d(1)

    def forward(self, y):
        y_org = y
        # 空间增强分支
        _, in_c, _, _ = y.shape
        y = self.conv(y)
        y = torch.nn.functional.pad(y, (2, 2, 2, 2), mode='replicate')
        x_axis = torch.sum(self.sobel_x_cov(y), dim=1).unsqueeze(1)
        y_axis = torch.sum(self.sobel_y_cov(y), dim=1).unsqueeze(1)

        Gradient = torch.cat([x_axis, y_axis], dim=1)
        Gradient_1 = torch.cat([x_axis, y_axis], dim=1).unsqueeze(1).reshape(Gradient.shape[0], Gradient.shape[1], 1, -1).permute(0,3,1,2)
        Gradient_2 = torch.cat([x_axis, y_axis], dim=1).unsqueeze(2).reshape(Gradient.shape[0], 1, Gradient.shape[1], -1).permute(0,3,1,2)
        Structure_tensor_mine = (Gradient_1 @ Gradient_2)+0.1
        attention_map = Structure_tensor_mine[:, :, 0, 0] * Structure_tensor_mine[:, :, 1, 1]-Structure_tensor_mine[:,: , 0, 1] * Structure_tensor_mine[:, :, 1, 0]
        attention_map = attention_map.reshape(attention_map.shape[0], 1, self.patch_size+4, self.patch_size+4)
        attention_map = attention_map[:, :, 2:-2, 2:-2]
        attention_map = self.batchnorm(attention_map)

        # 可视化注意力特征
        # for i in range(8):
        #     img = attention_map[i].squeeze().unsqueeze(2).cpu().detach().numpy()
        #     plt.figure(figsize=(15, 10))
        #     plt.subplot(1, 2, 1)
        #     plt.axis("off")
        #     plt.title("Low-ResolutionInputs")
        #     plt.imshow(img)
        #
        #     plt.subplot(1, 2, 2)
        #     plt.axis("off")
        #     plt.title("Low-ResolutionInputs")
        #     plt.imshow(y_org[i].squeeze().permute(1,2,0).cpu().detach().numpy())
        #     plt.show()

        return attention_map

class SADCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, drop=0.0, attn_drop=0.0, drop_path=0.0, use_fft=False, in_hsi=102,in_msi=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_fft = use_fft
        self.in_hsi = in_hsi
        self.in_msi = in_msi

        self.group_num = next(g for g in range(3, self.in_hsi + 1) if self.in_hsi % g == 0)

        self.channel_mapper_hsi = nn.Conv2d(self.in_hsi, dim, 1) # 这里输入的是超分得到的LRHSI和对齐后的HSI的结果
        self.channel_mapper_msi = nn.Conv2d(self.in_msi, dim, 1)

        self.channel_mapper_out = nn.Sequential(
            nn.Conv2d(dim, self.in_hsi, 1),
            nn.GroupNorm(num_groups=self.group_num, num_channels=self.in_hsi),
            nn.ReLU()
        )
        # nn.Conv2d(dim, self.in_hsi, 1)

        # norm & attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        if self.use_fft:
            self.attn = DeformCenterFFTCrossAttention(dim=dim, num_heads=num_heads,
                                                      attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = DeformCenterCrossAttention(dim=dim, num_heads=num_heads,
                                                   attn_drop=attn_drop, proj_drop=drop)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x_hsi, x_msi):
        B,C_hsi,H,W = x_hsi.shape
        _,C_msi,_,_ = x_msi.shape
        x_hsi = x_hsi.reshape(B,C_hsi,H//self.num_heads, self.num_heads,W//self.num_heads, self.num_heads).permute(0,3,5,1,2,4).reshape(B*self.num_heads*self.num_heads,C_hsi,H//self.num_heads,W//self.num_heads)
        x_msi = x_msi.reshape(B,C_msi,H//self.num_heads, self.num_heads,W//self.num_heads, self.num_heads).permute(0,3,5,1,2,4).reshape(B*self.num_heads*self.num_heads,C_msi,H//self.num_heads,W//self.num_heads)
        

        # 第一步，压缩通道
        x_hsi = self.channel_mapper_hsi(x_hsi)
        x_msi = self.channel_mapper_msi(x_msi)

        # flatten
        N = H*W // (self.num_heads*self.num_heads)
        x_hsi_flat = x_hsi.permute(0,2,3,1).reshape(-1,N,self.dim)
        x_msi_flat = x_msi.permute(0,2,3,1).reshape(-1,N,self.dim)

        # attention
        x_attn = self.attn(self.norm1(x_hsi_flat), self.norm1(x_msi_flat))
        x_attn = self.proj_drop(x_attn)
        x_out = x_hsi_flat + self.drop_path(x_attn)

        # MLP
        x_mlp = self.mlp(self.norm2(x_out))
        x_out = x_out + self.drop_path(x_mlp)
        x_out = x_out.reshape(B,self.num_heads,self.num_heads,H//self.num_heads,W//self.num_heads,self.dim).permute(0,5,1,3,2,4).reshape(B,self.dim,H,W).contiguous()
        return x_out




class spectral_token_compensation(nn.Module):
    """
    光谱token补偿模块：从 ori_hsi_feature 提取全局光谱 token
    ori_hsi_feature: [B, C, H, W], conv(ori_hsi) 得到
    """

    def __init__(self, dim):
        super(spectral_token_compensation, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = KANLinear(dim, dim // 4)
        self.gelu = nn.GELU()
        self.linear2 = KANLinear(dim // 4, dim)

    def forward(self, ori_hsi_feature):
        # [B, C, 1, 1]
        # self.token_compensation = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(dim, dim // 8, kernel_size=1),
        #     nn.BatchNorm2d(dim // 8),
        #     nn.GELU(),
        #     nn.Conv2d(dim // 8, dim, kernel_size=1),
        # )

        x = self.avg_pool(ori_hsi_feature)

        # ---- flatten for Linear ----
        x = x.squeeze(-1).squeeze(-1)       # [B, C]

        # ---- spectral token proj ----
        x = self.linear1(x)                  # [B, C]
        x = self.gelu(x)
        x = self.linear2(x)                  # [B, C]
        # ---- reshape back ----
        x = x.unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]

        return x

class SpecCrossAttention(nn.Module):
    """
    Cross Attention + FFN 完整模块 
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads
        bias (bool): If True, add a learnable bias to projection
    """

    def __init__(self, dim, num_heads, size, bias):
        super(SpecCrossAttention, self).__init__()
        self.size = size
        self.num_heads = num_heads
        self.dim = dim
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.hsi_feature = nn.Conv1d(in_channels=self.size*self.size, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.ori_hsi_feature = nn.Conv1d(in_channels=self.size*self.size // 16, out_channels=dim, kernel_size=3, stride = 1, padding=1, bias=bias)

        # --- Attention projections ---
        self.qkv = nn.Conv1d(dim, dim * 3, kernel_size=1, bias=bias)
        self.q = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv1d(dim, self.size*self.size, kernel_size=1, bias=bias)

        # --- Norm layers ---
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # --- FFN ---
        ffn_hidden = dim * 4
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, ffn_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(ffn_hidden, dim, kernel_size=1)
        )

        # --- dropout (optional) ---
        self.dropout = nn.Dropout(0.1)



    def forward(self, f_hsi, ori_hsi_feature):
        b, c, h, w = f_hsi.shape
        _, _, h1, w1 = ori_hsi_feature.shape
        # f_hsi = f_img

        shortcut = f_hsi  # for residual connection

        # ------- Spectral feature extraction -------
        f_hsi_reshaped = rearrange(f_hsi, 'b c h w -> b c (h w)').permute(0,2,1)  # [B, H*W, C]
        
        f_hsi = self.hsi_feature(f_hsi_reshaped)  # [b, dim, c]
        qkv = self.qkv(f_hsi)# [b, dim * 3, c]

        _, k, v = qkv.chunk(3, dim=1)


        ori_hsi_reshape = rearrange(ori_hsi_feature, 'b c h1 w1 -> b c (h1 w1)').permute(0,2,1)  # [B, H1*W1, C]
        ori_hsi_f = self.ori_hsi_feature(ori_hsi_reshape)  # [b, dim, c]
        q = self.q(ori_hsi_f) # [b, dim, c]

        # -------------------- Cross Attention (C x C) --------------------
        # q, k, v: [B, dim, C]  --> split heads
        b, _, c_tokens = q.shape
        head_dim = self.dim // self.num_heads

        # reshape to multi-head: [B, heads, head_dim, C]
        q = q.view(b, self.num_heads, head_dim, c_tokens)
        k = k.view(b, self.num_heads, head_dim, c_tokens)
        v = v.view(b, self.num_heads, head_dim, c_tokens)

        # attention score: [B, heads, C, C]
        attn = torch.matmul(q.transpose(-2, -1), k) * self.temperature
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # output: [B, heads, C, head_dim]
        out = torch.matmul(attn, v.transpose(-2, -1))
        # restore: [B, dim, C]
        out = out.transpose(-2, -1).reshape(b, self.dim, c_tokens)

        # projection back to spectral tokens
        out = self.project_out(out)               # [B, size*size, C]

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        out = self.dropout(out) + shortcut

        # ------- FFN block -------
        out_ln = rearrange(out, 'b c h w -> b (h w) c')
        out_ln = self.norm2(out_ln)
        out_ln = rearrange(out_ln, 'b (h w) c -> b c h w', h=h, w=w)

        ffn_out = self.ffn(out_ln)
        out = out + self.dropout(ffn_out)

        # # spectral token compensation
        # spec_token = self.(ori_hsi_feature)
        # out = out + spec_token * out

        return out # [B, C, H, W]


class VectorCompressionAttention(nn.Module):
    """
    向量压缩注意力 (保持不变)
    负责提供纯净的全局光谱特征校准。
    """
    def __init__(self, in_channels):
        super().__init__()
        # 确保通道数是偶数
        self.half_dim = in_channels // 2

        # 压缩激活网络
        self.compress_activate = nn.Sequential(
            nn.Linear(self.half_dim, self.half_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.half_dim // 4, self.half_dim, bias=False), 
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.shape
        # 1. 池化 -> [B, C]
        y = self.avg_pool(x).view(b, c)

        # 2. Split 分组
        y1, y2 = torch.split(y, self.half_dim, dim=1)

        # 3. 计算门控权重 (Group 1)
        gate = self.compress_activate(y1) 

        # 4. 加权 (Group 2) -> 得到纯净的光谱特征
        y_out = y2 * gate 

        # 输出 [B, C//2, 1, 1]
        return y_out.view(b, self.half_dim, 1, 1)


class ResBlock_v1(nn.Module):
    """
    Large Kernel Dual-Branch ResBlock
    空间分支：使用 7x7 DWConv 替代普通 3x3，在不增加深度的情况下极大提升感受野。
    光谱分支：使用向量压缩注意力，防止光谱失真。
    """
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True):
        super(ResBlock_v1, self).__init__()
        self.strides = strides
        
        # ============================================================
        # 1. 空间分支 (Spatial Branch) - 升级版
        # 使用 7x7 DWConv (大感受野) + 1x1 Conv (通道融合)
        # 这种结构类似 ConvNeXt 块，极其高效且强大
        # ============================================================
        self.spatial_branch = nn.Sequential(
            # 步骤A: 7x7 Depthwise Conv (提取宽广的空间上下文)
            # groups=in_channel 保证了它是 DWConv，不破坏通道独立性
            nn.Conv2d(in_channel, in_channel, kernel_size=7, stride=strides, padding=3, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            # 这里可以放一个 GELU，但在 DWConv 后直接接 Linear 往往效果更好，我们保留 ReLU 以保持兼容性
            nn.ReLU(inplace=True),
            
            # 步骤B: 1x1 Pointwise Conv (负责通道混合和特征映射)
            # 替代了原本第二层 3x3 卷积的功能
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        # ============================================================
        # 2. 光谱分支 (Spectral Branch) - 保持并行
        # ============================================================
        self.spectral_branch = VectorCompressionAttention(in_channel)

        # 3. 融合层 (Fusion)
        # 将空间特征(out_channel) 和 光谱特征(in//2) 拼接后融合
        concat_dim = out_channel + (in_channel // 2)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(concat_dim, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        # 4. Shortcut (残差连接)
        if not same_shape or strides != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.final_act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. 计算 Shortcut
        residual = self.shortcut(x)

        # 2. 空间分支 (大核提取)
        out_spatial = self.spatial_branch(x)

        # 3. 光谱分支 (向量校准)
        out_spectral = self.spectral_branch(x)
        
        # 4. 对齐尺寸 (处理 stride > 1 的情况)
        H_new, W_new = out_spatial.shape[-2], out_spatial.shape[-1]
        out_spectral = out_spectral.expand(-1, -1, H_new, W_new)

        # 5. 拼接
        out_cat = torch.cat([out_spatial, out_spectral], dim=1)

        # 6. 融合
        out_fused = self.fusion_conv(out_cat)

        # 7. 输出
        return self.final_act(out_fused + residual)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True):
        super(ResBlock, self).__init__()
        self.same_shape = same_shape
        # if not same_shape:
        #     strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)

        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
        return F.relu(out + x)

# 1. 辅助类：支持 [N, C, H, W] 格式的 LayerNorm
# ConvNeXt 官方推荐使用 LayerNorm，但 PyTorch 的 LayerNorm 默认是对最后也是维度的
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W] -> [N, H, W, C] -> LN -> [N, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
# 3. 新设计的 ConvNeXt 融合模块
class ConvNeXt_Fusion(nn.Module):
    """
    ConvNeXt Style Dual-Branch Block
    Spatial Branch: ConvNeXt Block (7x7 DW -> LN -> 1x1 PW -> GELU -> 1x1 PW)
    Spectral Branch: VectorCompressionAttention
    """
    def __init__(self, in_channel, out_channel, strides=1, drop_path=0.):
        super().__init__()
        
        # --- 下采样层 (Shortcut path) ---
        # 如果 stride > 1 或者输入输出通道不一致，Shortcut 需要调整尺寸
        self.downsample = None
        if strides > 1 or in_channel != out_channel:
            self.downsample = nn.Sequential(
                # ConvNeXt 风格通常用 2x2, stride=2 的卷积做下采样
                # 这里为了兼容性，使用 1x1 或者根据 stride 调整
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False),
                LayerNorm2d(out_channel) 
            )

        # --- 空间分支 (Spatial Branch) - ConvNeXt Style ---
        # 1. Depthwise Conv 7x7
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=7, padding=3, 
                                groups=in_channel, stride=strides, bias=False) 
        self.norm = LayerNorm2d(in_channel)
        
        # 2. Pointwise Conv (Inverted Bottleneck: Expansion)
        # ConvNeXt 通常将内部维度扩大 4 倍
        hidden_dim = 4 * in_channel
        self.pwconv1 = nn.Conv2d(in_channel, hidden_dim, 1) 
        self.act = nn.GELU()
        
        # 3. Pointwise Conv (Projection)
        self.pwconv2 = nn.Conv2d(hidden_dim, out_channel, 1)
        
        # DropPath (正则化，可选)
        self.drop_path = nn.Identity() # 这里简化处理，实际训练建议配合 timm 的 DropPath

        # --- 光谱分支 (Spectral Branch) ---
        self.spectral_branch = VectorCompressionAttention(in_channel)

        # --- 融合层 (Fusion) ---
        # 将 ConvNeXt 提取的空间特征(out_channel) + 光谱特征(in//2) 融合
        concat_dim = out_channel + (in_channel // 2)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(concat_dim, out_channel, 1, bias=False),
            LayerNorm2d(out_channel), 
            nn.GELU() # 融合后加一个激活函数
        )
        
        # 这一层用于调整最终残差相加时的缩放（ConvNeXt有个LayerScale的概念，这里简化略去）

    def forward(self, x):
        input_tensor = x

        # ===========================
        # 1. 空间分支 (ConvNeXt)
        # ===========================
        x_spatial = self.dwconv(x)
        x_spatial = self.norm(x_spatial)
        x_spatial = self.pwconv1(x_spatial)
        x_spatial = self.act(x_spatial)
        x_spatial = self.pwconv2(x_spatial)
        
        # ===========================
        # 2. 光谱分支 (Spectral)
        # ===========================
        # 注意：光谱分支通常不需要 stride，因为它本来就是 Global 的 (1x1)
        x_spectral = self.spectral_branch(x)

        # ===========================
        # 3. 对齐与拼接
        # ===========================
        # x_spatial 现在的尺寸是 [B, out_c, H_new, W_new]
        H_new, W_new = x_spatial.shape[-2], x_spatial.shape[-1]
        
        # 广播光谱特征到空间尺寸
        x_spectral_expanded = x_spectral.expand(-1, -1, H_new, W_new)
        
        # 拼接
        x_cat = torch.cat([x_spatial, x_spectral_expanded], dim=1)
        
        # 融合
        x_fused = self.fusion_conv(x_cat)
        
        # ===========================
        # 4. 残差连接 (Residual)
        # ===========================
        if self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            
        # 最终输出 = Shortcut + (Spatial & Spectral Fused)
        return input_tensor + self.drop_path(x_fused)


class ConvNeXt_FusionBlock(nn.Module):
    """
    ConvNeXt Style Dual-Branch Block
    Spatial Branch: ConvNeXt Block (7x7 DW -> LN -> 1x1 PW -> GELU -> 1x1 PW)
    Spectral Branch: VectorCompressionAttention
    """
    def __init__(self, in_channel, out_channel, strides=1, depth=3, drop_path=0.):
        super().__init__()
        self.depth = depth
        self.strides = strides
        self.blocks = nn.ModuleList()
        self.blocks.append(
            ConvNeXt_Fusion(in_channel, out_channel, strides=self.strides)
        )
        for _ in range(depth-1):
            self.blocks.append(
                ConvNeXt_Fusion(out_channel, out_channel, strides=self.strides)
            )
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class FusionModule(nn.Module):
    """
    融合模块：将HSI和MSI特征融合为联合特征
    """
    def __init__(self, in_hsi, in_msi, dim, depth=1, num_heads=8, img_size=128):
        super(FusionModule, self).__init__()
        
        self.depth = depth
        self.num_heads = num_heads

        self.hsi_feature = nn.Conv2d(in_hsi * 2, dim, 3,1,1)
        self.msi_feature = nn.Conv2d(in_msi, dim, 3,1,1)
        self.lrhsi_feature = nn.Conv2d(in_hsi, dim, 1,1,0)

        self.output_mapper = nn.Conv2d(dim, in_hsi, 1,1,0)

        self.st = StructureTensor(in_msi, patch_size=img_size)

        self.SpaBlock = nn.ModuleList()
        self.SpeBlock = nn.ModuleList()
        for i in range(depth):
            self.SpaBlock.append(
                SADCrossAttentionBlock(
                    dim=dim,
                    num_heads=self.num_heads,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    use_fft=True,
                    in_hsi=dim,
                    in_msi=dim
                )
            )
        for i in range(depth):
            self.SpeBlock.append(
                SpecCrossAttention(
                    dim=dim,
                    num_heads=self.num_heads,
                    size=128,
                    bias=True
                )
            )

        self.FusBlock_HSI = nn.ModuleList([
            ConvNeXt_FusionBlock(dim*2, dim*2, strides=1),
            nn.ReLU(),
            ConvNeXt_FusionBlock(dim*2, dim, strides=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        ])

    def forward(self, hsi_feat, msi_feat, ori_hsi):
        # 通道映射
        hsi_mapped = self.hsi_feature(hsi_feat)
        msi_mapped = self.msi_feature(msi_feat)
        ori_hsi_feat = self.lrhsi_feature(ori_hsi)

        msi_atten = self.st(msi_feat)
        msi_mapped = msi_mapped * msi_atten + msi_mapped

        for spa_block, spe_block in zip(self.SpaBlock, self.SpeBlock):
            # 空间交互
            hsi_mapped = spa_block(hsi_mapped, msi_mapped)

            # 光谱交互
            hsi_mapped = spe_block(hsi_mapped, ori_hsi_feat)

        # 残差连接融合
        hsi_mapped = torch.cat([hsi_mapped, msi_mapped], dim=1)
        for block in self.FusBlock_HSI:
            hsi_mapped = block(hsi_mapped)

        # 输出
        out = self.output_mapper(hsi_mapped)

        return out

# ====== 测试代码 ======
if __name__ == "__main__":
    B, C_hsi, C_msi, H, W = 1, 64, 4, 128, 128

    hsi = torch.randn(B, C_hsi * 2, H, W).to('cuda')
    msi = torch.randn(B, C_msi, H, W).to('cuda')
    ori_hsi = torch.randn(B, C_hsi, H//4, W//4).to('cuda')

    model = FusionModule(in_hsi=C_hsi, in_msi=C_msi, dim=64).to('cuda')

    out = model(hsi, msi, ori_hsi)
    print("FusionModule output shape:", out.shape)