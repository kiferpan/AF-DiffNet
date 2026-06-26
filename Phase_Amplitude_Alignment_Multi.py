### 思路，首先将输入特征映射到频域中
### 然后对频域特征显性地进行相位和幅度的分离
### 接着，对相位和幅度分别进行对齐
### 并采用GFNet的频域全局卷积的思路取代自注意力矩阵
### 最后将对齐后的相位和幅度重新组合并映射回空域
### 对于MLP，采用FreqMLP的思路，在频域中对特征进行处理
import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath,  trunc_normal_


class AmplitudeSEFusion(nn.Module):
    """
    使用标准通道注意力（SE）融合 amplitude，
    输入先用 1×1 卷积进行通道压缩。
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.conv_reduce = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0)
        # Step2: 标准 SE 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )
    def forward(self, amp_x, amp_y):
        # amp_x, amp_y: [B, H, W, C]
        B, H, W, C = amp_x.shape

        # 1. concat 并 1x1 conv 压缩到 C 通道
        cat = torch.cat([amp_x, amp_y], dim=-1)  # [B, H, W, 2C]
        fused = self.conv_reduce(cat.permute(0, 3, 1, 2))  # [B, C, H, W]
        pooled = self.avg_pool(fused) # [B, C, 1, 1]
        pooled = pooled.view(B, C)
        alpha = self.fc(pooled).view(B, C, 1, 1)  # [B, C]

        A_fused = alpha * fused # [B, C, H, W]
        # GPT推荐
        # # 最终 amplitude 融合结果（你需要的唯一 amplitude）
        # A_fused = alpha * amp_x + (1 - alpha) * amp_y
        return A_fused

class PhaseSpatialAttention(nn.Module):
    """
    Phase Spatial Attention (PSA) 模块
    - 先融合两个相位（相量加权）
    - 再用 3x3 空间卷积生成注意力调整相位
    """
    def __init__(self, dim):
        super().__init__()
        # 3x3 conv 生成空间注意力权重 α
        # 输入通道为 cos+sin 拼接 → 2C
        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, phase_x, phase_y):
        """
        phase_x, phase_y: [B, H, W, C]
        return: phase_final: [B, H, W, C]
        """
        B, H, W, C = phase_x.shape
        # 1. 相量融合
        phase = self.fusion_conv(torch.cat([phase_x, phase_y], dim=-1).permute(0, 3, 1, 2))  # [B,C,H,W]
        # 2. 相量表示
        cos_f, sin_f = torch.cos(phase), torch.sin(phase)

        # 归一化
        norm = torch.sqrt(cos_f**2 + sin_f**2 + 1e-6)
        cos_f = cos_f / norm
        sin_f = sin_f / norm

        # 3. 构造卷积输入 [B, 2C, H, W]
        x_conv = torch.cat([cos_f, sin_f], dim=1)  # [B,1C,H,W]

        # 4. 生成空间注意力权重 α
        alpha = self.sigmoid(self.conv(x_conv))  # [B,C,H,W]
        # alpha = alpha.permute(0,2,3,1)          

        # 5. 调整相量
        cos_f = cos_f * alpha
        sin_f = sin_f * alpha   

        # 归一化 + atan2
        norm = torch.sqrt(cos_f**2 + sin_f**2 + 1e-6)
        cos_f = cos_f / norm
        sin_f = sin_f / norm
        phase_final = torch.atan2(sin_f, cos_f)  # [B, C, H, W]

        return phase_final



### 原版 GFNet 频域全局卷积模块，取代Transformer中的自注意力机制
class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.h = h
        self.w = w

        if dim == 64: #96 for large model, 64 for small and base model
            self.h = 16 # img_size/patch
            self.w = 9 # (img_size/2*patch)+1            
        if dim ==128:
            self.h = 28 #H
            self.w = 15 #(W/2)+1, this is due to rfft2
        if dim == 96: #96 for large model, 64 for small and base model
            self.h = 56 #H
            self.w = 29 #(W/2)+1            
        if dim ==192:
            self.h = 28 #H
            self.w = 15 #(W/2)+1, this is due to rfft2
        self.complex_weight = nn.Parameter(torch.randn(self.h, self.w, dim, 2, dtype=torch.float32) * 0.02)

        self.fusion_amplitude = AmplitudeSEFusion(dim=dim) # nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0)
        self.fusion_phase = PhaseSpatialAttention(dim=dim)# nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C) # C * patch_size
        y = y.view(B, a, b, C)

        x = x.to(torch.float32)
        y = y.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        y = torch.fft.rfft2(y, dim=(1, 2), norm='ortho')

        amplitude_x = torch.abs(x) # [b,16,9,64]
        phase_x = torch.angle(x)    # [b,16,9,64]

        amplitude_y = torch.abs(y) # [b,16,9,64]
        phase_y = torch.angle(y) # [b,16,9,64]

        # 对齐相位和幅度
        # amplitude = torch.cat((amplitude_x, amplitude_y), dim=-1)  # 将幅度拼接在一起
        # phase = torch.cat((phase_x, phase_y), dim=-1)  # 将相位拼接在一起

        # amplitude = amplitude.permute(0, 3, 1, 2)  # → [B, 2C, H, W']
        # phase = phase.permute(0, 3, 1, 2)

        # 使用卷积层进行融合
        amplitude = self.fusion_amplitude(amplitude_x, amplitude_y)
        phase = self.fusion_phase(phase_x,phase_y)

        amplitude = amplitude.permute(0, 2, 3, 1)
        phase = phase.permute(0, 2, 3, 1)

        # 构造新的复数频谱
        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)

        x = torch.complex(real, imag)
        #############################################
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x

# MLP模块，用于Transformer结构中的前馈神经网络部分
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 分组线性层，用于FreqMLP模块
class GroupLinear(nn.Module):
    def __init__(self, in_features, out_features, num_groups):
        super(GroupLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.Tensor(num_groups, in_features // num_groups, out_features // num_groups))
        self.bias = nn.Parameter(torch.Tensor(num_groups, out_features // num_groups))
        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Kaiming (He) 初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # 使用统一的偏置初始化
        fan_in = self.in_features // self.num_groups
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        B, N, C = x.shape
        G = self.num_groups
        x = x.view(B, N, G, C // G)
        x = torch.einsum('b n g c, g c o -> b n g o', x, self.weight)
        x = x + self.bias.view(1, 1, G, -1)
        x = x.reshape(B, N, -1)
        return x

# FreqMLP模块，在频域中对特征进行处理
class FreqMLP(nn.Module):
    def __init__(self, dim, out_channels, num_groups):
        super(FreqMLP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, 1, 1, 0),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # nn.Conv2d(dim, out_channels, 1, 1, 0),
            )
        self.group_linear1 = GroupLinear(dim, dim, num_groups)
        self.gelu = nn.GELU()
        self.group_linear2 = GroupLinear(dim, dim, num_groups)
        self.fft_norm = 'ortho'

    def forward(self, x):
        x = self.conv1(x)
        batch, c, h, w = x.shape
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm) # (batch, c, h, w/2+1, 2) 将输入张量进行实部和虚部的傅里叶变换，得到复数表示的频率域数据ffted
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1) # 将频率域数据的实部和虚部分别堆叠起来
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:]) # 将堆叠后的数据拉平 (batch, c*2, h, w/2+1)
        ffted = ffted.view(batch, -1, h * (w // 2 + 1)).permute(0, 2, 1).contiguous() # (batch, h*(w/2+1), c*2)

        ffted = self.group_linear1(ffted)
        ffted = self.gelu(ffted)
        ffted = self.group_linear2(ffted)
        

        ffted = ffted.permute(0, 2, 1).contiguous()  # (batch, c*2, h*(w/2+1))
        ffted = ffted.view(batch, c * 2, h, w // 2 + 1)  # (batch, c*2, h, w/2+1)


        ffted = ffted.view(batch, c, 2, h, w // 2 + 1)
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()  # (batch, c, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])  # (batch, c, h, w/2+1)
        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        output = self.conv2(output)
        return output

# Stem模块，用于初始的特征提取和生成图像token, 输出尺寸应该为原始图像尺寸的一半
class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        x = self.conv(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class PatchEmbedStem(nn.Module):
    """Patch Embedding Stem：划分图像为非重叠小块"""
    def __init__(self, in_channels, embed_dim, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels*(patch_size**2), embed_dim,
            kernel_size=3,
            stride=1,  # 非重叠划分
            padding=1
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        H_ = H // self.patch_size
        W_ = W // self.patch_size
        x = x.view(B, C, H_, self.patch_size, W_, self.patch_size)  # (B, C, H//patch_size, patch_size, W//patch_size, patch_size)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()  # (B, C, H//patch_size, W//patch_size, patch_size, patch_size)        
        x = x.view(B, C*self.patch_size*self.patch_size, H_, W_,)  
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        x = self.norm(x)
        return x, H_, W_

# # 原版GFNet中的Patch Embedding模块
# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x

# Transformer结构中的基本Block模块
class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

# 适配FreqMLP的Transformer Block模块
class FreqBlock(nn.Module):
    def __init__(self, dim, out_channels, patch_size=8, drop_path=0., norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FreqMLP(dim=dim, out_channels=out_channels, num_groups=4)
    def forward(self, x, y):
        # x为图像

        x = self.norm1(x)
        y = self.norm1(y)
        x = self.filter(x,y)

        x = x + self.drop_path(x)
        x = self.norm2(x)
        x = x.view(x.size(0), -1, int(math.sqrt(x.size(1))), int(math.sqrt(x.size(1))))
        x = x + self.drop_path(self.mlp(x))
        return x

class PhaseAmplitudeAlignmentBlock(nn.Module):
    def __init__(self, IN_CH_HSI, IN_CH_MSI, stem_hidden_dim, drop_path=0., norm_layer=nn.LayerNorm, img_size=128):
        super().__init__()
        self.patch_size = 8
        self.squeeze_channels = nn.Conv2d(IN_CH_MSI+IN_CH_HSI, stem_hidden_dim, 1, 1, 0)
        # self.stem = PatchEmbedStem(in_channels=stem_hidden_dim, embed_dim=stem_hidden_dim,patch_size=8)
        
        self.stem_x = PatchEmbedStem(in_channels=IN_CH_HSI, embed_dim=stem_hidden_dim,patch_size=self.patch_size)
        self.stem_y = PatchEmbedStem(in_channels=IN_CH_MSI, embed_dim=stem_hidden_dim,patch_size=self.patch_size)

        self.blk = FreqBlock(
            dim=stem_hidden_dim, 
            out_channels=stem_hidden_dim*(self.patch_size**2), 
            drop_path=drop_path, 
            patch_size=self.patch_size, 
            norm_layer=norm_layer,
            h=img_size//self.patch_size,
            w=img_size//(2 * self.patch_size) + 1)
        # self.patch_embed = PatchEmbed(
        #         img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim[0])
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
        self.conv = nn.Conv2d(stem_hidden_dim, IN_CH_HSI*(self.patch_size**2), kernel_size=1, stride=1, padding=0)
    def forward(self, x, y): # x: sr_hsi, y:hr_msi
        b,c,h,w = x.shape
        _,cy,_,_ = y.shape

        x,_,_ = self.stem_x(x)
        y,_,_ = self.stem_y(y)

        # x = torch.cat([x,y],dim=1)
        # x = self.squeeze_channels(x)
        # x, _, _ = self.stem(x) 
        
        x = self.blk(x,y)

        ##########################
        ## 现在改到还原原始尺寸 ##
        ##########################
        x = self.conv(x)

        x = x.view(b, c, self.patch_size, self.patch_size, h//self.patch_size, w//self.patch_size)  # (B, C, patch_size, patch_size, H//patch_size, W//patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()  # (B, C, H//patch_size, patch_size, W//patch_size, patch_size)
        x = x.view(b, c, h, w)  # (B, C, H, W)
        return x

if __name__ == "__main__":
    # 定义输入参数
    batch_size = 1
    stem_hidden_dim = 64  # Stem 隐藏层维度
    out_channels = 31  # 输出特征维度
    image_size = (512, 512)  # 输入图像尺寸 (H, W)

    model = PhaseAmplitudeAlignmentBlock(
        IN_CH_HSI=out_channels, 
        IN_CH_MSI=3,
        stem_hidden_dim=stem_hidden_dim, 
        drop_path=0.1, 
        img_size=512).to('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化输入张量
    x = torch.randn(batch_size, out_channels, *image_size).to('cuda' if torch.cuda.is_available() else 'cpu')  # 输入形状为 (B, C, H, W)
    y = torch.randn(batch_size, 3, *image_size).to('cuda' if torch.cuda.is_available() else 'cpu')
    # 前向传播
    output = model(x,y)

    # 打印输出形状
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)

    
    # 模型参数量统计
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    # 定义损失函数和优化器
    criterion = torch.nn.L1Loss()  # L1 损失
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adam 优化器
    # 计算损失
    loss = criterion(output, x)

    # 反向传播
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数
