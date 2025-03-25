import torch
from torch import nn as nn
from basicsr.archs.arch_util import Upsample, trunc_normal_, to_2tuple
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn import functional as F
from einops import rearrange

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)  # 在空间方向执行全局平均池化: (B,C,H,W)-->(B,C,1,1)
        y = y.squeeze(-1).permute(0, 2, 1)  # 将通道描述符去掉一维,便于在通道上执行卷积操作:(B,C,1,1)-->(B,C,1)-->(B,1,C)
        y = self.conv(y)  # 在通道维度上执行1D卷积操作,建模局部通道之间的相关性: (B,1,C)-->(B,1,C)
        y = self.sigmoid(y)  # 生成权重表示: (B,1,C)
        y = y.permute(0, 2, 1).unsqueeze(-1)  # 重塑shape: (B,1,C)-->(B,C,1)-->(B,C,1,1)
        return y

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=5, stride=1, padding=2)
        '''self.conv5x5 = nn.Sequential(
            nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        )'''
        self.ChannelAttention = ECAAttention()

    def forward(self, x):
        # (B,C,H,W)
        b, c, h, w = x.size()

        ### 坐标注意力模块  ###
        group_x = x.reshape(b * self.groups, -1, h, w)  # 在通道方向上将输入分为G组: (B,C,H,W)-->(B*G,C/G,H,W)
        x_h = self.pool_h(group_x)  # 使用全局平均池化压缩水平空间方向: (B*G,C/G,H,W)-->(B*G,C/G,H,1)
        x_w = self.pool_w(group_x).permute(0, 1, 3,
                                           2)  # 使用全局平均池化压缩垂直空间方向: (B*G,C/G,H,W)-->(B*G,C/G,1,W)-->(B*G,C/G,W,1)
        hw = self.conv1x1(
            torch.cat([x_h, x_w], dim=2))  # 将水平方向和垂直方向的全局特征进行拼接: (B*G,C/G,H+W,1), 然后通过1×1Conv进行变换,来编码空间水平和垂直方向上的特征
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 沿着空间方向将其分割为两个矩阵表示: x_h:(B*G,C/G,H,1); x_w:(B*G,C/G,W,1)

        ### 1×1分支和3×3分支的输出表示  ###
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        # 通过水平方向权重和垂直方向权重调整输入,得到1×1分支的输出: (B*G,C/G,H,W) * (B*G,C/G,H,1) * (B*G,C/G,1,W)=(B*G,C/G,H,W)
        x2 = self.conv3x3(group_x)  # 通过3×3卷积提取局部上下文信息: (B*G,C/G,H,W)-->(B*G,C/G,H,W)
        x3 = self.conv5x5(group_x)
        # 通道注意力权重
        c1 = self.ChannelAttention(x1.clone())
        c2 = self.ChannelAttention(x2.clone())
        c3 = self.ChannelAttention(x3.clone())
        group_x = group_x * (c1 + c2 + c3)

        ### 跨空间学习 ###
        ## 1×1分支生成通道描述符来调整3×3分支和5×5分支的输出
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # 对1×1分支的输出执行平均池化,然后通过softmax获得归一化后的通道描述符:
        # (B*G,C/G,H,W)-->agp-->(B*G,C/G,1,1)-->reshape-->(B*G,C/G,1)-->permute-->(B*G,1,C/G)
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x13 = x3.reshape(b * self.groups, c // self.groups, -1)
        # 将3×3分支的输出进行变换,以便与1×1分支生成的通道描述符进行相乘: (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        y1_2 = torch.bmm(x11, x12)  # 调整3×3分支
        y1_3 = torch.bmm(x11, x13)  # 调整5×5分支

        ## 3×3分支生成通道描述符来调整1×1分支和5×5分支的输出
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # 对3×3分支的输出执行平均池化,然后通过softmax获得归一化后的通道描述符:
        # (B*G,C/G,H,W)-->agp-->(B*G,C/G,1,1)-->reshape-->(B*G,C/G,1)-->permute-->(B*G,1,C/G)
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        # b*g, c//g, hw  # 将1×1分支的输出进行变换,以便与3×3分支生成的通道描述符进行相乘: (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        x23 = x3.reshape(b * self.groups, c // self.groups, -1)
        # 将5×5分支的输出进行变换,以便与3×3分支生成的通道描述符进行相乘: (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        y2_1 = torch.bmm(x21, x22)  # 调整1×1分支
        y2_3 = torch.bmm(x21, x23)  # 调整5×5分支

        ## 5×5分支生成通道描述符来调整1×1分支和3×3分支的输出
        x31 = self.softmax(self.agp(x3).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # 对5×5分支的输出执行平均池化,然后通过softmax获得归一化后的通道描述符:
        # (B*G,C/G,H,W)-->agp-->(B*G,C/G,1,1)-->reshape-->(B*G,C/G,1)-->permute-->(B*G,1,C/G)
        x32 = x1.reshape(b * self.groups, c // self.groups, -1)
        # 将1×1分支的输出进行变换,以便与5×5分支生成的通道描述符进行相乘: (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        x33 = x2.reshape(b * self.groups, c // self.groups, -1)
        # 将3×3分支的输出进行变换,以便与5×5分支生成的通道描述符进行相乘: (B*G,C/G,H,W)-->reshape-->(B*G,C/G,H*W)
        y3_1 = torch.bmm(x31, x32)  # 调整1×1分支
        y3_2 = torch.bmm(x31, x33)  # 调整3×3分支

        # 聚合两种尺度的空间位置信息, 通过sigmoid生成空间权重, 从而再次调整输入表示
        weights = (y1_2 + y1_3 + y2_1 + y2_3 + y3_1 + y3_2).reshape(b * self.groups, 1, h, w)
        # 将两种尺度下的空间位置信息进行聚合: (B*G,1,H*W)-->reshape-->(B*G,1,H,W)
        weights_ = weights.sigmoid()  # 通过sigmoid生成权重表示: (B*G,1,H,W)
        out = (group_x * weights_).reshape(b, c, h, w)
        # 通过空间权重再次校准输入: (B*G,C/G,H,W)*(B*G,1,H,W)==(B*G,C/G,H,W)-->reshape(B,C,H,W)
        return out

class MyBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, exp_ratio=2, res_scale=0.1):
        super(MyBlock, self).__init__()

        self.res_scale = nn.Parameter(torch.ones(1) * res_scale)
        self.res_scale_conv = nn.Parameter(torch.ones(1) * res_scale)

        self.conv1 = DepthwiseSeparableConv(n_feats, n_feats, kernel_size, stride=1, padding=kernel_size // 2)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv(n_feats, n_feats, kernel_size, stride=1, padding=kernel_size // 2)
        self.ema = EMA(n_feats, factor=18)
        self.conv3 = nn.Conv2d(n_feats, n_feats, 1, stride=1, padding=0)

    def forward(self, x):
        residual = x

        # 卷积提取特征
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x * self.res_scale_conv + residual

        y = self.conv3(self.ema(x))
        # 最终残差连接
        x = y * self.res_scale + x
        return x


@ARCH_REGISTRY.register()
class MyArch(nn.Module):
    """Example architecture.


        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        upscale (int): Upsampling factor. Default: 4.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 kernel_size=3,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 img_range=255.,
                 act='relu',
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(MyArch, self).__init__()
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.kernel_size = kernel_size
        # Head module
        m_head = [nn.Conv2d(num_in_ch, num_feat, self.kernel_size, 1, 1)]

        # Body module
        m_body = [
            MyBlock(
                num_feat, self.kernel_size
            ) for _ in range(num_block)
        ]

        self.conv_after_body = nn.Conv2d(num_feat, num_feat, self.kernel_size, 1, 1)

        m_tail = [
            # Upsample(scale=2, num_feat=num_feat),
            DySample(num_feat, 2, 'lp'),
            nn.Conv2d(num_feat, num_out_ch, self.kernel_size, 1, 1)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.head(x)
        res = x

        x = self.body(x)

        x = self.conv_after_body(x)

        x = x + res
        x = self.tail(x)
        x = x / self.img_range + self.mean
        return x
