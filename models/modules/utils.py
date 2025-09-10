import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from torchvision.ops import DeformConv2d


class DeformableConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        edge=True,
    ):
        super(DeformableConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=bias,
        )
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        if edge:
            self.offset_edge = nn.Conv2d(
                1,
                2 * self.kernel_size[0] * self.kernel_size[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )

    def forward(self, x, edge=None):
        offsets = self.offset_conv(x)
        if edge is not None:
            offsets_e = self.offset_edge(edge)
            offsets = offsets * offsets_e

        x = self.deform_conv(x, offsets)
        x = self.bn(x)
        x = self.act(x)
        return x


def _pair(x):
    if isinstance(x, int):
        return (x, x)
    return x


def image2patches(
    image,
    grid_h=2,
    grid_w=2,
    patch_ref=None,
    transformation="b c (hg h) (wg w) -> (b hg wg) c h w",
):
    if patch_ref is not None:
        grid_h, grid_w = (
            image.shape[-2] // patch_ref.shape[-2],
            image.shape[-1] // patch_ref.shape[-1],
        )
    patches = rearrange(image, transformation, hg=grid_h, wg=grid_w)
    return patches


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class LayerNorm(nn.LayerNorm):
    def __init__(self, inchannels):
        super().__init__(inchannels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


class SA(nn.Module):

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.3,
        ffn_expansion_factor: int = 4,
    ):
        super(SA, self).__init__()
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim**-0.5

        self.norm1 = LayerNorm(channels)
        self.norm2 = LayerNorm(channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_dropout = nn.Dropout(dropout)

        self.ffn = FeedForward(
            channels, ffn_expansion_factor=ffn_expansion_factor, bias=False
        )

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        x_norm1 = self.norm1(x)
        if mask is not None:
            mask = F.interpolate(
                mask, size=(h, w), mode="bilinear", align_corners=False
            )
        q = self.conv1(x_norm1)
        k = self.conv2(x_norm1)
        v = self.conv3(x_norm1)
        k = k * mask if mask is not None else k
        v = v * mask if mask is not None else v
        q = rearrange(q, "b (head d) h w -> b head d (h w)", head=self.num_heads)
        k = rearrange(k, "b (head d) h w -> b head d (h w)", head=self.num_heads)
        v = rearrange(v, "b (head d) h w -> b head d (h w)", head=self.num_heads)

        dots = (q@ k.transpose(-2, -1)) * self.scale

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b head d (h w) -> b (head d) h w", head=self.num_heads, h=h, w=w)

        attn_out = self.proj_dropout(self.proj(out))

        x = x + attn_out

        x_norm2 = self.norm2(x)

        ffn_out = self.ffn(x_norm2)

        x = x + ffn_out

        return x


def gauss_kernel(channels=3, kernel_size=5, sigma=1.0, device=None):
    coords = torch.arange(kernel_size, device=device).float() - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    gg = g.ger(g)
    gg = gg.view(1, 1, kernel_size, kernel_size)
    return gg.repeat(channels, 1, 1, 1)


def conv_gauss(x, kernel):
    kernel = kernel.to(x.device)
    padding = kernel.shape[-1] // 2
    return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])


def downsample(x):
    return F.avg_pool2d(x, 2)


def upsample(x, channels):
    return F.interpolate(x, scale_factor=2, mode="nearest")


def make_laplace_pyramid(img, level, channels):
    device = img.device
    current = img
    pyr = []

    for _ in range(level):
        kernel = gauss_kernel(channels, device=device)
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down, channels)

        if up.shape[2:] != current.shape[2:]:
            up = F.interpolate(up, size=current.shape[2:])

        diff = current - up
        pyr.append(diff)
        current = down

    pyr.append(current)
    return pyr
