"""
Containing the source code of the components of your model. 
Each component must be implemented as a class or a function in PyTorch or TensorFlow/Keras. 
Your implementation of the model should not depend on NumPy (or other Python libraries not part of PyTorch / TensorFlow) in any way unless otherwise approved by the teaching staff.
"""

from dataset import *

# =============================================================================
# MODULES   compact StyleGAN2: mapping network -> style-modulated convs + noise -> skip-RGB generator,
#           residual discriminator with minibatch-stddev layer, adaptive augmentation
# =============================================================================
SQ2 = math.sqrt(2)


def ch(res):
    return min(CHANNEL_BASE // res, MAX_CH)


class EqLinear(nn.Module):
    """Linear layer with equalised learning rate (weights rescaled at run time)."""
    def __init__(self, i, o, bias_init=0.0, lr_mul=1.0, act=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i) / lr_mul)
        self.bias = nn.Parameter(torch.full([o], float(bias_init)))
        self.w_gain, self.b_gain, self.act = lr_mul / math.sqrt(i), lr_mul, act

    def forward(self, x):
        x = F.linear(x, self.weight * self.w_gain, self.bias * self.b_gain)
        return F.leaky_relu(x, 0.2) * SQ2 if self.act else x


class EqConv2d(nn.Module):
    """Plain conv with equalised learning rate (discriminator)."""
    def __init__(self, i, o, k, bias=True, act=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i, k, k))
        self.bias = nn.Parameter(torch.zeros(o)) if bias else None
        self.gain, self.pad, self.act = 1 / math.sqrt(i * k * k), k // 2, act

    def forward(self, x):
        x = F.conv2d(x, self.weight * self.gain, self.bias, padding=self.pad)
        return F.leaky_relu(x, 0.2) * SQ2 if self.act else x


class ModConv2d(nn.Module):
    """Weight (de)modulation. The style w becomes a per-input-channel scale s of a shared kernel;
    demodulation then renormalises every output channel so activations keep unit variance."""
    def __init__(self, cin, cout, k, demod=True, up=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin, k, k))
        self.affine = EqLinear(W_DIM, cin, bias_init=1.0)        # w -> style s
        self.gain, self.pad, self.demod, self.up = 1 / math.sqrt(cin * k * k), k // 2, demod, up

    def forward(self, x, w):
        s = self.affine(w)                                        # [B, cin]
        W = self.weight * self.gain                               # [cout, cin, k, k]
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.conv2d(x * s[:, :, None, None], W, padding=self.pad)
        if self.demod:
            d = torch.rsqrt(s.square() @ W.square().sum([2, 3]).t() + 1e-8)    # [B, cout]
            x = x * d[:, :, None, None]
        return x


class StyleConv(nn.Module):
    """modulated 3x3 conv (optionally 2x upsampling) + per-pixel noise + bias + leaky ReLU"""
    def __init__(self, cin, cout, up=False):
        super().__init__()
        self.conv = ModConv2d(cin, cout, 3, up=up)
        self.noise_strength = nn.Parameter(torch.zeros([]))
        self.bias = nn.Parameter(torch.zeros(cout))

    def forward(self, x, w, noise_mode="random"):
        x = self.conv(x, w)
        if noise_mode == "random":
            x = x + torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) * self.noise_strength
        return F.leaky_relu(x + self.bias.view(1, -1, 1, 1), 0.2) * SQ2


class ToImg(nn.Module):
    """1x1 modulated conv -> 1-channel (greyscale) image contribution"""
    def __init__(self, cin):
        super().__init__()
        self.conv = ModConv2d(cin, 1, 1, demod=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, w):
        return self.conv(x, w) + self.bias.view(1, -1, 1, 1)


class Mapping(nn.Module):
    """z -> w. Pixel-norm on z, then a small MLP with a 100x smaller learning rate."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(*[EqLinear(Z_DIM if i == 0 else W_DIM, W_DIM, lr_mul=0.01, act=True)
                                   for i in range(MAP_LAYERS)])

    def forward(self, z):
        return self.net(z * torch.rsqrt(z.square().mean(1, keepdim=True) + 1e-8))


class Generator(nn.Module):
    def __init__(self, res):
        super().__init__()
        self.res, self.log2 = res, int(math.log2(res))
        self.num_ws = 1 + 2 * (self.log2 - 2)                     # one style per conv layer
        self.mapping = Mapping()
        self.const = nn.Parameter(torch.randn(ch(4), 4, 4))       # learned 4x4 starting tensor (no input image)
        self.conv4, self.rgb4 = StyleConv(ch(4), ch(4)), ToImg(ch(4))
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(conv0=StyleConv(ch(2 ** (i - 1)), ch(2 ** i), up=True),
                               conv1=StyleConv(ch(2 ** i), ch(2 ** i)),
                               rgb=ToImg(ch(2 ** i))))
            for i in range(3, self.log2 + 1)])
        self.register_buffer("w_avg", torch.zeros(W_DIM))         # running mean of w (for truncation)

    def make_ws(self, z, mix_prob=0.0, psi=1.0):
        """z -> per-layer styles [B, num_ws, W_DIM], with optional style mixing and truncation."""
        w = self.mapping(z)
        if self.training:
            self.w_avg.copy_(w.detach().mean(0).lerp(self.w_avg, 0.995))
        ws = w[:, None].repeat(1, self.num_ws, 1)
        if mix_prob > 0:                                          # style mixing: later layers take a second w
            B, dev = z.shape[0], z.device
            w2 = self.mapping(torch.randn_like(z))[:, None]
            cut = torch.randint(1, self.num_ws, (B, 1), device=dev)
            swap = (torch.arange(self.num_ws, device=dev)[None] >= cut) & (torch.rand(B, 1, device=dev) < mix_prob)
            ws = torch.where(swap[:, :, None], w2.expand_as(ws), ws)
        if psi != 1.0:                                            # truncation trick (inference only)
            ws = self.w_avg + psi * (ws - self.w_avg)
        return ws

    def synthesis(self, ws, noise_mode="random"):
        x = self.const[None].expand(ws.shape[0], -1, -1, -1)
        x = self.conv4(x, ws[:, 0], noise_mode)
        img = self.rgb4(x, ws[:, 0])
        for k, b in enumerate(self.blocks):
            x = b["conv0"](x, ws[:, 1 + 2 * k], noise_mode)
            x = b["conv1"](x, ws[:, 2 + 2 * k], noise_mode)
            img = F.interpolate(img, scale_factor=2, mode="bilinear", align_corners=False) + b["rgb"](x, ws[:, 2 + 2 * k])
        return img

    def forward(self, z, psi=1.0, mix_prob=0.0, noise_mode="random"):
        return self.synthesis(self.make_ws(z, mix_prob, psi), noise_mode)


class MinibatchStd(nn.Module):
    """Appends one feature map holding the average per-feature std within each group of images.
    It lets D see how diverse a batch is, so a generator that outputs near-identical images is punished."""
    def __init__(self, group=4):
        super().__init__()
        self.group = group

    def forward(self, x):
        B, C, H, W = x.shape
        G_ = min(self.group, B)
        while B % G_:
            G_ -= 1
        y = x.reshape(G_, B // G_, C, H, W)
        y = y - y.mean(0, keepdim=True)
        y = (y.square().mean(0) + 1e-8).sqrt().mean([1, 2, 3])   # [B // G_]
        y = y.view(-1, 1, 1, 1).repeat(G_, 1, H, W)               # [B, 1, H, W]
        return torch.cat([x, y], 1)


class DBlock(nn.Module):
    """residual block that halves the resolution"""
    def __init__(self, cin, cout):
        super().__init__()
        self.conv0 = EqConv2d(cin, cin, 3)
        self.conv1 = EqConv2d(cin, cout, 3)
        self.skip = EqConv2d(cin, cout, 1, bias=False, act=False)

    def forward(self, x):
        y = F.avg_pool2d(self.conv1(self.conv0(x)), 2)
        return (y + F.avg_pool2d(self.skip(x), 2)) / SQ2


class Discriminator(nn.Module):
    def __init__(self, res):
        super().__init__()
        log2 = int(math.log2(res))
        self.from_img = EqConv2d(1, ch(res), 1)
        self.blocks = nn.Sequential(*[DBlock(ch(2 ** i), ch(2 ** (i - 1))) for i in range(log2, 2, -1)])
        self.mbstd = MinibatchStd(4)
        self.conv = EqConv2d(ch(4) + 1, ch(4), 3)
        self.fc = EqLinear(ch(4) * 16, ch(4), act=True)
        self.out = EqLinear(ch(4), 1)

    def forward(self, img):
        x = self.conv(self.mbstd(self.blocks(self.from_img(img))))
        return self.out(self.fc(x.flatten(1))).squeeze(1)


def augment(x, p):
    """Differentiable augmentation, each image augmented with probability p (adaptive, see TRAINING).
    Small shifts / zooms / rotations and a brightness gain. No flips: a left-right flip of a sagittal
    slice would swap front and back of the head. Black stays black."""
    if p <= 0:
        return x
    B, dev = x.shape[0], x.device
    u = lambda: torch.rand(B, device=dev) * 2 - 1
    g = (torch.rand(B, device=dev) < p).float()
    ang, scl = u() * (10 * math.pi / 180) * g, torch.exp(u() * 0.1 * g)
    tx, ty = u() * 0.25 * g, u() * 0.25 * g                       # +-12.5% of the image size
    cos, sin = torch.cos(ang) / scl, torch.sin(ang) / scl
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x + 1, grid, mode="bilinear", padding_mode="zeros", align_corners=False) - 1
    gain = torch.exp(u()[:, None, None, None] * 0.22 * (torch.rand(B, 1, 1, 1, device=dev) < p).float())
    return (((x + 1) / 2) * gain).clamp(0, 1) * 2 - 1


def build_models(res):
    G, D = Generator(res).to(DEVICE), Discriminator(res).to(DEVICE)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    print(f"G {sum(p.numel() for p in G.parameters()) / 1e6:.1f}M params | "
          f"D {sum(p.numel() for p in D.parameters()) / 1e6:.1f}M params | {G.num_ws} style layers")
    return G, D, G_ema