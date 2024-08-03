import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
import numpy as np


class EqualizedWeight(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.c = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=0):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding=0):
        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim)
        )

    def forward(self, x):
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        return self.mapping(x)


class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, demodulate=True, eps=1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2

        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x, s):
        b, _, h, w = x.shape
        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)
        return x.reshape(-1, self.out_features, h, w)


class StyleBlock(nn.Module):
    def __init__(self, W_DIM, in_features, out_features):
        super().__init__()
        self.to_styles = EqualizedLinear(W_DIM, in_features, bias=1)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):
        s = self.to_styles(w)
        # print(x.shape, s.shape)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])


class ToRGB(nn.Module):
    def __init__(self, W_DIM, features):
        super().__init__()
        self.to_style = EqualizedLinear(W_DIM, features, bias=1)
        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])


class PathLengthPenalty(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w, x):
        device = x.device
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)

        output = (x * y).sum() / sqrt(image_size)
        sqrt(image_size)

        gradients, *_ = torch.autograd.grad(outputs=output, inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)
        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            loss = torch.mean((norm - 1) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1-self.beta)
        self.steps.add_(1.)

        return loss


class GeneratorBlock(nn.Module):
    def __init__(self, W_DIM, in_features, out_features):
        super().__init__()
        self.style_block1 = StyleBlock(W_DIM, in_features, out_features)
        self.style_block2 = StyleBlock(W_DIM, out_features, out_features)
        self.to_rgb = ToRGB(W_DIM, out_features)

    def forward(self, x, w, noise):
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])
        rgb = self.to_rgb(x, w)
        return x, rgb


class Generator(nn.Module):
    def __init__(self, log_resolution, W_DIM, n_features=8, max_features=256):
        super().__init__()
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        # self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))
        # self.style_block = StyleBlock(W_DIM, features[0], features[0])
        # self.to_rgb = ToRGB(W_DIM, features[0])

        self.initial_constant_layers = []
        self.style_block_layers = nn.ModuleList([])
        self.to_rgb_layers = nn.ModuleList([])

        for i in range(0, self.n_blocks - 1):
            init_features = features[i]
            self.initial_constant_layers.append(nn.Parameter(torch.randn((1, init_features, 4, 4))))
            self.style_block_layers.append(StyleBlock(W_DIM, init_features, init_features))
            self.to_rgb_layers.append(ToRGB(W_DIM, init_features))

        blocks = [GeneratorBlock(W_DIM, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, w, input_noise, device, step):
        batch_size = w.shape[1]

        x = self.initial_constant_layers[step-1].expand(batch_size, -1, -1, -1).half().to(device)
        x = self.style_block_layers[step-1](x, w[0], input_noise[0][1])
        rgb = self.to_rgb_layers[step-1](x, w[0])

        control = 1
        for i in range(step, self.n_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, rgb_new = self.blocks[i - 1](x, w[control], input_noise[control])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear") + rgb_new
            control += 1

        return torch.tanh(rgb)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            EqualizedConv2d(in_features, out_features, kernel_size=1)
        )

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.scale = 1 / sqrt(2)

    def forward(self, x):
        residual = self.residual(x)

        x = self.block(x)
        x = self.down_sample(x)
        return (x + residual) * self.scale


class Discriminator(nn.Module):
    def __init__(self, log_resolution, n_features=8, max_features=256):
        super().__init__()
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True)
        )
        n_blocks = len(features) - 1
        self.n_blocks = n_blocks
        # blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        # self.blocks = nn.Sequential(*blocks)
        self.blocks = nn.ModuleList([])
        for i in range(n_blocks):
            self.blocks.append(DiscriminatorBlock(features[i], features[i + 1]))

        self.conv_layers = nn.ModuleList([])
        self.final_layers = nn.ModuleList([])
        for i in range(1, n_blocks + 1):
            final_features = features[i] + 1
            # print('i: ', final_features)
            self.conv_layers.append(EqualizedConv2d(final_features, final_features, 3))
            self.final_layers.append(EqualizedLinear(2 * 2 * final_features, 1))

        # final_features = features[-1] + 1
        # self.conv = EqualizedConv2d(final_features, final_features, 3)
        # self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, step):
        x = self.from_rgb(x)
        for i in range(step):
            x = self.blocks[i](x)

        x = self.minibatch_std(x)
        x = self.conv_layers[step - 1](x)
        x = x.reshape(x.shape[0], -1)
        return self.final_layers[step - 1](x)

