import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
import numpy as np

factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


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
        self.outputFeature = out_features
        self.input_features = in_features
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        f = nn.Linear(x.shape[1], out_features=self.input_features).to('cuda')
        x = f(x).to('cuda')
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


class GenBlock(nn.Module):
    def __init__(self, w_dim, in_features, out_features):
        super().__init__()
        self.style_block1 = StyleBlock(w_dim, in_features, out_features)
        self.style_block2 = StyleBlock(w_dim, out_features, out_features)
        self.to_rgb = ToRGB(w_dim, out_features)

    def forward(self, x, w, noise):
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])
        rgb = self.to_rgb(x, w)
        return x, rgb


class DiscBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            EqualizedConv2d(in_features, out_features, kernel_size=1)
        )

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.scale = 1 / sqrt(2)

    def forward(self, x):
        residual = self.residual(x)

        x = self.block(x)
        x = self.down_sample(x)
        return (x + residual) * self.scale


class Generator(nn.Module):
    def __init__(self, log_resolution, w_dim, n_features=8, max_features=256):
        super().__init__()
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution, -1, -1)]
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        self.style_block = StyleBlock(w_dim, features[0], features[0])
        self.initial_rgb = ToRGB(w_dim, features[0])
        self.prog_blocks, self.rgb_layers = (nn.ModuleList([]), nn.ModuleList([self.initial_rgb]))

        print('features', features)
        self.n_blocks = len(features)
        for i in range(1, self.n_blocks):
            in_feat = features[i - 1]
            out_feat = features[i]
            print('IN: ', in_feat, 'OUT: ', out_feat )
            self.prog_blocks.append(GenBlock(w_dim, in_feat, out_feat))
            self.rgb_layers.append(ToRGB(w_dim, out_feat))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, w, input_noise, alpha, steps):
        cur_step = len(self.prog_blocks) - steps - 1
        batch_size = w.shape[1]
        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.initial_rgb(x, w[0])

        print('current step', cur_step, steps)

        if steps == 0:
            return torch.tanh(rgb)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            # upscaled = F.interpolate(x, scale_factor=float(2), mode='bilinear')
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            print(x.shape, w[step].shape)
            x, rgb_new = self.prog_blocks[step - 1](x, w[step], input_noise[step])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear") + rgb_new
            print(rgb.shape)

        final_upscaled = self.rgb_layers[steps - 1](x, w[0])


        return self.fade_in(alpha, final_upscaled, rgb)


class Discriminator(nn.Module):
    def __init__(self, log_resolution, n_features=64, max_features=256):
        super().__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True)
        )
        features = [int(min(max_features, n_features * (2 ** i))) for i in range(log_resolution + 1)]
        self.n_blocks = len(features) - 1

        for i in range(1, self.n_blocks):
            in_feat = features[i]
            out_feat = features[i + 1]
            self.prog_blocks.append(DiscBlock(in_feat, out_feat))
            self.rgb_layers.append(nn.Sequential(EqualizedConv2d(3, in_feat, 1),
                                                 nn.LeakyReLU(0.2, True)))
        self.rgb_layers.append(self.from_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        final_features = features[-1] + 1
        self.conv = EqualizedConv2d(final_features, final_features, 2)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps

        if steps == 0:
            x = self.from_rgb(x)
            x = self.minibatch_std(x)
            x = self.conv(x)
            x = x.reshape(x.shape[0], -1)
            return self.final(x)

        x = self.leaky(self.rgb_layers[cur_step](x))
        # downscaled = self.leaky(self.rgb_layers[cur_step + 1](x))
        x = self.prog_blocks[cur_step](x)
        # x = self.fade_in(alpha, downscaled, x)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            x = self.prog_blocks[step](x)

        x = self.minibatch_std(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)


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

