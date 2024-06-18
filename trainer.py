from classes import Generator, Discriminator
# from torchinfo import summary
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
from load_data import load_data


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def gradient_penalty(D, xr, xf, batch_size):
    t = torch.rand(batch_size, 1)
    t = t.expand_as(xr)

    mid = t * xr + (1 - t) * xf

    mid.requires_grad_()

    pred = D(mid)
    grads = torch.autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
    return gp


def train_WGAN(dataloader):
    model_path = "./models/"

    batch_size = 64
    num_epochs = 30

    generator = Generator()
    discriminator = Discriminator()

    generator.apply(weights_init)
    discriminator.apply(weights_init)
    lr_g = 0.0001
    lr_d = 0.0001
    beta1 = 0.5
    d_updates = 3
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr_g, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr_d, betas=(beta1, 0.999))
    fixed_noise_input = torch.randn(batch_size, 64, 1, 1)
    # summary(generator, (64, 1, 1))
    # summary(discriminator, (3, 32, 32))

    print(generator)
    print(discriminator)

    train_losses_d = []
    train_losses_g = []

    for epoch in range(num_epochs):
        with tqdm.tqdm(dataloader, unit='batch') as tepoch:
            for i, data in enumerate(tepoch):
                train_loss_g = train_loss_d = 0

                generator.zero_grad()
                discriminator.zero_grad()

                real = data[0]

                d_success = discriminator(real)
                loss_r = -d_success.mean()

                noise = fixed_noise_input
                fake = generator(noise)
                d_failure = discriminator(fake)
                loss_f = d_failure.mean()

                # gp = gradient_penalty(discriminator, d_success, d_failure.detach(), batch_size)
                # loss_D = loss_r + loss_f + 0.2 * gp

                ###
                eps_shape = [batch_size] + [1] * (len(data[0].shape)-1)
                eps = torch.rand(eps_shape)
                fake = eps * real + (1-eps)*fake
                output = discriminator(fake)
                grad = torch.autograd.grad(outputs=output, inputs=fake,
                                            grad_outputs=torch.ones_like(output.size),
                                            create_graph=True, retain_graph=True, only_inputs=True,
                                            allow_unused=True)[0]
                d_grad_penalty = ((grad.norm(2,dim=1)-1)**2).mean()
                loss_D = (loss_r-loss_f) + d_grad_penalty.mean()*10
                ###

                loss_D.backward()
                optimizerD.step()

                z = torch.randn(batch_size, 2)
                xf = generator(z)
                predf = discriminator(xf)
                # maximize predf.mean()
                loss_G = -predf.mean()
                # optimize
                loss_G.backward()
                optimizerG.step()
