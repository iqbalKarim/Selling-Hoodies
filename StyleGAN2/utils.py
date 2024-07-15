import torch
from config import *
import os

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def get_noise(batch_size):

    noise = []
    resolution = 4

    for i in range(LOG_RESOLUTION):
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)

        noise.append((n1, n2))

        resolution *= 2

    return noise


def save_everything(critic, gen, path_length_penalty, mapping_network, opt_critic, opt_gen, opt_mapping_network, epoch):
    identifier = f"epoch{epoch}"
    if not os.path.exists(f'models2/{identifier}'):
        os.makedirs(f'models2/{identifier}')
    torch.save({
        'generator': gen.state_dict(),
        'discriminator': critic.state_dict(),
        'mapping': mapping_network.state_dict(),
        'plp': path_length_penalty.state_dict(),
        'g_optim': opt_gen.state_dict(),
        'd_optim': opt_critic.state_dict(),
        'map_optim': opt_mapping_network.state_dict()
    }, f'./models2/{identifier}/trained.pth')
