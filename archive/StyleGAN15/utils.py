import torch
import os
from torchvision.utils import save_image

LOG_RESOLUTION = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_examples(gen, steps, z_dim, n=100, device='cpu', uniq_path="saved_examples"):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, z_dim).to(device)
            img = gen(noise, alpha, steps)
            path = f'{uniq_path}/step{steps}'
            if not os.path.exists(path):
                os.makedirs(path)
            save_image(img*0.5+0.5, f"{path}/img_{i}.png")
    gen.train()


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    print(f'beta {beta.shape}')
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

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


def save_model(gen, crit, opt_gen, opt_crit, alpha,
               z_dim, w_dim, in_channels, img_channels,
               step, ada_p=0, identifier='STYLEGAN15'):
    if not os.path.exists(f'models/{identifier}'):
        os.makedirs(f'models/{identifier}')
    torch.save({
        'generator': gen.state_dict(),
        'discriminator': crit.state_dict(),
        'g_optim': opt_gen.state_dict(),
        'd_optim': opt_crit.state_dict(),
        'ada_prob': ada_p,
        'parameters': (step, alpha, z_dim, w_dim, in_channels, img_channels),
    }, f'./models/{identifier}/trained.pth')


def load_model(gen, identifier, with_critic=False, crit=None, with_optim=False, opt_gen=None, opt_crit=None):
    print('identifier', identifier)
    model = torch.load(f'./models/{identifier}/trained.pth')
    gen.load_state_dict(model['generator'])
    if with_critic:
        crit.load_state_dict(model['discriminator'])
    step, alpha, z_dim, w_dim, in_channels, img_channels = model['parameters']
    ada_prob = model['ada_prob']
    if with_optim:
        opt_gen.load_state_dict(model['g_optim'])
        opt_crit.load_state_dict(model['d_optim'])

    return step, alpha, ada_prob


def load_all(identifier):
    return torch.load(f'./models/{identifier}/trained.pth')