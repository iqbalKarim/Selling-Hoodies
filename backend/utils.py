import torch
from StyleGAN.classes import Generator
from torchvision.utils import make_grid

CHANNELS_IMG = 3
Z_DIM = 256
IN_CHANNELS = 256
W_DIM = 256


def generate_examples(gen, steps, n=5):
    alpha = 1.0
    imgs = []
    # for i in range(n):
    #     with torch.no_grad():
    #         noise = torch.randn(1, Z_DIM)
    #         imgs.append(gen(noise, alpha, steps))
    with torch.no_grad():
        noise = torch.randn(n, Z_DIM)
        imgs = gen(noise, alpha, steps)

    return imgs


def generate_grid(gen, step, cols=3, n=3):
    gen.eval()
    imgs = []
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(cols * cols, Z_DIM)
            samples = gen(noise, 1, step)
            grid = make_grid(samples, cols)
        imgs.append(grid)

    return imgs


def load_generator():
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG)
    model = torch.load("./models/trained.pth")
    gen.load_state_dict(model["generator"])
    gen.eval()
    return gen

def load_emnist_generator():
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG)
    model = torch.load("./models/emnist.pth")
    gen.load_state_dict(model["generator"])
    gen.eval()
    return gen

def generate_styleGAN1_examples(gen):
    return 'Gen these'